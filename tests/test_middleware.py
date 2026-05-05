"""Тести для DormouseClient middleware."""

import asyncio
from unittest.mock import MagicMock, patch

from dormouse.middleware import DormouseClient


def _make_mock_response(content: str):
    """Створює mock ChatCompletion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _make_mock_client(response_content: str = "This is a response."):
    """Створює mock OpenAI client."""
    mock_client = MagicMock()
    mock_client.__class__ = type("OpenAI", (), {})
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = MagicMock(
        return_value=_make_mock_response(response_content)
    )
    return mock_client


def _make_stream_chunks(texts: list[str]):
    """Створює mock streaming chunks."""
    chunks = []
    for text in texts:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = text
        chunks.append(chunk)
    # Фінальний chunk з content=None
    final = MagicMock()
    final.choices = [MagicMock()]
    final.choices[0].delta = MagicMock()
    final.choices[0].delta.content = None
    chunks.append(final)
    return chunks


class TestDormouseClientInit:
    def test_creates_with_defaults(self):
        mock = _make_mock_client()
        client = DormouseClient(mock)
        assert client._target == "cloud"
        assert client._log_savings is False

    def test_custom_config(self):
        mock = _make_mock_client()
        client = DormouseClient(
            mock, target="cloud", roles=("user", "system"),
            squeeze_system=True, log_savings=True,
        )
        assert client._target == "cloud"
        assert client._log_savings is True


class TestDormouseClientProxy:
    def test_getattr_proxies_to_original(self):
        """Невідомі атрибути проксяться на оригінальний клієнт."""
        mock = _make_mock_client()
        mock.models = MagicMock()
        mock.models.list = MagicMock(return_value=["gpt-4"])
        client = DormouseClient(mock)
        result = client.models.list()
        assert result == ["gpt-4"]


class TestDormouseClientSync:
    @patch("dormouse.middleware.squeeze")
    @patch("dormouse.middleware.map_to_en")
    @patch("dormouse.middleware.unsqueeze")
    def test_squeeze_and_unsqueeze_applied(self, mock_unsqueeze, mock_map, mock_squeeze):
        mock_squeeze.return_value = "нормалізований текст"
        mock_map.return_value = "normalized text"
        mock_unsqueeze.return_value = "відповідь українською"

        mock_openai = _make_mock_client("This is a response.")
        client = DormouseClient(mock_openai, target="cloud")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "шо там по деплою"}],
        )

        mock_squeeze.assert_called_once_with("шо там по деплою")
        mock_map.assert_called_once_with("нормалізований текст")
        mock_unsqueeze.assert_called_once_with("This is a response.", translate_fn=None)
        assert response.choices[0].message.content == "відповідь українською"

    @patch("dormouse.middleware.squeeze")
    @patch("dormouse.middleware.map_to_en")
    @patch("dormouse.middleware.unsqueeze")
    def test_system_message_skipped_by_default(self, mock_unsqueeze, mock_map, mock_squeeze):
        mock_squeeze.return_value = "нормалізований"
        mock_map.return_value = "normalized"
        mock_unsqueeze.return_value = "відповідь"

        mock_openai = _make_mock_client("Response.")
        client = DormouseClient(mock_openai, target="cloud")

        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "привіт"},
            ],
        )

        call_kwargs = mock_openai.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert messages[0]["content"] == "You are helpful."

    @patch("dormouse.middleware.squeeze")
    @patch("dormouse.middleware.map_to_en")
    @patch("dormouse.middleware.unsqueeze")
    def test_json_skipped(self, mock_unsqueeze, mock_map, mock_squeeze):
        mock_unsqueeze.return_value = "відповідь"

        mock_openai = _make_mock_client("Response.")
        client = DormouseClient(mock_openai, target="cloud")

        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": '{"key": "value"}'}],
        )

        mock_squeeze.assert_not_called()

    @patch("dormouse.middleware.squeeze", side_effect=Exception("boom"))
    @patch("dormouse.middleware.unsqueeze")
    def test_squeeze_error_passes_original(self, mock_unsqueeze, mock_squeeze):
        """Якщо squeeze впав — передаємо оригінальний текст."""
        mock_unsqueeze.return_value = "ok"
        mock_openai = _make_mock_client("Response.")
        client = DormouseClient(mock_openai, target="cloud")

        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "привіт"}],
        )

        call_kwargs = mock_openai.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert messages[0]["content"] == "привіт"

    @patch("dormouse.middleware.unsqueeze", side_effect=Exception("boom"))
    def test_unsqueeze_error_passes_original_response(self, mock_unsqueeze):
        """Якщо unsqueeze впав — повертаємо оригінальну відповідь."""
        mock_openai = _make_mock_client("Original response.")
        client = DormouseClient(mock_openai, target="cloud")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "шо там"}],
        )

        assert response.choices[0].message.content == "Original response."


class TestDormouseClientStreaming:
    @patch("dormouse.middleware.squeeze")
    @patch("dormouse.middleware.map_to_en")
    @patch("dormouse.middleware.unsqueeze", side_effect=lambda t: f"[ua]{t}[/ua]")
    def test_streaming_unsqueezes_by_sentence(self, mock_unsqueeze, mock_map, mock_squeeze):
        mock_squeeze.return_value = "normalized"
        mock_map.return_value = "mapped"

        chunks = _make_stream_chunks(["Hello", " world", ".", " Nice", " day", "."])
        mock_openai = _make_mock_client()
        mock_openai.chat.completions.create = MagicMock(return_value=iter(chunks))

        client = DormouseClient(mock_openai, target="cloud")
        result_texts = []
        for chunk in client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "привіт"}],
            stream=True,
        ):
            content = chunk.choices[0].delta.content
            if content is not None:
                result_texts.append(content)

        joined = "".join(result_texts)
        assert "[ua]" in joined

    @patch("dormouse.middleware.squeeze")
    @patch("dormouse.middleware.map_to_en")
    @patch("dormouse.middleware.unsqueeze", side_effect=Exception("boom"))
    def test_streaming_unsqueeze_error_passes_original(
        self, mock_unsqueeze, mock_map, mock_squeeze
    ):
        mock_squeeze.return_value = "normalized"
        mock_map.return_value = "mapped"

        chunks = _make_stream_chunks(["Hello."])
        mock_openai = _make_mock_client()
        mock_openai.chat.completions.create = MagicMock(return_value=iter(chunks))

        client = DormouseClient(mock_openai, target="cloud")
        result_texts = []
        for chunk in client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "привіт"}],
            stream=True,
        ):
            content = chunk.choices[0].delta.content
            if content is not None:
                result_texts.append(content)

        joined = "".join(result_texts)
        assert "Hello." in joined


class TestDormouseClientAsync:
    def test_async_client_detected(self):
        """AsyncOpenAI клієнт розпізнається автоматично."""
        mock_async = MagicMock()
        mock_async.__class__ = type("AsyncOpenAI", (), {})
        mock_async.chat = MagicMock()
        mock_async.chat.completions = MagicMock()

        client = DormouseClient(mock_async, target="cloud")
        assert client._is_async is True

    @patch("dormouse.middleware.squeeze", return_value="normalized")
    @patch("dormouse.middleware.map_to_en", return_value="mapped")
    @patch("dormouse.middleware.unsqueeze", return_value="відповідь")
    def test_async_create(self, mock_unsqueeze, mock_map, mock_squeeze):
        mock_response = _make_mock_response("Response.")

        mock_async = MagicMock()
        mock_async.__class__ = type("AsyncOpenAI", (), {})
        mock_async.chat = MagicMock()
        mock_async.chat.completions = MagicMock()

        async def mock_create(**kwargs):
            return mock_response

        mock_async.chat.completions.create = mock_create

        client = DormouseClient(mock_async, target="cloud")

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "привіт"}],
                )
            )
        finally:
            loop.close()

        assert result.choices[0].message.content == "відповідь"


class TestSavingsLogging:
    @patch("dormouse.middleware.squeeze", return_value="нормалізований")
    @patch("dormouse.middleware.map_to_en", return_value="normalized")
    @patch("dormouse.middleware.unsqueeze", return_value="відповідь")
    def test_log_savings_outputs_to_stderr(self, mock_unsqueeze, mock_map, mock_squeeze, capsys):
        mock_openai = _make_mock_client("Response.")
        client = DormouseClient(mock_openai, target="cloud", log_savings=True)

        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "шо там по деплою"}],
        )

        captured = capsys.readouterr()
        assert "[dormouse]" in captured.err

    @patch("dormouse.middleware.squeeze", return_value="нормалізований")
    @patch("dormouse.middleware.map_to_en", return_value="normalized")
    @patch("dormouse.middleware.unsqueeze", return_value="відповідь")
    def test_log_savings_disabled_silent(self, mock_unsqueeze, mock_map, mock_squeeze, capsys):
        mock_openai = _make_mock_client("Response.")
        client = DormouseClient(mock_openai, target="cloud", log_savings=False)

        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "шо там по деплою"}],
        )

        captured = capsys.readouterr()
        assert "[dormouse]" not in captured.err


class TestDormouseClientDetection:
    def test_anthropic_client_detected(self):
        """Anthropic клієнт розпізнається автоматично."""
        mock_anthropic = MagicMock()
        mock_anthropic.__class__ = type("Anthropic", (), {})
        mock_anthropic.messages = MagicMock()

        client = DormouseClient(mock_anthropic, target="cloud")
        assert client._provider == "anthropic"
        assert hasattr(client, "messages")

    def test_async_anthropic_client_detected(self):
        """AsyncAnthropic клієнт розпізнається автоматично."""
        mock_async = MagicMock()
        mock_async.__class__ = type("AsyncAnthropic", (), {})
        mock_async.messages = MagicMock()

        client = DormouseClient(mock_async, target="cloud")
        assert client._provider == "anthropic"
        assert client._is_async is True

    def test_openai_client_detected_as_openai(self):
        """OpenAI клієнт визначається як openai provider."""
        mock = _make_mock_client()
        client = DormouseClient(mock, target="cloud")
        assert client._provider == "openai"

    def test_unknown_client_passthrough(self):
        """Невідомий клієнт — pass-through з warning."""
        mock = MagicMock(spec=["close"])
        mock.__class__ = type("SomeUnknownClient", (), {})

        import logging
        with patch.object(logging.getLogger("dormouse"), "warning") as mock_warn:
            client = DormouseClient(mock, target="cloud")
            mock_warn.assert_called_once()
        assert client._provider == "unknown"
