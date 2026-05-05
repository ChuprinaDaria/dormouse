"""Тести для Anthropic proxy middleware."""

import asyncio
from unittest.mock import MagicMock, patch

from dormouse.anthropic_proxy import _AnthropicMessagesProxy
from dormouse.language_guard import LanguageGuard


def _make_anthropic_response(text: str):
    """Створює mock Anthropic Message response."""
    response = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = text
    response.content = [block]
    return response


def _make_anthropic_client(response_text: str = "This is a response."):
    """Створює mock Anthropic client."""
    mock_client = MagicMock()
    mock_client.__class__ = type("Anthropic", (), {})
    mock_client.messages = MagicMock()
    mock_client.messages.create = MagicMock(
        return_value=_make_anthropic_response(response_text)
    )
    return mock_client


class TestAnthropicProxySync:
    @patch("dormouse.anthropic_proxy.squeeze")
    @patch("dormouse.anthropic_proxy.map_to_en")
    @patch("dormouse.anthropic_proxy.unsqueeze")
    def test_squeeze_and_unsqueeze_applied(self, mock_unsqueeze, mock_map, mock_squeeze):
        mock_squeeze.return_value = "нормалізований текст"
        mock_map.return_value = "normalized text"
        mock_unsqueeze.return_value = "відповідь українською"

        mock_client = _make_anthropic_client("This is a response.")
        guard = LanguageGuard(roles=("user",), squeeze_system=False)
        proxy = _AnthropicMessagesProxy(
            mock_client.messages, guard, target="cloud", log_savings=False,
        )

        response = proxy.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "шо там по деплою"}],
        )

        mock_squeeze.assert_called_once_with("шо там по деплою")
        mock_map.assert_called_once_with("нормалізований текст")
        mock_unsqueeze.assert_called_once_with("This is a response.", translate_fn=None)
        assert response.content[0].text == "відповідь українською"

    @patch("dormouse.anthropic_proxy.squeeze")
    @patch("dormouse.anthropic_proxy.map_to_en")
    @patch("dormouse.anthropic_proxy.unsqueeze")
    def test_system_string_skipped_by_default(self, mock_unsqueeze, mock_map, mock_squeeze):
        """system param не чіпається коли squeeze_system=False."""
        mock_squeeze.return_value = "нормалізований"
        mock_map.return_value = "normalized"
        mock_unsqueeze.return_value = "відповідь"

        mock_client = _make_anthropic_client("Response.")
        guard = LanguageGuard(roles=("user",), squeeze_system=False)
        proxy = _AnthropicMessagesProxy(
            mock_client.messages, guard, target="cloud", log_savings=False,
        )

        proxy.create(
            model="claude-sonnet-4-20250514",
            system="Ти корисний помічник.",
            messages=[{"role": "user", "content": "привіт"}],
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "Ти корисний помічник."

    @patch("dormouse.anthropic_proxy.squeeze")
    @patch("dormouse.anthropic_proxy.map_to_en")
    @patch("dormouse.anthropic_proxy.unsqueeze")
    def test_system_string_squeezed_when_enabled(self, mock_unsqueeze, mock_map, mock_squeeze):
        """system param стискається коли squeeze_system=True."""
        mock_squeeze.return_value = "стиснутий"
        mock_map.return_value = "squeezed"
        mock_unsqueeze.return_value = "відповідь"

        mock_client = _make_anthropic_client("Response.")
        guard = LanguageGuard(roles=("user",), squeeze_system=True)
        proxy = _AnthropicMessagesProxy(
            mock_client.messages, guard, target="cloud", log_savings=False,
        )

        proxy.create(
            model="claude-sonnet-4-20250514",
            system="Ти корисний помічник.",
            messages=[{"role": "user", "content": "привіт"}],
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "squeezed"

    @patch("dormouse.anthropic_proxy.squeeze")
    @patch("dormouse.anthropic_proxy.map_to_en")
    @patch("dormouse.anthropic_proxy.unsqueeze")
    def test_system_content_blocks_squeezed(self, mock_unsqueeze, mock_map, mock_squeeze):
        """system як list of content blocks — text blocks стискаються."""
        mock_squeeze.return_value = "стиснутий"
        mock_map.return_value = "squeezed"
        mock_unsqueeze.return_value = "відповідь"

        mock_client = _make_anthropic_client("Response.")
        guard = LanguageGuard(roles=("user",), squeeze_system=True)
        proxy = _AnthropicMessagesProxy(
            mock_client.messages, guard, target="cloud", log_savings=False,
        )

        proxy.create(
            model="claude-sonnet-4-20250514",
            system=[
                {"type": "text", "text": "Ти корисний помічник."},
                {"type": "text", "text": "Відповідай коротко."},
            ],
            messages=[{"role": "user", "content": "привіт"}],
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        system_blocks = call_kwargs["system"]
        assert system_blocks[0]["text"] == "squeezed"
        assert system_blocks[1]["text"] == "squeezed"


class TestAnthropicContentBlocks:
    @patch("dormouse.anthropic_proxy.squeeze")
    @patch("dormouse.anthropic_proxy.map_to_en")
    @patch("dormouse.anthropic_proxy.unsqueeze")
    def test_content_blocks_text_squeezed(self, mock_unsqueeze, mock_map, mock_squeeze):
        """text blocks в user message стискаються, image blocks — ні."""
        mock_squeeze.return_value = "стиснутий"
        mock_map.return_value = "squeezed"
        mock_unsqueeze.return_value = "відповідь"

        mock_client = _make_anthropic_client("Response.")
        guard = LanguageGuard(roles=("user",))
        proxy = _AnthropicMessagesProxy(
            mock_client.messages, guard, target="cloud", log_savings=False,
        )

        proxy.create(
            model="claude-sonnet-4-20250514",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "опиши це зображення"},
                    {"type": "image", "source": {"type": "base64", "data": "abc123"}},
                ],
            }],
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        parts = call_kwargs["messages"][0]["content"]
        assert parts[0]["text"] == "squeezed"
        assert parts[1]["type"] == "image"
        assert parts[1]["source"] == {"type": "base64", "data": "abc123"}

    @patch("dormouse.anthropic_proxy.squeeze")
    @patch("dormouse.anthropic_proxy.map_to_en")
    @patch("dormouse.anthropic_proxy.unsqueeze")
    def test_tool_use_block_not_touched(self, mock_unsqueeze, mock_map, mock_squeeze):
        """tool_result blocks не обробляються, squeeze не викликається."""
        mock_unsqueeze.return_value = "відповідь"

        mock_client = _make_anthropic_client("Response.")
        guard = LanguageGuard(roles=("user",))
        proxy = _AnthropicMessagesProxy(
            mock_client.messages, guard, target="cloud", log_savings=False,
        )

        proxy.create(
            model="claude-sonnet-4-20250514",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "123", "content": "result data"},
                ],
            }],
        )

        mock_squeeze.assert_not_called()

    @patch("dormouse.anthropic_proxy.squeeze")
    @patch("dormouse.anthropic_proxy.map_to_en")
    @patch("dormouse.anthropic_proxy.unsqueeze")
    def test_assistant_role_skipped(self, mock_unsqueeze, mock_map, mock_squeeze):
        """assistant messages проходять без змін."""
        mock_unsqueeze.return_value = "відповідь"

        mock_client = _make_anthropic_client("Response.")
        guard = LanguageGuard(roles=("user",))
        proxy = _AnthropicMessagesProxy(
            mock_client.messages, guard, target="cloud", log_savings=False,
        )

        proxy.create(
            model="claude-sonnet-4-20250514",
            messages=[
                {"role": "assistant", "content": "попередня відповідь"},
                {"role": "user", "content": "привіт"},
            ],
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["messages"][0]["content"] == "попередня відповідь"

    @patch("dormouse.anthropic_proxy.squeeze")
    @patch("dormouse.anthropic_proxy.map_to_en")
    @patch("dormouse.anthropic_proxy.unsqueeze")
    def test_json_content_skipped(self, mock_unsqueeze, mock_map, mock_squeeze):
        """JSON контент не стискається."""
        mock_unsqueeze.return_value = "відповідь"

        mock_client = _make_anthropic_client("Response.")
        guard = LanguageGuard(roles=("user",))
        proxy = _AnthropicMessagesProxy(
            mock_client.messages, guard, target="cloud", log_savings=False,
        )

        proxy.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": '{"key": "value"}'}],
        )

        mock_squeeze.assert_not_called()

    @patch("dormouse.anthropic_proxy.squeeze")
    @patch("dormouse.anthropic_proxy.map_to_en")
    @patch("dormouse.anthropic_proxy.unsqueeze")
    def test_response_with_multiple_text_blocks_unsqueezed(
        self, mock_unsqueeze, mock_map, mock_squeeze,
    ):
        """Декілька text blocks + tool_use в response — тільки text unsqueezed."""
        mock_squeeze.return_value = "стиснутий"
        mock_map.return_value = "squeezed"
        mock_unsqueeze.return_value = "розтиснутий"

        response = MagicMock()
        block1 = MagicMock()
        block1.type = "text"
        block1.text = "First response."
        block2 = MagicMock()
        block2.type = "tool_use"
        block2.name = "search"
        block3 = MagicMock()
        block3.type = "text"
        block3.text = "Second response."
        response.content = [block1, block2, block3]

        mock_client = MagicMock()
        mock_client.__class__ = type("Anthropic", (), {})
        mock_client.messages = MagicMock()
        mock_client.messages.create = MagicMock(return_value=response)

        guard = LanguageGuard(roles=("user",))
        proxy = _AnthropicMessagesProxy(
            mock_client.messages, guard, target="cloud", log_savings=False,
        )

        result = proxy.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "привіт"}],
        )

        assert result.content[0].text == "розтиснутий"
        assert result.content[1].type == "tool_use"
        assert result.content[2].text == "розтиснутий"
        assert mock_unsqueeze.call_count == 2


class TestAnthropicErrorHandling:
    @patch("dormouse.anthropic_proxy.squeeze", side_effect=Exception("boom"))
    @patch("dormouse.anthropic_proxy.unsqueeze")
    def test_squeeze_error_passes_original(self, mock_unsqueeze, mock_squeeze):
        """Якщо squeeze впав — передаємо оригінальний текст."""
        mock_unsqueeze.return_value = "ok"
        mock_client = _make_anthropic_client("Response.")
        guard = LanguageGuard(roles=("user",))
        proxy = _AnthropicMessagesProxy(
            mock_client.messages, guard, target="cloud", log_savings=False,
        )

        proxy.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "привіт"}],
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["messages"][0]["content"] == "привіт"

    @patch("dormouse.anthropic_proxy.unsqueeze", side_effect=Exception("boom"))
    def test_unsqueeze_error_passes_original_response(self, mock_unsqueeze):
        """Якщо unsqueeze впав — повертаємо оригінальну відповідь."""
        mock_client = _make_anthropic_client("Original response.")
        guard = LanguageGuard(roles=("user",))
        proxy = _AnthropicMessagesProxy(
            mock_client.messages, guard, target="cloud", log_savings=False,
        )

        response = proxy.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "шо там"}],
        )

        assert response.content[0].text == "Original response."


def _make_anthropic_stream_events(texts: list[str]):
    """Генерує mock Anthropic streaming events.

    message_start → content_block_start → content_block_delta (per text) →
    content_block_stop → message_stop
    """
    import types

    events = []

    # message_start
    events.append(types.SimpleNamespace(type="message_start", message=None))

    # content_block_start
    events.append(types.SimpleNamespace(type="content_block_start", index=0))

    # content_block_delta per text
    for text in texts:
        delta = types.SimpleNamespace(text=text)
        events.append(types.SimpleNamespace(type="content_block_delta", delta=delta))

    # content_block_stop
    events.append(types.SimpleNamespace(type="content_block_stop", index=0))

    # message_stop
    events.append(types.SimpleNamespace(type="message_stop"))

    return events


class TestAnthropicStreaming:
    @patch("dormouse.anthropic_proxy.squeeze")
    @patch("dormouse.anthropic_proxy.map_to_en")
    @patch("dormouse.anthropic_proxy.unsqueeze", side_effect=lambda t: f"[ua]{t}[/ua]")
    def test_streaming_unsqueezes_by_sentence(self, mock_unsqueeze, mock_map, mock_squeeze):
        mock_squeeze.return_value = "normalized"
        mock_map.return_value = "mapped"

        events = _make_anthropic_stream_events(
            ["Hello", " world", ".", " Nice", " day", "."],
        )
        mock_client = _make_anthropic_client()
        mock_client.messages.create = MagicMock(return_value=iter(events))

        guard = LanguageGuard(roles=("user",))
        proxy = _AnthropicMessagesProxy(
            mock_client.messages, guard, target="cloud", log_savings=False,
        )

        result_texts = []
        for event in proxy.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "привіт"}],
            stream=True,
        ):
            if getattr(event, "type", None) == "content_block_delta":
                result_texts.append(event.delta.text)

        joined = "".join(result_texts)
        assert "[ua]" in joined
        # unsqueeze має бути викликано по реченнях
        assert mock_unsqueeze.call_count >= 2

    @patch("dormouse.anthropic_proxy.squeeze")
    @patch("dormouse.anthropic_proxy.map_to_en")
    @patch("dormouse.anthropic_proxy.unsqueeze", side_effect=Exception("boom"))
    def test_streaming_unsqueeze_error_passes_original(
        self, mock_unsqueeze, mock_map, mock_squeeze,
    ):
        mock_squeeze.return_value = "normalized"
        mock_map.return_value = "mapped"

        events = _make_anthropic_stream_events(["Hello."])
        mock_client = _make_anthropic_client()
        mock_client.messages.create = MagicMock(return_value=iter(events))

        guard = LanguageGuard(roles=("user",))
        proxy = _AnthropicMessagesProxy(
            mock_client.messages, guard, target="cloud", log_savings=False,
        )

        result_texts = []
        for event in proxy.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "привіт"}],
            stream=True,
        ):
            if getattr(event, "type", None) == "content_block_delta":
                result_texts.append(event.delta.text)

        joined = "".join(result_texts)
        assert "Hello." in joined

    @patch("dormouse.anthropic_proxy.squeeze")
    @patch("dormouse.anthropic_proxy.map_to_en")
    @patch("dormouse.anthropic_proxy.unsqueeze", side_effect=lambda t: t)
    def test_streaming_passes_non_delta_events(self, mock_unsqueeze, mock_map, mock_squeeze):
        """message_start, content_block_start, content_block_stop, message_stop присутні."""
        mock_squeeze.return_value = "normalized"
        mock_map.return_value = "mapped"

        events = _make_anthropic_stream_events(["Hello."])
        mock_client = _make_anthropic_client()
        mock_client.messages.create = MagicMock(return_value=iter(events))

        guard = LanguageGuard(roles=("user",))
        proxy = _AnthropicMessagesProxy(
            mock_client.messages, guard, target="cloud", log_savings=False,
        )

        event_types = []
        for event in proxy.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "привіт"}],
            stream=True,
        ):
            event_types.append(event.type)

        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_stop" in event_types
        assert "message_stop" in event_types


class TestAnthropicAsync:
    def test_async_proxy_exists(self):
        """Імпорт _AsyncAnthropicMessagesProxy працює."""
        from dormouse.anthropic_proxy import _AsyncAnthropicMessagesProxy
        assert _AsyncAnthropicMessagesProxy is not None

    @patch("dormouse.anthropic_proxy.squeeze", return_value="normalized")
    @patch("dormouse.anthropic_proxy.map_to_en", return_value="mapped")
    @patch("dormouse.anthropic_proxy.unsqueeze", return_value="відповідь")
    def test_async_create(self, mock_unsqueeze, mock_map, mock_squeeze):
        from dormouse.anthropic_proxy import _AsyncAnthropicMessagesProxy

        mock_response = _make_anthropic_response("Response.")

        mock_client = MagicMock()
        mock_client.__class__ = type("AsyncAnthropic", (), {})
        mock_client.messages = MagicMock()

        async def mock_create(**kwargs):
            return mock_response

        mock_client.messages.create = mock_create

        guard = LanguageGuard(roles=("user",))
        proxy = _AsyncAnthropicMessagesProxy(
            mock_client.messages, guard, target="cloud", log_savings=False,
        )

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                proxy.create(
                    model="claude-sonnet-4-20250514",
                    messages=[{"role": "user", "content": "привіт"}],
                )
            )
        finally:
            loop.close()

        assert result.content[0].text == "відповідь"

    @patch("dormouse.anthropic_proxy.squeeze", return_value="normalized")
    @patch("dormouse.anthropic_proxy.map_to_en", return_value="mapped")
    @patch("dormouse.anthropic_proxy.unsqueeze", side_effect=lambda t: f"[ua]{t}[/ua]")
    def test_async_streaming(self, mock_unsqueeze, mock_map, mock_squeeze):
        """Async streaming працює через async for."""
        from dormouse.anthropic_proxy import _AsyncAnthropicMessagesProxy

        events = _make_anthropic_stream_events(["Hello", " world", "."])

        mock_messages = MagicMock()

        async def _async_iter_events():
            for event in events:
                yield event

        async def mock_create(**kwargs):
            return _async_iter_events()

        mock_messages.create = mock_create

        guard = LanguageGuard(roles=("user",), squeeze_system=False)
        proxy = _AsyncAnthropicMessagesProxy(
            mock_messages, guard, target="cloud", log_savings=False,
        )

        async def run():
            result_texts = []
            stream = await proxy.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                messages=[{"role": "user", "content": "привіт"}],
                stream=True,
            )
            async for event in stream:
                if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                    result_texts.append(event.delta.text)
            return result_texts

        loop = asyncio.new_event_loop()
        try:
            result_texts = loop.run_until_complete(run())
        finally:
            loop.close()

        joined = "".join(result_texts)
        assert "[ua]" in joined
