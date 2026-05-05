"""SDK Wrapper для OpenAI / Anthropic — прозоре стиснення українських промптів.

Використання (OpenAI):
    from openai import OpenAI
    from dormouse.middleware import DormouseClient

    client = DormouseClient(OpenAI(), target="cloud")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "шо там по деплою"}]
    )

Використання (Anthropic):
    from anthropic import Anthropic
    from dormouse.middleware import DormouseClient

    client = DormouseClient(Anthropic(), target="cloud")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "шо там по деплою"}]
    )
"""

import logging

from dormouse.language_guard import LanguageGuard
from dormouse.mapper import map_to_en
from dormouse.optimizer import squeeze
from dormouse.stream_buffer import StreamBuffer
from dormouse.unsqueeze import _TRANSLATE_PROMPT, unsqueeze

logger = logging.getLogger("dormouse")


class _CompletionsProxy:
    """Проксі для client.chat.completions — перехоплює create()."""

    def __init__(self, original_completions, guard: LanguageGuard, target: str,
                 log_savings: bool, llm_unsqueeze: bool = False):
        self._original = original_completions
        self._guard = guard
        self._target = target
        self._log_savings = log_savings
        self._llm_unsqueeze = llm_unsqueeze

    def create(self, **kwargs):
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        model = kwargs.get("model", "")

        # Squeeze outgoing messages
        processed_messages = self._process_messages(messages)
        kwargs["messages"] = processed_messages

        if stream:
            return self._handle_stream(**kwargs)

        # Non-streaming
        response = self._original.create(**kwargs)
        self._unsqueeze_response(response, model=model)
        if self._log_savings:
            self._log_token_savings(messages, processed_messages)
        return response

    def _process_messages(self, messages: list[dict]) -> list[dict]:
        """Обробляє messages: squeeze + bridge для підходящих."""
        result = []
        for msg in messages:
            if not self._guard.should_process_message(msg):
                result.append(msg)
                continue

            processed = dict(msg)
            content = msg.get("content")

            if isinstance(content, list):
                # Multimodal: обробляємо тільки text parts
                new_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        new_parts.append({
                            **part,
                            "text": self._squeeze_text(part.get("text", "")),
                        })
                    else:
                        new_parts.append(part)
                processed["content"] = new_parts
            elif isinstance(content, str):
                processed["content"] = self._squeeze_text(content)

            result.append(processed)
        return result

    def _squeeze_text(self, text: str) -> str:
        """squeeze + bridge з fallback на оригінал при помилці."""
        try:
            squeezed = squeeze(text)
            if self._target == "cloud":
                squeezed = map_to_en(squeezed)
            return squeezed
        except Exception:
            logger.warning("dormouse: squeeze failed, passing original text", exc_info=True)
            return text

    def _make_translate_fn(self, model: str):
        """Створює translate_fn для LLM fallback unsqueeze."""
        if not self._llm_unsqueeze or not model:
            return None
        original = self._original

        def translate(text: str) -> str:
            resp = original.create(
                model=model,
                messages=[{"role": "user", "content": _TRANSLATE_PROMPT.format(text=text)}],
                temperature=0.1,
            )
            return resp.choices[0].message.content

        return translate

    def _unsqueeze_response(self, response, model: str = "") -> None:
        """unsqueeze відповіді моделі з fallback."""
        try:
            translate_fn = self._make_translate_fn(model)
            for choice in response.choices:
                if choice.message and choice.message.content:
                    choice.message.content = unsqueeze(
                        choice.message.content, translate_fn=translate_fn,
                    )
        except Exception:
            logger.warning(
                "dormouse: unsqueeze failed, returning original response", exc_info=True
            )

    def _unsqueeze_text(self, text: str) -> str:
        """unsqueeze з fallback на оригінал (streaming — без LLM)."""
        try:
            return unsqueeze(text)
        except Exception:
            logger.warning("dormouse: unsqueeze failed, passing original", exc_info=True)
            return text

    def _log_token_savings(self, original_msgs, processed_msgs):
        """Логує економію токенів в stderr."""
        import sys

        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")

            orig_text = " ".join(
                m.get("content", "") for m in original_msgs
                if isinstance(m.get("content"), str)
            )
            proc_text = " ".join(
                m.get("content", "") for m in processed_msgs
                if isinstance(m.get("content"), str)
            )

            orig_tokens = len(enc.encode(orig_text))
            proc_tokens = len(enc.encode(proc_text))
            pct = round((orig_tokens - proc_tokens) / orig_tokens * 100) if orig_tokens > 0 else 0

            print(
                f"[dormouse] {orig_tokens} tokens → {proc_tokens} tokens (-{pct}%)",
                file=sys.stderr,
            )
        except ImportError:
            orig_words = sum(
                len(m.get("content", "").split()) for m in original_msgs
                if isinstance(m.get("content"), str)
            )
            proc_words = sum(
                len(m.get("content", "").split()) for m in processed_msgs
                if isinstance(m.get("content"), str)
            )
            print(
                f"[dormouse] ~{orig_words} words → ~{proc_words} words",
                file=sys.stderr,
            )

    def _handle_stream(self, **kwargs):
        """Streaming з буферизацією unsqueeze по реченнях."""
        raw_stream = self._original.create(**kwargs)
        buffer = StreamBuffer()

        for chunk in raw_stream:
            content = None
            if chunk.choices and chunk.choices[0].delta:
                content = chunk.choices[0].delta.content

            if content is None:
                # Flush buffer on final chunk
                for sentence in buffer.flush():
                    unsqueezed = self._unsqueeze_text(sentence)
                    chunk.choices[0].delta.content = unsqueezed
                    yield chunk
                continue

            for sentence in buffer.feed(content):
                unsqueezed = self._unsqueeze_text(sentence)
                chunk.choices[0].delta.content = unsqueezed
                yield chunk

    def __getattr__(self, name):
        return getattr(self._original, name)


class _AsyncCompletionsProxy:
    """Async проксі для client.chat.completions."""

    def __init__(self, original_completions, guard: LanguageGuard, target: str,
                 log_savings: bool, llm_unsqueeze: bool = False):
        self._original = original_completions
        self._sync = _CompletionsProxy(original_completions, guard, target, log_savings,
                                       llm_unsqueeze=llm_unsqueeze)

    async def create(self, **kwargs):
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        model = kwargs.get("model", "")

        processed_messages = self._sync._process_messages(messages)
        kwargs["messages"] = processed_messages

        if stream:
            return self._handle_stream(**kwargs)

        response = await self._original.create(**kwargs)
        self._sync._unsqueeze_response(response, model=model)
        return response

    async def _handle_stream(self, **kwargs):
        raw_stream = await self._original.create(**kwargs)
        buffer = StreamBuffer()

        async for chunk in raw_stream:
            content = None
            if chunk.choices and chunk.choices[0].delta:
                content = chunk.choices[0].delta.content

            if content is None:
                for sentence in buffer.flush():
                    unsqueezed = self._sync._unsqueeze_text(sentence)
                    chunk.choices[0].delta.content = unsqueezed
                    yield chunk
                continue

            for sentence in buffer.feed(content):
                unsqueezed = self._sync._unsqueeze_text(sentence)
                chunk.choices[0].delta.content = unsqueezed
                yield chunk

    def __getattr__(self, name):
        return getattr(self._original, name)


class _ChatProxy:
    """Проксі для client.chat — перехоплює completions."""

    def __init__(self, original_chat, guard: LanguageGuard, target: str, log_savings: bool,
                 async_mode: bool = False, llm_unsqueeze: bool = False):
        self._original = original_chat
        if async_mode:
            self.completions = _AsyncCompletionsProxy(
                original_chat.completions, guard, target, log_savings,
                llm_unsqueeze=llm_unsqueeze,
            )
        else:
            self.completions = _CompletionsProxy(
                original_chat.completions, guard, target, log_savings,
                llm_unsqueeze=llm_unsqueeze,
            )

    def __getattr__(self, name):
        return getattr(self._original, name)


class DormouseClient:
    """Drop-in wrapper для OpenAI та Anthropic SDK.

    Прозоро стискає українські промпти і розтискає відповіді.
    Автоматично визначає провайдера за class name клієнта:
    - OpenAI / AsyncOpenAI → self.chat.completions.create(...)
    - Anthropic / AsyncAnthropic → self.messages.create(...)
    - Невідомий клієнт → pass-through з warning
    """

    def __init__(
        self,
        client,
        target: str = "cloud",
        roles: tuple[str, ...] = ("user",),
        squeeze_system: bool = False,
        skip_patterns: tuple[str, ...] = ("json", "code"),
        log_savings: bool = False,
        llm_unsqueeze: bool = False,
    ):
        self._client = client
        self._target = target
        self._log_savings = log_savings
        self._llm_unsqueeze = llm_unsqueeze
        self._is_async = "Async" in client.__class__.__name__
        self._guard = LanguageGuard(
            roles=roles,
            squeeze_system=squeeze_system,
            skip_patterns=skip_patterns,
        )

        name = client.__class__.__name__
        if "Anthropic" in name:
            self._provider = "anthropic"
            from dormouse.anthropic_proxy import (
                _AnthropicMessagesProxy,
                _AsyncAnthropicMessagesProxy,
            )
            if self._is_async:
                self.messages = _AsyncAnthropicMessagesProxy(
                    client.messages, self._guard, target, log_savings,
                    llm_unsqueeze=llm_unsqueeze,
                )
            else:
                self.messages = _AnthropicMessagesProxy(
                    client.messages, self._guard, target, log_savings,
                    llm_unsqueeze=llm_unsqueeze,
                )
        elif hasattr(client, "chat"):
            self._provider = "openai"
            self.chat = _ChatProxy(
                client.chat, self._guard, target, log_savings,
                async_mode=self._is_async,
                llm_unsqueeze=llm_unsqueeze,
            )
        else:
            self._provider = "unknown"
            logger.warning(
                "dormouse: unknown client type %r, acting as pass-through", name,
            )

    def __getattr__(self, name):
        return getattr(self._client, name)


def openrouter_headers(referer: str = "", title: str = "") -> dict:
    """Генерує extra headers для OpenRouter.

    Використання:
        from openai import OpenAI
        from dormouse.middleware import DormouseClient, openrouter_headers

        client = DormouseClient(
            OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key="sk-or-...",
                default_headers=openrouter_headers(
                    referer="https://myapp.com",
                    title="MyApp",
                ),
            ),
            target="cloud",
        )
    """
    headers: dict[str, str] = {}
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers
