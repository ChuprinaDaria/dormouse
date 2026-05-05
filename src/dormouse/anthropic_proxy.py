"""SDK Proxy для Anthropic — прозоре стиснення українських промптів.

Використання:
    from anthropic import Anthropic
    from dormouse.anthropic_proxy import _AnthropicMessagesProxy

    client = Anthropic()
    guard = LanguageGuard(roles=("user",))
    proxy = _AnthropicMessagesProxy(client.messages, guard, target="cloud", log_savings=False)
    response = proxy.create(model="claude-sonnet-4-20250514", messages=[...])
"""

import logging
import types

from dormouse.language_guard import LanguageGuard
from dormouse.mapper import map_to_en
from dormouse.optimizer import squeeze
from dormouse.stream_buffer import StreamBuffer
from dormouse.unsqueeze import _TRANSLATE_PROMPT, unsqueeze

logger = logging.getLogger("dormouse")


class _AnthropicMessagesProxy:
    """Проксі для client.messages — перехоплює create()."""

    def __init__(
        self,
        original_messages,
        guard: LanguageGuard,
        target: str,
        log_savings: bool,
        llm_unsqueeze: bool = False,
    ):
        self._original = original_messages
        self._guard = guard
        self._target = target
        self._log_savings = log_savings
        self._llm_unsqueeze = llm_unsqueeze

    def create(self, **kwargs):
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        model = kwargs.get("model", "")

        # Squeeze system prompt
        if "system" in kwargs:
            kwargs["system"] = self._process_system(kwargs["system"])

        # Squeeze messages
        kwargs["messages"] = self._process_messages(messages)

        if stream:
            return self._handle_stream(**kwargs)

        response = self._original.create(**kwargs)
        self._unsqueeze_response(response, model=model)
        return response

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

    def _unsqueeze_text(self, text: str) -> str:
        """unsqueeze з fallback на оригінал."""
        try:
            return unsqueeze(text)
        except Exception:
            logger.warning("dormouse: unsqueeze failed, passing original", exc_info=True)
            return text

    def _process_system(self, system):
        """Обробляє system prompt: string або list of content blocks."""
        if not self._guard.squeeze_system:
            return system

        if isinstance(system, str):
            if self._guard.should_process(system):
                return self._squeeze_text(system)
            return system

        if isinstance(system, list):
            result = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if self._guard.should_process(text):
                        result.append({**block, "text": self._squeeze_text(text)})
                    else:
                        result.append(block)
                else:
                    result.append(block)
            return result

        return system

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
                new_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "")
                        if self._guard.should_process(text):
                            new_parts.append({
                                **part,
                                "text": self._squeeze_text(text),
                            })
                        else:
                            new_parts.append(part)
                    else:
                        new_parts.append(part)
                processed["content"] = new_parts
            elif isinstance(content, str):
                processed["content"] = self._squeeze_text(content)

            result.append(processed)
        return result

    def _make_translate_fn(self, model: str):
        """Створює translate_fn для LLM fallback unsqueeze."""
        if not self._llm_unsqueeze or not model:
            return None
        original = self._original

        def translate(text: str) -> str:
            resp = original.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": _TRANSLATE_PROMPT.format(text=text)}],
                temperature=0.1,
            )
            return resp.content[0].text

        return translate

    def _unsqueeze_response(self, response, model: str = "") -> None:
        """unsqueeze відповіді моделі — тільки text blocks."""
        try:
            translate_fn = self._make_translate_fn(model)
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    block.text = unsqueeze(block.text, translate_fn=translate_fn)
        except Exception:
            logger.warning(
                "dormouse: unsqueeze failed, returning original response", exc_info=True
            )

    def _handle_stream(self, **kwargs):
        """Streaming з буферизацією unsqueeze по реченнях."""
        raw_stream = self._original.create(**kwargs)
        buffer = StreamBuffer()

        for event in raw_stream:
            event_type = getattr(event, "type", None)

            if event_type == "content_block_delta":
                delta_text = getattr(event.delta, "text", None)
                if delta_text is not None:
                    for sentence in buffer.feed(delta_text):
                        unsqueezed = self._unsqueeze_text(sentence)
                        event.delta.text = unsqueezed
                        yield event
                    continue

            if event_type in ("content_block_stop", "message_stop"):
                for sentence in buffer.flush():
                    unsqueezed = self._unsqueeze_text(sentence)
                    delta = types.SimpleNamespace(text=unsqueezed)
                    flush_event = types.SimpleNamespace(
                        type="content_block_delta", delta=delta,
                    )
                    yield flush_event
                yield event
                continue

            yield event

    def __getattr__(self, name):
        return getattr(self._original, name)


class _AsyncAnthropicMessagesProxy:
    """Async проксі для client.messages."""

    def __init__(
        self,
        original_messages,
        guard: LanguageGuard,
        target: str,
        log_savings: bool,
        llm_unsqueeze: bool = False,
    ):
        self._original = original_messages
        self._sync = _AnthropicMessagesProxy(
            original_messages, guard, target, log_savings, llm_unsqueeze=llm_unsqueeze,
        )

    async def create(self, **kwargs):
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        model = kwargs.get("model", "")

        if "system" in kwargs:
            kwargs["system"] = self._sync._process_system(kwargs["system"])

        kwargs["messages"] = self._sync._process_messages(messages)

        if stream:
            return self._handle_stream(**kwargs)

        response = await self._original.create(**kwargs)
        self._sync._unsqueeze_response(response, model=model)
        return response

    async def _handle_stream(self, **kwargs):
        raw_stream = await self._original.create(**kwargs)
        buffer = StreamBuffer()

        async for event in raw_stream:
            event_type = getattr(event, "type", None)

            if event_type == "content_block_delta":
                delta_text = getattr(event.delta, "text", None)
                if delta_text is not None:
                    for sentence in buffer.feed(delta_text):
                        unsqueezed = self._sync._unsqueeze_text(sentence)
                        event.delta.text = unsqueezed
                        yield event
                    continue

            if event_type in ("content_block_stop", "message_stop"):
                for sentence in buffer.flush():
                    unsqueezed = self._sync._unsqueeze_text(sentence)
                    delta = types.SimpleNamespace(text=unsqueezed)
                    flush_event = types.SimpleNamespace(
                        type="content_block_delta", delta=delta,
                    )
                    yield flush_event
                yield event
                continue

            yield event

    def __getattr__(self, name):
        return getattr(self._original, name)
