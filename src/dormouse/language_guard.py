"""Детекція мови та формату тексту.

Вирішує чи обробляти message через dormouse pipeline.
Пропускає JSON, код, англійський текст, порожні повідомлення.
"""

import re

# Патерни коду: рядок починається з ключового слова
_CODE_PATTERNS = re.compile(
    r"^\s*(def |class |import |from .+ import |function |const |let |var |async def )",
    re.MULTILINE,
)


class LanguageGuard:
    """Визначає чи потрібно обробляти текст через squeeze pipeline."""

    def __init__(
        self,
        roles: tuple[str, ...] = ("user",),
        squeeze_system: bool = False,
        skip_patterns: tuple[str, ...] = ("json", "code"),
    ):
        self.roles = roles
        self.squeeze_system = squeeze_system
        self.skip_patterns = skip_patterns

    def should_process(self, text: str | None) -> bool:
        """Чи потрібно обробляти цей текст."""
        if text is None or not isinstance(text, str):
            return False

        stripped = text.strip()
        if not stripped:
            return False

        # JSON detection
        if "json" in self.skip_patterns:
            if stripped.startswith(("{", "[")):
                return False

        # Code detection
        if "code" in self.skip_patterns:
            if _CODE_PATTERNS.search(stripped):
                return False

        # Language detection: >70% латиниці = англійська
        latin = sum(1 for c in stripped if c.isalpha() and ord(c) < 128)
        alpha = sum(1 for c in stripped if c.isalpha())
        if alpha > 0 and latin / alpha > 0.7:
            return False

        return True

    def should_process_message(self, message: dict) -> bool:
        """Чи потрібно обробляти цей message з OpenAI API."""
        role = message.get("role", "")
        content = message.get("content")

        # Role check
        if role == "system":
            if not self.squeeze_system:
                return False
        elif role not in self.roles:
            return False

        # Content check
        if content is None:
            return False

        # Multimodal: list of content parts
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    if self.should_process(text):
                        return True
            return False

        return self.should_process(content)
