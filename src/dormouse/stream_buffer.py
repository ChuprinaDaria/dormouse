"""Буферизація streaming відповідей для unsqueeze по реченнях.

Накопичує токени в буфер, виділяє повні речення по розділових знаках,
щоб unsqueeze працював з повними реченнями а не з окремими словами.
"""

import re

# Кінець речення: крапка/!/? після літери або закриваючих дужок,
# але НЕ крапка всередині числа (1.5) або URL (example.com)
_SENTENCE_END = re.compile(
    r"(?<=[a-zA-Zа-яіїєґА-ЯІЇЄҐ)\]\"'])"  # перед знаком — літера або дужка
    r"[.!?]"                                  # розділовий знак
    r"(?=\s|$)"                               # після — пр��біл або кінець
    r"|"                                       # або
    r"\.{3}"                                   # еліпсис ...
    r"(?=\s|$)"                                # після — пробіл ��бо кінець
    r"|"                                       # а��о
    r"\n"                                      # новий рядок
)


class StreamBuffer:
    """Буферизує streaming токени і виділяє повні речення."""

    def __init__(self):
        self._buffer = ""

    def feed(self, token: str | None) -> list[str]:
        """Додає токен в буфер, повертає завершені речення."""
        if not token:
            return []

        self._buffer += token
        return self._extract_sentences()

    def flush(self) -> list[str]:
        """Повертає залишок буфера."""
        if self._buffer:
            result = self._buffer
            self._buffer = ""
            return [result]
        return []

    def _extract_sentences(self) -> list[str]:
        """Виділяє завершені речення з буфера."""
        sentences = []

        while True:
            match = _SENTENCE_END.search(self._buffer)
            if not match:
                break

            end_pos = match.end()
            sentence = self._buffer[:end_pos]
            self._buffer = self._buffer[end_pos:]
            sentences.append(sentence)

        return sentences
