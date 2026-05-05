"""Семантична компресія українського тексту.

Шар 2 pipeline: видаляє надлишковість зі стандартного (вже нормалізованого) тексту.
Два режими: rule-based (0 залежностей) і LLMLingua-2 (optional).

v2: pre-compiled regex для filler phrases.
"""

import re

# Українські стоп-слова (не несуть семантичного навантаження в промптах)
STOP_WORDS = frozenset([
    "я", "ти", "він", "вона", "воно", "ми", "ви", "вони",
    "це", "той", "цей", "цього", "цьому", "цю", "ця",
    "свій", "своє", "свого", "своєму", "свою",
    "ось", "от", "ж", "же", "ну", "так", "ні",
    "і", "й", "та", "а", "але", "або", "чи",
    "в", "у", "на", "з", "із", "до", "від", "за", "по", "для",
    "не", "ні", "би", "б",
])

# Вставні конструкції та ввічливі обгортки
FILLER_PHRASES = [
    "я думаю що",
    "я думаю",
    "мені здається що",
    "мені здається",
    "як на мене",
    "на мою думку",
    "якщо чесно",
    "чесно кажучи",
    "якщо можна",
    "чи не могли б ви",
    "чи не міг би ти",
    "мені потрібно",
    "я хочу щоб",
    "я хотів би",
    "я хотіла б",
    "можете допомогти",
]

# Pre-compiled patterns (longest first)
_FILLER_PATTERNS: list[re.Pattern] = [
    re.compile(re.escape(phrase), re.IGNORECASE)
    for phrase in sorted(FILLER_PHRASES, key=len, reverse=True)
]

# Підсилювачі (не додають інформації)
INTENSIFIERS = frozenset([
    "дуже", "досить", "надзвичайно", "абсолютно",
    "зовсім", "цілком", "повністю", "справді",
])

# Pre-compiled cleanup patterns
_RE_DOUBLE_PUNCT = re.compile(r"[,;]{2,}")
_RE_LEADING_PUNCT = re.compile(r"^\s*[,;]\s*")
_RE_TRAILING_PUNCT = re.compile(r"\s*[,;]\s*$")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:])")
_RE_DOUBLE_SPACE = re.compile(r"\s{2,}")


def _remove_filler_phrases(text: str) -> str:
    """Видаляє вставні конструкції (pre-compiled, longest first)."""
    for pattern in _FILLER_PATTERNS:
        text = pattern.sub("", text)
    return text


def _remove_intensifiers(text: str) -> str:
    """Видаляє підсилювачі."""
    words = text.split()
    return " ".join(w for w in words if w.lower() not in INTENSIFIERS)


def _dedup_consecutive(text: str) -> str:
    """Видаляє повтори слів поспіль: 'дуже дуже' → 'дуже'."""
    words = text.split()
    if not words:
        return text
    result = [words[0]]
    for w in words[1:]:
        if w.lower() != result[-1].lower():
            result.append(w)
    return " ".join(result)


def _cleanup(text: str) -> str:
    """Фінальна чистка: подвійні пробіли, осиротіла пунктуація."""
    text = _RE_DOUBLE_PUNCT.sub(",", text)
    text = _RE_LEADING_PUNCT.sub("", text)
    text = _RE_TRAILING_PUNCT.sub("", text)
    text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _RE_DOUBLE_SPACE.sub(" ", text)
    return text.strip()


def compress(text: str, mode: str = "rule") -> str:
    """Компресія тексту.

    Args:
        text: Нормалізований текст (після rule_engine).
        mode: "rule" (0 залежностей) або "llmlingua" (потребує dormouse[compress]).
    """
    if not text or not text.strip():
        return text

    if mode == "llmlingua":
        return _compress_llmlingua(text)

    # Rule-based компресія
    text = _remove_filler_phrases(text)
    text = _remove_intensifiers(text)
    text = _dedup_consecutive(text)
    text = _cleanup(text)

    return text


def _compress_llmlingua(text: str) -> str:
    """Компресія через LLMLingua-2 (optional dependency)."""
    try:
        from llmlingua import PromptCompressor
    except ImportError:
        raise ImportError(
            "LLMLingua не встановлено. "
            "Встанови: pip install dormouse[compress]"
        )

    compressor = PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True,
    )
    result = compressor.compress_prompt(
        [text],
        rate=0.5,
        force_tokens=["\n"],
    )
    return result["compressed_prompt"]
