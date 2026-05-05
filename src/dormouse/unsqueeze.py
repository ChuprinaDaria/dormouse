"""Зворотній маппінг EN→UA для відповідей моделі.

Коли модель відповідає англійською (після squeeze target="cloud"),
unsqueeze перекладає назад в українську через SQLite лексикон.

v2: фільтрація EN стоп-слів, пріоритетний вибір UA форми.
v3: морфологічне узгодження (прийменник→відмінок, дієслово→об'єкт, рід).
v4: LLM fallback — якщо >30% слів не перекладено, translate_fn доперекладає.
"""

import logging
import re
import sqlite3
from collections.abc import Callable

from dormouse.lexicon_db import get_lexicon
from dormouse.morphology import get_morph

logger = logging.getLogger("dormouse")

# Поріг: якщо більше 30% слів залишились латиницею — якість низька
_UNTRANSLATED_THRESHOLD = 0.3

_TRANSLATE_PROMPT = (
    "Переклади цей англійський текст на українську мову. "
    "Технічні терміни (назви сервісів, команди, код) залишай як є. "
    "Відповідай ТІЛЬКИ перекладом, без пояснень.\n\n{text}"
)

# Артиклі — в українській не існують, просто видаляємо
_EN_ARTICLES = frozenset({"the", "a", "an"})

# EN функціональні слова → UA переклад (модель додає їх сама)
_EN_FUNCTION_WORDS: dict[str, str] = {
    # Заперечення
    "not": "не", "don't": "не", "doesn't": "не", "didn't": "не",
    "isn't": "не", "aren't": "не", "wasn't": "не", "weren't": "не",
    "won't": "не", "can't": "не можна", "cannot": "не можна",
    # Сполучники
    "and": "і", "or": "або", "but": "але",
    # Присудкові (теперішній час — зайві в UA)
    "is": "", "are": "", "am": "",
    # Минулий/майбутній — несуть значення
    "was": "був", "were": "були",
    "will": "буде", "would": "б",
    "could": "міг", "should": "слід",
    # Допоміжні
    "do": "", "does": "", "did": "",
    "has": "має", "have": "маємо",
    "it": "це",
    # Інше
    "been": "", "being": "",
}

# Зворотна сумісність
_EN_STOPWORDS = _EN_ARTICLES

_conn = None


def _get_conn():
    global _conn
    if _conn is None:
        _conn = get_lexicon()
    return _conn


_CYRILLIC_RE = re.compile(r"[а-яіїєґ]", re.IGNORECASE)
_LATIN_RE = re.compile(r"[a-z]", re.IGNORECASE)


def _is_mixed_script(word: str) -> bool:
    """Перевіряє чи слово містить і кирилицю і латиницю (OPUS garbage)."""
    return bool(_CYRILLIC_RE.search(word) and _LATIN_RE.search(word))


def _pick_best_ua(candidates: list[tuple[str, str | None]]) -> str:
    """Вибирає найкращу UA форму з кандидатів.

    Пріоритет:
    1. Normalized форма з найбільшою частотою (більше слів → краща = канонічніша)
    2. Канонічні (normalized=None) без мішаного скрипту
    3. Найкоротше серед рівних
    """
    # Normalized — рахуємо скільки слів вказують на кожну форму
    freq: dict[str, int] = {}
    for _, n in candidates:
        if n and not _is_mixed_script(n):
            freq[n] = freq.get(n, 0) + 1

    if freq:
        # Сортуємо: спочатку найчастіші, потім найкоротші
        return min(freq, key=lambda x: (-freq[x], len(x)))

    # Канонічні — без mixed script garbage
    canonical = [w for w, n in candidates if n is None and not _is_mixed_script(w)]
    if canonical:
        return min(canonical, key=len)

    # Fallback — будь-що чисте
    for w, n in candidates:
        ua = n if n else w
        if not _is_mixed_script(ua):
            return ua
    return candidates[0][1] or candidates[0][0]


# Прийменник → відмінок наступного іменника
_PREP_CASE: dict[str, str] = {
    "в": "loct", "у": "loct",
    "на": "loct",
    "з": "gent", "із": "gent", "зі": "gent",
    "до": "gent",
    "від": "gent",
    "за": "accs",
    "для": "gent",
    "без": "gent",
    "про": "accs",
    "під": "accs",
    "над": "ablt",
    "між": "ablt",
    "через": "accs",
    "по": "datv",
    "після": "gent",
}

_UA_WORD_RE = re.compile(r"[а-яіїєґ]", re.IGNORECASE)


def _strip_punct(token: str) -> tuple[str, str, str]:
    """Розділяє токен на (prefix, word, suffix)."""
    stripped = token.strip(".,!?;:()\"'—–-…")
    if not stripped:
        return "", token, ""
    idx = token.index(stripped)
    end = idx + len(stripped)
    return token[:idx], stripped, token[end:]


def _morph_postprocess(tokens: list[str]) -> list[str]:
    """Морфологічне узгодження UA токенів.

    Правила:
    1. Прийменник + іменник → правильний відмінок
    2. Дієслово (наказовий/інфінітив) + іменник → знахідний
    3. Іменник + дієслово мин. часу → узгодження роду
    """
    morph = get_morph()
    if not morph:
        return tokens

    result = []
    for i, token in enumerate(tokens):
        _, clean, _ = _strip_punct(token)
        if not clean or not _UA_WORD_RE.match(clean):
            result.append(token)
            continue

        # Попередній чистий токен
        prev_clean = ""
        if result:
            _, prev_clean, _ = _strip_punct(result[-1])

        prev_lower = prev_clean.lower()
        curr_parse = morph.parse(clean)[0]

        # Правило 1: прийменник + іменник → відмінок
        if prev_lower in _PREP_CASE and curr_parse.tag.POS == "NOUN":
            case = _PREP_CASE[prev_lower]
            inflected = curr_parse.inflect({case})
            if inflected and inflected.word != clean:
                token = token.replace(clean, inflected.word)
                result.append(token)
                continue

        # Правило 2: дієслово + іменник → знахідний (наказ., інфін., минулий час)
        if prev_clean and _UA_WORD_RE.match(prev_clean):
            prev_parses = morph.parse(prev_clean)
            is_action_verb = any(
                p.tag.POS in ("VERB", "INFN")
                for p in prev_parses
            )
            if is_action_verb and curr_parse.tag.POS == "NOUN":
                inflected = curr_parse.inflect({"accs"})
                if inflected and inflected.word != clean:
                    token = token.replace(clean, inflected.word)
                    result.append(token)
                    continue

        # Правило 3: іменник + дієслово мин. часу → узгодження роду
        if (
            curr_parse.tag.POS == "VERB"
            and curr_parse.tag.tense == "past"
            and prev_clean
            and _UA_WORD_RE.match(prev_clean)
        ):
            prev_parse = morph.parse(prev_clean)[0]
            if prev_parse.tag.POS == "NOUN" and prev_parse.tag.gender:
                gender = prev_parse.tag.gender
                target = {"past", gender}
                if prev_parse.tag.number == "plur":
                    target = {"past", "plur"}
                inflected = curr_parse.inflect(target)
                if inflected and inflected.word != clean:
                    token = token.replace(clean, inflected.word)

        result.append(token)

    return result


def _build_en_to_ua(conn: sqlite3.Connection) -> dict[str, str]:
    """Будує зворотній словник EN→UA з лексикону.

    Збирає всіх кандидатів для кожного EN слова, потім вибирає найкращого.
    """
    candidates: dict[str, list[tuple[str, str | None]]] = {}

    rows = conn.execute(
        "SELECT word, normalized, en_compressed FROM lexicon "
        "WHERE en_compressed IS NOT NULL"
    ).fetchall()

    for row in rows:
        word = row["word"]
        normalized = row["normalized"]
        en_val = row["en_compressed"]
        if not en_val:
            continue
        key = en_val.lower()
        if key not in candidates:
            candidates[key] = []
        candidates[key].append((word, normalized))

    return {en: _pick_best_ua(cands) for en, cands in candidates.items()}


_en_to_ua_cache: dict[str, str] | None = None


def _get_en_to_ua() -> dict[str, str]:
    global _en_to_ua_cache
    if _en_to_ua_cache is None:
        conn = _get_conn()
        _en_to_ua_cache = _build_en_to_ua(conn)
    return _en_to_ua_cache


def _untranslated_ratio(text: str) -> float:
    """Частка слів що залишились латиницею (не перекладені)."""
    words = text.split()
    if not words:
        return 0.0
    en_count = 0
    for w in words:
        clean = w.strip(".,!?;:()\"'—–-…")
        if clean and _LATIN_RE.match(clean) and not _CYRILLIC_RE.search(clean):
            en_count += 1
    return en_count / len(words)


def unsqueeze(
    text: str,
    *,
    translate_fn: Callable[[str], str] | None = None,
) -> str:
    """Перекладає EN відповідь моделі назад в українську.

    Args:
        text: Англійський текст (відповідь Claude/GPT/local model).
        translate_fn: Опціональна функція LLM-перекладу (str→str).
            Якщо передана і word-by-word якість низька (>30% EN слів) —
            fallback на LLM переклад оригінального тексту.

    Returns:
        Текст українською.
    """
    if not text or not text.strip():
        return text

    original_text = text
    en_to_ua = _get_en_to_ua()

    # Longest match first (фрази перед словами)
    for phrase in sorted(en_to_ua, key=len, reverse=True):
        if " " not in phrase:
            continue  # single words handled below
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        text = pattern.sub(en_to_ua[phrase], text)

    # Single words
    words = text.split()
    result = []
    for word in words:
        stripped = word.strip(".,!?;:()\"'—–-…")
        if not stripped:
            result.append(word)
            continue
        prefix = word[: word.index(stripped)] if stripped in word else ""
        end = word.rindex(stripped) + len(stripped) if stripped in word else len(word)
        suffix = word[end:]

        lookup = stripped.lower()

        # Артиклі — просто пропускаємо
        if lookup in _EN_ARTICLES:
            continue

        # Функціональні слова — перекладаємо (порожній рядок = пропуск)
        if lookup in _EN_FUNCTION_WORDS:
            ua = _EN_FUNCTION_WORDS[lookup]
            if ua:
                result.append(prefix + ua + suffix)
            continue

        if lookup in en_to_ua:
            result.append(prefix + en_to_ua[lookup] + suffix)
        else:
            result.append(word)

    result = _morph_postprocess(result)
    word_by_word = " ".join(result)

    # Hybrid: якщо якість низька і є LLM — перекладаємо оригінал
    if translate_fn and _untranslated_ratio(word_by_word) > _UNTRANSLATED_THRESHOLD:
        try:
            llm_result = translate_fn(original_text)
            if llm_result and llm_result.strip():
                logger.debug("dormouse: unsqueeze LLM fallback (ratio=%.0f%%)",
                             _untranslated_ratio(word_by_word) * 100)
                return llm_result
        except Exception:
            logger.warning("dormouse: LLM unsqueeze failed, using word-by-word", exc_info=True)

    return word_by_word
