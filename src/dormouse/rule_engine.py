"""Rule-based нормалізація українського тексту (baseline).

v4: pymorphy3 через shared morphology module + кешування правил і regex.
"""

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from dormouse.morphology import get_morph, inflect_replacement

LEXICON_PATH = files("dormouse.data").joinpath("replacements.json")


@dataclass
class CrackResult:
    text: str
    original: str
    replacements_made: int

    @property
    def changed(self) -> bool:
        return self.text != self.original


# --- Кеш правил і compiled regex ---

_rules_cache: dict | None = None
_rules_cache_path: Path | None = None
_compiled_expressions: dict[int, list[tuple[re.Pattern, str | None]]] | None = None
_compiled_patterns: list[tuple[re.Pattern, str]] | None = None
_word_map: dict[str, str | None] | None = None
_lemma_map: dict[str, str | None] | None = None


def _load_rules(path=None) -> dict:
    """Завантажує правила замін з JSON (з кешуванням)."""
    global _rules_cache, _rules_cache_path
    p = path or LEXICON_PATH
    if _rules_cache is not None and _rules_cache_path == p:
        return _rules_cache
    # importlib.resources Traversable або Path — обидва підтримують open()
    with p.open(encoding="utf-8") as f:
        _rules_cache = json.load(f)
    _rules_cache_path = p
    return _rules_cache


def _get_compiled(path: Path | None = None):
    """Повертає pre-compiled regex та word maps. Кешує після першого виклику."""
    global _compiled_expressions, _compiled_patterns, _word_map, _lemma_map

    if _compiled_expressions is not None and _rules_cache_path == (path or LEXICON_PATH):
        return

    rules = _load_rules(path)

    # Групуємо по ngram
    groups: dict[int, list[dict]] = {}
    for r in rules["replacements"]:
        n = r.get("ngram", 1)
        groups.setdefault(n, []).append(r)

    # Pre-compile expression regex (ngram > 1)
    _compiled_expressions = {}
    for n, exprs in groups.items():
        if n <= 1:
            continue
        compiled = []
        for expr in exprs:
            pat = re.compile(r"\b" + re.escape(expr["from"]) + r"\b", re.IGNORECASE)
            compiled.append((pat, expr["to"]))
        _compiled_expressions[n] = compiled

    # Pre-compile pattern regex
    _compiled_patterns = []
    for p in rules.get("patterns", []):
        _compiled_patterns.append((re.compile(p["regex"], re.IGNORECASE), p["to"]))

    # Word map (unigrams)
    unigrams = groups.get(1, [])
    _word_map = {r["from"].lower(): r["to"] for r in unigrams}

    # Lemma map (pymorphy3)
    morph = get_morph()
    _lemma_map = {}
    if morph:
        for from_word, to_word in _word_map.items():
            lemma = morph.parse(from_word)[0].normal_form
            if lemma not in _lemma_map:
                _lemma_map[lemma] = to_word


def _apply_expressions(text: str, ngram: int) -> tuple[str, int]:
    """Замінює багатослівні вирази через pre-compiled regex."""
    count = 0
    for pattern, replacement in _compiled_expressions.get(ngram, []):
        repl = replacement if replacement is not None else ""
        new_text, n = pattern.subn(repl, text)
        count += n
        text = new_text
    return text, count


def _apply_word_replacements(text: str) -> tuple[str, int]:
    """Замінює слова зі словника. pymorphy3: лематизація + inflect."""
    morph = get_morph()
    count = 0
    words = text.split()
    result = []

    for word in words:
        stripped = word.strip(".,!?;:()\"'«»—–-…")
        if not stripped or stripped not in word:
            result.append(word)
            continue
        prefix = word[: word.index(stripped)]
        end = word.rindex(stripped) + len(stripped)
        suffix = word[end:]

        lookup = stripped.lower()

        # 1. Exact match (пріоритет)
        if lookup in _word_map:
            replacement = _word_map[lookup]
            if replacement is None:
                count += 1
                continue
            result.append(prefix + replacement + suffix)
            count += 1
            continue

        # 2. Lemma match через pymorphy3
        if _lemma_map and morph:
            lemma = morph.parse(lookup)[0].normal_form
            if lemma in _lemma_map:
                replacement = _lemma_map[lemma]
                if replacement is None:
                    count += 1
                    continue
                inflected = inflect_replacement(stripped, replacement)
                result.append(prefix + inflected + suffix)
                count += 1
                continue

        result.append(word)

    return " ".join(result), count


def _apply_patterns(text: str) -> tuple[str, int]:
    """Застосовує pre-compiled regex-патерни."""
    count = 0
    for pattern, replacement in _compiled_patterns:
        new_text, n = pattern.subn(replacement, text)
        count += n
        text = new_text
    return text.strip(), count


def crack(text: str, rules_path: Path | None = None) -> str:
    """Нормалізує текст: замінює суржик, видаляє fillers.

    Повертає очищений текст. Для детального результату — crack_open().
    """
    return crack_open(text, rules_path).text


_APOSTROPHE_RE = re.compile(r"[\u2018\u2019\u02BC\u02B9\u0060\u00B4]")


def crack_open(text: str, rules_path: Path | None = None) -> CrackResult:
    """Нормалізує текст з детальним результатом.

    Порядок: апострофи → триграми → біграми → уніграми (з лематизацією) → regex-патерни.
    """
    original = text
    _get_compiled(rules_path)

    # Нормалізація апострофів: ' ' ʼ ʹ ` ´ → ' (U+0027)
    text = _APOSTROPHE_RE.sub("'", text)

    total_count = 0

    # Longest match first: триграми → біграми
    for n in sorted((k for k in _compiled_expressions if k > 1), reverse=True):
        text, count = _apply_expressions(text, n)
        total_count += count

    # Уніграми (окремі слова + lemma fallback)
    text, count = _apply_word_replacements(text)
    total_count += count

    # Regex-патерни (cleanup)
    text, pattern_count = _apply_patterns(text)
    total_count += pattern_count

    return CrackResult(
        text=text,
        original=original,
        replacements_made=total_count,
    )
