"""Composition of UA→EN training targets from clean Ukrainian text.

Deterministic path: crack_open → compress → strict lexicon mapping. Unlike
mapper.map_to_en, strict_map_to_en returns None instead of transliterating —
a transliterated target is garbage supervision for the seq2seq.

Expression matching goes through exact n-gram lookups against the word index
(greedy, longest first) instead of mapper's per-expression regex scan: the
regex scan is fine for one runtime query but far too slow for composing tens
of thousands of training windows.
"""

from __future__ import annotations

import re
import sqlite3

from dormouse.compressor import compress
from dormouse.lexicon_db import lookup_batch
from dormouse.morphology import get_morph
from dormouse.rule_engine import crack_open

_RE_CYRILLIC = re.compile(r"[а-яіїєґА-ЯІЇЄҐ]")
_RE_LATIN_TOKEN = re.compile(r"^[a-zA-Z0-9?.!,'\"\-&:;%$#+/()]+$")
_STRIP_CHARS = ".,!?;:()\"'«»—–-…"

MAX_EXPR_NGRAM = 4


def _match_expressions(words: list[str], conn: sqlite3.Connection) -> list[str]:
    """Жадібна заміна n-грам (4→2) точним lookup по word-індексу лексикону.

    Повертає список токенів, де знайдені вирази вже замінені на EN.
    """
    lowered = [w.strip(_STRIP_CHARS).lower() for w in words]
    result: list[str] = []
    i = 0
    while i < len(words):
        replaced = False
        for n in range(min(MAX_EXPR_NGRAM, len(words) - i), 1, -1):
            chunk = " ".join(lowered[i : i + n])
            if not chunk:
                continue
            row = lookup_batch(conn, [chunk]).get(chunk)
            if row and row["en_compressed"]:
                result.append(row["en_compressed"])
                i += n
                replaced = True
                break
        if not replaced:
            result.append(words[i])
            i += 1
    return result


def strict_map_to_en(text: str, conn: sqlite3.Connection) -> str | None:
    """Як mapper.map_to_en, але без транслітерації: None якщо хоч одне
    кириличне слово не знайдено ні в лексиконі, ні через lemma fallback."""
    if not text or not text.strip():
        return None

    words = _match_expressions(text.split(), conn)

    clean_words = [w.strip(_STRIP_CHARS).lower() for w in words]
    clean_words = [w for w in clean_words if w]
    found = lookup_batch(conn, clean_words)

    morph = get_morph()
    lemma_found: dict[str, str] = {}
    if morph:
        missing = [w for w in clean_words if w not in found or not found[w]["en_compressed"]]
        lemma_to_original = {}
        for w in missing:
            if not _RE_CYRILLIC.search(w):
                continue
            lemma = morph.parse(w)[0].normal_form
            if lemma != w:
                lemma_to_original.setdefault(lemma, w)
        if lemma_to_original:
            for lemma, row in lookup_batch(conn, list(lemma_to_original.keys())).items():
                if row["en_compressed"]:
                    lemma_found[lemma_to_original[lemma]] = row["en_compressed"]

    result = []
    for word in words:
        stripped = word.strip(_STRIP_CHARS)
        if not stripped:
            continue
        lookup = stripped.lower()
        row = found.get(lookup)
        if row and row["en_compressed"]:
            result.append(row["en_compressed"])
        elif lookup in lemma_found:
            result.append(lemma_found[lookup])
        elif _RE_CYRILLIC.search(stripped):
            return None  # немає перекладу — не вигадуємо
        else:
            result.append(word)  # латинський токен — зберігаємо з пунктуацією

    return " ".join(result).lower().strip() or None


def compose_en(clean_ua: str, conn: sqlite3.Connection) -> str | None:
    """Чиста UA → стислий EN: crack_open → compress → strict map."""
    text = crack_open(clean_ua).text
    text = compress(text)
    return strict_map_to_en(text, conn)


def target_ok(src: str, tgt: str | None) -> bool:
    """Продакшн-гейт приймання (дзеркалить mapper._try_seq2seq_expressions):
    всі токени латиницею, довжина 0 < len(tgt) <= len(src)+2, без <unk>."""
    if not tgt:
        return False
    tgt_words = tgt.split()
    if not tgt_words or len(tgt_words) > len(src.split()) + 2:
        return False
    if "<unk>" in tgt:
        return False
    return all(_RE_LATIN_TOKEN.match(w) for w in tgt_words)
