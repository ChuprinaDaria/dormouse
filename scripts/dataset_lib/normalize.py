"""Text normalization and language gating for dataset pairs."""

from __future__ import annotations

import re
import unicodedata

from dormouse.rule_engine import _APOSTROPHE_RE

_RE_SPACES = re.compile(r"\s+")
_RE_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_UA_MARKERS = set("іїєґІЇЄҐ")
_RU_MARKERS = set("ыэъёЫЭЪЁ")


def normalize(text: str) -> str:
    """NFC + apostrophe normalization + trim + collapse whitespace.

    Apostrophes are folded to U+0027 with the same regex crack_open uses, so
    stored pairs match what the runtime pipeline sees.
    """
    text = unicodedata.normalize("NFC", text)
    text = _APOSTROPHE_RE.sub("'", text)
    return _RE_SPACES.sub(" ", text).strip()


def is_ukrainian(text: str, min_cyrillic_share: float = 0.6) -> bool:
    """Heuristic uk gate: mostly Cyrillic letters, UA markers outweigh RU markers.

    Dependency-free by design — langid models are overkill for filtering GEC
    corpora that are already predominantly Ukrainian.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    cyr = sum(1 for ch in letters if _RE_CYRILLIC.match(ch))
    if cyr / len(letters) < min_cyrillic_share:
        return False
    ua = sum(1 for ch in text if ch in _UA_MARKERS) + text.count("'")
    ru = sum(1 for ch in text if ch in _RU_MARKERS)
    return ua >= ru
