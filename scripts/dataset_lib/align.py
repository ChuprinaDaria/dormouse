"""Token-level alignment of (dirty, clean) sentence pairs into expression windows.

The deployed seq2seq sees 2-4 word windows (mapper._try_seq2seq_expressions),
so sentence/document GEC pairs must be cut into short aligned windows around
the actual edits instead of being fed whole.
"""

from __future__ import annotations

from difflib import SequenceMatcher

WHOLE_PAIR_MAX_WORDS = 6


def extract_windows(
    dirty: str,
    clean: str,
    max_n: int = 4,
    ctx: int = 1,
) -> list[tuple[str, str]]:
    """Вирізає (dirty_window, clean_window) навколо кожного зміненого регіону.

    ±ctx незмінних слів контексту; dirty-вікно обмежене max_n словами (як
    n-gram сканер у mapper). Коротка пара цілком (≤ WHOLE_PAIR_MAX_WORDS слів)
    додається як окреме вікно.
    """
    dirty_words = dirty.split()
    clean_words = clean.split()
    windows: list[tuple[str, str]] = []

    if 0 < len(dirty_words) <= WHOLE_PAIR_MAX_WORDS and clean_words:
        windows.append((dirty, clean))

    matcher = SequenceMatcher(a=dirty_words, b=clean_words, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        d_lo, d_hi = max(0, i1 - ctx), min(len(dirty_words), i2 + ctx)
        c_lo, c_hi = max(0, j1 - ctx), min(len(clean_words), j2 + ctx)
        if d_hi - d_lo == 0 or d_hi - d_lo > max_n:
            continue
        c_win = clean_words[c_lo:c_hi]
        if not c_win:
            continue
        windows.append((" ".join(dirty_words[d_lo:d_hi]), " ".join(c_win)))

    # дедуп у межах пари, зі збереженням порядку
    seen = set()
    out = []
    for w in windows:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out
