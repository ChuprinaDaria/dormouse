"""Corruption utilities: rule inversion (clean → surzhyk/slang) and noise ops.

build_inverse_rules() is shared by generate_expression_pairs.py (dirty variants
of lexicon expressions) and synth_corrupt.py (full sentence corruption).
"""

from __future__ import annotations

import json
import random
from importlib.resources import files


def load_replacements() -> dict:
    """Бандлений replacements.json — той самий, що використовує rule_engine."""
    path = files("dormouse.data").joinpath("replacements.json")
    return json.loads(path.read_text(encoding="utf-8"))


def build_inverse_rules(replacements: dict | None = None) -> dict[str, list[str]]:
    """Інверсна мапа clean → [dirty...] з правил, де to != null.

    Пряма мапа many-to-one (шо/чо/чьо → що), тому інверсія повертає список
    брудних варіантів; вибір конкретного — справа генератора (rng.choice).
    """
    if replacements is None:
        replacements = load_replacements()
    inverse: dict[str, list[str]] = {}
    for rule in replacements["replacements"]:
        clean = rule.get("to")
        if not clean:
            continue  # філери (to: null) інжектуються, не інвертуються
        inverse.setdefault(clean.lower(), []).append(rule["from"].lower())
    return inverse


def dirty_variant(
    text: str, inverse: dict[str, list[str]], rng: random.Random, max_swaps: int = 2
) -> str | None:
    """Замінює до max_swaps слів на брудні варіанти. None якщо нема що міняти."""
    words = text.split()
    candidates = [i for i, w in enumerate(words) if w.lower() in inverse]
    if not candidates:
        return None
    rng.shuffle(candidates)
    for i in candidates[:max_swaps]:
        words[i] = rng.choice(inverse[words[i].lower()])
    result = " ".join(words)
    return result if result != text else None
