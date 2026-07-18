"""Corruption utilities: rule inversion (clean → surzhyk/slang) and noise ops.

build_inverse_rules() is shared by generate_expression_pairs.py (dirty variants
of lexicon expressions) and synth_corrupt.py (full sentence corruption via the
Corruptor class).
"""

from __future__ import annotations

import json
import random
import re
from importlib.resources import files

from dormouse.compressor import FILLER_PHRASES, INTENSIFIERS
from dormouse.mapper import _TRANSLIT
from dormouse.morphology import get_morph

# Сусідні клавіші розкладки ЙЦУКЕН (горизонталь + вертикаль)
_QWERTY_ROWS = ["йцукенгшщзхї", "фівапролджє", "ячсмитьбю"]
KEYBOARD_NEIGHBORS: dict[str, str] = {}
for _r, _row in enumerate(_QWERTY_ROWS):
    for _c, _ch in enumerate(_row):
        neighbors = []
        if _c > 0:
            neighbors.append(_row[_c - 1])
        if _c < len(_row) - 1:
            neighbors.append(_row[_c + 1])
        for _other in (_QWERTY_ROWS[_r - 1] if _r > 0 else "",
                       _QWERTY_ROWS[_r + 1] if _r < len(_QWERTY_ROWS) - 1 else ""):
            if _c < len(_other):
                neighbors.append(_other[_c])
        KEYBOARD_NEIGHBORS[_ch] = "".join(neighbors)

_RE_WORD = re.compile(r"^[а-яіїєґ']+$")

# 22 філери з to:null у replacements.json інжектуються разом з цими
_INSERTABLE_FILLERS = ["ну", "коротше", "типу", "канєшна", "вообщем"]


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


def null_fillers(replacements: dict | None = None) -> list[str]:
    """Філери з to:null — брудні слова, які нормалізація просто видаляє."""
    if replacements is None:
        replacements = load_replacements()
    return [r["from"].lower() for r in replacements["replacements"] if not r.get("to")]


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


class Corruptor:
    """Псує чисті речення за конфігом з імовірностями. Детермінований по rng."""

    def __init__(self, cfg: dict, rng: random.Random):
        self.cfg = cfg["ops"]
        self.min_ops = cfg.get("min_ops", 1)
        self.rng = rng
        replacements = load_replacements()
        self.inverse = build_inverse_rules(replacements)
        self.fillers = null_fillers(replacements) + _INSERTABLE_FILLERS
        self.intensifiers = sorted(INTENSIFIERS)
        self.filler_phrases = FILLER_PHRASES
        self._lemma_inverse_cache: dict[str, list[str] | None] = {}

    def _p(self, op: str, default: float = 0.0) -> float:
        return self.cfg.get(op, {}).get("p", default)

    def _lemma_inverse(self, word: str) -> list[str] | None:
        """Інверсія через лему: 'щось' у формі 'щосьому' теж знаходить варіанти."""
        if word in self._lemma_inverse_cache:
            return self._lemma_inverse_cache[word]
        result = None
        morph = get_morph()
        if morph:
            lemma = morph.parse(word)[0].normal_form
            if lemma in self.inverse:
                result = self.inverse[lemma]
        self._lemma_inverse_cache[word] = result
        return result

    def _op_rule_inversion(self, words: list[str]) -> tuple[list[str], int]:
        max_swaps = self.cfg.get("rule_inversion", {}).get("max_per_sentence", 2)
        applied = 0
        candidates = []
        for i, w in enumerate(words):
            low = w.lower()
            if low in self.inverse:
                candidates.append((i, self.inverse[low]))
            else:
                by_lemma = self._lemma_inverse(low)
                if by_lemma:
                    candidates.append((i, by_lemma))
        self.rng.shuffle(candidates)
        for i, variants in candidates[:max_swaps]:
            words[i] = self.rng.choice(variants)
            applied += 1
        return words, applied

    def _op_typo_neighbor(self, words: list[str]) -> tuple[list[str], int]:
        p = self._p("typo_neighbor")
        applied = 0
        for i, w in enumerate(words):
            if len(w) >= 4 and _RE_WORD.match(w.lower()) and self.rng.random() < p:
                pos = self.rng.randrange(1, len(w) - 1)
                ch = w[pos].lower()
                if ch in KEYBOARD_NEIGHBORS and KEYBOARD_NEIGHBORS[ch]:
                    typo = self.rng.choice(KEYBOARD_NEIGHBORS[ch])
                    words[i] = w[:pos] + typo + w[pos + 1:]
                    applied += 1
        return words, applied

    def _op_char_drop(self, words: list[str]) -> tuple[list[str], int]:
        p = self._p("char_drop")
        applied = 0
        for i, w in enumerate(words):
            if len(w) >= 5 and _RE_WORD.match(w.lower()) and self.rng.random() < p:
                pos = self.rng.randrange(1, len(w) - 1)
                words[i] = w[:pos] + w[pos + 1:]
                applied += 1
        return words, applied

    def _op_translit_word(self, words: list[str]) -> tuple[list[str], int]:
        p = self._p("translit_word")
        applied = 0
        for i, w in enumerate(words):
            if _RE_WORD.match(w.lower()) and self.rng.random() < p:
                words[i] = "".join(_TRANSLIT.get(ch, ch) for ch in w.lower())
                applied += 1
        return words, applied

    def _op_filler_insert(self, words: list[str]) -> tuple[list[str], int]:
        if self.rng.random() >= self._p("filler_insert"):
            return words, 0
        # 50/50: брудне слово-філер або ввічлива фраза-обгортка на початку
        if self.rng.random() < 0.5:
            filler = self.rng.choice(self.fillers)
            pos = self.rng.randrange(0, min(3, len(words) + 1))
            words.insert(pos, filler)
        else:
            words[:0] = self.rng.choice(self.filler_phrases).split()
        return words, 1

    def _op_intensifier_insert(self, words: list[str]) -> tuple[list[str], int]:
        if self.rng.random() >= self._p("intensifier_insert"):
            return words, 0
        pos = self.rng.randrange(0, len(words) + 1)
        words.insert(pos, self.rng.choice(self.intensifiers))
        return words, 1

    def corrupt(self, clean: str) -> tuple[str, list[str]] | None:
        """Повертає (dirty, застосовані ops) або None якщо застосовано < min_ops."""
        words = clean.split()
        if not words:
            return None
        applied_ops: list[str] = []

        if self.rng.random() < self._p("rule_inversion"):
            words, n = self._op_rule_inversion(words)
            applied_ops.extend(["rule_inversion"] * n)
        for op in ("typo_neighbor", "char_drop", "translit_word",
                   "filler_insert", "intensifier_insert"):
            words, n = getattr(self, f"_op_{op}")(words)
            applied_ops.extend([op] * n)

        dirty = " ".join(words)
        if len(applied_ops) < self.min_ops or dirty == clean:
            return None
        return dirty, applied_ops
