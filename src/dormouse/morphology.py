"""Спільний модуль морфологічного аналізу (pymorphy3).

Єдиний _get_morph() для всього проєкту — rule_engine, search, mapper.
"""

import re

# pymorphy3 — optional, fallback на exact match
_morph = None

# Грамеми для іменників (відмінок + число)
_NOUN_GRAMMEMES = frozenset({
    "nomn", "gent", "datv", "accs", "ablt", "loct", "voct",
    "plur", "sing",
})
# Грамеми для дієслів (час, особа, число, рід)
_VERB_GRAMMEMES = frozenset({
    "plur", "sing",
    "1per", "2per", "3per",
    "past", "pres", "futr",
    "masc", "femn", "neut",
    "impr",
})


def get_morph():
    """Lazy init pymorphy3. Повертає MorphAnalyzer або None."""
    global _morph
    if _morph is None:
        try:
            import pymorphy3
            _morph = pymorphy3.MorphAnalyzer(lang="uk")
        except ImportError:
            _morph = False  # не встановлено
    return _morph if _morph else None


def lemmatize(word: str) -> str | None:
    """Повертає лему слова або None якщо pymorphy3 недоступний."""
    morph = get_morph()
    if not morph:
        return None
    return morph.parse(word)[0].normal_form


def lemmatize_text(text: str) -> str:
    """Лематизує кожне слово в тексті."""
    morph = get_morph()
    if not morph:
        return text
    words = re.findall(r"[а-яіїєґА-ЯІЇЄҐёыэъʼ']+|\S+", text.lower())
    lemmas = []
    for w in words:
        if re.match(r"[а-яіїєґ]", w):
            lemmas.append(morph.parse(w)[0].normal_form)
        else:
            lemmas.append(w)
    return " ".join(lemmas)


def inflect_replacement(word_in: str, replacement: str) -> str:
    """Ставить replacement у граматичну форму word_in через pymorphy3."""
    morph = get_morph()
    if not morph:
        return replacement

    p_in = morph.parse(word_in)[0]
    p_out = morph.parse(replacement)[0]

    pos_in = p_in.tag.POS
    grammemes = set(p_in.tag.grammemes)

    if pos_in in ("VERB", "INFN", "GRND", "PRTF", "PRTS"):
        target = grammemes & _VERB_GRAMMEMES
    else:
        target = grammemes & _NOUN_GRAMMEMES

    if not target:
        return replacement

    inflected = p_out.inflect(target)
    return inflected.word if inflected else replacement
