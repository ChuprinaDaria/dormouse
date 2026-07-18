"""PII-скраб для датасет-пайплайну і колектора пар.

Замінює персональні дані плейсхолдерами-одиничними токенами (<PHONE>, <NAME>…),
які переживають токенізацію WordVocab (lower().split()). Порядок правил
фіксований: IBAN і картки до телефонів, бо довгі цифрові послідовності
перетинаються з телефонними патернами.

Імена — евристика на pymorphy3 (грамеми Name/Surn/Patr), без важких NER-моделей.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dormouse.morphology import get_morph

# UA IBAN: "UA" + 27 цифр (29 символів)
_RE_IBAN = re.compile(r"\bUA\d{27}\b")

# Кандидати в номери карток: 13-19 цифр з пробілами/дефісами, далі Luhn-перевірка
_RE_CARD = re.compile(r"\b(?:\d[ \-]?){12,18}\d\b")

# Українські телефони: +380 XX XXX XX XX, 0XX-XXX-XX-XX, (0XX) XXX XX XX...
_RE_PHONE = re.compile(
    r"(?:\+?38[\s\-]?)?\(?0\d{2}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b"
    r"|\+\d{10,14}\b"
)

_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

_RE_URL = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
_RE_URL_SENSITIVE = re.compile(r"[?&#][^\s]*?(?:token|key|secret|sig|password|auth)[^\s]*", re.I)
_RE_URL_DOMAIN = re.compile(r"^(https?://[^/\s?#]+)", re.IGNORECASE)

# Telegram/соцмережеві хендли
_RE_HANDLE = re.compile(r"(?<![\w@])@[A-Za-z][A-Za-z0-9_]{4,31}\b")

# Адреси: вул./просп./пров./бульв. + назва [+ номер будинку]
_RE_ADDR = re.compile(
    r"\b(?:вул|просп|пров|бульв|пл)\.\s*[А-ЯІЇЄҐ][\w'\-]+(?:\s*,?\s*(?:буд\.\s*)?\d+[а-яА-Я]?)?",
)

_NAME_GRAMMEMES = frozenset({"Name", "Surn", "Patr"})
_RE_CAPITALIZED = re.compile(r"^[А-ЯІЇЄҐ][а-яіїєґ'\-]+$")
_RE_SENT_END = re.compile(r"[.!?…]\s*$")


@dataclass
class ScrubResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.counts)


def _luhn_ok(digits: str) -> bool:
    """Luhn-чексума — відсікає випадкові цифрові послідовності від карток."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _scrub_urls(text: str, counts: dict[str, int]) -> str:
    def repl(m: re.Match) -> str:
        url = m.group(0)
        if _RE_URL_SENSITIVE.search(url):
            counts["<URL>"] = counts.get("<URL>", 0) + 1
            return "<URL>"
        # без токенів — лишаємо тільки домен (шлях може містити slug з іменами)
        domain = _RE_URL_DOMAIN.match(url)
        if domain and domain.group(1) != url.rstrip("/"):
            counts["<URL>"] = counts.get("<URL>", 0) + 1
            return domain.group(1)
        return url

    return _RE_URL.sub(repl, text)


def _scrub_cards(text: str, counts: dict[str, int]) -> str:
    def repl(m: re.Match) -> str:
        digits = re.sub(r"[ \-]", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            counts["<CARD>"] = counts.get("<CARD>", 0) + 1
            return "<CARD>"
        return m.group(0)

    return _RE_CARD.sub(repl, text)


def _scrub_names(text: str, counts: dict[str, int], min_score: float = 0.4) -> str:
    """Евристика імен: капіталізоване слово не на початку речення з грамемами
    Name/Surn/Patr у найкращому розборі pymorphy3."""
    morph = get_morph()
    if morph is None:
        return text

    tokens = text.split(" ")
    out = []
    sentence_start = True
    for tok in tokens:
        core = tok.strip(".,!?;:()\"'«»")
        is_candidate = bool(_RE_CAPITALIZED.match(core))
        if is_candidate:
            parse = morph.parse(core)[0]
            grammemes = set(parse.tag.grammemes)
            is_name = bool(grammemes & _NAME_GRAMMEMES) and parse.score >= min_score
            # на початку речення вимагаємо Surn/Patr (Іван на старті — часто
            # звичайне слово з великої букви)
            if is_name and sentence_start and not (grammemes & {"Surn", "Patr"}):
                is_name = False
            if is_name:
                counts["<NAME>"] = counts.get("<NAME>", 0) + 1
                out.append(tok.replace(core, "<NAME>"))
                sentence_start = False
                continue
        out.append(tok)
        sentence_start = bool(_RE_SENT_END.search(tok)) if tok else sentence_start
    return " ".join(out)


def _count_sub(pattern: re.Pattern, placeholder: str, text: str, counts: dict[str, int]) -> str:
    text, n = pattern.subn(placeholder, text)
    if n:
        counts[placeholder] = counts.get(placeholder, 0) + n
    return text


def scrub(text: str, names: bool = True) -> ScrubResult:
    """Повний PII-скраб тексту. Порядок: IBAN → картки → телефони → email →
    URL → хендли → адреси → імена."""
    counts: dict[str, int] = {}
    text = _count_sub(_RE_IBAN, "<IBAN>", text, counts)
    text = _scrub_cards(text, counts)
    text = _count_sub(_RE_PHONE, "<PHONE>", text, counts)
    text = _count_sub(_RE_EMAIL, "<EMAIL>", text, counts)
    text = _scrub_urls(text, counts)
    text = _count_sub(_RE_HANDLE, "<HANDLE>", text, counts)
    text = _count_sub(_RE_ADDR, "<ADDR>", text, counts)
    if names:
        text = _scrub_names(text, counts)
    return ScrubResult(text=text, counts=counts)


def scrub_pair(
    dirty: str, clean: str, max_placeholder_share: float = 0.5
) -> tuple[str, str] | None:
    """Скрабить обидві сторони пари. None (дропнути пару), якщо будь-яка
    сторона стала переважно плейсхолдерами — така пара нічому не вчить."""
    results = (scrub(dirty), scrub(clean))
    for res in results:
        words = res.text.split()
        if not words:
            return None
        placeholders = sum(1 for w in words if w.startswith("<") and w.endswith(">"))
        if placeholders / len(words) > max_placeholder_share:
            return None
    return results[0].text, results[1].text
