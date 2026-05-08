"""UA→EN маппінг для LLM через SQLite лексикон.

Шар 3 pipeline: переклад українського тексту на англійську.
Максимальне стиснення для cloud моделей (OpenAI/Claude/Gemini).

Всі маппінги живуть в lexicon.db — B-Tree індекс, <1ms на запит.
v2: pymorphy3 lemma fallback — "помилці" знаходить "помилка"→"error".
"""

import re

from dormouse.lexicon_db import get_expressions, get_lexicon, lookup_batch
from dormouse.morphology import get_morph

# Транслітерація кирилиця → латиниця (fallback)
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g",
    "д": "d", "е": "e", "є": "ye", "ж": "zh", "з": "z",
    "и": "y", "і": "i", "ї": "yi", "й": "y", "к": "k",
    "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
    "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ь": "", "ю": "yu", "я": "ya", "ъ": "", "ы": "y",
    "э": "e", "ё": "yo",
}

# Lazy connection
_conn = None


def _get_conn():
    global _conn
    if _conn is None:
        _conn = get_lexicon()
    return _conn


def _transliterate(text: str) -> str:
    """Транслітерація кирилиці в латиницю."""
    result = []
    for ch in text:
        low = ch.lower()
        if low in _TRANSLIT:
            tr = _TRANSLIT[low]
            result.append(tr.upper() if ch.isupper() and tr else tr)
        else:
            result.append(ch)
    return "".join(result)


def _apply_expressions(text: str) -> str:
    """Замінює багатослівні вирази (біграми, триграми) через SQLite."""
    conn = _get_conn()

    for ngram in (3, 2):
        expressions = get_expressions(conn, ngram)
        for expr in expressions:
            en_val = expr["en_compressed"]
            if not en_val:
                continue
            pattern = re.compile(r"\b" + re.escape(expr["word"]) + r"\b", re.IGNORECASE)
            text = pattern.sub(en_val, text)
    return text


def _try_seq2seq_expressions(text: str) -> str:
    """Перекладає короткі вирази (2-4 слова) через seq2seq v2.

    Сканує текст вікном 4→3→2 слова, перекладає вирази де seq2seq впевнений.
    Замінені вирази позначає, щоб word-by-word їх не чіпав.
    """
    try:
        from dormouse.seq2seq import translate_expression
    except ImportError:
        return text

    words = text.split()
    if len(words) < 2:
        return text

    replaced = [False] * len(words)
    result = list(words)

    for ngram_size in (4, 3, 2):
        for i in range(len(words) - ngram_size + 1):
            if any(replaced[i: i + ngram_size]):
                continue

            chunk = " ".join(words[i: i + ngram_size])
            # Пропускаємо якщо вже латиниця
            if not re.search(r"[а-яіїєґ]", chunk, re.IGNORECASE):
                continue

            translated = translate_expression(chunk)
            if not translated:
                continue

            # Перевірка якості: всі слова латиницею, кількість слів адекватна
            tr_words = translated.split()
            all_latin = all(re.match(r"^[a-zA-Z0-9?.!,'\"-]+$", w) for w in tr_words)
            reasonable_len = 0 < len(tr_words) <= ngram_size + 2

            if all_latin and reasonable_len:
                result[i] = translated
                for j in range(i + 1, i + ngram_size):
                    result[j] = ""
                    replaced[j] = True
                replaced[i] = True

    return " ".join(w for w in result if w)


def map_to_en(text: str, use_seq2seq: bool = False) -> str:
    """Перекладає/маппить українській текст на англійську.

    Максимальне стиснення для cloud моделей.
    Pipeline: [seq2seq v2 (вирази)] → lexicon expressions → word-by-word → transliteration.

    Args:
        text: Нормалізований і стиснутий текст.
        use_seq2seq: Увімкнути seq2seq v2 для перекладу виразів (потребує натреновану модель).
    """
    if not text or not text.strip():
        return text

    # 0. Спробувати seq2seq v2 на коротких виразах (2-4 слова)
    if use_seq2seq:
        text = _try_seq2seq_expressions(text)

    conn = _get_conn()
    col = "en_compressed"

    # 1. Спочатку multi-word expressions (longest first)
    text = _apply_expressions(text)

    # 2. Потім single words через batch lookup
    words = text.split()
    clean_words = []
    for w in words:
        stripped = w.strip(".,!?;:()\"'«»—–-…")
        if stripped:
            clean_words.append(stripped.lower())

    # Один SQL запит на всі слова
    found = lookup_batch(conn, clean_words)

    # 2.1. Lemma fallback — шукаємо леми для слів яких нема в лексиконі
    morph = get_morph()
    lemma_found: dict[str, str] = {}
    if morph:
        missing = [w for w in clean_words if w not in found or not found[w][col]]
        if missing:
            # Лематизуємо і шукаємо леми
            lemma_to_original: dict[str, str] = {}
            for w in missing:
                lemma = morph.parse(w)[0].normal_form
                if lemma != w and lemma not in lemma_to_original:
                    lemma_to_original[lemma] = w

            if lemma_to_original:
                lemma_rows = lookup_batch(conn, list(lemma_to_original.keys()))
                for lemma, row in lemma_rows.items():
                    if row[col]:
                        original_word = lemma_to_original[lemma]
                        lemma_found[original_word] = row[col]

    # 3. Замінюємо
    result = []
    for word in words:
        stripped = word.strip(".,!?;:()\"'«»—–-…")
        if not stripped:
            result.append(word)
            continue

        prefix = word[: word.index(stripped)] if stripped in word else ""
        end = word.rindex(stripped) + len(stripped) if stripped in word else len(word)
        suffix = word[end:]

        lookup = stripped.lower()

        # Пряме знаходження
        row = found.get(lookup)
        if row and row[col]:
            result.append(prefix + row[col] + suffix)
        # Lemma fallback
        elif lookup in lemma_found:
            result.append(prefix + lemma_found[lookup] + suffix)
        elif re.search(r"[а-яіїєґА-ЯІЇЄҐ]", stripped):
            # Fallback: транслітерація
            result.append(prefix + _transliterate(stripped) + suffix)
        else:
            result.append(word)

    return " ".join(result).strip()


# --- Словники для build_lexicon.py (генерація lexicon.db) ---

COMPRESSED_MAP: dict[str, str] = {
    # Привітання / прощання
    "привіт": "hi", "вітаю": "hi",
    "добрий день": "hi", "добрий ранок": "hi", "добрий вечір": "hi",
    "бувай": "bye", "до побачення": "bye",
    "дякую": "thx", "будь ласка": "pls",
    "так": "y", "ні": "n", "добре": "ok", "гаразд": "ok",
    "зрозумів": "got it", "зрозуміла": "got it", "зрозуміло": "clear",
    "як справи": "how?", "що нового": "news?", "яка ціна": "price?",
    "скільки": "how much",
    "нормально": "ok", "погано": "bad", "чудово": "great", "класно": "cool",
    "дивно": "strange", "важливо": "important",
    # Мат
    "бля": "damn", "блять": "damn", "нахуй": "fuck off",
    "хуйня": "bs", "піздєц": "disaster", "піздець": "disaster",
    "заєбісь": "awesome", "поєбало": "broke", "ебать": "wow", "сука": "damn",
    # Меми
    "кукусь": "hi", "кринж": "cringe", "рофл": "lol",
    "вайб": "vibe", "тригер": "trigger", "токсик": "toxic",
    "чілити": "chill", "флексити": "flex",
    # Дії
    "допоможи": "help", "допоможіть": "help",
    "зроби": "do", "зробити": "do", "зробимо": "do",
    "виправ": "fix", "виправити": "fix", "виправлення": "fix",
    "перевір": "check", "перевірити": "check",
    "оновити": "update", "оновлення": "update",
    "видалити": "delete", "створити": "create",
    "знайти": "find", "шукати": "search",
    "надіслати": "send", "отримати": "get",
    "обговорити": "discuss", "обговоримо": "discuss",
    "подивитись": "look", "подивись": "look",
    "працює": "works", "працювати": "work", "не працює": "broken",
    "розгортання": "deploy", "розгортати": "deploy",
    "впасти": "crash", "впало": "crashed", "впав": "crashed",
    # Загальне
    "проблема": "issue", "помилка": "error", "помилки": "errors",
    "функція": "feature", "функції": "features",
    "задача": "task", "задачу": "task", "задачі": "tasks",
    "завдання": "task", "завдань": "tasks",
    "зустріч": "meeting", "зустрічі": "meetings",
    "дзвінок": "call", "відгук": "feedback",
    "програма": "app", "програму": "app",
    "користувач": "user", "користувачі": "users", "користувачів": "users",
    "інформація": "info", "комп'ютер": "pc",
    "крайній термін": "deadline",
    "налаштування": "setup", "налаштовувати": "setup",
    # Прийменники та функціональні слова
    "в": "in", "у": "in", "на": "on", "з": "with", "із": "with",
    "до": "to", "від": "from", "за": "for", "по": "by",
    "для": "for", "через": "via", "між": "between",
    "під": "under", "над": "above", "без": "without", "про": "about",
    # Займенники / вказівні
    "це": "this", "цей": "this", "ця": "this", "ці": "these",
    "той": "that", "та": "and", "ті": "those",
    "він": "he", "вона": "she", "воно": "it", "вони": "they",
    "я": "I", "ми": "we", "ви": "you",
    "мій": "my", "твій": "your", "наш": "our", "їх": "their",
    # Зв'язки
    "тому що": "because", "якщо": "if", "коли": "when",
    "також": "also", "навіть": "even", "мабуть": "maybe",
    "звісно": "sure", "треба": "need", "можна": "can",
    "зараз": "now", "потім": "later", "завжди": "always",
    "ніколи": "never", "іноді": "sometimes",
    "багато": "many", "трохи": "few", "занадто": "too much",
    "краще": "better", "більше": "more", "менше": "less",
    "напевно": "probably", "наче": "seems", "але": "but",
    "що": "that", "як": "how", "де": "where", "хто": "who", "чому": "why",
    "тут": "here", "там": "there", "цього": "this",
    "нічого": "nothing", "щось": "smth",
    "не знаю": "idk", "на мою думку": "imo", "до речі": "btw",
    # Дієслова
    "робити": "do", "роблю": "doing", "робимо": "doing",
    "розумію": "understand", "розумієш": "understand",
    "дивлюсь": "looking", "дивись": "look",
    "говорити": "say", "сказати": "say",
    "писати": "write", "читати": "read",
    "думати": "think", "знати": "know",
    "хотіти": "want", "могти": "can", "бути": "be",
    "є": "is", "був": "was", "буде": "will",
    # Часті слова після нормалізації
    "взагалі": "generally", "видалення": "deletion",
    "додай": "add", "додати": "add", "додаємо": "add",
    "запуск": "launch", "запустити": "run",
    "зміна": "change", "змінити": "change", "зміни": "changes",
    "вибір": "choice", "вибрати": "choose",
    "замість": "instead", "тобто": "i.e.",
    "нарешті": "finally", "спочатку": "first",
    "разом": "together", "окремо": "separately",
    "раніше": "earlier", "пізніше": "later",
    "проте": "however", "однак": "however",
    "причина": "reason", "результат": "result",
    "приклад": "example", "наприклад": "eg",
    "замовлення": "order", "клієнт": "client",
    "сервер": "server", "запит": "request", "відповідь": "response",
    "файл": "file", "папка": "folder", "список": "list",
    "кнопка": "button", "сторінка": "page", "форма": "form",
    "база": "database", "таблиця": "table", "поле": "field",
    "рядок": "row", "стовпець": "column",
    # UA-GEC
    "кілька": "several", "передусім": "first",
    "загалом": "overall", "стежити": "monitor",
    "зазвичай": "usually", "потреба": "need",
    "щодня": "daily", "оскільки": "since",
    "отже": "so", "внесок": "contribution",
    "перебуває": "is located", "залежно": "depending",
    "розв'язати": "solve", "насамперед": "primarily",
    "здебільшого": "mostly", "набагато": "much",
    "доволі": "quite", "справді": "indeed",
}

