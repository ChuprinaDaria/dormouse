"""Парсери файлів для семантичного пошуку.

Підтримка: TXT, CSV, XLSX (openpyxl), PDF (pymupdf).
Chunking: мердж коротких, розбиття довгих по реченнях.
"""

from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path

# Евристичні назви текстових колонок (укр + англ)
_TEXT_COLUMN_NAMES = [
    "text", "content", "message", "comment", "body", "description",
    "текст", "зміст", "коментар", "опис", "повідомлення",
    "name", "назва",
]


class UnsupportedFormat(Exception):
    """Непідтримуваний формат файлу."""


def parse_file(path: str | Path, **kwargs) -> list[str]:
    """Диспетчер: обирає парсер за розширенням файлу."""
    path = Path(path)
    ext = path.suffix.lower()
    dispatch = {
        ".txt": parse_txt,
        ".csv": parse_csv,
        ".xlsx": parse_xlsx,
        ".pdf": parse_pdf,
    }
    parser = dispatch.get(ext)
    if parser is None:
        raise UnsupportedFormat(f"Формат {ext} не підтримується. Доступні: {', '.join(dispatch)}")
    return parser(path, **kwargs)


def parse_txt(path: str | Path, encoding: str = "utf-8") -> list[str]:
    """Розбиття TXT по порожніх рядках (\\n\\s*\\n)."""
    path = Path(path)
    content = path.read_text(encoding=encoding)
    if not content.strip():
        return []
    paragraphs = re.split(r"\n\s*\n", content)
    return [p.strip() for p in paragraphs if p.strip()]


def parse_csv(
    path: str | Path,
    column: str | None = None,
    encoding: str = "utf-8",
) -> list[str]:
    """CSV з автодетектом текстової колонки та encoding fallback."""
    path = Path(path)
    raw = path.read_bytes()

    # Encoding fallback: utf-8 → cp1251 → latin-1
    for enc in [encoding, "cp1251", "latin-1"]:
        try:
            text = raw.decode(enc)
            # Перевірка що декодування дало щось осмислене
            text.encode("utf-8")
            break
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
    else:
        text = raw.decode("latin-1")  # latin-1 ніколи не фейлить

    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        return []

    # Визначення текстової колонки
    if column is None:
        column = _find_text_column(reader.fieldnames)
    if column is None:
        # Якщо не знайшли — беремо першу колонку
        column = reader.fieldnames[0]

    results = []
    for row in reader:
        val = row.get(column, "")
        if val and val.strip():
            results.append(val.strip())
    return results


def parse_xlsx(path: str | Path, sheet: str | None = None, column: str | None = None) -> list[str]:
    """Excel через openpyxl (lazy import)."""
    try:
        import openpyxl
    except ImportError:
        raise ImportError(
            "openpyxl потрібен для .xlsx файлів: pip install openpyxl"
        ) from None

    path = Path(path)
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet] if sheet else wb.active

    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not rows:
        return []

    headers = [str(h).strip() if h else "" for h in rows[0]]

    if column is None:
        column = _find_text_column(headers)
    if column is None:
        column = headers[0]

    try:
        col_idx = headers.index(column)
    except ValueError:
        raise ValueError(f"Колонка '{column}' не знайдена. Доступні: {headers}") from None

    results = []
    for row in rows[1:]:
        if col_idx < len(row) and row[col_idx]:
            val = str(row[col_idx]).strip()
            if val:
                results.append(val)
    return results


def parse_pdf(path: str | Path) -> list[str]:
    """PDF через pymupdf (lazy import)."""
    try:
        import pymupdf  # noqa: F811
    except ImportError:
        raise ImportError(
            "pymupdf потрібен для .pdf файлів: pip install pymupdf"
        ) from None

    path = Path(path)
    doc = pymupdf.open(str(path))
    texts = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            texts.append(text)
    doc.close()

    if not texts:
        warnings.warn(f"PDF {path.name} не містить тексту (можливо скановані сторінки)")
    return texts


def chunk_texts(
    texts: list[str],
    min_words: int = 50,
    max_words: int = 300,
) -> list[str]:
    """Мердж коротких текстів, розбиття довгих по реченнях.

    - Тексти коротші за min_words мерджаться з наступними
    - Тексти довші за max_words розбиваються по реченнях
    - Нормальні тексти залишаються як є
    """
    if not texts:
        return []

    # Фільтруємо порожні
    texts = [t.strip() for t in texts if t.strip()]
    if not texts:
        return []

    # Крок 1: мердж коротких
    merged: list[str] = []
    buffer = ""
    for text in texts:
        if buffer:
            buffer = buffer + " " + text
        else:
            buffer = text

        if len(buffer.split()) >= min_words:
            merged.append(buffer)
            buffer = ""

    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)

    # Крок 2: розбиття довгих
    result: list[str] = []
    for text in merged:
        words = text.split()
        if len(words) <= max_words:
            result.append(text)
        else:
            result.extend(_split_by_sentences(text, max_words))

    return result


def _split_by_sentences(text: str, max_words: int) -> list[str]:
    """Розбиття тексту по реченнях з обмеженням max_words."""
    # Розбиваємо по кінцях речень (.!?), зберігаючи пунктуацію
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 1:
        # Немає речень — ріжемо по словах
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_words):
            chunks.append(" ".join(words[i:i + max_words]))
        return chunks

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate.split()) > max_words and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def _find_text_column(fieldnames: list[str] | tuple[str, ...]) -> str | None:
    """Евристика: знайти текстову колонку за назвою."""
    lower_names = {name.lower().strip(): name for name in fieldnames if name}
    for candidate in _TEXT_COLUMN_NAMES:
        if candidate in lower_names:
            return lower_names[candidate]
    return None
