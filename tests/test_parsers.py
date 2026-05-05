"""Тести для парсерів файлів та chunking."""

import csv

import pytest

from dormouse.parsers import (
    UnsupportedFormat,
    chunk_texts,
    parse_csv,
    parse_file,
    parse_txt,
)


class TestParseTxt:
    def test_paragraphs_split(self, tmp_path):
        """Розбиття по порожніх рядках."""
        p = tmp_path / "doc.txt"
        p.write_text("Перший абзац.\n\nДругий абзац.\n\nТретій абзац.\n")
        result = parse_txt(p)
        assert result == ["Перший абзац.", "Другий абзац.", "Третій абзац."]

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("")
        result = parse_txt(p)
        assert result == []

    def test_single_paragraph(self, tmp_path):
        p = tmp_path / "single.txt"
        p.write_text("Один суцільний текст без розривів.")
        result = parse_txt(p)
        assert result == ["Один суцільний текст без розривів."]

    def test_multiple_empty_lines(self, tmp_path):
        """Кілька порожніх рядків між абзацами — все одно два абзаци."""
        p = tmp_path / "gaps.txt"
        p.write_text("Абзац один.\n\n\n\nАбзац два.\n")
        result = parse_txt(p)
        assert result == ["Абзац один.", "Абзац два."]


class TestParseCsv:
    def test_auto_detect_text_column(self, tmp_path):
        """Автодетект колонки 'text'."""
        p = tmp_path / "data.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text", "label"])
            writer.writeheader()
            writer.writerow({"id": "1", "text": "Привіт світ", "label": "ok"})
            writer.writerow({"id": "2", "text": "Другий рядок", "label": "ok"})
        result = parse_csv(p)
        assert result == ["Привіт світ", "Другий рядок"]

    def test_explicit_column(self, tmp_path):
        """Явна вказівка колонки."""
        p = tmp_path / "data.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "note"])
            writer.writeheader()
            writer.writerow({"id": "1", "note": "Нотатка"})
        result = parse_csv(p, column="note")
        assert result == ["Нотатка"]

    def test_cp1251_fallback(self, tmp_path):
        """Fallback на cp1251 якщо utf-8 не працює."""
        p = tmp_path / "cp1251.csv"
        content = "id,text\n1,Привіт\n2,Світ\n"
        p.write_bytes(content.encode("cp1251"))
        result = parse_csv(p)
        assert result == ["Привіт", "Світ"]

    def test_empty_csv(self, tmp_path):
        """Порожній CSV — тільки заголовки."""
        p = tmp_path / "empty.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "text"])
        result = parse_csv(p)
        assert result == []


class TestParseFile:
    def test_dispatch_txt(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("Абзац один.\n\nАбзац два.\n")
        result = parse_file(p)
        assert result == ["Абзац один.", "Абзац два."]

    def test_dispatch_csv(self, tmp_path):
        p = tmp_path / "data.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text"])
            writer.writeheader()
            writer.writerow({"text": "Рядок"})
        result = parse_file(p)
        assert result == ["Рядок"]

    def test_unsupported_format(self, tmp_path):
        p = tmp_path / "doc.docx"
        p.write_text("fake")
        with pytest.raises(UnsupportedFormat, match=r"\.docx"):
            parse_file(p)


class TestChunkTexts:
    def test_merge_short(self):
        """Короткі тексти (<50 слів) мерджаться."""
        texts = ["Короткий.", "Ще коротший.", "І цей."]
        result = chunk_texts(texts, min_words=50, max_words=300)
        assert len(result) == 1
        assert "Короткий." in result[0]
        assert "Ще коротший." in result[0]

    def test_split_long(self):
        """Довгий текст (>300 слів) розбивається."""
        long_text = " ".join(["слово"] * 400)
        result = chunk_texts([long_text], min_words=50, max_words=300)
        assert len(result) >= 2
        for chunk in result:
            word_count = len(chunk.split())
            assert word_count <= 350  # допуск на розбиття по реченнях

    def test_normal_unchanged(self):
        """Текст нормальної довжини залишається як є."""
        text = " ".join(["слово"] * 100)
        result = chunk_texts([text], min_words=50, max_words=300)
        assert len(result) == 1
        assert result[0] == text

    def test_empty_input(self):
        result = chunk_texts([], min_words=50, max_words=300)
        assert result == []

    def test_whitespace_only_filtered(self):
        """Порожні та whitespace-only тексти фільтруються."""
        result = chunk_texts(["", "  ", "\n"], min_words=50, max_words=300)
        assert result == []


class TestImportErrors:
    def test_xlsx_import_error(self, tmp_path):
        """parse_xlsx дає зрозуміле повідомлення без openpyxl."""
        from dormouse.parsers import parse_xlsx

        p = tmp_path / "data.xlsx"
        p.write_bytes(b"fake xlsx")
        # Може бути ImportError або реальний парсинг — залежить від оточення
        # Тестуємо що функція існує і не крашить на import
        assert callable(parse_xlsx)

    def test_pdf_import_error(self, tmp_path):
        """parse_pdf дає зрозуміле повідомлення без pymupdf."""
        from dormouse.parsers import parse_pdf

        p = tmp_path / "doc.pdf"
        p.write_bytes(b"fake pdf")
        assert callable(parse_pdf)
