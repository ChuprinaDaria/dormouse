"""Тести для роботи з БД."""

import sqlite3

from dormouse.db import get_connection


class TestDatabase:
    """Створення БД і таблиць."""

    def test_creates_tables(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = get_connection(db_path)

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t["name"] for t in tables]

        assert "raw_texts" in table_names
        assert "lexicon" in table_names
        assert "pairs" in table_names
        assert "normal_texts" in table_names
        conn.close()

    def test_raw_texts_has_corpus(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = get_connection(db_path)

        conn.execute(
            "INSERT INTO raw_texts (source, text, corpus, word_count) VALUES (?, ?, ?, ?)",
            ("test", "тест текст", "розмовний", 2),
        )
        conn.commit()

        row = conn.execute("SELECT corpus FROM raw_texts WHERE id = 1").fetchone()
        assert row["corpus"] == "розмовний"
        conn.close()

    def test_insert_pair(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = get_connection(db_path)

        conn.execute(
            "INSERT INTO pairs (source_text, optimized_text, style, method) VALUES (?, ?, ?, ?)",
            ("шо там", "що там", "суржик", "rule"),
        )
        conn.commit()

        row = conn.execute("SELECT * FROM pairs WHERE id = 1").fetchone()
        assert row["source_text"] == "шо там"
        assert row["optimized_text"] == "що там"
        conn.close()

    def test_lexicon_unique_word(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = get_connection(db_path)

        conn.execute(
            "INSERT INTO lexicon (word, type) VALUES (?, ?)",
            ("шо", "surzhyk"),
        )
        conn.commit()

        try:
            conn.execute(
                "INSERT INTO lexicon (word, type) VALUES (?, ?)",
                ("шо", "surzhyk"),
            )
            assert False, "Мав бути IntegrityError"
        except sqlite3.IntegrityError:
            pass

        conn.close()
