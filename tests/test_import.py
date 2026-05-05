"""Тести для скриптів імпорту та генерації пар."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from dormouse.db import get_connection


class TestImportManual:
    """Тести для import_manual.py."""

    def test_import_valid_jsonl(self, tmp_path):
        jsonl_path = tmp_path / "test.jsonl"
        entries = [
            {
                "source": "threads",
                "original": "шо там по тому баґу",
                "optimized": "що з багом",
                "style": "розмовна",
                "lexicon_items": ["шо"],
                "notes": "тест",
            },
            {
                "source": "threads",
                "original": "ну крч все норм",
                "optimized": "все нормально",
                "style": "розмовна",
                "lexicon_items": ["крч", "норм"],
            },
        ]
        jsonl_path.write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in entries),
            encoding="utf-8",
        )

        db_path = tmp_path / "test.db"
        conn = get_connection(db_path)

        from import_manual import import_lexicon, import_pair

        for entry in entries:
            import_pair(conn, entry)
            if entry.get("lexicon_items"):
                import_lexicon(conn, entry["lexicon_items"], "threads")

        conn.commit()

        pairs = conn.execute("SELECT * FROM pairs").fetchall()
        assert len(pairs) == 2
        assert pairs[0]["source_text"] == "шо там по тому баґу"
        assert pairs[0]["method"] == "manual"
        assert pairs[0]["validated"] == 1

        lexicon = conn.execute("SELECT * FROM lexicon ORDER BY word").fetchall()
        words = [r["word"] for r in lexicon]
        assert "шо" in words
        assert "крч" in words
        assert "норм" in words

        conn.close()

    def test_duplicate_pair_skipped(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = get_connection(db_path)

        from import_manual import import_pair

        entry = {
            "original": "тест текст",
            "optimized": "тест",
            "style": "розмовна",
        }

        assert import_pair(conn, entry) is True
        assert import_pair(conn, entry) is False

        conn.close()

    def test_validate_entry(self):
        from import_manual import validate_entry

        assert validate_entry({"original": "a", "optimized": "b", "style": "розмовна"}, 1) == []

        errors = validate_entry({"original": "a", "optimized": "b", "style": "invalid"}, 1)
        assert len(errors) == 1

        errors = validate_entry({"original": "a", "style": "розмовна"}, 1)
        assert len(errors) == 1


class TestGeneratePairs:
    """Тести для генерації пар через rule-based."""

    def test_surzhyk_text_creates_pair(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = get_connection(db_path)

        conn.execute(
            "INSERT INTO raw_texts (source, text, word_count) VALUES (?, ?, ?)",
            ("test", "шо там по тому баґу пофікси плз", 7),
        )
        conn.commit()

        rows = conn.execute(
            "SELECT id, text, source, source_name FROM raw_texts WHERE processed = FALSE"
        ).fetchall()

        from dormouse.rule_engine import crack_open

        for row in rows:
            result = crack_open(row["text"])
            if result.changed:
                conn.execute(
                    "INSERT INTO pairs (source_text, optimized_text, style, method)"
                    " VALUES (?, ?, ?, 'rule')",
                    (row["text"], result.text, "розмовна"),
                )
            conn.execute("UPDATE raw_texts SET processed = TRUE WHERE id = ?", (row["id"],))

        conn.commit()

        pairs = conn.execute("SELECT * FROM pairs").fetchall()
        assert len(pairs) == 1
        assert "що" in pairs[0]["optimized_text"]
        assert pairs[0]["method"] == "rule"

        raw = conn.execute("SELECT processed FROM raw_texts WHERE id = 1").fetchone()
        assert raw["processed"] == 1

        conn.close()

    def test_clean_text_no_pair(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = get_connection(db_path)

        conn.execute(
            "INSERT INTO raw_texts (source, text, word_count) VALUES (?, ?, ?)",
            ("test", "Це нормальний український текст без суржику", 6),
        )
        conn.commit()

        from dormouse.rule_engine import crack_open

        row = conn.execute("SELECT * FROM raw_texts WHERE id = 1").fetchone()
        result = crack_open(row["text"])
        assert result.changed is False

        pairs = conn.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
        assert pairs == 0

        conn.close()


class TestCollectTelegram:
    """Тести для допоміжних функцій collect_telegram."""

    def test_hash_author(self):
        from collect_telegram import hash_author

        h = hash_author(12345)
        assert isinstance(h, str)
        assert len(h) == 16
        assert hash_author(12345) == h
        assert hash_author(None) is None

    def test_is_valid(self):
        from collect_telegram import is_valid

        assert is_valid("") is False
        assert is_valid("ок") is False  # менше 2 слів
        assert is_valid("це повідомлення") is True
        assert is_valid("http " * 6) is False  # спам з лінками
        assert is_valid("this is english text with words") is True  # v2: без фільтра кирилиці
