"""Тести для FTS5 full-text search."""

import importlib.util

import pytest

from dormouse.search import (
    count,
    get_search_db,
    index_batch,
    index_text,
    search,
    search_exact,
)

HAS_PYMORPHY = importlib.util.find_spec("pymorphy3") is not None


class TestIndex:
    def test_index_single(self, tmp_path):
        conn = get_search_db(tmp_path / "test_search.db")
        normalized = index_text(conn, "шо там по баґу", source="test")
        conn.commit()
        assert "що" in normalized
        assert count(conn) == 1
        conn.close()

    def test_index_batch(self, tmp_path):
        conn = get_search_db(tmp_path / "test_search.db")
        texts = [
            {"text": "пофікси баг плз", "source": "chat"},
            {"text": "деплой впав знову", "source": "chat"},
            {"text": "нормальний текст без суржику", "source": "news"},
        ]
        n = index_batch(conn, texts)
        assert n == 3
        assert count(conn) == 3
        conn.close()


class TestSearch:
    def _build_index(self, tmp_path):
        conn = get_search_db(tmp_path / "test_search.db")
        texts = [
            {"text": "шо там по баґу, пофікси плз", "source": "chat"},
            {"text": "деплой впав, треба фіксити", "source": "chat"},
            {"text": "юзер каже шо нічо не працює", "source": "chat"},
            {"text": "Верховна Рада прийняла закон", "source": "news"},
            {"text": "оновлення програмного забезпечення", "source": "docs"},
        ]
        index_batch(conn, texts)
        return conn

    @pytest.mark.skipif(not HAS_PYMORPHY, reason="pymorphy3 not installed")
    def test_search_surzhyk_finds_normalized(self, tmp_path):
        """Пошук 'помилка' знаходить текст з 'баґу'."""
        conn = self._build_index(tmp_path)
        results = search(conn, "помилка")
        assert len(results) >= 1
        assert any("баґу" in r["original"] for r in results)
        conn.close()

    @pytest.mark.skipif(not HAS_PYMORPHY, reason="pymorphy3 not installed")
    def test_search_slang_finds_normalized(self, tmp_path):
        """Пошук 'баг' знаходить текст — бо 'баг' нормалізується в 'помилка'."""
        conn = self._build_index(tmp_path)
        results = search(conn, "баг")
        assert len(results) >= 1
        conn.close()

    def test_search_no_results(self, tmp_path):
        conn = self._build_index(tmp_path)
        results = search(conn, "квантова фізика")
        assert len(results) == 0
        conn.close()

    def test_search_exact(self, tmp_path):
        conn = self._build_index(tmp_path)
        results = search_exact(conn, "деплой впав")
        assert len(results) >= 1
        assert any("деплой" in r["original"] for r in results)
        conn.close()

    def test_search_returns_original(self, tmp_path):
        """Результат містить оригінальний текст (суржик)."""
        conn = self._build_index(tmp_path)
        results = search(conn, "закон")
        for r in results:
            assert r["original"]
            assert r["normalized"]
        conn.close()

    def test_empty_query(self, tmp_path):
        conn = self._build_index(tmp_path)
        assert search(conn, "") == []
        assert search(conn, "   ") == []
        conn.close()
