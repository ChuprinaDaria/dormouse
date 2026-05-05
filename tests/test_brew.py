"""Тести для brew() — LLM-powered пошук і витягування."""

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from dormouse.cli import main
from dormouse.teapot import Teapot


# --- Mock LLM ---


class MockLLM:
    """Mock LLM: classify повертає True якщо 'богиня' або 'афродіта' в тексті."""

    def classify(self, text: str, query: str) -> bool:
        lower = text.lower()
        return "богиня" in lower or "афродіта" in lower

    def extract(self, text: str, schema: dict) -> dict | None:
        if "Афродіта" in text:
            return {"name": "Афродіта", "domain": "кохання"}
        if "Артеміда" in text:
            return {"name": "Артеміда", "domain": "полювання"}
        return None


class RejectAllLLM:
    """Mock LLM: завжди відхиляє."""

    def classify(self, text: str, query: str) -> bool:
        return False

    def extract(self, text: str, schema: dict) -> dict | None:
        return None


def _make_embedder(dim=3):
    """Мок-ембедер."""
    emb = MagicMock()
    call_count = [0]

    def fake_embed(texts):
        vectors = []
        for _ in texts:
            call_count[0] += 1
            vectors.append([float(call_count[0])] * dim)
        return vectors

    def fake_embed_one(text):
        return fake_embed([text])[0]

    emb.embed = fake_embed
    emb.embed_one = fake_embed_one
    emb.cosine_similarity = MagicMock(side_effect=lambda a, b: 0.9)
    return emb


# --- Tests ---


class TestBrewEmpty:
    def test_empty_index(self, tmp_path):
        """brew на порожньому індексі -> порожній список."""
        t = Teapot(db_path=tmp_path / "test.db", embedder=_make_embedder())
        results = t.brew("щось", llm=MockLLM())
        assert results == []


class TestBrewHierarchical:
    def _build(self, tmp_path, texts):
        t = Teapot(db_path=tmp_path / "test.db", embedder=_make_embedder())
        t.stir(texts, min_words=1)
        return t

    def test_finds_relevant_chunks(self, tmp_path):
        t = self._build(tmp_path, [
            "Афродіта — богиня кохання.",
            "Зевс — бог грому.",
            "Рецепт борщу з буряком.",
        ])
        results = t.brew("богині", llm=MockLLM())
        assert len(results) >= 1
        texts = [r["text"] for r in results]
        assert any("Афродіта" in txt for txt in texts)

    def test_filters_irrelevant(self, tmp_path):
        t = self._build(tmp_path, [
            "Афродіта — богиня кохання.",
            "Рецепт борщу.",
            "Погода на завтра.",
        ])
        results = t.brew("богині", llm=MockLLM())
        texts = [r["text"] for r in results]
        assert not any("борщу" in txt for txt in texts)
        assert not any("Погода" in txt for txt in texts)

    def test_returns_dict_with_text_score_chunk_id(self, tmp_path):
        t = self._build(tmp_path, ["Афродіта — богиня кохання."])
        results = t.brew("богині", llm=MockLLM())
        assert len(results) == 1
        r = results[0]
        assert "text" in r
        assert "score" in r
        assert "chunk_id" in r

    def test_limit_caps_results(self, tmp_path):
        t = self._build(tmp_path, [
            "Афродіта — богиня кохання.",
            "Артеміда — богиня полювання.",
            "Гера — богиня шлюбу.",
        ])
        results = t.brew("богині", llm=MockLLM(), limit=1)
        assert len(results) <= 1

    def test_reject_all_returns_empty(self, tmp_path):
        t = self._build(tmp_path, ["Афродіта — богиня кохання."])
        results = t.brew("богині", llm=RejectAllLLM())
        assert results == []


class TestBrewExtract:
    def _build(self, tmp_path, texts):
        t = Teapot(db_path=tmp_path / "test.db", embedder=_make_embedder())
        t.stir(texts, min_words=1)
        return t

    def test_extract_returns_structured_data(self, tmp_path):
        t = self._build(tmp_path, ["Афродіта — богиня кохання і краси."])
        results = t.brew(
            "богині",
            llm=MockLLM(),
            extract={"name": str, "domain": str},
        )
        assert len(results) == 1
        assert results[0]["name"] == "Афродіта"
        assert results[0]["domain"] == "кохання"

    def test_extract_skips_none(self, tmp_path):
        t = self._build(tmp_path, [
            "Афродіта — богиня кохання.",
            "Зевс — бог грому.",
        ])
        results = t.brew(
            "богині",
            llm=MockLLM(),
            extract={"name": str, "domain": str},
        )
        assert all(r["name"] != "Зевс" for r in results)

    def test_extract_deduplicates(self, tmp_path):
        t = self._build(tmp_path, [
            "Афродіта — богиня кохання.",
            "Богиня Афродіта, покровителька кохання.",
        ])
        results = t.brew(
            "богині",
            llm=MockLLM(),
            extract={"name": str, "domain": str},
        )
        names = [r["name"] for r in results]
        assert names.count("Афродіта") == 1


class TestBrewStrategies:
    def _build(self, tmp_path, texts):
        t = Teapot(db_path=tmp_path / "test.db", embedder=_make_embedder())
        t.stir(texts, min_words=1)
        return t

    def test_filter_strategy_skips_classify(self, tmp_path):
        llm = MagicMock()
        llm.classify = MagicMock(side_effect=AssertionError("should not be called"))
        llm.extract = MagicMock(return_value=None)

        t = self._build(tmp_path, ["Текст один.", "Текст два."])
        results = t.brew("запит", llm=llm, strategy="filter")
        llm.classify.assert_not_called()
        assert len(results) > 0

    def test_full_scan_checks_all(self, tmp_path):
        call_count = [0]

        class CountingLLM:
            def classify(self, text, query):
                call_count[0] += 1
                return True

        t = self._build(tmp_path, ["Один", "Два", "Три"])
        t.brew("запит", llm=CountingLLM(), strategy="full_scan")
        assert call_count[0] == 3


class TestBrewCLI:
    def test_brew_requires_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["brew", "запит"])
        assert result.exit_code != 0

    def test_brew_missing_file(self):
        runner = CliRunner()
        result = runner.invoke(main, ["brew", "запит", "--file", "/nonexistent"])
        assert result.exit_code != 0


class TestPublicAPI:
    def test_brew_importable(self):
        from dormouse import brew
        assert callable(brew)

    def test_local_llm_importable(self):
        from dormouse import LocalLLM
        assert callable(LocalLLM)
