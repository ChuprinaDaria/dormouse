"""Тести для Teapot — ядро семантичного пошуку."""

from unittest.mock import MagicMock, patch

import pytest

from dormouse.teapot import Teapot, _deserialize_embedding, _serialize_embedding

# --- Helpers ---


def _make_embedder(dim=3):
    """Мок-ембедер для тестів."""
    emb = MagicMock()
    call_count = [0]

    def fake_embed(texts):
        vectors = []
        for i, _ in enumerate(texts):
            call_count[0] += 1
            vectors.append([float(call_count[0])] * dim)
        return vectors

    def fake_embed_one(text):
        return fake_embed([text])[0]

    emb.embed = fake_embed
    emb.embed_one = fake_embed_one
    emb.cosine_similarity = MagicMock(side_effect=lambda a, b: 0.9)
    return emb


# --- Task 4: stir + storage ---


class TestSerializeEmbedding:
    def test_roundtrip(self):
        vec = [0.1, 0.2, 0.3, 0.4]
        blob = _serialize_embedding(vec)
        restored = _deserialize_embedding(blob)
        assert len(restored) == 4
        assert restored == pytest.approx(vec)


class TestStir:
    def test_stir_list(self, tmp_path):
        """stir list[str] -> count() == len."""
        t = Teapot(db_path=tmp_path / "test.db", embedder=_make_embedder())
        texts = ["Привіт світ", "Як справи", "Все добре"]
        t.stir(texts, min_words=1)
        assert t.count() == 3

    def test_stir_file(self, tmp_path):
        """stir з файлом -> parse_file викликається."""
        p = tmp_path / "doc.txt"
        p.write_text("Перший абзац.\n\nДругий абзац.\n")
        t = Teapot(db_path=tmp_path / "test.db", embedder=_make_embedder())
        with patch("dormouse.teapot.parse_file", return_value=["Перший", "Другий"]) as mock_pf:
            t.stir(str(p), min_words=1)
            mock_pf.assert_called_once_with(str(p))
        assert t.count() == 2

    def test_stir_returns_self(self, tmp_path):
        """stir повертає self для chaining."""
        t = Teapot(db_path=tmp_path / "test.db", embedder=_make_embedder())
        result = t.stir(["тест"], min_words=1)
        assert result is t

    def test_stir_chaining(self, tmp_path):
        """stir().stir() — ланцюжок викликів."""
        t = Teapot(db_path=tmp_path / "test.db", embedder=_make_embedder())
        t.stir(["перший"], min_words=1).stir(["другий"], min_words=1)
        assert t.count() == 2

    def test_stir_empty_list(self, tmp_path):
        """stir порожнього списку -> count() == 0."""
        t = Teapot(db_path=tmp_path / "test.db", embedder=_make_embedder())
        t.stir([])
        assert t.count() == 0

    def test_stir_without_embedder(self, tmp_path):
        """stir без ембедера -> тільки FTS5, без embeddings."""
        t = Teapot(db_path=tmp_path / "test.db")
        t.stir(["Привіт світ"], min_words=1)
        # count() рахує embeddings, без ембедера їх 0
        assert t.count() == 0

    def test_stir_file_path_object(self, tmp_path):
        """stir з Path об'єктом."""
        p = tmp_path / "doc.txt"
        p.write_text("Один абзац.\n\nДва абзаци.\n")
        t = Teapot(db_path=tmp_path / "test.db", embedder=_make_embedder())
        with patch("dormouse.teapot.parse_file", return_value=["Один", "Два"]) as mock_pf:
            t.stir(p, min_words=1)
            mock_pf.assert_called_once_with(p)
        assert t.count() == 2


# --- Task 5: mumble (hybrid search) ---


class TestMumble:
    def _build_teapot(self, tmp_path, embedder=None):
        emb = embedder or _make_embedder()
        t = Teapot(db_path=tmp_path / "test.db", embedder=emb)
        t.stir(["Привіт світ", "Як справи", "Все добре", "Тестовий текст"], min_words=1)
        return t

    def test_returns_sorted_by_score(self, tmp_path):
        """Результати відсортовані за score (desc)."""
        emb = MagicMock()
        scores = [0.3, 0.9, 0.5, 0.7]
        call_idx = [0]

        def fake_embed(texts):
            return [[float(i)] * 3 for i in range(len(texts))]

        emb.embed = fake_embed
        emb.embed_one = lambda t: [1.0, 1.0, 1.0]

        def fake_cosine(a, b):
            idx = call_idx[0]
            call_idx[0] += 1
            return scores[idx] if idx < len(scores) else 0.0

        emb.cosine_similarity = fake_cosine

        t = Teapot(db_path=tmp_path / "test.db", embedder=emb)
        t.stir(["один", "два", "три", "чотири"], min_words=1)

        results = t.mumble("запит")
        assert len(results) > 0
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]

    def test_returns_text_and_score_keys(self, tmp_path):
        """Результати мають ключі text, score, chunk_id."""
        t = self._build_teapot(tmp_path)
        results = t.mumble("привіт")
        assert len(results) > 0
        for r in results:
            assert "text" in r
            assert "score" in r
            assert "chunk_id" in r

    def test_top_k_limits_results(self, tmp_path):
        """top_k обмежує кількість результатів."""
        t = self._build_teapot(tmp_path)
        results = t.mumble("привіт", top_k=2)
        assert len(results) <= 2

    def test_empty_index_returns_empty(self, tmp_path):
        """Порожній індекс -> порожній результат."""
        t = Teapot(db_path=tmp_path / "test.db", embedder=_make_embedder())
        results = t.mumble("привіт")
        assert results == []

    def test_fallback_fts5_without_embedder(self, tmp_path):
        """Без ембедера -> fallback на FTS5 search()."""
        t = Teapot(db_path=tmp_path / "test.db")
        t.stir(["тестовий текст для пошуку"], min_words=1)
        with patch("dormouse.teapot.fts_search") as mock_search:
            mock_search.return_value = [
                {"original": "тестовий текст", "rank": -1.5},
            ]
            results = t.mumble("тестовий")
            mock_search.assert_called_once()
            assert len(results) == 1
            assert "text" in results[0]
            assert "score" in results[0]


# --- Task 6: sip (classification) ---


class TestSip:
    def test_classifies_by_topics(self, tmp_path):
        """sip класифікує тексти по темам."""
        emb = MagicMock()

        def fake_embed(texts):
            return [[float(i + 1)] * 3 for i in range(len(texts))]

        emb.embed = fake_embed
        emb.embed_one = lambda t: [1.0, 1.0, 1.0]

        # Повертаємо високий score для першої теми, низький для другої
        scores = iter([0.9, 0.1, 0.8, 0.2])
        emb.cosine_similarity = lambda a, b: next(scores)

        t = Teapot(db_path=tmp_path / "test.db", embedder=emb)
        result = t.sip(
            ["текст один", "текст два"],
            topics=["тема А", "тема Б"],
            threshold=0.0,
        )
        assert "тема А" in result
        assert "тема Б" in result

    def test_threshold_filters(self, tmp_path):
        """threshold фільтрує слабкі матчі."""
        emb = MagicMock()
        emb.embed = lambda texts: [[1.0] * 3 for _ in texts]
        emb.embed_one = lambda t: [1.0] * 3
        emb.cosine_similarity = lambda a, b: 0.3  # нижче за threshold

        t = Teapot(db_path=tmp_path / "test.db", embedder=emb)
        result = t.sip(
            ["текст"],
            topics=["тема"],
            threshold=0.5,
        )
        assert result["тема"] == []

    def test_text_matches_multiple_topics(self, tmp_path):
        """Текст може матчити кілька тем."""
        emb = MagicMock()
        emb.embed = lambda texts: [[1.0] * 3 for _ in texts]
        # Високий score для обох тем
        emb.cosine_similarity = lambda a, b: 0.8

        t = Teapot(db_path=tmp_path / "test.db", embedder=emb)
        result = t.sip(
            ["текст"],
            topics=["тема А", "тема Б"],
            threshold=0.5,
        )
        assert len(result["тема А"]) == 1
        assert len(result["тема Б"]) == 1

    def test_raises_without_embedder(self, tmp_path):
        """sip без ембедера -> ValueError."""
        t = Teapot(db_path=tmp_path / "test.db")
        with pytest.raises(ValueError, match="embedder"):
            t.sip(["текст"], topics=["тема"])

    def test_sip_with_file(self, tmp_path):
        """sip з файлом."""
        p = tmp_path / "doc.txt"
        p.write_text("Абзац один.\n\nАбзац два.\n")
        emb = MagicMock()
        emb.embed = lambda texts: [[1.0] * 3 for _ in texts]
        emb.cosine_similarity = lambda a, b: 0.7

        t = Teapot(db_path=tmp_path / "test.db", embedder=emb)
        with patch("dormouse.teapot.parse_file", return_value=["Абзац один", "Абзац два"]):
            result = t.sip(p, topics=["тема"], threshold=0.5)
        assert len(result["тема"]) == 2
