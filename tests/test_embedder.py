"""Тести для Embedder — обгортка над sentence-transformers."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from dormouse.embedder import Embedder


class TestCosine:
    def test_identical_vectors(self):
        assert Embedder.cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert Embedder.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert Embedder.cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        sim = Embedder.cosine_similarity([1, 1], [1, 0])
        assert 0.5 < sim < 1.0


class TestEmbedderLocal:
    @patch("dormouse.embedder.SentenceTransformer")
    def test_embed_returns_vectors(self, mock_st_cls):
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_st_cls.return_value = mock_model

        emb = Embedder()
        result = emb.embed(["привіт", "світ"])

        assert len(result) == 2
        assert len(result[0]) == 3
        mock_model.encode.assert_called_once()

    @patch("dormouse.embedder.SentenceTransformer")
    def test_embed_one(self, mock_st_cls):
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_st_cls.return_value = mock_model

        emb = Embedder()
        result = emb.embed_one("привіт")

        assert len(result) == 3
        assert result == [0.1, 0.2, 0.3]


class TestEmbedderFallback:
    def test_no_sentence_transformers_raises(self):
        """Без sentence-transformers — ImportError з підказкою pip install."""
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with patch("dormouse.embedder.SentenceTransformer", None):
                emb = Embedder()
                with pytest.raises(ImportError, match="dormouse"):
                    emb.embed(["test"])
