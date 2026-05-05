"""Embeddings-based класифікація текстів (локально, без API)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SniffResult:
    text: str
    category: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "category": self.category,
            "confidence": round(self.confidence, 3),
        }


# Кеш моделі — завантажуємо один раз
_model = None


def _get_model():
    """Завантажує sentence-transformers модель (lazy loading)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
        )
    return _model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity між вектором a і матрицею b."""
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def sniff(
    items: list[str],
    categories: list[str],
    squeeze_first: bool = False,
) -> list[SniffResult]:
    """Класифікує тексти по категоріях через embeddings.

    Args:
        items: Тексти для класифікації.
        categories: Список категорій.
        optimize_first: Оптимізувати тексти перед класифікацією.

    Returns:
        Список SniffResult.

    Потребує: pip install dormouse[embeddings]
    """
    if squeeze_first:
        from dormouse.optimizer import squeeze_batch

        items = squeeze_batch(items)

    try:
        model = _get_model()
    except ImportError:
        raise ImportError(
            "sentence-transformers потрібен для sniff(). "
            "Встанови: pip install dormouse[embeddings]"
        )

    # Кодуємо все разом для ефективності
    item_embeddings = model.encode(items)
    cat_embeddings = model.encode(categories)

    results = []
    for i, item in enumerate(items):
        sims = _cosine_similarity(
            item_embeddings[i], cat_embeddings,
        ).flatten()
        best_idx = int(np.argmax(sims))
        results.append(SniffResult(
            text=item,
            category=categories[best_idx],
            confidence=float(sims[best_idx]),
        ))

    return results
