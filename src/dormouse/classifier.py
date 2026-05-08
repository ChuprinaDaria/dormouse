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


def _parse_examples(categories: dict[str, str | list[str]]) -> tuple[list[str], list[list[str]]]:
    """Парсить приклади з dict категорій.

    Returns:
        (cat_names, list_of_example_lists)
    """
    cat_names = list(categories.keys())
    cat_examples: list[list[str]] = []

    for name in cat_names:
        examples = categories[name]
        if isinstance(examples, str):
            examples = [e.strip() for e in examples.split(",") if e.strip()]
            # Якщо одна строка без ком — split по пробілах
            if len(examples) == 1 and " " in examples[0]:
                examples = examples[0].split()
        cat_examples.append(examples)

    return cat_names, cat_examples


def sniff(
    items: list[str],
    categories: list[str] | dict[str, str | list[str]],
    squeeze_first: bool = False,
) -> list[SniffResult]:
    """Класифікує тексти по категоріях через embeddings.

    Args:
        items: Тексти для класифікації.
        categories: Категорії. Два формати:
            - list[str] — назви категорій (cosine до назви).
            - dict[str, str | list[str]] — приклади для кожної категорії.
              Значення: список прикладів або рядок через кому/пробіли.
              Приклад: {"Напої": "сік морс лимонад компот", "Десерти": ["торт", "чізкейк"]}
        squeeze_first: Оптимізувати тексти перед класифікацією.

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

    item_embeddings = model.encode(items)

    # Example-based: mean cosine до кожного прикладу
    if isinstance(categories, dict):
        cat_names, cat_examples = _parse_examples(categories)

        # Encode всі приклади одним батчем
        all_examples = [ex for examples in cat_examples for ex in examples]
        all_example_embs = model.encode(all_examples)

        # Розрізаємо назад по категоріях
        cat_emb_groups: list[np.ndarray] = []
        offset = 0
        for examples in cat_examples:
            cat_emb_groups.append(all_example_embs[offset:offset + len(examples)])
            offset += len(examples)

        results = []
        for i, item in enumerate(items):
            best_cat = 0
            best_score = -1.0
            for c, embs in enumerate(cat_emb_groups):
                sims = _cosine_similarity(item_embeddings[i], embs).flatten()
                score = float(np.mean(sims))
                if score > best_score:
                    best_score = score
                    best_cat = c
            results.append(SniffResult(
                text=item,
                category=cat_names[best_cat],
                confidence=best_score,
            ))
        return results

    # List-based: cosine до назви категорії (backward compatible)
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
