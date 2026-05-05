"""Embedder — обгортка для локальних embedding моделей.

Використовує sentence-transformers для ембедінгів без зовнішніх API.
"""

import math

# Lazy import — заповнюється при першому виклику _init_local
SentenceTransformer = None

DEFAULT_LOCAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class Embedder:
    """Обгортка для embedding моделей (sentence-transformers).

    Args:
        model: Назва моделі. Default: paraphrase-multilingual-MiniLM-L12-v2.
    """

    def __init__(self, model: str | None = None):
        self.model_name = model
        self._model = None

    def _init_local(self):
        """Ініціалізація sentence-transformers моделі."""
        global SentenceTransformer
        if SentenceTransformer is None:
            try:
                from sentence_transformers import SentenceTransformer as ST

                SentenceTransformer = ST
            except ImportError:
                raise ImportError(
                    "sentence-transformers не встановлений. "
                    "Встановіть: pip install dormouse[embeddings]"
                )
        model_name = self.model_name or DEFAULT_LOCAL_MODEL
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Батч-ембедінг текстів.

        Args:
            texts: Список текстів для ембедінгу.

        Returns:
            Список векторів (list of float lists).
        """
        if self._model is None:
            self._init_local()
        vectors = self._model.encode(texts)
        return [list(v) for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        """Ембедінг одного тексту.

        Args:
            text: Текст для ембедінгу.

        Returns:
            Вектор (list of floats).
        """
        return self.embed([text])[0]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Косинусна подібність між двома векторами.

        Реалізація без numpy — тільки math.sqrt.
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
