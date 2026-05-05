"""Класифікація стилю тексту: норм / розмовна / суржик / мем.

sklearn baseline: TF-IDF + LogisticRegression.
Потрібен для визначення — чи варто оптимізувати текст.

Використовує pickle для збереження sklearn моделі — стандартний підхід
для sklearn pipeline. Файли моделей генеруються локально, не з зовнішніх джерел.
"""

import pickle
from pathlib import Path

import numpy as np

STYLES = ["норм", "розмовна", "суржик", "мем"]
MODEL_DIR = Path(__file__).resolve().parent / "models"


class NoseKit:
    """TF-IDF + LogisticRegression для детекції стилю."""

    def __init__(self):
        self.vectorizer = None
        self.clf = None

    def train(self, texts: list[str], labels: list[str]):
        """Тренує класифікатор на текстах і мітках."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
        )
        X = self.vectorizer.fit_transform(texts)
        self.clf = LogisticRegression(max_iter=1000, random_state=42)
        self.clf.fit(X, np.array(labels))

    def predict(self, text: str) -> str:
        """Повертає стиль тексту."""
        if self.clf is None:
            raise RuntimeError("Модель не натренована. Викличте train() або load().")
        X = self.vectorizer.transform([text])
        return self.clf.predict(X)[0]

    def predict_proba(self, text: str) -> dict[str, float]:
        """Повертає ймовірності для кожного стилю."""
        if self.clf is None:
            raise RuntimeError("Модель не натренована.")
        X = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(X)[0]
        return dict(zip(self.clf.classes_, [round(p, 3) for p in probs]))

    def save(self, path: Path | None = None):
        """Зберігає модель на диск (pickle — стандарт для sklearn)."""
        p = path or MODEL_DIR / "style_classifier.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "clf": self.clf}, f)

    def load(self, path: Path | None = None):
        """Завантажує локально збережену модель."""
        p = path or MODEL_DIR / "style_classifier.pkl"
        with open(p, "rb") as f:
            data = pickle.load(f)  # noqa: S301 — довіряємо локальним файлам
        self.vectorizer = data["vectorizer"]
        self.clf = data["clf"]

    @property
    def is_trained(self) -> bool:
        return self.clf is not None


def needs_cracking(text: str, classifier: NoseKit | None = None) -> bool:
    """Чи потребує текст оптимізації (не 'норм')."""
    if classifier is None or not classifier.is_trained:
        # Fallback: евристика через rule_engine
        from dormouse.rule_engine import crack_open

        return crack_open(text).changed

    style = classifier.predict(text)
    return style != "норм"
