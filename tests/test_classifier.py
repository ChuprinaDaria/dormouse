"""Тести для sniff() — embeddings-based класифікація."""

from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")

from dormouse.classifier import SniffResult, _parse_examples, sniff  # noqa: E402


@pytest.fixture
def mock_model():
    """Мок sentence-transformers моделі з детермінованими embeddings."""
    model = MagicMock()

    # Словник: текст → embedding (імітація семантичної близькості)
    embeddings = {
        # Страви (кластер "гарячі страви")
        "Борщ український з яловичиною": [0.9, 0.1, 0.0, 0.0],
        "борщ": [0.85, 0.1, 0.05, 0.0],
        "суп": [0.8, 0.15, 0.0, 0.05],
        "юшка": [0.75, 0.2, 0.0, 0.05],
        "харчо": [0.82, 0.12, 0.03, 0.03],
        # Кластер "салати"
        "Салат Цезар з куркою": [0.1, 0.9, 0.0, 0.0],
        "салат": [0.1, 0.85, 0.0, 0.05],
        "цезар": [0.12, 0.8, 0.05, 0.03],
        "вінегрет": [0.15, 0.78, 0.02, 0.05],
        # Кластер "напої"
        "Морс із лісових ягід": [0.0, 0.0, 0.9, 0.1],
        "Лимонад класичний": [0.05, 0.0, 0.85, 0.1],
        "сік": [0.0, 0.05, 0.88, 0.07],
        "морс": [0.0, 0.03, 0.85, 0.12],
        "лимонад": [0.05, 0.0, 0.82, 0.13],
        "компот": [0.03, 0.02, 0.8, 0.15],
        # Кластер "десерти"
        "Чізкейк Нью-Йорк": [0.0, 0.0, 0.1, 0.9],
        "торт": [0.0, 0.05, 0.05, 0.9],
        "чізкейк": [0.0, 0.0, 0.1, 0.9],
        "еклер": [0.05, 0.0, 0.08, 0.87],
        # Назви категорій (абстрактні — далекі від реальних страв)
        "Гарячі страви": [0.4, 0.3, 0.2, 0.1],
        "Салати": [0.25, 0.4, 0.2, 0.15],
        "Напої": [0.2, 0.2, 0.4, 0.2],
        "Десерти": [0.15, 0.2, 0.25, 0.4],
    }

    def fake_encode(texts):
        result = []
        for t in texts:
            if t in embeddings:
                result.append(embeddings[t])
            else:
                result.append([0.25, 0.25, 0.25, 0.25])
        return np.array(result)

    model.encode = fake_encode
    return model


class TestSniffResult:
    def test_to_dict(self):
        r = SniffResult(text="Борщ", category="Гарячі страви", confidence=0.95123)
        d = r.to_dict()
        assert d == {"text": "Борщ", "category": "Гарячі страви", "confidence": 0.951}


class TestSniffListCategories:
    """Старий API: categories як list[str]."""

    @patch("dormouse.classifier._get_model")
    def test_basic_classification(self, mock_get, mock_model):
        mock_get.return_value = mock_model

        results = sniff(
            ["Борщ український з яловичиною"],
            ["Гарячі страви", "Салати", "Напої", "Десерти"],
        )

        assert len(results) == 1
        assert results[0].text == "Борщ український з яловичиною"
        assert isinstance(results[0].confidence, float)

    @patch("dormouse.classifier._get_model")
    def test_multiple_items(self, mock_get, mock_model):
        mock_get.return_value = mock_model

        results = sniff(
            ["Борщ український з яловичиною", "Морс із лісових ягід"],
            ["Гарячі страви", "Напої"],
        )

        assert len(results) == 2


class TestSniffDictCategories:
    """Новий API: categories як dict з прикладами."""

    @patch("dormouse.classifier._get_model")
    def test_example_based_classification(self, mock_get, mock_model):
        mock_get.return_value = mock_model

        categories = {
            "Гарячі страви": "борщ суп юшка харчо",
            "Салати": "салат цезар вінегрет",
            "Напої": "сік морс лимонад компот",
            "Десерти": "торт чізкейк еклер",
        }

        results = sniff(
            ["Борщ український з яловичиною", "Чізкейк Нью-Йорк",
             "Морс із лісових ягід", "Салат Цезар з куркою"],
            categories,
        )

        assert len(results) == 4
        assert results[0].category == "Гарячі страви"
        assert results[1].category == "Десерти"
        assert results[2].category == "Напої"
        assert results[3].category == "Салати"

    @patch("dormouse.classifier._get_model")
    def test_list_examples(self, mock_get, mock_model):
        mock_get.return_value = mock_model

        categories = {
            "Гарячі страви": ["борщ", "суп", "юшка"],
            "Напої": ["сік", "морс", "компот"],
        }

        results = sniff(["Борщ український з яловичиною"], categories)
        assert results[0].category == "Гарячі страви"

    @patch("dormouse.classifier._get_model")
    def test_comma_separated_examples(self, mock_get, mock_model):
        mock_get.return_value = mock_model

        categories = {
            "Гарячі страви": "борщ, суп, юшка, харчо",
            "Десерти": "торт, чізкейк, еклер",
        }

        results = sniff(["Чізкейк Нью-Йорк"], categories)
        assert results[0].category == "Десерти"

    @patch("dormouse.classifier._get_model")
    def test_example_based_beats_name_based(self, mock_get, mock_model):
        """Example-based має давати кращі результати ніж name-based."""
        mock_get.return_value = mock_model

        items = ["Борщ український з яловичиною", "Лимонад класичний"]
        cats_list = ["Гарячі страви", "Напої"]
        cats_dict = {
            "Гарячі страви": "борщ суп юшка харчо",
            "Напої": "сік морс лимонад компот",
        }

        results_list = sniff(items, cats_list)
        results_dict = sniff(items, cats_dict)

        # Example-based має вищу confidence
        for r_dict, r_list in zip(results_dict, results_list):
            assert r_dict.confidence >= r_list.confidence - 0.01


class TestParseExamples:
    def test_string_examples_space_separated(self):
        names, examples = _parse_examples({"A": "борщ суп"})
        assert names == ["A"]
        assert examples == [["борщ", "суп"]]

    def test_string_examples_comma_separated(self):
        names, examples = _parse_examples({"A": "борщ, суп, юшка"})
        assert names == ["A"]
        assert examples == [["борщ", "суп", "юшка"]]

    def test_list_examples(self):
        names, examples = _parse_examples(
            {"A": ["борщ", "суп"], "B": ["сік", "морс"]}
        )
        assert names == ["A", "B"]
        assert examples == [["борщ", "суп"], ["сік", "морс"]]

    def test_preserves_order(self):
        names, _ = _parse_examples({"C": "x", "A": "y", "B": "z"})
        assert names == ["C", "A", "B"]
