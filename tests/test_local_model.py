"""Тести для local_model.py."""

from dormouse.local_model import is_ollama_available, list_models


class TestLocalModel:
    def test_ollama_check(self):
        # Просто перевіряємо що функція не падає
        result = is_ollama_available()
        assert isinstance(result, bool)

    def test_list_models(self):
        models = list_models()
        assert isinstance(models, list)
