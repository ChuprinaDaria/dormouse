"""Тести для LocalLLM — уніфікований LLM інтерфейс."""

from unittest.mock import patch

import pytest

from dormouse.local_model import LocalLLM, MODEL_ALIASES


class TestLocalLLMInit:
    def test_ollama_backend_default(self):
        """Дефолтний backend = ollama."""
        llm = LocalLLM("qwen2.5:1.5b")
        assert llm._backend == "ollama"

    def test_hf_backend(self):
        """HF backend ініціалізується."""
        llm = LocalLLM("Qwen3-4B", backend="hf")
        assert llm._backend == "hf"

    def test_model_alias_resolved(self):
        """MamayLM-4B розпізнається через MODEL_ALIASES."""
        llm = LocalLLM("MamayLM-4B", backend="ollama")
        expected = MODEL_ALIASES["MamayLM-4B"]["ollama"]
        assert llm._resolved_model == expected

    def test_unknown_model_used_as_is(self):
        """Невідома модель передається як є."""
        llm = LocalLLM("my-custom-model:latest")
        assert llm._resolved_model == "my-custom-model:latest"

    def test_invalid_backend_raises(self):
        """Невідомий backend -> ValueError."""
        with pytest.raises(ValueError, match="backend"):
            LocalLLM("test", backend="invalid")


class TestClassify:
    def test_parses_yes(self):
        llm = LocalLLM("test-model")
        with patch.object(llm, "ask", return_value="yes"):
            assert llm.classify("Афродіта — богиня кохання", "богині") is True

    def test_parses_no(self):
        llm = LocalLLM("test-model")
        with patch.object(llm, "ask", return_value="no"):
            assert llm.classify("Рецепт борщу", "богині") is False

    def test_parses_yes_with_noise(self):
        llm = LocalLLM("test-model")
        with patch.object(llm, "ask", return_value="Yes."):
            assert llm.classify("текст", "запит") is True

    def test_parses_tak(self):
        llm = LocalLLM("test-model")
        with patch.object(llm, "ask", return_value="так"):
            assert llm.classify("текст", "запит") is True

    def test_parses_garbage_as_no(self):
        llm = LocalLLM("test-model")
        with patch.object(llm, "ask", return_value="maybe possibly"):
            assert llm.classify("текст", "запит") is False

    def test_prompt_structure(self):
        llm = LocalLLM("test-model")
        captured = {}

        def capture_ask(prompt, system=None):
            captured["prompt"] = prompt
            return "no"

        with patch.object(llm, "ask", side_effect=capture_ask):
            llm.classify("тестовий текст", "мій запит")
        assert "мій запит" in captured["prompt"]
        assert "тестовий текст" in captured["prompt"]
        assert "yes or no" in captured["prompt"]


class TestExtract:
    def test_parses_json(self):
        llm = LocalLLM("test-model")
        with patch.object(llm, "ask", return_value='{"name": "Афродіта", "domain": "кохання"}'):
            result = llm.extract("текст", {"name": str, "domain": str})
        assert result == {"name": "Афродіта", "domain": "кохання"}

    def test_parses_json_with_preamble(self):
        llm = LocalLLM("test-model")
        response = 'Ось результат:\n{"name": "Зевс"}\nГотово.'
        with patch.object(llm, "ask", return_value=response):
            result = llm.extract("текст", {"name": str})
        assert result == {"name": "Зевс"}

    def test_null_response(self):
        llm = LocalLLM("test-model")
        with patch.object(llm, "ask", return_value="null"):
            result = llm.extract("борщ", {"name": str})
        assert result is None

    def test_garbage_response(self):
        llm = LocalLLM("test-model")
        with patch.object(llm, "ask", return_value="я не знаю"):
            result = llm.extract("текст", {"name": str})
        assert result is None

    def test_prompt_includes_schema(self):
        llm = LocalLLM("test-model")
        captured = {}

        def capture_ask(prompt, system=None):
            captured["prompt"] = prompt
            return "null"

        with patch.object(llm, "ask", side_effect=capture_ask):
            llm.extract("текст", {"name": str, "price": float})
        assert "name (text)" in captured["prompt"]
        assert "price (number)" in captured["prompt"]
