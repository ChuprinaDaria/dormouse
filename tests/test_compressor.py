"""Тести для compressor.py."""

from dormouse.compressor import compress


class TestRuleCompression:
    def test_filler_phrases_removed(self):
        result = compress("я думаю що це правильно")
        assert "я думаю що" not in result
        assert "правильно" in result

    def test_intensifiers_removed(self):
        result = compress("це дуже важливе завдання")
        assert "дуже" not in result
        assert "важливе" in result

    def test_dedup_consecutive(self):
        result = compress("дуже дуже важливо")
        # "дуже" видаляється як intensifier, але якщо б залишилось — дедуп
        assert "важливо" in result

    def test_clean_text_passes(self):
        text = "написати функцію обробки даних"
        result = compress(text)
        assert "функцію" in result
        assert "обробки" in result

    def test_empty_text(self):
        assert compress("") == ""
        assert compress("   ") == "   "

    def test_cleanup_double_spaces(self):
        result = compress("слово  слово")
        assert "  " not in result

    def test_polite_wrapper(self):
        result = compress("чи не могли б ви допомогти")
        assert "чи не могли б ви" not in result
        assert "допомогти" in result
