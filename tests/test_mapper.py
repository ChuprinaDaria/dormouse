"""Тести для mapper.py."""

from dormouse.mapper import map_to_en


class TestCloudMode:
    def test_basic_mapping(self):
        result = map_to_en("привіт")
        assert result == "hi"

    def test_phrase_mapping(self):
        result = map_to_en("як справи")
        assert result == "how?"

    def test_mixed_text(self):
        result = map_to_en("привіт, як справи")
        assert "hi" in result
        assert "how?" in result

    def test_unknown_word_transliterated(self):
        result = map_to_en("кавоварка")
        # Невідоме слово — транслітерується
        assert "а" not in result  # кирилиці не має залишитись

    def test_empty(self):
        assert map_to_en("") == ""


class TestTransliteration:
    def test_cyrillic_fallback(self):
        result = map_to_en("тест")
        assert result == "test"

    def test_preserves_latin(self):
        result = map_to_en("API endpoint")
        assert result == "API endpoint"
