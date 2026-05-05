"""Тести для rule-based нормалізації."""

from dormouse.rule_engine import crack, crack_open


class TestWordReplacements:
    """Заміна окремих слів зі словника."""

    def test_surzhyk_basic(self):
        assert crack("шо там") == "що там"

    def test_surzhyk_case_insensitive(self):
        assert "що" in crack("Шо там")

    def test_filler_removed(self):
        result = crack("крч пофікси це плз")
        assert "крч" not in result
        assert "плз" not in result
        assert "виправ" in result

    def test_slang_replaced(self):
        assert crack("пофікси баг") == "виправ помилка"

    def test_clean_text_unchanged(self):
        text = "Це нормальний український текст"
        assert crack(text) == text

    def test_multiple_replacements(self):
        result = crack("шо там по тому баґу, пофікси плз")
        assert "що" in result
        assert "виправ" in result
        assert "плз" not in result

    def test_punctuation_preserved(self):
        result = crack("шо?")
        assert result == "що?"


class TestExpressions:
    """Багатослівні вирази (ngram > 1)."""

    def test_bigram_replaced(self):
        result = crack("да ладно ти серйозно")
        assert "та годі" in result

    def test_bigram_potому_що(self):
        result = crack("зробив потому що треба")
        assert "тому що" in result

    def test_trigram_preserved(self):
        """'будь ласка' — стандартна UA, не видаляється."""
        result = crack("підкажіть будь ласка як зробити")
        assert "будь ласка" in result

    def test_expression_before_words(self):
        """Вирази мають матчитись ДО окремих слів."""
        result = crack("потому шо ваще не понятно")
        assert "тому що" in result
        assert "взагалі" in result
        assert "зрозуміло" in result


class TestPatterns:
    """Regex-патерни."""

    def test_nu_at_start(self):
        result = crack("ну давай зробимо")
        assert not result.startswith("ну")

    def test_double_spaces_cleaned(self):
        assert "  " not in crack("слово  слово")


class TestVerbose:
    """Детальний результат."""

    def test_changed_flag(self):
        result = crack_open("шо там")
        assert result.changed is True
        assert result.replacements_made > 0

    def test_unchanged_flag(self):
        result = crack_open("нормальний текст")
        assert result.changed is False
        assert result.replacements_made == 0

    def test_original_preserved(self):
        original = "шо там по баґу"
        result = crack_open(original)
        assert result.original == original


class TestEdgeCases:
    """Крайні випадки."""

    def test_empty_string(self):
        assert crack("") == ""

    def test_single_word_filler(self):
        result = crack("крч")
        assert result == ""

    def test_numbers_and_special(self):
        text = "версія 2.0.1 — реліз"
        assert crack(text) == text
