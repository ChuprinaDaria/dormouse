"""Тести для optimizer.py."""

from dormouse.optimizer import SqueezedText, squeeze, squeeze_batch


class TestSqueeze:
    def test_basic(self):
        result = squeeze("шо там по баґу")
        assert isinstance(result, str)
        assert "що" in result

    def test_verbose(self):
        result = squeeze("шо там", verbose=True)
        assert isinstance(result, SqueezedText)
        assert "що" in result.text
        assert result.original == "шо там"
        assert result.method == "rule"
        assert result.replacements_made > 0

    def test_empty(self):
        assert squeeze("") == ""
        result = squeeze("", verbose=True)
        assert isinstance(result, SqueezedText)
        assert result.text == ""

    def test_clean_text_passes_through(self):
        text = "написати функцію обробки даних"
        result = squeeze(text)
        assert "функцію" in result

    def test_batch(self):
        results = squeeze_batch(["шо", "нормальний текст"])
        assert len(results) == 2
        assert results[0] == "що"


class TestTargetCloud:
    def test_cloud_returns_english(self):
        result = squeeze("привіт як справи", target="cloud")
        assert isinstance(result, str)

    def test_cloud_verbose(self):
        result = squeeze("привіт", target="cloud", verbose=True)
        assert isinstance(result, SqueezedText)
        assert result.original == "привіт"


class TestTargetNone:
    def test_none_target_stays_ukrainian(self):
        result = squeeze("привіт")
        assert isinstance(result, str)

    def test_none_target_verbose(self):
        result = squeeze("привіт", verbose=True)
        assert isinstance(result, SqueezedText)
