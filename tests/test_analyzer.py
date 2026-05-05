"""Тести для analyzer.py."""

import pytest


class TestNibble:
    def test_nibble_basic(self):
        pytest.importorskip("tiktoken")
        from dormouse.analyzer import nibble

        stats = nibble("шо там по тому баґу пофікси плз")
        assert stats.gpt4 > 0
        assert stats.gpt4_squeezed > 0
        assert stats.gpt4_squeezed <= stats.gpt4
        assert stats.savings_gpt4_pct >= 0
        assert stats.squeezed_text != ""

    def test_nibble_clean_text(self):
        pytest.importorskip("tiktoken")
        from dormouse.analyzer import nibble

        stats = nibble("нормальний текст")
        assert stats.gpt4 > 0
        assert stats.savings_gpt4_pct >= 0

    def test_to_dict(self):
        pytest.importorskip("tiktoken")
        from dormouse.analyzer import nibble

        stats = nibble("тест")
        d = stats.to_dict()
        assert "gpt4" in d
        assert "savings_gpt4_pct" in d

    def test_batch(self):
        pytest.importorskip("tiktoken")
        from dormouse.analyzer import nibble_batch

        results = nibble_batch(["текст один", "текст два"])
        assert len(results) == 2
