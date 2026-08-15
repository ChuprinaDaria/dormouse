"""Tests for the MT translator wrapper — mocks out optimum/transformers."""

from unittest.mock import MagicMock, patch

from dormouse import mt_translator
from dormouse.mt_translator import MTTranslator, get_translator


def _reset_singleton():
    mt_translator._INSTANCES.clear()


def test_get_translator_returns_none_when_ensure_fails(monkeypatch):
    _reset_singleton()
    monkeypatch.setattr(
        mt_translator, "ensure_mt_model",
        MagicMock(side_effect=NotImplementedError("no pin yet")),
    )
    assert get_translator("uk-en") is None


def test_get_translator_caches_instance(monkeypatch, tmp_path):
    _reset_singleton()
    monkeypatch.setattr(mt_translator, "ensure_mt_model", MagicMock(return_value=tmp_path))
    inst1 = get_translator("uk-en")
    inst2 = get_translator("uk-en")
    assert inst1 is inst2
    assert isinstance(inst1, MTTranslator)


def test_translate_swallows_load_errors_and_returns_none(tmp_path):
    _reset_singleton()
    inst = MTTranslator("uk-en", tmp_path)
    with patch.object(inst, "_lazy_load", side_effect=RuntimeError("boom")):
        assert inst.translate("hello") is None


def test_translate_returns_decoded_text(tmp_path):
    _reset_singleton()
    inst = MTTranslator("uk-en", tmp_path)
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    mock_tokenizer.decode.return_value = "hello world"
    mock_model = MagicMock()
    mock_model.generate.return_value = [[1, 2, 3]]
    with patch.object(inst, "_lazy_load"):
        inst._tokenizer = mock_tokenizer
        inst._model = mock_model
        assert inst.translate("привіт світ") == "hello world"
    mock_model.generate.assert_called_once()
