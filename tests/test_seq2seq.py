"""Тести для seq2seq моделі (ExpressionTranslator, word-level)."""

import pytest

torch = pytest.importorskip("torch")

from dormouse.seq2seq import ExpressionTranslator, WordVocab


class TestWordVocab:
    def test_build_and_encode(self):
        vocab = WordVocab(min_freq=1)
        vocab.build(["привіт як справи", "привіт тест"])
        encoded = vocab.encode("привіт")
        assert encoded[0] == WordVocab.SOS
        assert encoded[-1] == WordVocab.EOS
        assert len(encoded) == 3  # SOS + привіт + EOS

    def test_decode(self):
        vocab = WordVocab(min_freq=1)
        vocab.build(["привіт як справи"])
        encoded = vocab.encode("привіт як")
        decoded = vocab.decode(encoded)
        assert decoded == "привіт як"

    def test_unk_token(self):
        vocab = WordVocab(min_freq=1)
        vocab.build(["привіт"])
        encoded = vocab.encode("невідоме слово")
        # Unknown words get UNK token
        assert WordVocab.UNK in encoded

    def test_save_load(self, tmp_path):
        vocab = WordVocab(min_freq=1)
        vocab.build(["привіт тест"])
        path = tmp_path / "vocab.json"
        vocab.save(path)

        vocab2 = WordVocab()
        vocab2.load(path)
        assert len(vocab2) == len(vocab)
        assert vocab2.encode("привіт") == vocab.encode("привіт")


class TestExpressionTranslator:
    def test_forward_shape(self):
        src_vocab = WordVocab(min_freq=1)
        src_vocab.build(["привіт як справи"])
        tgt_vocab = WordVocab(min_freq=1)
        tgt_vocab.build(["hello how are you"])

        model = ExpressionTranslator(len(src_vocab), len(tgt_vocab))
        src = torch.tensor([src_vocab.encode("привіт як")])
        tgt = torch.tensor([tgt_vocab.encode("hello how")])

        output = model(src, tgt, teacher_forcing_ratio=1.0)
        assert output.shape[0] == 1
        assert output.shape[1] == tgt.shape[1]
        assert output.shape[2] == len(tgt_vocab)

    def test_translate(self):
        src_vocab = WordVocab(min_freq=1)
        src_vocab.build(["привіт як справи"])
        tgt_vocab = WordVocab(min_freq=1)
        tgt_vocab.build(["hello how are you"])

        model = ExpressionTranslator(len(src_vocab), len(tgt_vocab))
        src_ids = torch.tensor(src_vocab.encode("привіт"))
        result = model.translate(src_ids, tgt_vocab)
        assert isinstance(result, str)

    def test_wake_up_expr_missing(self, tmp_path):
        from dormouse.seq2seq import wake_up_expr

        result = wake_up_expr(tmp_path)
        assert result is None
