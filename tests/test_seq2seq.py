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


class TestSubwordVocab:
    TEXTS = [
        "шо там по багу",
        "що там з багом",
        "треба виправити помилку",
        "помилка у формі замовлення",
        "як справи взагалі нормально",
        "де моє замовлення зараз",
    ]

    def _vocab(self):
        from dormouse.seq2seq import SubwordVocab

        v = SubwordVocab()
        v.train(self.TEXTS, vocab_size=300)
        return v

    def test_special_token_ids_match_wordvocab(self):
        from dormouse.seq2seq import SubwordVocab

        assert (SubwordVocab.PAD, SubwordVocab.SOS, SubwordVocab.EOS, SubwordVocab.UNK) == (
            WordVocab.PAD, WordVocab.SOS, WordVocab.EOS, WordVocab.UNK,
        )

    def test_roundtrip(self):
        v = self._vocab()
        for text in self.TEXTS:
            assert v.decode(v.encode(text)) == text

    def test_typo_and_digits_no_unk(self):
        v = self._vocab()
        # одруківка і число зі знайомих символів — жодного UNK
        for text in ["памилка", "багу 4512"]:
            ids = v.encode("памилка")
            assert v.UNK not in ids

    def test_digits_roundtrip(self):
        v = self._vocab()
        assert v.decode(v.encode("замовлення 4512")) == "замовлення 4512"

    def test_unknown_char_is_unk(self):
        v = self._vocab()
        ids = v.encode("参")
        assert v.UNK in ids

    def test_save_load_roundtrip(self, tmp_path):
        from dormouse.seq2seq import SubwordVocab

        v = self._vocab()
        v.save(tmp_path / "vocab.json")
        v2 = SubwordVocab()
        v2.load(tmp_path / "vocab.json")
        text = "шо там по багу"
        assert v2.encode(text) == v.encode(text)
        assert v2.decode(v2.encode(text)) == text

    def test_encode_lowercases(self):
        v = self._vocab()
        assert v.encode("ШО ТАМ") == v.encode("шо там")


class TestLoadVocab:
    def test_word_file_word_config(self, tmp_path):
        from dormouse.seq2seq import load_vocab

        v = WordVocab(min_freq=1)
        v.build(["слово тест"])
        v.save(tmp_path / "v.json")
        loaded = load_vocab(tmp_path / "v.json", "word")
        assert isinstance(loaded, WordVocab)

    def test_bpe_file_bpe_config(self, tmp_path):
        from dormouse.seq2seq import SubwordVocab, load_vocab

        v = SubwordVocab()
        v.train(["слово тест"], vocab_size=50)
        v.save(tmp_path / "v.json")
        loaded = load_vocab(tmp_path / "v.json", "bpe")
        assert isinstance(loaded, SubwordVocab)

    def test_mixed_formats_raise(self, tmp_path):
        import pytest as _pytest

        from dormouse.seq2seq import SubwordVocab, load_vocab

        word = WordVocab(min_freq=1)
        word.build(["слово тест"])
        word.save(tmp_path / "word.json")
        with _pytest.raises(ValueError):
            load_vocab(tmp_path / "word.json", "bpe")

        bpe = SubwordVocab()
        bpe.train(["слово тест"], vocab_size=50)
        bpe.save(tmp_path / "bpe.json")
        with _pytest.raises(ValueError):
            load_vocab(tmp_path / "bpe.json", "word")
