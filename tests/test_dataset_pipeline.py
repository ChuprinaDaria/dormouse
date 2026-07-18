"""Тести dataset-пайплайну (scripts/dataset_lib). CI-safe: без даних і мережі."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dataset_lib.align import extract_windows  # noqa: E402
from dataset_lib.compose import strict_map_to_en, target_ok  # noqa: E402
from dataset_lib.corrupt import build_inverse_rules, dirty_variant  # noqa: E402
from dataset_lib.filters import passes_filters  # noqa: E402
from dataset_lib.io import sha256_text  # noqa: E402
from dataset_lib.normalize import is_ukrainian, normalize  # noqa: E402


class TestNormalize:
    def test_nfc_and_spaces(self):
        assert normalize("слово   слово\n\nще") == "слово слово ще"

    def test_apostrophe_folding(self):
        # U+2019 і U+02BC → U+0027 (як у crack_open)
        assert normalize("комп’ютер") == "комп'ютер"
        assert normalize("компʼютер") == "комп'ютер"

    def test_idempotent(self):
        text = normalize("  комп’ютер   працює  ")
        assert normalize(text) == text


class TestIsUkrainian:
    def test_ukrainian(self):
        assert is_ukrainian("це українське речення з літерою ї")

    def test_russian_rejected(self):
        assert not is_ukrainian("это русский текст с буквой ы и э")

    def test_latin_rejected(self):
        assert not is_ukrainian("this is english text")

    def test_empty_rejected(self):
        assert not is_ukrainian("")


class TestFilters:
    def test_identical_dropped(self):
        ok, reason = passes_filters("текст", "текст", set(), "h1")
        assert not ok and reason == "identical"

    def test_too_long_dropped(self):
        long = "слово " * 200
        ok, reason = passes_filters(long, "текст", set(), "h1")
        assert not ok and reason == "too_long"

    def test_duplicate_dropped(self):
        ok, reason = passes_filters("брудний текст", "чистий текст", {"h1"}, "h1")
        assert not ok and reason == "duplicate"

    def test_frozen_dropped(self):
        ok, reason = passes_filters("брудний текст", "чистий текст", set(), "h1", {"h1"})
        assert not ok and reason == "frozen"

    def test_good_pair_passes(self):
        ok, reason = passes_filters("шо там по багу", "що там з багом", set(), "h1")
        assert ok and reason == ""


class TestAlign:
    def test_single_edit_window(self):
        dirty = "вчора ми ходили в кіно і бачили шось цікаве там ввечері"
        clean = "вчора ми ходили в кіно і бачили щось цікаве там ввечері"
        windows = extract_windows(dirty, clean)
        assert ("бачили шось цікаве", "бачили щось цікаве") in windows

    def test_short_pair_whole(self):
        windows = extract_windows("шо там", "що там")
        assert ("шо там", "що там") in windows

    def test_windows_capped_at_max_n(self):
        dirty = "а б в г д е ж з" + " зовсім інший текст повністю"
        for w_dirty, _ in extract_windows(dirty, "інша послідовність слів тут"):
            assert len(w_dirty.split()) <= 4

    def test_no_edits_no_windows(self):
        long_same = "одне і те саме речення без жодних змін узагалі тут"
        assert extract_windows(long_same, long_same) == []


class TestCompose:
    def _conn(self, tmp_path):
        from dormouse.lexicon_db import get_lexicon

        conn = get_lexicon(tmp_path / "lex.db")
        rows = [
            ("помилка", "error", 1),
            ("виправити", "fix", 1),
            ("треба", "need", 1),
            ("як справи", "how?", 2),
        ]
        for word, en, ngram in rows:
            conn.execute(
                "INSERT INTO lexicon (word, normalized, en_compressed, ngram) "
                "VALUES (?, NULL, ?, ?)",
                (word, en, ngram),
            )
        conn.commit()
        return conn

    def test_word_lookup(self, tmp_path):
        conn = self._conn(tmp_path)
        assert strict_map_to_en("треба виправити", conn) == "need fix"

    def test_expression_lookup(self, tmp_path):
        conn = self._conn(tmp_path)
        assert strict_map_to_en("як справи", conn) == "how?"

    def test_lemma_fallback(self, tmp_path):
        conn = self._conn(tmp_path)
        # "помилку" немає, лема "помилка" є
        assert strict_map_to_en("виправити помилку", conn) == "fix error"

    def test_unknown_word_returns_none(self, tmp_path):
        conn = self._conn(tmp_path)
        assert strict_map_to_en("треба надзвичайнослово", conn) is None

    def test_no_translit_garbage(self, tmp_path):
        conn = self._conn(tmp_path)
        assert strict_map_to_en("бозна-що", conn) is None


class TestTargetOk:
    def test_good_target(self):
        assert target_ok("шо там по багу", "bug status?")

    def test_cyrillic_target_rejected(self):
        assert not target_ok("шо там", "що там")

    def test_too_long_target_rejected(self):
        assert not target_ok("шо там", "a b c d e f")

    def test_empty_rejected(self):
        assert not target_ok("шо там", None)
        assert not target_ok("шо там", "")


class TestCorrupt:
    def test_inverse_rules_from_bundled_json(self):
        inverse = build_inverse_rules()
        assert "що" in inverse
        assert set(inverse["що"]) >= {"шо", "чо"}
        assert all(variants for variants in inverse.values())

    def test_no_null_targets(self):
        inverse = build_inverse_rules()
        assert None not in inverse
        assert "" not in inverse

    def test_dirty_variant_deterministic(self):
        inverse = build_inverse_rules()
        text = "що там взагалі відбувається"
        v1 = dirty_variant(text, inverse, random.Random(7))
        v2 = dirty_variant(text, inverse, random.Random(7))
        assert v1 == v2
        assert v1 is not None and v1 != text

    def test_dirty_variant_none_when_no_candidates(self):
        inverse = build_inverse_rules()
        assert dirty_variant("qwerty asdf", inverse, random.Random(1)) is None


class TestCorruptor:
    CFG = {
        "min_ops": 1,
        "ops": {
            "rule_inversion": {"p": 1.0, "max_per_sentence": 2},
            "typo_neighbor": {"p": 0.1},
            "char_drop": {"p": 0.1},
            "translit_word": {"p": 0.05},
            "filler_insert": {"p": 0.5, "max": 1},
            "intensifier_insert": {"p": 0.3},
        },
    }

    def _corruptor(self, seed=7):
        from dataset_lib.corrupt import Corruptor

        return Corruptor(self.CFG, random.Random(seed))

    def test_deterministic_by_seed(self):
        text = "що там взагалі відбувається з цим завданням"
        r1 = self._corruptor(5).corrupt(text)
        r2 = self._corruptor(5).corrupt(text)
        assert r1 == r2

    def test_corruption_changes_text(self):
        result = self._corruptor().corrupt("що там взагалі відбувається")
        assert result is not None
        dirty, ops = result
        assert dirty != "що там взагалі відбувається"
        assert ops

    def test_min_ops_returns_none(self):
        from dataset_lib.corrupt import Corruptor

        cfg = {"min_ops": 1, "ops": {op: {"p": 0.0} for op in self.CFG["ops"]}}
        assert Corruptor(cfg, random.Random(1)).corrupt("чисте речення тут") is None

    def test_keyboard_neighbors_symmetric(self):
        from dataset_lib.corrupt import KEYBOARD_NEIGHBORS

        for ch, neighbors in KEYBOARD_NEIGHBORS.items():
            for n in neighbors:
                assert ch in KEYBOARD_NEIGHBORS[n], f"{ch}<->{n} не симетричні"

    def test_null_fillers_loaded(self):
        from dataset_lib.corrupt import null_fillers

        fillers = null_fillers()
        assert fillers
        assert all(isinstance(f, str) and f for f in fillers)


class TestHash:
    def test_sha256_stable(self):
        assert sha256_text("текст") == sha256_text("текст")
        assert sha256_text("текст") != sha256_text("інший")


class TestCheckpointRoundtrip:
    def test_save_checkpoint_loads_via_wake_up_expr(self, tmp_path):
        """Замок на деплой-сумісність: чекпоінт з train_expressions мусить
        вантажитись продакшн-лоадером wake_up_expr і перекладати."""
        pytest.importorskip("torch")
        import dormouse.seq2seq as seq2seq
        from train_expressions import save_checkpoint

        src_vocab = seq2seq.WordVocab(min_freq=1)
        src_vocab.build(["шо там", "як справи", "шо там по багу"])
        tgt_vocab = seq2seq.WordVocab(min_freq=1)
        tgt_vocab.build(["status?", "how?", "bug status?"])

        model_cfg = {"embed_dim": 16, "hidden_dim": 32, "dropout": 0.0}
        model = seq2seq.ExpressionTranslator(
            len(src_vocab), len(tgt_vocab), **model_cfg
        )
        save_checkpoint(model, src_vocab, tgt_vocab, model_cfg, tmp_path)

        for name in ("expr_seq2seq.pt", "expr_config.json",
                     "expr_vocab_src.json", "expr_vocab_tgt.json"):
            assert (tmp_path / name).exists()

        seq2seq._expr_cache = None
        loaded = seq2seq.wake_up_expr(model_dir=tmp_path)
        assert loaded is not None
        result = seq2seq.translate_expression("шо там", model_dir=tmp_path)
        assert result is None or isinstance(result, str)
        seq2seq._expr_cache = None

    def test_saved_keys_are_short_format(self, tmp_path):
        pytest.importorskip("torch")
        import torch

        import dormouse.seq2seq as seq2seq
        from train_expressions import save_checkpoint

        vocab = seq2seq.WordVocab(min_freq=1)
        vocab.build(["слово тест"])
        cfg = {"embed_dim": 16, "hidden_dim": 32, "dropout": 0.0}
        model = seq2seq.ExpressionTranslator(len(vocab), len(vocab), **cfg)
        save_checkpoint(model, vocab, vocab, cfg, tmp_path)
        seq2seq._expr_cache = None

        state = torch.load(tmp_path / "expr_seq2seq.pt", weights_only=True)
        prefixes = ("enc.emb.", "enc.rnn.", "enc.fc.", "dec.emb.",
                    "dec.attn.a.", "dec.attn.v.", "dec.rnn.", "dec.fc.")
        for key in state:
            assert key.startswith(prefixes), f"довгий ключ у чекпоінті: {key}"
