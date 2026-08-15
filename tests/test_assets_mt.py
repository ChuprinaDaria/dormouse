"""Tests for MT asset resolution — pure-python, no torch/onnxruntime."""

from unittest.mock import patch

import pytest

from dormouse import assets


def test_ensure_mt_model_raises_when_manifest_empty():
    """Fresh clone: manifest is empty for a direction ⇒ NotImplementedError."""
    with patch.dict(assets.MT_ASSET_MANIFEST, {"uk-en": {}, "en-uk": {}}, clear=True):
        with pytest.raises(NotImplementedError):
            assets.ensure_mt_model("uk-en")


def test_ensure_mt_model_env_override(tmp_path, monkeypatch):
    """DORMOUSE_MT_DIR_UK_EN short-circuits everything else."""
    override = tmp_path / "mt-uk-en"
    override.mkdir()
    monkeypatch.setenv("DORMOUSE_MT_DIR_UK_EN", str(override))
    assert assets.ensure_mt_model("uk-en") == override


def test_ensure_mt_model_env_missing_raises(monkeypatch):
    monkeypatch.setenv("DORMOUSE_MT_DIR_UK_EN", "/nonexistent/path/here")
    with pytest.raises(FileNotFoundError):
        assets.ensure_mt_model("uk-en")


def test_offline_and_missing_cache_raises(tmp_path, monkeypatch):
    """Manifest exists with a fake pin, cache empty, offline → FileNotFoundError."""
    fake_hash = "a" * 64
    monkeypatch.setenv("DORMOUSE_OFFLINE", "1")
    monkeypatch.setenv("DORMOUSE_CACHE_DIR", str(tmp_path))
    with patch.dict(
        assets.MT_ASSET_MANIFEST,
        {"uk-en": {"model.onnx": (fake_hash, 1000)}, "en-uk": {}},
        clear=True,
    ):
        with pytest.raises(FileNotFoundError):
            assets.ensure_mt_model("uk-en")


def test_sha256_mismatch_deletes_and_raises(tmp_path, monkeypatch):
    """Cached file with wrong hash → attempts re-download, still mismatches → ValueError."""
    good_hash = "b" * 64
    monkeypatch.delenv("DORMOUSE_OFFLINE", raising=False)
    monkeypatch.setenv("DORMOUSE_CACHE_DIR", str(tmp_path))
    cache = tmp_path / "mt" / assets.MT_VERSION / "uk-en"
    cache.mkdir(parents=True)
    bad_file = cache / "model.onnx"
    bad_file.write_bytes(b"garbage")

    def fake_download(url, dest):
        dest.write_bytes(b"still-garbage")
        return True

    with patch.dict(
        assets.MT_ASSET_MANIFEST,
        {"uk-en": {"model.onnx": (good_hash, 1)}, "en-uk": {}},
        clear=True,
    ), patch.object(assets, "_download", side_effect=fake_download):
        with pytest.raises(ValueError, match="sha256 mismatch"):
            assets.ensure_mt_model("uk-en")
    assert not bad_file.exists()  # bad file cleaned up
