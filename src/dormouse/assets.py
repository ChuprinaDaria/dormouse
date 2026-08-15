"""Lazy-download та кешування даних dormouse.

Великі файли (lexicon.db, seq2seq model) завантажуються при першому
використанні з GitHub Releases або HuggingFace Hub.

MT-моделі (ONNX INT8) живуть окремо: `Dariachup/dormouse-mt-{direction}`,
беруться через `ensure_mt_model()` з sha256-верифікацією.
"""

import hashlib
import os
import sys
import urllib.request
from importlib.resources import files
from pathlib import Path

VERSION = "0.3.0"
MT_VERSION = "v0.7.0"

# SHA256 і розмір для кожного файлу MT-моделі. Оновлювати при кожному
# новому HF push. None = pin ще не знятий; ensure_mt_model raise-ить
# NotImplementedError, поки напрямок не прописано.
MT_ASSET_MANIFEST: dict[str, dict[str, tuple[str, int] | None]] = {
    "uk-en": {
        "config.json": ("25fb03a1b0bb78ab5ea1479c625289d3eb00fab3c5541b4025f95c59907e604a", 1287),
        "generation_config.json": ("9c988bb041710f15898b986795d8dc6ed73d7caa5886df6626fac4006aeda3e7", 994),
        "model.safetensors": ("c04b95c3fce45343edd542973543ee6fe4118c6055fb339791fad19b627ad82a", 302959564),
        "source.spm": ("7abd810b2df512e6fe177f8732f6a0b3107614c84cc52aba20d805b4cb0d7b6b", 1007605),
        "target.spm": ("4dc9157e60a15b157c2f3d5f892379b03aed111c9fe8ac7cecba5f2aa56379ef", 808645),
        "tokenizer_config.json": ("655e36db6b722e47c9283fbcb3f3f0e5e36babf26627aebe378183e1f3b69823", 849),
        "vocab.json": ("e8b45188b1db6a4e0f960813e1bd493948ab0bc4d57b069ea03e77a78b7567d4", 2490886),
    },
    "en-uk": {
        "config.json": ("25fb03a1b0bb78ab5ea1479c625289d3eb00fab3c5541b4025f95c59907e604a", 1287),
        "generation_config.json": ("3011df47504d1638158179f949456c6e97cbac0daeb8c025e44ba1500778fac0", 994),
        "model.safetensors": ("63eb7a9f3e10e40b270dc933cf906a39035f636f409eb930dc4ebe0bbef80d7e", 302959564),
        "source.spm": ("4dc9157e60a15b157c2f3d5f892379b03aed111c9fe8ac7cecba5f2aa56379ef", 808645),
        "target.spm": ("7abd810b2df512e6fe177f8732f6a0b3107614c84cc52aba20d805b4cb0d7b6b", 1007605),
        "tokenizer_config.json": ("488da4caa6829d110a01adb6b3c0853dbb9ee5a426ba371e9025ac5b241f3de6", 849),
        "vocab.json": ("e8b45188b1db6a4e0f960813e1bd493948ab0bc4d57b069ea03e77a78b7567d4", 2490886),
    },
}

_GITHUB_BASE = (
    "https://github.com/ChuprinaDaria/dormouse/releases/download/v{version}"
)
_HF_BASE = "https://huggingface.co/Dariachup/dormouse/resolve/main"
_HF_MT_BASE = "https://huggingface.co/Dariachup/dormouse-mt-{direction}/resolve/main"

_DEFAULT_CACHE = Path("~/.cache/dormouse").expanduser()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_dir() -> Path:
    """Кеш директорія з версійним підкаталогом."""
    base = Path(os.environ.get("DORMOUSE_CACHE_DIR", str(_DEFAULT_CACHE)))
    return base / f"v{VERSION}"


def _data_dir() -> Path | None:
    """Dev mode: локальна директорія з даними (скіпає download)."""
    val = os.environ.get("DORMOUSE_DATA_DIR")
    if val:
        return Path(val)
    return None


def _download(url: str, dest: Path) -> bool:
    """Завантажує файл з URL в dest. Повертає True при успіху."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")

        req = urllib.request.Request(url, headers={
            "User-Agent": "dormouse/" + VERSION,
            "Accept": "application/octet-stream",
        })
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block = 64 * 1024

            name = dest.name
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(block)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total and sys.stderr.isatty():
                        pct = downloaded * 100 // total
                        mb = downloaded / 1024 / 1024
                        total_mb = total / 1024 / 1024
                        sys.stderr.write(
                            f"\r  Downloading {name}... "
                            f"{mb:.1f}/{total_mb:.1f}MB [{pct}%]"
                        )
                        sys.stderr.flush()

            if sys.stderr.isatty() and total:
                sys.stderr.write("\n")

        tmp.rename(dest)
        return True
    except (OSError, urllib.error.URLError):
        # Cleanup tmp якщо лишився
        tmp = dest.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()
        return False


def get_asset(name: str) -> Path:
    """Повертає шлях до asset файлу. Завантажує якщо нема в кеші.

    Порядок пошуку:
    1. DORMOUSE_DATA_DIR (dev mode)
    2. Cache (~/.cache/dormouse/v{version}/)
    3. Download: GitHub Releases → HuggingFace fallback

    Raises:
        FileNotFoundError: Якщо файл не знайдено і download невдалий.
    """
    # Dev mode — пряме посилання на локальні дані
    dev = _data_dir()
    if dev:
        # Шукаємо в різних піддиректоріях dev dir
        candidates = [
            dev / "db" / name,
            dev / name,
            dev / "lexicon" / name,
        ]
        for p in candidates:
            if p.exists():
                return p

    # Bundled package data (бандлені файли в dormouse/data/)
    try:
        pkg_file = files("dormouse.data").joinpath(name)
        pkg_path = Path(str(pkg_file))
        if pkg_path.exists():
            return pkg_path
    except (TypeError, FileNotFoundError):
        pass

    # Cache
    cache = _cache_dir()
    cached = cache / name
    if cached.exists():
        return cached

    # Offline mode
    if os.environ.get("DORMOUSE_OFFLINE"):
        raise FileNotFoundError(
            f"{name} not found in cache. "
            f"Set DORMOUSE_DATA_DIR or disable DORMOUSE_OFFLINE."
        )

    # Download
    github_url = _GITHUB_BASE.format(version=VERSION) + "/" + name
    if _download(github_url, cached):
        return cached

    hf_url = _HF_BASE + "/" + name
    if _download(hf_url, cached):
        return cached

    raise FileNotFoundError(
        f"Failed to download {name}. Check internet connection or set DORMOUSE_DATA_DIR."
    )


def ensure_assets(names: list[str]) -> dict[str, Path]:
    """Завантажує кілька assets за один виклик.

    Returns:
        {name: Path} для кожного файлу.
    """
    return {name: get_asset(name) for name in names}


def _mt_cache_dir(direction: str) -> Path:
    base = Path(os.environ.get("DORMOUSE_CACHE_DIR", str(_DEFAULT_CACHE)))
    return base / "mt" / MT_VERSION / direction


def ensure_mt_model(direction: str) -> Path:
    """Завантажує ONNX MT-модель для напрямку (uk-en / en-uk).

    Порядок:
    1. DORMOUSE_MT_DIR env var (dev mode override для одного напрямку).
    2. Cache ~/.cache/dormouse/mt/{MT_VERSION}/{direction}/.
    3. Download з HF Hub `Dariachup/dormouse-mt-{direction}`.

    Кожен файл верифікується за sha256 з MT_ASSET_MANIFEST. Якщо hash не
    збігається — файл видаляється і робиться повторний download. Порожня
    маніфест-мапа для напрямку піднімає NotImplementedError (модель ще не
    задеплоєна).

    Returns:
        Path директорії з готовими файлами моделі.
    """
    env_dir = os.environ.get(f"DORMOUSE_MT_DIR_{direction.replace('-', '_').upper()}")
    if env_dir:
        p = Path(env_dir)
        if not p.exists():
            raise FileNotFoundError(f"DORMOUSE_MT_DIR override does not exist: {p}")
        return p

    manifest = MT_ASSET_MANIFEST.get(direction) or {}
    if not manifest:
        raise NotImplementedError(
            f"MT model for direction={direction!r} not yet published to HF. "
            "Once export_onnx.py + upload is done, pin the sha256s in "
            "MT_ASSET_MANIFEST and retry."
        )

    cache = _mt_cache_dir(direction)
    cache.mkdir(parents=True, exist_ok=True)

    for relpath, entry in manifest.items():
        expected_sha, _size = entry
        local = cache / relpath
        if local.exists() and _sha256_file(local) == expected_sha:
            continue

        if os.environ.get("DORMOUSE_OFFLINE"):
            raise FileNotFoundError(
                f"MT model file missing from cache: {local} (offline mode)"
            )

        url = _HF_MT_BASE.format(direction=direction) + "/" + relpath
        if not _download(url, local):
            raise FileNotFoundError(f"failed to download {url}")
        actual = _sha256_file(local)
        if actual != expected_sha:
            local.unlink(missing_ok=True)
            raise ValueError(
                f"sha256 mismatch for {relpath}: expected {expected_sha}, got {actual}"
            )

    return cache
