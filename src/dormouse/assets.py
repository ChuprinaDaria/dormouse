"""Lazy-download та кешування даних dormouse.

Великі файли (lexicon.db, seq2seq model) завантажуються при першому
використанні з GitHub Releases або HuggingFace Hub.
"""

import os
import sys
import urllib.request
from pathlib import Path

VERSION = "0.3.0"

_GITHUB_BASE = (
    "https://github.com/ChuprinaDaria/dormouse/releases/download/v{version}"
)
_HF_BASE = "https://huggingface.co/ChuprinaDaria/dormouse/resolve/main"

_DEFAULT_CACHE = Path("~/.cache/dormouse").expanduser()


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

        req = urllib.request.Request(url, headers={"User-Agent": "dormouse/" + VERSION})
        with urllib.request.urlopen(req, timeout=60) as resp:
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
