"""JSONL i/o, hashing and manifest helpers for the dataset pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PAIRS_DIR = DATA_DIR / "pairs"
TRAIN_DIR = DATA_DIR / "train"
EVAL_DIR = DATA_DIR / "eval"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
ASSETS_DIR = DATA_DIR / "assets"

MANIFEST_PATH = RAW_DIR / "MANIFEST.md"


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest_append(source: str, version: str, license_: str, files: list[Path]) -> None:
    """Append a dataset entry to data/raw/MANIFEST.md (creates it with a header)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        MANIFEST_PATH.write_text(
            "# Raw datasets manifest\n\n"
            "| Source | Version | License | Date | File | sha256 |\n"
            "|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    today = date.today().isoformat()
    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        for path in files:
            rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
            f.write(
                f"| {source} | {version} | {license_} | {today} | {rel} | {sha256_file(path)} |\n"
            )
