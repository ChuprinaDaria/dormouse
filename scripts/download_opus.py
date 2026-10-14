"""Download parallel uk-en data from OPUS (OpenSubtitles) and Tatoeba.

Fetches Moses-format aligned text from OPUS and sentence pairs from Tatoeba,
filters by word length (short conversational phrases), corrupts the UA side
with Corruptor (so the model trains on dirty→real_EN), and outputs Layer-A
pairs with an 'en' field for direct-EN composition.

Output: data/pairs/opus_subs.jsonl   (OpenSubtitles, corrupted)
        data/pairs/tatoeba.jsonl     (Tatoeba, corrupted)
        data/raw/opus_subs/          (cached raw downloads)
        data/raw/tatoeba/

Usage:
    python scripts/download_opus.py [--max-opus 150000] [--max-tatoeba 30000]
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_lib.corrupt import Corruptor  # noqa: E402
from dataset_lib.io import (  # noqa: E402
    PAIRS_DIR,
    RAW_DIR,
    REPO_ROOT,
    load_frozen_hashes,
    manifest_append,
    sha256_text,
    write_jsonl,
)
from dataset_lib.normalize import is_ukrainian, normalize  # noqa: E402

# --- URLs ---
# OPUS OpenSubtitles v2018 Moses en-uk (zipped parallel txt)
OPUS_URL = "https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-uk.txt.zip"
# Tatoeba challenge, ukr-eng (TSV from OPUS/Tatoeba)
TATOEBA_URL = "https://object.pouta.csc.fi/Tatoeba-Challenge-v2023-09-26/ukr-eng.tar"

# --- Filters ---
MAX_UK_WORDS = 6
MAX_EN_WORDS_DELTA = 2  # len(en) <= len(uk) + delta
MIN_WORDS = 2


def _word_count(s: str) -> int:
    return len(s.split())


def length_ok(uk: str, en: str) -> bool:
    uk_wc = _word_count(uk)
    en_wc = _word_count(en)
    return (MIN_WORDS <= uk_wc <= MAX_UK_WORDS and
            en_wc <= uk_wc + MAX_EN_WORDS_DELTA and
            en_wc >= 1)


# --- Download helpers ---

def download_opus(out_dir: Path) -> tuple[Path, Path]:
    """Download and extract OpenSubtitles Moses files. Returns (uk_path, en_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "en-uk.txt.zip"

    uk_path = out_dir / "OpenSubtitles.en-uk.uk"
    en_path = out_dir / "OpenSubtitles.en-uk.en"

    if uk_path.exists() and en_path.exists():
        print(f"OPUS files already cached at {out_dir}")
        return uk_path, en_path

    print(f"Downloading OpenSubtitles en-uk from OPUS...")
    urlretrieve(OPUS_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        uk_name = next(n for n in names if n.endswith(".uk"))
        en_name = next(n for n in names if n.endswith(".en"))
        with open(uk_path, "wb") as f:
            f.write(zf.read(uk_name))
        with open(en_path, "wb") as f:
            f.write(zf.read(en_name))
    zip_path.unlink()
    print(f"Extracted: {uk_path.name}, {en_path.name}")
    return uk_path, en_path


def download_tatoeba(out_dir: Path) -> tuple[Path, Path]:
    """Download Tatoeba challenge ukr-eng. Returns (uk_path, en_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    uk_path = out_dir / "tatoeba.uk"
    en_path = out_dir / "tatoeba.en"

    if uk_path.exists() and en_path.exists():
        print(f"Tatoeba files already cached at {out_dir}")
        return uk_path, en_path

    tar_path = out_dir / "ukr-eng.tar"
    print(f"Downloading Tatoeba ukr-eng...")
    urlretrieve(TATOEBA_URL, tar_path)

    import tarfile
    with tarfile.open(tar_path) as tf:
        # Tatoeba challenge has train/dev/test splits as TSV
        # We want all available pairs
        uk_lines = []
        en_lines = []
        for member in tf.getmembers():
            if not member.isfile():
                continue
            # Files like data/ukr-eng/train.src, train.trg, etc.
            f = tf.extractfile(member)
            if f is None:
                continue
            content = f.read()
            # TSV format: id\tlang\ttext or just src/trg parallel files
            name = member.name.lower()
            if name.endswith(".src") or name.endswith(".src.gz"):
                data = gzip.decompress(content) if name.endswith(".gz") else content
                uk_lines.extend(data.decode("utf-8").splitlines())
            elif name.endswith(".trg") or name.endswith(".trg.gz"):
                data = gzip.decompress(content) if name.endswith(".gz") else content
                en_lines.extend(data.decode("utf-8").splitlines())

    if not uk_lines or len(uk_lines) != len(en_lines):
        # Fallback: try single TSV file
        print(f"  Trying TSV fallback (got {len(uk_lines)} uk, {len(en_lines)} en)")
        with tarfile.open(tar_path) as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                content = f.read()
                name = member.name.lower()
                if "tsv" in name or "txt" in name:
                    data = gzip.decompress(content) if name.endswith(".gz") else content
                    for line in data.decode("utf-8").splitlines():
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            uk_lines.append(parts[0])
                            en_lines.append(parts[1])

    uk_path.write_text("\n".join(uk_lines), encoding="utf-8")
    en_path.write_text("\n".join(en_lines), encoding="utf-8")
    tar_path.unlink()
    print(f"Tatoeba: {len(uk_lines)} pairs extracted")
    return uk_path, en_path


# --- Processing ---

def load_corruptor() -> Corruptor:
    """Load Corruptor with synth_corrupt.yaml config."""
    cfg_path = REPO_ROOT / "scripts/configs/synth_corrupt.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return Corruptor(cfg, random.Random(1337))


def process_parallel(
    uk_path: Path,
    en_path: Path,
    source_name: str,
    license_: str,
    corruptor: Corruptor,
    max_pairs: int,
    frozen: set[str],
    existing_hashes: set[str],
) -> dict:
    """Filter + corrupt parallel pairs, write to data/pairs/<source>.jsonl."""
    report = {
        "total_lines": 0,
        "kept": 0,
        "dropped": {
            "length_filter": 0,
            "not_ukrainian": 0,
            "corrupt_failed": 0,
            "duplicate": 0,
            "frozen": 0,
            "normalize_empty": 0,
        },
    }

    records = []
    seen = set(existing_hashes)
    rng = random.Random(42)

    with open(uk_path, encoding="utf-8") as f_uk, open(en_path, encoding="utf-8") as f_en:
        for uk_raw, en_raw in zip(f_uk, f_en):
            report["total_lines"] += 1
            if len(records) >= max_pairs:
                break

            uk_raw = uk_raw.strip()
            en_raw = en_raw.strip()
            if not uk_raw or not en_raw:
                report["dropped"]["normalize_empty"] += 1
                continue

            uk = normalize(uk_raw)
            en = en_raw.lower().strip()

            if not uk or not en:
                report["dropped"]["normalize_empty"] += 1
                continue

            if not length_ok(uk, en):
                report["dropped"]["length_filter"] += 1
                continue

            if not is_ukrainian(uk):
                report["dropped"]["not_ukrainian"] += 1
                continue

            # Corrupt the UA side
            result = corruptor.corrupt(uk)
            if result is None:
                # Use original as-is (already "clean" input, still valid pair)
                dirty = uk
            else:
                dirty, _ops = result

            dirty_hash = sha256_text(dirty.lower())
            if dirty_hash in frozen:
                report["dropped"]["frozen"] += 1
                continue
            if dirty_hash in seen:
                report["dropped"]["duplicate"] += 1
                continue
            seen.add(dirty_hash)

            records.append({
                "dirty": dirty,
                "clean": uk,
                "en": en,
                "source": source_name,
                "license": license_,
            })
            report["kept"] += 1

    # Shuffle to avoid position bias from alphabetical/temporal ordering
    rng.shuffle(records)

    out_path = PAIRS_DIR / f"{source_name}.jsonl"
    write_jsonl(out_path, records)

    report_path = PAIRS_DIR / f"report_{source_name}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-opus", type=int, default=150_000)
    parser.add_argument("--max-tatoeba", type=int, default=30_000)
    parser.add_argument("--skip-opus", action="store_true", help="skip OpenSubtitles")
    parser.add_argument("--skip-tatoeba", action="store_true", help="skip Tatoeba")
    args = parser.parse_args()

    frozen = load_frozen_hashes()
    if frozen:
        print(f"Frozen hashes: {len(frozen)}")

    corruptor = load_corruptor()
    print("Corruptor loaded")

    # Collect existing hashes from other pairs for cross-source dedup
    existing: set[str] = set()
    if PAIRS_DIR.exists():
        from dataset_lib.io import read_jsonl
        for path in PAIRS_DIR.glob("*.jsonl"):
            if path.stem in ("opus_subs", "tatoeba"):
                continue
            for rec in read_jsonl(path):
                existing.add(sha256_text(rec["dirty"].lower()))
    print(f"Cross-source dedup hashes: {len(existing)}")

    if not args.skip_opus:
        print("\n=== OpenSubtitles uk-en ===")
        opus_dir = RAW_DIR / "opus_subs"
        uk_path, en_path = download_opus(opus_dir)
        report = process_parallel(
            uk_path, en_path,
            source_name="opus_subs",
            license_="opensubtitles (check per-file)",
            corruptor=corruptor,
            max_pairs=args.max_opus,
            frozen=frozen,
            existing_hashes=existing,
        )
        print(json.dumps(report, indent=2))
        manifest_append("OpenSubtitles v2018 uk-en (OPUS)", "v2018", "opensubtitles", [uk_path, en_path])

        # Add opus hashes to existing for tatoeba dedup
        for rec in (PAIRS_DIR / "opus_subs.jsonl").open(encoding="utf-8"):
            r = json.loads(rec)
            existing.add(sha256_text(r["dirty"].lower()))

    if not args.skip_tatoeba:
        print("\n=== Tatoeba uk-en ===")
        tat_dir = RAW_DIR / "tatoeba"
        uk_path, en_path = download_tatoeba(tat_dir)
        report = process_parallel(
            uk_path, en_path,
            source_name="tatoeba",
            license_="CC BY 2.0",
            corruptor=corruptor,
            max_pairs=args.max_tatoeba,
            frozen=frozen,
            existing_hashes=existing,
        )
        print(json.dumps(report, indent=2))
        manifest_append("Tatoeba Challenge ukr-eng", "2023-09-26", "CC BY 2.0", [uk_path, en_path])


if __name__ == "__main__":
    main()
