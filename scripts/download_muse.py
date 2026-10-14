"""Download MUSE uk-en bilingual dictionary and import into lexicon.db.

MUSE (Facebook Research) provides word-level translation dictionaries. We import
uk→en pairs that aren't already in the lexicon, filtered for quality:
- Only single-word entries (ngram=1)
- Skip if uk word already exists in lexicon
- Skip entries with >3 translations (ambiguous — take top 2 by position)
- Lowercase, Latin-only EN side

Output: updates lexicon.db in-place, reports additions.

Usage:
    python scripts/download_muse.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_lib.io import RAW_DIR, manifest_append  # noqa: E402

from dormouse.lexicon_db import get_lexicon  # noqa: E402

# MUSE dictionary URL (uk-en, full)
MUSE_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries/uk-en.txt"

_RE_LATIN = re.compile(r"^[a-z\-]+$")
_RE_CYRILLIC_WORD = re.compile(r"^[а-яіїєґ'\-]+$")

MAX_TRANSLATIONS = 2  # per uk word, take top N by position in file


def download_muse(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "uk-en.txt"
    if path.exists():
        print(f"MUSE dict cached at {path}")
        return path
    print("Downloading MUSE uk-en dictionary...")
    urlretrieve(MUSE_URL, path)
    print(f"Downloaded: {path}")
    return path


def parse_muse(path: Path) -> dict[str, list[str]]:
    """Parse MUSE TSV: uk_word \\t en_word. Returns {uk: [en1, en2, ...]}."""
    entries: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            uk = parts[0].lower().strip()
            en = parts[1].lower().strip()
            if not uk or not en:
                continue
            if not _RE_CYRILLIC_WORD.match(uk):
                continue
            if not _RE_LATIN.match(en):
                continue
            if len(uk) < 2 or len(en) < 2:
                continue
            entries.setdefault(uk, []).append(en)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-translations", type=int, default=MAX_TRANSLATIONS)
    args = parser.parse_args()

    muse_dir = RAW_DIR / "muse"
    path = download_muse(muse_dir)
    entries = parse_muse(path)
    print(f"MUSE entries parsed: {len(entries)} unique uk words")

    conn = get_lexicon()

    # Get existing words
    existing = set()
    for row in conn.execute("SELECT word FROM lexicon WHERE ngram = 1"):
        existing.add(row[0].lower())
    print(f"Existing single-word entries in lexicon: {len(existing)}")

    # Filter and prepare inserts
    to_insert = []
    skipped = {"already_exists": 0, "too_ambiguous": 0}

    for uk, en_list in entries.items():
        if uk in existing:
            skipped["already_exists"] += 1
            continue
        # Take top N translations
        translations = en_list[:args.max_translations]
        # Use first translation as primary
        en_compressed = translations[0]
        to_insert.append({
            "word": uk,
            "normalized": uk,
            "en_compressed": en_compressed,
            "en_full": ", ".join(translations) if len(translations) > 1 else en_compressed,
            "type": "word",
            "ngram": 1,
            "is_filler": False,
            "source": "muse",
        })

    print(f"New entries to insert: {len(to_insert)}")
    print(f"Skipped: {json.dumps(skipped)}")

    if args.dry_run:
        print("DRY RUN — no changes made")
        print(f"Sample (first 10):")
        for entry in to_insert[:10]:
            print(f"  {entry['word']} → {entry['en_compressed']}")
        return

    # Insert
    conn.executemany(
        """INSERT INTO lexicon (word, normalized, en_compressed, en_full, type, ngram, is_filler, source)
           VALUES (:word, :normalized, :en_compressed, :en_full, :type, :ngram, :is_filler, :source)""",
        to_insert,
    )
    conn.commit()
    print(f"Inserted {len(to_insert)} entries into lexicon.db")

    new_total = conn.execute("SELECT COUNT(*) FROM lexicon").fetchone()[0]
    print(f"New lexicon total: {new_total}")

    manifest_append("MUSE uk-en (Facebook Research)", "v1", "CC BY-NC 4.0", [path])


if __name__ == "__main__":
    main()
