"""Unify raw dataset records into canonical Layer-A pairs.

Input: data/raw/<source>/*.jsonl with at least {"dirty","clean"} per record.
Output: data/pairs/<source>.jsonl with {"dirty","clean","source","license"}
plus data/pairs/report_<source>.json (kept/dropped counts by reason).

Per record: normalize (NFC + apostrophes + whitespace) → filters (identical,
length, language, dedup) → PII scrub. Document-level records are split into
sentence pairs first when both sides have the same sentence count.

Usage:
    python scripts/unify_pairs.py --source ua_gec --license "CC BY 4.0"
    python scripts/unify_pairs.py --source omnigec --license "see MANIFEST"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_lib.filters import passes_filters  # noqa: E402
from dataset_lib.io import PAIRS_DIR, RAW_DIR, read_jsonl, sha256_text, write_jsonl  # noqa: E402
from dataset_lib.normalize import normalize  # noqa: E402

from dormouse.pii import scrub_pair  # noqa: E402

_RE_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯІЇЄҐA-Z«\"])")


def sentence_pairs(dirty: str, clean: str) -> list[tuple[str, str]]:
    """Розбиває документ-пару на реченнєві пари, якщо кількість речень
    збігається (інакше — повертає документ цілком)."""
    out = []
    for d_line, c_line in zip(dirty.splitlines(), clean.splitlines()):
        d_sents = [s for s in _RE_SENT_SPLIT.split(d_line) if s.strip()]
        c_sents = [s for s in _RE_SENT_SPLIT.split(c_line) if s.strip()]
        if len(d_sents) == len(c_sents):
            out.extend(zip(d_sents, c_sents))
        else:
            out.append((d_line, c_line))
    return [(d, c) for d, c in out if d.strip() and c.strip()]


def unify(source: str, license_: str, in_dir: Path, existing_hashes: set[str]) -> dict:
    records = []
    report = {"input_records": 0, "candidate_pairs": 0, "kept": 0, "dropped": {}}
    seen = set(existing_hashes)

    def drop(reason: str) -> None:
        report["dropped"][reason] = report["dropped"].get(reason, 0) + 1

    for path in sorted(in_dir.glob("*.jsonl")):
        for rec in read_jsonl(path):
            report["input_records"] += 1
            for d_raw, c_raw in sentence_pairs(rec["dirty"], rec["clean"]):
                report["candidate_pairs"] += 1
                dirty, clean = normalize(d_raw), normalize(c_raw)
                dirty_hash = sha256_text(dirty.lower())
                ok, reason = passes_filters(dirty, clean, seen, dirty_hash)
                if not ok:
                    drop(reason)
                    continue
                scrubbed = scrub_pair(dirty, clean)
                if scrubbed is None:
                    drop("pii_dominated")
                    continue
                seen.add(dirty_hash)
                records.append(
                    {
                        "dirty": scrubbed[0],
                        "clean": scrubbed[1],
                        "source": source,
                        "license": license_,
                    }
                )
                report["kept"] += 1

    out_path = PAIRS_DIR / f"{source}.jsonl"
    write_jsonl(out_path, records)
    report_path = PAIRS_DIR / f"report_{source}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def load_existing_hashes(exclude: str) -> set[str]:
    """Хеші dirty з усіх інших pairs-файлів — крос-джерельний дедуп."""
    hashes = set()
    if PAIRS_DIR.exists():
        for path in PAIRS_DIR.glob("*.jsonl"):
            if path.stem == exclude:
                continue
            for rec in read_jsonl(path):
                hashes.add(sha256_text(rec["dirty"].lower()))
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="source name = raw subdir = output stem")
    parser.add_argument("--license", required=True)
    parser.add_argument("--in-dir", type=Path, default=None)
    args = parser.parse_args()

    in_dir = args.in_dir or RAW_DIR / args.source
    if not in_dir.exists():
        sys.exit(f"{in_dir} does not exist — run the download script first")

    report = unify(args.source, args.license, in_dir, load_existing_hashes(args.source))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
