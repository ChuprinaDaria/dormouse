"""Compose Layer-B training pairs (dirty UA → compressed EN) from Layer-A pairs.

For each (dirty, clean) pair: cut aligned expression windows (2-4 words, the
range the deployed model sees), derive the EN target from the CLEAN side via
crack_open → compress → strict lexicon map, and keep only windows passing the
production acceptance gate (all-Latin, sane length). Pairs whose dirty hash is
in the frozen eval set are excluded.

Input:  data/pairs/<source>.jsonl
Output: data/train/<source>.jsonl  {"src","tgt","clean","source","license"}
        data/train/report_<source>.json

Usage:
    DORMOUSE_DATA_DIR=data/assets python scripts/compose_train.py --source ua_gec
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_lib.align import extract_windows  # noqa: E402
from dataset_lib.compose import compose_en, target_ok  # noqa: E402
from dataset_lib.io import (  # noqa: E402
    PAIRS_DIR,
    TRAIN_DIR,
    load_frozen_hashes,
    read_jsonl,
    sha256_text,
    write_jsonl,
)

from dormouse.lexicon_db import get_lexicon  # noqa: E402


def compose_source(source: str, conn, frozen: set[str]) -> dict:
    report = {
        "input_pairs": 0, "windows": 0, "kept": 0,
        "dropped": {"compose_failed": 0, "gate_rejected": 0, "frozen": 0, "duplicate": 0},
    }
    records = []
    seen: set[str] = set()

    for rec in read_jsonl(PAIRS_DIR / f"{source}.jsonl"):
        report["input_pairs"] += 1
        for src_win, clean_win in extract_windows(rec["dirty"], rec["clean"]):
            report["windows"] += 1
            src = src_win.lower()
            key = sha256_text(src)
            if key in seen:
                report["dropped"]["duplicate"] += 1
                continue
            if key in frozen:
                report["dropped"]["frozen"] += 1
                continue
            tgt = compose_en(clean_win, conn)
            if tgt is None:
                report["dropped"]["compose_failed"] += 1
                continue
            if not target_ok(src, tgt):
                report["dropped"]["gate_rejected"] += 1
                continue
            seen.add(key)
            records.append(
                {
                    "src": src,
                    "tgt": tgt,
                    "clean": clean_win,
                    "source": rec.get("source", source),
                    "license": rec.get("license", ""),
                }
            )
            report["kept"] += 1

    write_jsonl(TRAIN_DIR / f"{source}.jsonl", records)
    (TRAIN_DIR / f"report_{source}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="pairs file stem, e.g. ua_gec")
    args = parser.parse_args()

    if not (PAIRS_DIR / f"{args.source}.jsonl").exists():
        sys.exit(f"data/pairs/{args.source}.jsonl not found — run unify_pairs.py first")

    conn = get_lexicon()
    frozen = load_frozen_hashes()
    if frozen:
        print(f"frozen hashes loaded: {len(frozen)}")
    report = compose_source(args.source, conn, frozen)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
