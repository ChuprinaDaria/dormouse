"""Create the frozen eval set (Layer B) BEFORE any training.

Stratified sample across data/train/*.jsonl sources with per-source floors:
the lexicon stratum guards the currently-shipped behavior, GEC strata measure
the new normalization ability. Synthetic pairs are never eligible.

Outputs (all committed to git):
    data/eval/frozen_v1.jsonl       — the eval pairs
    data/eval/frozen_v1.hashes.txt  — sha256 of each src; every train-writing
                                      script drops matching pairs forever
    data/eval/FROZEN.md             — sha256 of the jsonl, counts, provenance

The jsonl is immutable once created: this script refuses to overwrite it.
A future frozen_v2 is a new file, not an edit.

Usage:
    python scripts/make_frozen_eval.py [--n 1500] [--seed 20260718]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_lib.io import EVAL_DIR, TRAIN_DIR, read_jsonl, sha256_file, sha256_text  # noqa: E402

EXCLUDED_SOURCES = {"synth"}
FLOORS = {"lexicon": 400}
DEFAULT_GEC_FLOOR = 300


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()

    out_jsonl = EVAL_DIR / "frozen_v1.jsonl"
    if out_jsonl.exists():
        sys.exit(f"{out_jsonl} already exists — frozen sets are immutable, create frozen_v2")

    rng = random.Random(args.seed)
    by_source: dict[str, list[dict]] = {}
    for path in sorted(TRAIN_DIR.glob("*.jsonl")):
        source = path.stem
        if source in EXCLUDED_SOURCES:
            continue
        rows = list(read_jsonl(path))
        if rows:
            by_source[source] = rows

    if not by_source:
        sys.exit("no train sources found — run compose_train.py / generate_expression_pairs.py")

    # флори, потім пропорційний розподіл залишку
    quotas = {src: min(FLOORS.get(src, DEFAULT_GEC_FLOOR), len(rows))
              for src, rows in by_source.items()}
    remaining = args.n - sum(quotas.values())
    if remaining > 0:
        total = sum(len(r) for r in by_source.values())
        for src, rows in by_source.items():
            extra = int(remaining * len(rows) / total)
            quotas[src] = min(quotas[src] + extra, len(rows))

    sample: list[dict] = []
    for src, rows in sorted(by_source.items()):
        rng.shuffle(rows)
        sample.extend(rows[: quotas[src]])
    rng.shuffle(sample)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in sample:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    hashes_path = EVAL_DIR / "frozen_v1.hashes.txt"
    hashes_path.write_text(
        "\n".join(sha256_text(rec["src"].lower()) for rec in sample) + "\n", encoding="utf-8"
    )

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    counts: dict[str, int] = {}
    for rec in sample:
        counts[rec["source"]] = counts.get(rec["source"], 0) + 1

    (EVAL_DIR / "FROZEN.md").write_text(
        f"""# Frozen eval set v1

**Immutable.** Never edit `frozen_v1.jsonl`; a future set is `frozen_v2.jsonl`.
Every train-writing script excludes pairs whose sha256(lower(src)) appears in
`frozen_v1.hashes.txt`.

- created: {date.today().isoformat()}
- generating commit: {commit}
- seed: {args.seed}
- pairs: {len(sample)}
- sha256(frozen_v1.jsonl): `{sha256_file(out_jsonl)}`
- per-source counts: {json.dumps(counts, ensure_ascii=False)}

Sources and licenses: see `data/raw/MANIFEST.md`. UA-GEC-derived pairs are
CC BY 4.0 (Grammarly UA-GEC corpus); lexicon-derived pairs are MIT (dormouse).

TODO (frozen_v2): ~200 hand-written pairs from the shops/LMS/support domains.
""",
        encoding="utf-8",
    )
    print(f"{out_jsonl}: {len(sample)} pairs; counts: {counts}")


if __name__ == "__main__":
    main()
