"""Generate synthetic (dirty, clean) pairs by corrupting clean sentences.

Sources: clean-sentence files from downloads (UA-GEC clean side, Brown-UK,
UberText social sample) listed in the config with sampling weights. Ops:
rule inversion (clean → surzhyk/slang via inverted replacement rules),
ЙЦУКЕН neighbor typos, char drops, word transliteration, filler/intensifier
injection (compress() removes exactly these, so composed EN targets teach the
model to drop them).

Output is Layer A (data/pairs/synth.jsonl) — EN targets come later from
compose_train.py like every other source. A .meta.json sidecar records the
config, seed and rule-file hash for reproducibility.

Usage:
    python scripts/synth_corrupt.py [--config scripts/configs/synth_corrupt.yaml]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_lib.corrupt import Corruptor  # noqa: E402
from dataset_lib.io import REPO_ROOT, sha256_text, write_jsonl  # noqa: E402
from dataset_lib.normalize import is_ukrainian, normalize  # noqa: E402

from dormouse.rule_engine import LEXICON_PATH  # noqa: E402


def load_sentences(cfg: dict) -> list[tuple[str, float]]:
    """(речення, вага джерела) з усіх наявних input-файлів конфігу."""
    out = []
    for inp in cfg["inputs"]:
        path = REPO_ROOT / inp["path"]
        if not path.exists():
            print(f"warning: {path} missing, skipping", file=sys.stderr)
            continue
        weight = inp.get("weight", 1.0)
        with open(path, encoding="utf-8") as f:
            for line in f:
                sent = normalize(line)
                if sent:
                    out.append((sent, weight))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "scripts/configs/synth_corrupt.yaml"
    )
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rng = random.Random(cfg.get("seed", 42))
    corruptor = Corruptor(cfg, rng)

    sentences = load_sentences(cfg)
    if not sentences:
        sys.exit("no input sentences — run the download scripts first")
    rng.shuffle(sentences)

    min_words = cfg.get("min_sentence_words", 2)
    max_words = cfg.get("max_sentence_words", 24)
    max_pairs = cfg.get("max_pairs", 20000)

    records = []
    op_counts: dict[str, int] = {}
    seen: set[str] = set()
    skipped = {"length": 0, "not_ukrainian": 0, "no_ops": 0, "duplicate": 0}

    for sent, weight in sentences:
        if len(records) >= max_pairs:
            break
        if weight < 1.0 and rng.random() > weight:
            continue
        n_words = len(sent.split())
        if not (min_words <= n_words <= max_words):
            skipped["length"] += 1
            continue
        if not is_ukrainian(sent):
            skipped["not_ukrainian"] += 1
            continue
        result = corruptor.corrupt(sent)
        if result is None:
            skipped["no_ops"] += 1
            continue
        dirty, ops = result
        key = sha256_text(dirty.lower())
        if key in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(key)
        for op in ops:
            op_counts[op] = op_counts.get(op, 0) + 1
        records.append({"dirty": dirty, "clean": sent, "source": "synth", "license": "mixed"})

    out_path = REPO_ROOT / cfg["output"]
    n = write_jsonl(out_path, records)

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    meta = {
        "config": cfg,
        "git_commit": commit,
        "replacements_sha256": sha256_text(LEXICON_PATH.read_text(encoding="utf-8")),
        "pairs": n,
        "op_counts": op_counts,
        "skipped": skipped,
    }
    Path(str(out_path) + ".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"pairs": n, "op_counts": op_counts, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
