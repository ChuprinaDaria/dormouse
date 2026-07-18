"""Evaluate a seq2seq checkpoint on the frozen eval set via the production path.

Uses wake_up_expr/translate_expression — the exact code the mapper calls — so
metrics measure deployed behavior, not a lab harness. Metrics:

- exact_match:      normalized prediction == target
- latin_ok_rate:    prediction passes mapper's acceptance gate
- none_rate:        model returned None (empty/all-UNK output)
- token_savings_pct: 1 - tokens(pred)/tokens(src), tiktoken cl100k_base
                     (same encoding optimizer.py uses); None-predictions
                     count as 0 savings
- per_source breakdown of exact_match

Usage:
    # baseline of the shipped model (assets extracted by fetch_assets.py):
    python scripts/eval_frozen.py --model-dir data/assets \
        --out data/eval/baseline_v0.4.2.json --model-label assets-v0.4.2
    # a freshly trained checkpoint:
    python scripts/eval_frozen.py --model-dir data/checkpoints/run_X --out ...
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_lib.compose import target_ok  # noqa: E402
from dataset_lib.io import EVAL_DIR, read_jsonl, sha256_file  # noqa: E402


def _token_count(enc, text: str) -> int:
    return len(enc.encode(text))


def evaluate(model_dir: Path, eval_path: Path) -> dict:
    import dormouse.seq2seq as seq2seq

    seq2seq._expr_cache = None  # інакше wake_up_expr поверне попередню модель
    model = seq2seq.wake_up_expr(model_dir=model_dir)
    if model is None:
        sys.exit(f"model failed to load from {model_dir}")

    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None
        print("warning: tiktoken unavailable, token_savings_pct will be null", file=sys.stderr)

    n = exact = latin_ok = none = 0
    src_tokens = pred_tokens = 0
    per_source: dict[str, dict[str, int]] = {}
    digits = {"n": 0, "exact": 0}  # окремий зріз: пари з цифрами в src

    for rec in read_jsonl(eval_path):
        n += 1
        src, tgt = rec["src"], rec["tgt"]
        stats = per_source.setdefault(rec.get("source", "unknown"), {"n": 0, "exact": 0})
        stats["n"] += 1
        has_digits = any(ch.isdigit() for ch in src)
        if has_digits:
            digits["n"] += 1

        pred = seq2seq.translate_expression(src, model_dir=model_dir)
        if pred is None:
            none += 1
        else:
            pred = pred.strip().lower()
            if pred == tgt.strip().lower():
                exact += 1
                stats["exact"] += 1
                if has_digits:
                    digits["exact"] += 1
            if target_ok(src, pred):
                latin_ok += 1
        if enc:
            s_tok = _token_count(enc, src)
            src_tokens += s_tok
            pred_tokens += _token_count(enc, pred) if pred else s_tok

    return {
        "n": n,
        "exact_match": round(exact / n, 4),
        "latin_ok_rate": round(latin_ok / n, 4),
        "none_rate": round(none / n, 4),
        "token_savings_pct": (
            round(100 * (1 - pred_tokens / src_tokens), 2) if enc and src_tokens else None
        ),
        "per_source": {
            src: {"n": s["n"], "exact_match": round(s["exact"] / s["n"], 4)}
            for src, s in sorted(per_source.items())
        },
        "has_digits": {
            "n": digits["n"],
            "exact_match": round(digits["exact"] / digits["n"], 4) if digits["n"] else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--eval", type=Path, default=EVAL_DIR / "frozen_v1.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-label", default=None)
    args = parser.parse_args()

    metrics = evaluate(args.model_dir, args.eval)
    result = {
        "model": {
            "label": args.model_label or str(args.model_dir),
            "checkpoint_sha256": sha256_file(args.model_dir / "expr_seq2seq.pt"),
        },
        "eval_set": {
            "path": str(args.eval),
            "sha256": sha256_file(args.eval),
            "n": metrics["n"],
        },
        "metrics": {k: v for k, v in metrics.items() if k != "n"},
        "created_at": date.today().isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
