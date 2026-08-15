#!/usr/bin/env python3
"""Count UK-input vs EN-translated tokens across every family of models the
user is likely to route through — so we can honestly show which cloud model
benefits the most from the dormouse translator, not just quote "% saved" once.

Tokenizers used:
- OpenAI: tiktoken (exact for OpenAI models)
- Anthropic: Claude token API is subscription-only here; we fall back to
  cl100k_base as a rough proxy and label it as approximate.
- Google Gemini: same story; we use gemma-2-9b-it tokenizer via HF as a
  close-enough proxy (Google's SentencePiece family).
- Qwen / Mistral / Llama / Gemma: HF tokenizers pulled via the Rust
  `tokenizers` package (no torch, no SIGILL on Phenom II).

Uses the same 20 real UK phrases + their ft_v06 English translations that
we already produced in the pipeline demo — no new inference needed.

    python scripts/tokenize_benchmark.py
"""
from __future__ import annotations

import json
from pathlib import Path

import tiktoken
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_JSONL = ROOT / "data/exports/agent_pipeline_demo_v06.jsonl"
OUT_JSONL = ROOT / "data/exports/tokenize_benchmark.jsonl"
OUT_TXT = ROOT / "data/exports/tokenize_benchmark.txt"

# (family label, encoding source, exact/approx). Order = presentation order.
TIKTOKEN_ENCS = [
    # (display_label, encoding_name, exact_for)
    ("OpenAI GPT-5.6 / 5.5 / 4.1 / 4o / o1", "o200k_base", "exact"),
    ("OpenAI GPT-4 / 3.5",                    "cl100k_base", "exact"),
    ("OpenAI GPT-3 davinci (legacy)",         "p50k_base",   "exact"),
]

# HF tokenizers — pulled via Rust `tokenizers`, no torch. All public.
HF_TOKENIZERS = [
    ("Anthropic Claude Opus 5 / Opus 4.8 / Sonnet 5", "cl100k_base_proxy", "approx"),
    ("Qwen 2.5 (7B/72B, incl. code)",                  "Qwen/Qwen2.5-7B-Instruct", "exact"),
    ("Qwen 3 (7B/32B/235B)",                           "Qwen/Qwen3-8B", "exact"),
    ("Mistral 7B v0.3",                                 "mistralai/Mistral-7B-Instruct-v0.3", "exact"),
    ("Mistral Nemo / Ministral",                        "mistralai/Mistral-Nemo-Instruct-2407", "exact"),
    ("Google Gemma 2 9B / Gemini approx",               "unsloth/gemma-2-9b-it", "approx"),
    ("Meta Llama 3.1 8B / 70B",                         "NousResearch/Meta-Llama-3.1-8B-Instruct", "exact"),
    ("Meta Llama 3 (legacy)",                           "NousResearch/Meta-Llama-3-8B", "exact"),
]


def load_pairs() -> list[dict]:
    return [json.loads(l) for l in PIPELINE_JSONL.read_text().splitlines() if l.strip()]


def build_counters() -> dict[str, tuple[str, callable, str]]:
    """Return {label: (family, encode_fn, exact_or_approx)}."""
    counters: dict[str, tuple[str, callable, str]] = {}
    for label, enc_name, kind in TIKTOKEN_ENCS:
        enc = tiktoken.get_encoding(enc_name)
        counters[label] = (enc_name, lambda t, e=enc: len(e.encode(t)), kind)

    for label, model_id, kind in HF_TOKENIZERS:
        if model_id == "cl100k_base_proxy":
            enc = tiktoken.get_encoding("cl100k_base")
            counters[label] = ("cl100k_base (proxy)", lambda t, e=enc: len(e.encode(t)), kind)
        else:
            print(f"  loading {model_id}...", flush=True)
            tok = Tokenizer.from_pretrained(model_id)
            counters[label] = (model_id, lambda t, tk=tok: len(tk.encode(t).ids), kind)
    return counters


def main() -> None:
    pairs = load_pairs()
    print(f"loaded {len(pairs)} UK/EN pairs from pipeline demo", flush=True)

    counters = build_counters()
    print(f"loaded {len(counters)} tokenizers\n", flush=True)

    # per-model total across all 12 messages (user-input side only)
    totals: dict[str, dict[str, int]] = {
        label: {"uk": 0, "en": 0, "saved_pct": 0.0, "source": src, "kind": kind}
        for label, (src, _, kind) in counters.items()
    }

    per_row: list[dict] = []
    for r in pairs:
        uk, en = r["user_uk"], r["prompt_en"]
        entry = {"uk": uk, "en": en, "tokens": {}}
        for label, (_src, enc, _kind) in counters.items():
            uk_n, en_n = enc(uk), enc(en)
            entry["tokens"][label] = {"uk": uk_n, "en": en_n}
            totals[label]["uk"] += uk_n
            totals[label]["en"] += en_n
        per_row.append(entry)

    OUT_JSONL.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in per_row))

    # sort by absolute savings (uk - en), then by relative
    def saved(label: str) -> float:
        t = totals[label]
        return (1 - t["en"] / t["uk"]) * 100 if t["uk"] else 0.0

    sorted_labels = sorted(counters.keys(), key=saved, reverse=True)

    lines = [
        f"# Tokenization benchmark — dormouse ft_v06 uk→en",
        f"# {len(pairs)} real UK user prompts vs their English translations",
        "",
        f"{'model':52} {'src':>4} → {'tgt':>4}  {'saved':>7}  {'kind':>7}",
        "-" * 88,
    ]
    for label in sorted_labels:
        t = totals[label]
        s = saved(label)
        lines.append(
            f"{label:52} {t['uk']:>4} → {t['en']:>4}  {s:>6.1f}%  {t['kind']:>7}"
        )

    OUT_TXT.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {OUT_TXT}")
    print(f"      {OUT_JSONL}")


if __name__ == "__main__":
    main()
