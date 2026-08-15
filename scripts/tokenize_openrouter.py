#!/usr/bin/env python3
"""Measure real prompt-token cost of Ukrainian vs dormouse-English input.

Unlike scripts/tokenize_benchmark.py (local tokenisers, approximate for
Anthropic/Google), this asks every provider directly through OpenRouter and
reads the token count they actually billed. One request per (model, phrase,
language); the completion is discarded, only `usage.prompt_tokens` matters.

    python scripts/tokenize_openrouter.py [--limit N] [--out PREFIX]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

API = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    ("OpenAI GPT-5.5", "openai/gpt-5.5"),
    ("OpenAI GPT-4.1", "openai/gpt-4.1"),
    ("OpenAI GPT-4", "openai/gpt-4"),
    ("Anthropic Claude Opus 5", "anthropic/claude-opus-5"),
    ("Anthropic Claude Sonnet 5", "anthropic/claude-sonnet-5"),
    ("Anthropic Claude Opus 4.8", "anthropic/claude-opus-4.8"),
    ("Google Gemini 3.7 Flash", "google/gemini-3.7-flash"),
    ("Google Gemini 3.1 Pro", "google/gemini-3.1-pro-preview"),
]

# Real customer-intent messages (shop / support chats), plus the general
# prompts already used in the v0.6 pipeline demo.
EXTRA_UK = [
    "Доброго дня, а скільки коштує доставка в Україну і за скільки днів прийде?",
    "хочу замовити два браслети, знижка якась є на два?",
    "а можна оплатити при отриманні? бо карткою не хочу",
    "замовляла ще тиждень тому, де моє замовлення, номер 1042",
    "а якщо розмір не підійде, можна поміняти або повернути?",
    "скажіть будь ласка цей перстень є в сріблі чи тільки латунь?",
    "шо по гарантії якщо застібка зламається?",
    "можете зробити на замовлення з гравіюванням імені?",
    "скиньте реквізити куди платити і я сьогодні оплачу",
    "чи є у вас щось до 200 злотих в подарунок дівчині",
]

_print_lock = threading.Lock()


def load_phrases(limit: int | None) -> list[str]:
    out: list[str] = []
    demo = ROOT / "data/exports/agent_pipeline_demo_v06.jsonl"
    if demo.exists():
        with demo.open(encoding="utf-8") as fh:
            for line in fh:
                uk = json.loads(line).get("user_uk", "").strip()
                if uk:
                    out.append(uk)
    out.extend(EXTRA_UK)
    return out[:limit] if limit else out


def prompt_tokens(model: str, text: str, key: str) -> int | None:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": text}],
        # Gemini omits `usage` entirely when it stops at max_tokens=1, so ask
        # for a few tokens we then throw away.
        "max_tokens": 16,
        "usage": {"include": True},
    }).encode()
    req = urllib.request.Request(
        API, data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r).get("usage", {}).get("prompt_tokens")
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503) and attempt < 2:
                continue
            with _print_lock:
                print(f"  ! {model}: HTTP {e.code} {e.read()[:120]!r}", file=sys.stderr)
            return None
        except Exception as e:  # noqa: BLE001
            if attempt < 2:
                continue
            with _print_lock:
                print(f"  ! {model}: {e}", file=sys.stderr)
            return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "data/exports/tokenize_openrouter"))
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set")

    from dormouse.mt_translator import get_translator
    tr = get_translator("uk-en")
    if tr is None:
        sys.exit("dormouse uk-en translator unavailable")

    uk_phrases = load_phrases(args.limit)
    print(f"[translating {len(uk_phrases)} phrases with dormouse v0.7 uk-en]")
    pairs = [(uk, tr.translate(uk) or "") for uk in uk_phrases]
    for uk, en in pairs:
        print(f"  UK: {uk}\n  EN: {en}")

    jobs = [
        (label, mid, idx, lang, text)
        for label, mid in MODELS
        for idx, (uk, en) in enumerate(pairs)
        for lang, text in (("uk", uk), ("en", en))
    ]
    print(f"\n[querying {len(jobs)} prompts across {len(MODELS)} models]")

    results: dict[tuple[str, int, str], int] = {}

    def run(job):
        label, mid, idx, lang, text = job
        n = prompt_tokens(mid, text, key)
        if n is not None:
            results[(label, idx, lang)] = n

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, jobs))

    # Every request carries a fixed chat-template overhead (role markers etc).
    # It inflates both columns equally and therefore *understates* the saving,
    # so measure it once per model with a single-token prompt and subtract.
    print("\n[measuring chat-template overhead]")
    overhead = {}
    for label, mid in MODELS:
        n = prompt_tokens(mid, "x", key)
        overhead[label] = max((n or 1) - 1, 0)
        print(f"  {label}: {overhead[label]}")

    rows = []
    for label, _mid in MODELS:
        n = len(pairs)
        off = overhead[label] * n
        uk_tot = sum(results.get((label, i, "uk"), 0) for i in range(n)) - off
        en_tot = sum(results.get((label, i, "en"), 0) for i in range(n)) - off
        missing = sum(
            1 for i in range(n)
            if (label, i, "uk") not in results or (label, i, "en") not in results
        )
        saved = (uk_tot - en_tot) / uk_tot * 100 if uk_tot else 0.0
        rows.append((label, uk_tot, en_tot, saved, missing))

    width = max(len(r[0]) for r in rows)
    lines = [
        f"| {'model':<{width}} | UK in | EN in | saved |",
        f"|{'-' * (width + 2)}|------:|------:|------:|",
    ]
    for label, uk_tot, en_tot, saved, missing in rows:
        flag = f"  ({missing} failed)" if missing else ""
        lines.append(f"| {label:<{width}} | {uk_tot:5d} | {en_tot:5d} | {saved:4.1f}% |{flag}")
    table = "\n".join(lines)
    print("\n" + table)

    out = Path(args.out)
    out.with_suffix(".txt").write_text(
        f"dormouse v0.7 — real prompt tokens via OpenRouter\n"
        f"{len(pairs)} Ukrainian phrases, each sent raw (UK) and dormouse-translated (EN).\n\n"
        + table + "\n", encoding="utf-8",
    )
    out.with_suffix(".jsonl").write_text(
        "\n".join(
            json.dumps({
                "model": label, "phrase_idx": i,
                "uk": pairs[i][0], "en": pairs[i][1],
                "uk_tokens": results.get((label, i, "uk")),
                "en_tokens": results.get((label, i, "en")),
            }, ensure_ascii=False)
            for label, _ in MODELS for i in range(len(pairs))
        ) + "\n", encoding="utf-8",
    )
    print(f"\n[written {out.with_suffix('.txt')} and .jsonl]")


if __name__ == "__main__":
    main()
