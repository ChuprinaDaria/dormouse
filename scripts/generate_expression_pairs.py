"""Regenerate expression training pairs from lexicon.db (Layer B directly).

Replacement for the original private-repo 28K pairs, which are not available
in the public repo. Every lexicon row with an en_compressed value becomes a
(src=UA expression, tgt=EN) pair; optionally adds surzhyk/slang variants of
the UA side via rule inversion, and inflected unigram forms via pymorphy3.

Output: data/train/lexicon.jsonl {"src","tgt","clean","source","license"}

Usage:
    DORMOUSE_DATA_DIR=data/assets python scripts/generate_expression_pairs.py \
        [--dirty-variant-share 0.3] [--expand-unigrams --max-forms 4] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_lib.compose import target_ok  # noqa: E402
from dataset_lib.corrupt import build_inverse_rules, dirty_variant  # noqa: E402
from dataset_lib.io import TRAIN_DIR, load_frozen_hashes, sha256_text, write_jsonl  # noqa: E402

from dormouse.lexicon_db import get_lexicon  # noqa: E402
from dormouse.morphology import get_morph  # noqa: E402


def inflected_forms(word: str, max_forms: int) -> list[str]:
    """Словоформи уніграми через pymorphy3 lexeme (без леми-дубліката)."""
    morph = get_morph()
    if morph is None:
        return []
    parse = morph.parse(word)[0]
    if parse.tag.POS not in ("NOUN", "VERB", "INFN", "ADJF"):
        return []
    forms = []
    for lex in parse.lexeme:
        form = lex.word.lower()
        if form != word and form not in forms:
            forms.append(form)
        if len(forms) >= max_forms:
            break
    return forms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirty-variant-share", type=float, default=0.3,
                        help="частка виразів, для яких додається брудний варіант")
    parser.add_argument("--expand-unigrams", action="store_true",
                        help="додати словоформи уніграм через pymorphy3")
    parser.add_argument("--max-forms", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    conn = get_lexicon()
    inverse = build_inverse_rules()
    frozen = load_frozen_hashes()

    rows = conn.execute(
        "SELECT word, en_compressed, ngram FROM lexicon "
        "WHERE en_compressed IS NOT NULL AND en_compressed != '' ORDER BY word"
    ).fetchall()

    records = []
    seen: set[str] = set()
    stats = {"base": 0, "dirty_variants": 0, "inflected": 0, "gate_rejected": 0, "frozen": 0}

    def add(src: str, tgt: str, kind: str) -> None:
        src = src.lower().strip()
        tgt = tgt.lower().strip()
        if src in seen:
            return
        if sha256_text(src) in frozen:
            stats["frozen"] += 1
            return
        if not target_ok(src, tgt):
            stats["gate_rejected"] += 1
            return
        seen.add(src)
        records.append(
            {"src": src, "tgt": tgt, "clean": src, "source": "lexicon", "license": "MIT"}
        )
        stats[kind] += 1

    for row in rows:
        word, en = row["word"], row["en_compressed"]
        add(word, en, "base")

        if rng.random() < args.dirty_variant_share:
            variant = dirty_variant(word, inverse, rng)
            if variant:
                add(variant, en, "dirty_variants")

        if args.expand_unigrams and row["ngram"] == 1 and " " not in word:
            for form in inflected_forms(word, args.max_forms):
                add(form, en, "inflected")

    n = write_jsonl(TRAIN_DIR / "lexicon.jsonl", records)
    stats["total"] = n
    (TRAIN_DIR / "report_lexicon.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
