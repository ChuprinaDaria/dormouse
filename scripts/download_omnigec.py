"""Download the Ukrainian subset of OmniGEC from the Hugging Face Hub.

OmniGEC (UNLP 2025, arXiv:2509.14504) is a silver multilingual GEC collection;
the Ukrainian-relevant parts are UberText-GEC (Telegram texts auto-corrected
with GPT-4o-mini) plus the Ukrainian slices of Reddit-MultiGEC and
WikiEdits-MultiGEC, published under the lang-uk organization.

Needs network access to huggingface.co — run locally, not in a sandboxed
session. The exact dataset ids are resolved at runtime against the hub (search
for "UberText-GEC" / "MultiGEC" under lang-uk) rather than hard-coded, per the
"don't guess hub paths" rule; the resolved id + revision land in MANIFEST.md.

Writes: data/raw/omnigec/{dataset}.jsonl with {"dirty","clean","subset"}.

Usage:
    pip install datasets huggingface_hub
    python scripts/download_omnigec.py [--max-records 200000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_lib.io import RAW_DIR, manifest_append, write_jsonl  # noqa: E402

CANDIDATE_QUERIES = ["UberText-GEC", "Reddit-MultiGEC", "WikiEdits-MultiGEC"]

# Column-name candidates across OmniGEC configs (source/corrected naming varies)
SRC_KEYS = ("source", "src", "original", "text", "sentence")
TGT_KEYS = ("target", "tgt", "corrected", "correction", "corrected_text")


def _pick(row: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in row and isinstance(row[k], str):
            return row[k]
    return None


def resolve_datasets() -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    ids = []
    for query in CANDIDATE_QUERIES:
        hits = api.list_datasets(search=query, author="lang-uk", limit=5)
        for hit in hits:
            ids.append(hit.id)
    if not ids:
        sys.exit("no OmniGEC datasets found under lang-uk — check hub access")
    return sorted(set(ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-records", type=int, default=200_000, help="cap per dataset")
    parser.add_argument("--ids", nargs="*", help="override resolved dataset ids")
    args = parser.parse_args()

    from datasets import load_dataset

    dataset_ids = args.ids or resolve_datasets()
    print(f"resolved datasets: {dataset_ids}")

    out_dir = RAW_DIR / "omnigec"
    for ds_id in dataset_ids:
        ds = load_dataset(ds_id, split="train")
        # Ukrainian-only slice where a language column exists (MultiGEC sets)
        lang_col = next((c for c in ("language", "lang") if c in ds.column_names), None)
        if lang_col:
            ds = ds.filter(lambda row: str(row[lang_col]).lower().startswith("uk"))
        records = []
        for row in ds:
            dirty, clean = _pick(row, SRC_KEYS), _pick(row, TGT_KEYS)
            if dirty and clean:
                records.append({"dirty": dirty, "clean": clean, "subset": ds_id})
            if len(records) >= args.max_records:
                break
        path = out_dir / f"{ds_id.split('/')[-1].lower()}.jsonl"
        n = write_jsonl(path, records)
        print(f"{path}: {n} records")
        manifest_append(
            source=f"OmniGEC / {ds_id} (HF)", version="main", license_="see dataset card",
            files=[path],
        )


if __name__ == "__main__":
    main()
