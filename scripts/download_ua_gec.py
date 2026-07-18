"""Download UA-GEC 2.x via the ua-gec pip package (corpus data is bundled).

Writes document-level (dirty, clean) records per partition:
    data/raw/ua_gec/train.jsonl, data/raw/ua_gec/test.jsonl
and clean sentences for synthetic corruption:
    data/raw/clean_sentences/ua_gec_clean.txt

License: CC BY 4.0 (https://github.com/grammarly/ua-gec).

Usage:
    pip install ua-gec
    python scripts/download_ua_gec.py [--layer gec-fluency]
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_lib.io import RAW_DIR, manifest_append, write_jsonl  # noqa: E402

_RE_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯІЇЄҐA-Z«\"])")


def split_sentences(text: str) -> list[str]:
    sentences = []
    for line in text.splitlines():
        sentences.extend(s.strip() for s in _RE_SENT_SPLIT.split(line) if s.strip())
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layer", default="gec-fluency", choices=["gec-fluency", "gec-only"],
        help="annotation layer; gec-fluency includes fluency edits (recommended)",
    )
    args = parser.parse_args()

    from ua_gec import Corpus

    out_dir = RAW_DIR / "ua_gec"
    written = []
    clean_sentences: list[str] = []
    for partition in ("train", "test"):
        corpus = Corpus(partition=partition, annotation_layer=args.layer)
        records = []
        for doc in corpus:
            records.append(
                {
                    "dirty": doc.source,
                    "clean": doc.target,
                    "doc_id": doc.doc_id,
                    "partition": partition,
                }
            )
            if partition == "train":
                clean_sentences.extend(split_sentences(doc.target))
        path = out_dir / f"{partition}.jsonl"
        n = write_jsonl(path, records)
        written.append(path)
        print(f"{path}: {n} documents")

    sent_path = RAW_DIR / "clean_sentences" / "ua_gec_clean.txt"
    sent_path.parent.mkdir(parents=True, exist_ok=True)
    sent_path.write_text("\n".join(clean_sentences) + "\n", encoding="utf-8")
    written.append(sent_path)
    print(f"{sent_path}: {len(clean_sentences)} sentences")

    manifest_append(
        source=f"UA-GEC ({args.layer}, ua-gec PyPI package)",
        version=pkg_version("ua-gec"),
        license_="CC BY 4.0",
        files=written,
    )


if __name__ == "__main__":
    main()
