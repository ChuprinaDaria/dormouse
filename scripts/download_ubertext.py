"""Download a capped sample of the UberText 2.0 social-media subcorpus.

Source: https://lang.org.ua/en/ubertext/ (UberText 2.0, lang-uk). Only the
social subcorpus is fetched, and only a capped sample is kept — it feeds
synthetic corruption and lexicon-coverage analysis, not paired training data.

Needs network access to lang.org.ua — run locally, not in a sandboxed session.
The download URL must be taken from the UberText page (it requires accepting
the terms); pass it via --url.

Writes:
    data/raw/clean_sentences/ubertext_social.txt  (one sentence per line, capped)

Usage:
    python scripts/download_ubertext.py --url <social subcorpus .txt.bz2 url> \
        [--max-sentences 200000]
"""

from __future__ import annotations

import argparse
import bz2
import lzma
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_lib.io import RAW_DIR, manifest_append  # noqa: E402
from dataset_lib.normalize import is_ukrainian, normalize  # noqa: E402


def _open_stream(path: Path):
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if path.suffix in (".xz", ".lzma"):
        return lzma.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="social subcorpus archive URL")
    parser.add_argument("--max-sentences", type=int, default=200_000)
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-words", type=int, default=30)
    args = parser.parse_args()

    archive = RAW_DIR / "ubertext" / Path(args.url).name
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print(f"downloading {args.url} …")
        urllib.request.urlretrieve(args.url, archive)

    out = RAW_DIR / "clean_sentences" / "ubertext_social.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with _open_stream(archive) as src, open(out, "w", encoding="utf-8") as dst:
        for line in src:
            sent = normalize(line)
            words = sent.split()
            if not (args.min_words <= len(words) <= args.max_words):
                continue
            if not is_ukrainian(sent):
                continue
            dst.write(sent + "\n")
            kept += 1
            if kept >= args.max_sentences:
                break
    print(f"{out}: {kept} sentences")

    manifest_append(
        source="UberText 2.0 social subcorpus (lang.org.ua, capped sample)",
        version=Path(args.url).name, license_="CC BY-NC 4.0 (research)", files=[out],
    )


if __name__ == "__main__":
    main()
