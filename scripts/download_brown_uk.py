"""Download Brown-UK (БрУК) — balanced corpus of clean modern Ukrainian.

Source: https://github.com/brown-uk/corpus (CC BY-NC-SA 4.0 for texts).
Only the "good" fragments (verified, standard Ukrainian) are used as clean
sentences for synthetic corruption.

NOTE (license): Brown-UK texts are CC BY-NC-SA — fine as *training input* for
an internal model, but do not redistribute the sentences themselves in the
public repo or the published dataset without checking the terms.

Needs network access to github.com — run locally, not in a sandboxed session.

Writes: data/raw/clean_sentences/brown_uk.txt (one sentence per line).

Usage:
    python scripts/download_brown_uk.py [--repo-dir /tmp/brown-uk]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_lib.io import RAW_DIR, manifest_append  # noqa: E402

REPO_URL = "https://github.com/brown-uk/corpus.git"
_RE_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯІЇЄҐ«\"])")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path("/tmp/brown-uk-corpus"))
    args = parser.parse_args()

    if not args.repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(args.repo_dir)], check=True
        )

    # "good" fragments live in data/good/*.txt (plain text alongside .xml markup)
    good_dir = args.repo_dir / "data" / "good"
    txt_files = sorted(good_dir.glob("*.txt"))
    if not txt_files:
        sys.exit(f"no .txt files under {good_dir} — repo layout changed?")

    sentences: list[str] = []
    for path in txt_files:
        text = path.read_text(encoding="utf-8")
        for sent in _RE_SENT_SPLIT.split(text):
            sent = " ".join(sent.split())
            if sent:
                sentences.append(sent)

    out = RAW_DIR / "clean_sentences" / "brown_uk.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    print(f"{out}: {len(sentences)} sentences from {len(txt_files)} files")

    rev = subprocess.run(
        ["git", "-C", str(args.repo_dir), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    manifest_append(
        source="Brown-UK (github.com/brown-uk/corpus, good fragments)",
        version=rev, license_="CC BY-NC-SA 4.0", files=[out],
    )


if __name__ == "__main__":
    main()
