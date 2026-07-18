"""Fetch the current model + lexicon assets from the dormouse-ua==0.4.2 wheel.

The 0.4.2 wheel on PyPI bundles all runtime assets (expr_seq2seq.pt, vocabs,
config, lexicon.db); later wheels (0.4.3+) ship without them. This extracts
them into data/assets/ so the eval/train pipeline can run without access to
GitHub Releases or HuggingFace (both may be blocked in sandboxed environments).

Usage:
    python scripts/fetch_assets.py [--version 0.4.2] [--out data/assets]

Then point the pipeline at them:
    DORMOUSE_DATA_DIR=data/assets python scripts/eval_frozen.py ...
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_lib.io import ASSETS_DIR, manifest_append, sha256_file  # noqa: E402

ASSET_NAMES = [
    "expr_seq2seq.pt",
    "expr_config.json",
    "expr_vocab_src.json",
    "expr_vocab_tgt.json",
    "lexicon.db",
    "replacements.json",
]


def fetch(version: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                sys.executable, "-m", "pip", "download",
                f"dormouse-ua=={version}", "--no-deps", "-d", tmp, "-q",
            ],
            check=True,
        )
        wheel = next(Path(tmp).glob("*.whl"))
        extracted = []
        with zipfile.ZipFile(wheel) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if member.startswith("dormouse/data/") and name in ASSET_NAMES:
                    target = out_dir / name
                    target.write_bytes(zf.read(member))
                    extracted.append(target)
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="0.4.2")
    parser.add_argument("--out", type=Path, default=ASSETS_DIR)
    args = parser.parse_args()

    files = fetch(args.version, args.out)
    if not files:
        sys.exit("no assets found in wheel — check the version")
    for f in sorted(files):
        print(f"{f}  {f.stat().st_size / 1e6:.1f}MB  sha256={sha256_file(f)[:12]}…")
    manifest_append(
        source="dormouse-ua wheel (PyPI)", version=args.version, license_="MIT", files=files,
    )
    print(f"\n{len(files)} assets → {args.out}. Use: DORMOUSE_DATA_DIR={args.out}")


if __name__ == "__main__":
    main()
