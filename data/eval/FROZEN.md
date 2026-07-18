# Frozen eval set v1

**Immutable.** Never edit `frozen_v1.jsonl`; a future set is `frozen_v2.jsonl`.
Every train-writing script excludes pairs whose sha256(lower(src)) appears in
`frozen_v1.hashes.txt`.

- created: 2026-07-18
- generating commit: 834049b
- seed: 20260718
- pairs: 1499
- sha256(frozen_v1.jsonl): `0853690a2c269b1d94b24b0f08171ab26107b454ba331f8020d084ac33ea44cb`
- per-source counts: {"lexicon": 1101, "ua_gec": 398}

Sources and licenses: see `data/raw/MANIFEST.md`. UA-GEC-derived pairs are
CC BY 4.0 (Grammarly UA-GEC corpus); lexicon-derived pairs are MIT (dormouse).

TODO (frozen_v2): ~200 hand-written pairs from the shops/LMS/support domains.
