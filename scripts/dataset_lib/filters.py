"""Pair filters for the unification step. Every drop returns a reason for the report."""

from __future__ import annotations

from dataset_lib.normalize import is_ukrainian

MAX_LEN = 512


def passes_filters(
    dirty: str,
    clean: str,
    seen_hashes: set[str],
    dirty_hash: str,
    frozen_hashes: set[str] | None = None,
) -> tuple[bool, str]:
    """Перевіряє пару після нормалізації. (True, "") або (False, причина)."""
    if not dirty or not clean:
        return False, "empty"
    if dirty == clean:
        return False, "identical"
    if len(dirty) > MAX_LEN or len(clean) > MAX_LEN:
        return False, "too_long"
    if not is_ukrainian(dirty):
        return False, "not_ukrainian"
    if dirty_hash in seen_hashes:
        return False, "duplicate"
    if frozen_hashes and dirty_hash in frozen_hashes:
        return False, "frozen"
    return True, ""
