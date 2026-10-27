"""The quick subset: a small, fixed part of a build for checking a system before the full run.

The 30 s clips fall into strata: one per capture condition for the clips where the target
speaks, and one for the target-absent clips. The quick subset keeps every ``QUICK_EVERY``-th
clip of each stratum in manifest order, starting with the first, so every capture condition
and the target-absent case are in it. Long files are left out: a few of them would hold a large
share of the subset's words and swing every number. ``runners/common.py`` holds the same rule,
because the runners run without this package.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

QUICK_EVERY = 8


def stratum(row: Mapping) -> str:
    """The stratum of a manifest row: ``long``, ``target_absent`` or ``clip:<capture>``."""
    if row.get("long"):
        return "long"
    if row.get("target_absent"):
        return "target_absent"
    return f"clip:{(row.get('buckets') or {}).get('capture')}"


def quick_ids(rows: Iterable[Mapping], every: int = QUICK_EVERY) -> list[str]:
    """The item ids of the quick subset, in manifest order. Long files are never in it."""
    if every < 1:
        raise ValueError("every must be at least 1")
    seen: dict[str, int] = {}
    ids: list[str] = []
    for row in rows:
        key = stratum(row)
        if key == "long":
            continue
        k = seen.get(key, 0)
        seen[key] = k + 1
        if k % every == 0:
            ids.append(str(row["id"]))
    return ids


__all__ = ["QUICK_EVERY", "quick_ids", "stratum"]
