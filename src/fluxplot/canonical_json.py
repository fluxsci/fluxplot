"""Canonical JSON serialization — part of the deterministic surface (P5).

Sorted keys + fixed float precision so the same inputs always produce byte-identical bytes (clean
git diffs, stable hashing). Floats are rounded to ``FLOAT_PRECISION`` decimals to neutralize
sub-ulp jitter from matplotlib's transforms across platforms.
"""
from __future__ import annotations

import json
import math

FLOAT_PRECISION = 4


def _round(obj):
    """Recursively round floats so serialization is platform-stable."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return obj
        r = round(obj, FLOAT_PRECISION)
        # normalize -0.0 → 0.0 for stable output
        return 0.0 if r == 0.0 else r
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v) for v in obj]
    return obj


def dumps(obj) -> str:
    """Serialize to canonical JSON text (sorted keys, rounded floats, trailing newline)."""
    return (
        json.dumps(
            _round(obj),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        )
        + "\n"
    )
