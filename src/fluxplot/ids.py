"""Deterministic semantic IDs — the public, long-lived join key (spec §7, open Q6).

An id is a dotted path of `[a-z0-9-]` segments; the dot is the only separator and the index is
0-based:

    control.line   control.point.3   axis.x.title   axis.x.tick.2   legend   annotation.peak

matplotlib's own ids use ``_`` + hex hashes (``line2d_7``, ``p3dcf139bf4``) and never contain dots,
so this namespace is provably disjoint from anything matplotlib autogenerates.
"""
from __future__ import annotations

import re

# Structural roots that stand alone at the head of an id (not series-scoped).
STRUCTURAL_ROOTS = frozenset(
    {
        "figure",
        "panel",
        "plot-area",
        "axis",
        "legend",
        "colorbar",
        "title",
        "background",
        "annotation",
        "reference-line",
        "highlight-region",
        "significance-bracket",
        "label",
    }
)

_SLUG_DROP = re.compile(r"[^a-z0-9-]+")
_SLUG_DASHES = re.compile(r"-+")
_SEGMENT_RE = re.compile(r"^[a-z0-9-]+$")


def slugify(name: object) -> str:
    """Turn an arbitrary series/category name into a stable id segment.

    Deterministic: lowercase → spaces/underscores to ``-`` → drop other non-``[a-z0-9-]`` →
    collapse repeated ``-`` → strip leading/trailing ``-``. Empty result falls back to ``series``.
    """
    s = str(name).strip().lower().replace("_", "-").replace(" ", "-")
    s = _SLUG_DROP.sub("-", s)
    s = _SLUG_DASHES.sub("-", s).strip("-")
    return s or "series"


def is_valid_segment(seg: str) -> bool:
    return bool(_SEGMENT_RE.match(seg))


def join(*segments: object) -> str:
    """Compose an id from segments, validating each is a legal segment."""
    parts = []
    for seg in segments:
        if seg is None:
            continue
        seg = str(seg)
        if not is_valid_segment(seg):
            raise ValueError(f"invalid id segment: {seg!r}")
        parts.append(seg)
    if not parts:
        raise ValueError("an id needs at least one segment")
    return ".".join(parts)


def series_root(series: object) -> str:
    """Slug for a series, disambiguated from structural roots.

    A series literally named "legend" becomes ``legend-series`` so structural roots stay unambiguous.
    """
    slug = slugify(series)
    return f"{slug}-series" if slug in STRUCTURAL_ROOTS else slug


def series_id(series: object, role: str | None = None, index: int | None = None) -> str:
    """``control`` → ``control``; +role → ``control.line``; +index → ``control.point.3``."""
    segs: list[object] = [series_root(series)]
    if role is not None:
        segs.append(role)
    if index is not None:
        segs.append(int(index))
    return join(*segs)


def axis_id(which: str, part: str | None = None, index: int | None = None) -> str:
    """``axis.x``, ``axis.x.title``, ``axis.x.tick.2`` (which ∈ x, y, x2, y2)."""
    segs: list[object] = ["axis", which]
    if part is not None:
        segs.append(part)
    if index is not None:
        segs.append(int(index))
    return join(*segs)


class IdAllocator:
    """Per-plot id allocation with deterministic collision handling.

    Same inputs → same ids. If a candidate id is already taken, later ones get a deterministic
    ``-2``, ``-3``, … suffix on the final segment (by insertion order).
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def take(self, candidate: str) -> str:
        if candidate not in self._seen:
            self._seen.add(candidate)
            return candidate
        head, _, tail = candidate.rpartition(".")
        i = 2
        while True:
            suffixed = f"{tail}-{i}"
            new = f"{head}.{suffixed}" if head else suffixed
            if new not in self._seen:
                self._seen.add(new)
                return new
            i += 1
