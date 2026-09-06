"""Save-time promotion of labeled ordinary matplotlib artists (plan §3).

A conventional labeled plot — ``ax.plot(t, y, label="Control")`` plus only ``fp.save`` — should
yield a named series with exact data, not an anonymous ``extra.*``. This module promotes raw
artists to series marks under strict rules:

- **Identity comes only from a public artist label** the user authored (non-empty, not
  ``_``-private). Never from color, draw order, geometry shape, linestyle or legend position.
- **Data comes only from the artist's own exact public state** (``get_data()``,
  ``get_offsets()``, bar patch geometry). Nothing is reconstructed or guessed.
- **Ambiguity declines**: duplicate labels, or a label colliding with an explicitly tagged
  series, leave every candidate in the honest ``extra.*`` fallback with one actionable warning.
- **Explicit helpers/tags always win** — an artist already in the registry is never touched.

Runs inside :func:`fluxplot.save` before gid assignment; the orphan sweep in ``tagger.py``
still rescues whatever this module (deliberately) does not claim.
"""
from __future__ import annotations

import warnings as _warnings

import numpy as np

from . import ids as _ids
from .descriptors import Mark

#: Additive manifest provenance for auto-promoted series: how identity and data were captured.
AUTO_CAPTURE = {"identity": "artist-label", "data": "artist"}


def is_colorbar_axes(ax) -> bool:
    """True if this Axes is a colorbar (added by fig.colorbar), not a plot area."""
    return getattr(ax, "_colorbar", None) is not None or ax.get_label() == "<colorbar>"


def _public_label(artist) -> str | None:
    """The artist's label iff it is public identity (non-empty, not ``_child0``/``_nolegend_``)."""
    lab = artist.get_label()
    if not isinstance(lab, str):
        return None
    if not lab.strip() or lab.startswith("_"):
        return None
    return lab


def _finite_offsets(coll):
    """Exact Nx2 finite offsets, or None (masked/non-finite would shift per-point indices)."""
    off = coll.get_offsets()
    if np.ma.isMaskedArray(off) and np.ma.is_masked(off):
        return None
    arr = np.asarray(off, dtype=float)
    if arr.ndim != 2 or arr.shape[-1] != 2 or arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr


def extract_xy(artist):
    from .data import artist_xy
    return artist_xy(artist)


def promote_labeled(fig, reg) -> list[str]:
    """Promote safe labeled raw artists on ``fig`` into ``reg``. Returns warning strings."""
    from matplotlib.collections import PathCollection, PolyCollection
    from matplotlib.container import BarContainer
    from matplotlib.lines import Line2D

    notes: list[str] = []
    already = {id(a) for m in reg.marks for a in m.artists}

    # candidates in deterministic order: per axes → lines, collections, bar containers
    candidates: list[tuple[str, str, object]] = []  # (label, adapter, artist/container)
    for ax in fig.axes:
        if is_colorbar_axes(ax):
            continue
        for ln in ax.lines:
            if id(ln) in already or not isinstance(ln, Line2D):
                continue
            lab = _public_label(ln)
            if lab is not None:
                candidates.append((lab, "line", ln))
        for coll in ax.collections:
            if id(coll) in already:
                continue
            lab = _public_label(coll)
            if lab is None:
                continue
            if isinstance(coll, PolyCollection):
                candidates.append((lab, "area", coll))
            elif isinstance(coll, PathCollection):
                candidates.append((lab, "point", coll))
        for cont in getattr(ax, "containers", []):
            if not isinstance(cont, BarContainer) or not cont.patches:
                continue
            if any(id(p) in already for p in cont.patches):
                continue
            lab = _public_label(cont)
            if lab is not None:
                candidates.append((lab, "bar", cont))

    # ambiguity: duplicate candidate labels, or a label whose slug collides with an
    # explicitly tagged series → decline ALL involved candidates (extra.* keeps them honest)
    explicit_slugs = {_ids.series_root(m.series) for m in reg.marks if m.series is not None}
    slug_counts: dict[str, int] = {}
    for lab, _, _ in candidates:
        slug = _ids.series_root(lab)
        slug_counts[slug] = slug_counts.get(slug, 0) + 1
    dropped = sorted(
        {
            lab
            for lab, _, _ in candidates
            if slug_counts[_ids.series_root(lab)] > 1 or _ids.series_root(lab) in explicit_slugs
        }
    )
    if dropped:
        msg = (
            f"fluxplot: not auto-promoting artist label(s) {dropped}: each duplicates another "
            "artist's label or an explicitly tagged series, so identity is ambiguous. The "
            "artists stay addressable as extra.*; give them distinct labels, or use the fp.* "
            "helpers / fp.tag(series=...) for explicit identity."
        )
        _warnings.warn(msg, UserWarning, stacklevel=3)
        notes.append(msg)
    skip = {_ids.series_root(lab) for lab in dropped}

    for lab, adapter, art in candidates:
        if _ids.series_root(lab) in skip:
            continue
        capture = {"capture": dict(AUTO_CAPTURE)}
        if adapter == "line":
            fx, fy = extract_xy(art)
            reg.add(
                Mark(role="line", series=lab, kind="line", x=fx, y=fy, label=lab,
                     artists=[art], live_data=True, data=capture)
            )
        elif adapter == "point":
            arr = _finite_offsets(art)
            if arr is None:
                msg = (
                    f"fluxplot: not auto-promoting scatter {lab!r}: offsets are masked or "
                    "non-finite, so per-point indices would not match the drawn points. It "
                    "stays addressable as extra.*; use fp.scatter for explicit tagging."
                )
                _warnings.warn(msg, UserWarning, stacklevel=3)
                notes.append(msg)
                continue
            reg.add(
                Mark(role="point", series=lab, kind="scatter",
                     x=[float(v) for v in arr[:, 0]], y=[float(v) for v in arr[:, 1]],
                     label=lab, artists=[art], indexed=True, live_data=True, data=capture)
            )
        elif adapter == "area":
            # exact fill geometry lives in the SVG; the original y1/y2 vectors are not
            # recoverable from the artist, and we do not invent them
            reg.add(Mark(role="area", series=lab, kind="area", label=lab, artists=[art], live_data=True, data=capture))
        elif adapter == "bar":
            from .data import bar_data
            cx, cy, bar = bar_data(art.patches, getattr(art, "orientation", "vertical"))
            capture["bar"] = bar
            reg.add(
                Mark(role="bar", series=lab, kind="bar", x=cx, y=cy, label=lab,
                     artists=list(art.patches), indexed=True, live_data=True, data=capture)
            )
    return notes
