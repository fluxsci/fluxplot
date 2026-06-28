"""Semantic descriptors — the in-memory record of "what each mark means", captured at plot time.

The registry (see ``tagger.py``) holds a list of :class:`Mark` per figure. Each Mark records the
meaning of one logical mark-group (a series' line, a series' points, a bar set, an overlay) plus the
matplotlib artist(s) that draw it, so :func:`fluxplot.save` can assign deterministic gids, capture
data, and emit the manifest. Geometry is never stored here — it stays authoritative in the SVG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass
class Mark:
    """One tagged logical mark-group (a series component or a standalone overlay)."""

    role: str  # line | point | bar | area | errorbar | box | annotation | reference-line | ...
    series: Optional[str] = None  # source series name (None for overlays / scaffold)
    name: Optional[str] = None  # identity for overlays, e.g. annotation "peak"
    kind: Optional[str] = None  # logical series kind: line | scatter | bar | errorbar | area
    label: Optional[str] = None  # legend label, if any
    x: Optional[Sequence[float]] = None  # captured data (the user's real arrays)
    y: Optional[Sequence[float]] = None
    artists: list = field(default_factory=list)  # mpl artist(s) this Mark draws (strong refs)
    indexed: bool = False  # True for point/bar groups → addressable per-index members
    split_use: bool = False  # True when one artist renders N <use> to split in post-process
    data: dict = field(default_factory=dict)  # extra identity (p, between, yerr, label_artist, …)

    # Resolved at save() time:
    gid: Optional[str] = None  # the group/primary id
    member_gids: list = field(default_factory=list)  # per-index member ids (bars), or per-point ids


@dataclass
class GuideTag:
    """A scaffold/guide element that was auto-tagged (axis, tick, label, gridline, spine, legend)."""

    gid: str
    role: str
    axis: Optional[str] = None  # "x" / "y" for axis-scoped guides
    text: Optional[str] = None  # for axis-title / tick-label text content
