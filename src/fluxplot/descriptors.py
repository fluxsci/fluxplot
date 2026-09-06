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
    live_data: bool = False  # refresh public artist data at save time
    axes: Any = None  # owning axes, captured before an artist can be detached
    split_use: bool = False  # True when one artist renders N <use> to split in post-process
    data: dict = field(default_factory=dict)  # extra identity (p, between, yerr, label_artist, …)

    # Resolved at save() time:
    gid: Optional[str] = None  # the group/primary id
    member_gids: list = field(default_factory=list)  # per-index member ids (bars), or per-point ids
    member_indices: list = field(default_factory=list)


@dataclass
class GuideTag:
    """A scaffold/guide element that was auto-tagged (axis, tick, label, gridline, spine, legend)."""

    gid: str
    role: str
    axis: Optional[str] = None  # "x" / "y" for axis-scoped guides
    text: Optional[str] = None  # for axis-title / tick-label text content
    index: Optional[int] = None  # per-index guides (gridline/tick/ticklabel/legend-entry)
    series: Optional[str] = None  # series this guide belongs to (legend swatch/label)
    data: dict = field(default_factory=dict)
    virtual: bool = False  # organizational guide without one SVG wrapper
    kind: Optional[str] = None  # data-kind hint (text|line|shape|container); auto-derived from role

    def __post_init__(self) -> None:
        if self.kind is None:
            from . import roles as _roles

            self.kind = _roles.kind_for_role(self.role)


def artist_kind(artist) -> Optional[str]:
    """Best-effort data-kind (text | line | shape) from a matplotlib artist's class.

    Used for marks whose role carries no static kind (``x-`` extension roles from
    :func:`fluxplot.tag`, the heterogeneous ``extra`` sweep). Containers / uninferable
    artists return ``None`` — the hint is then simply omitted (additive contract).
    """
    from matplotlib.collections import Collection, LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    if isinstance(artist, Text):
        return "text"
    if isinstance(artist, (Line2D, LineCollection)):
        return "line"
    if isinstance(artist, (Patch, Collection)):
        return "shape"
    return None


def mark_kind(mark: "Mark") -> Optional[str]:
    """Data-kind hint for a Mark: the role's static kind, else inferred from its artist."""
    from . import roles as _roles

    k = _roles.kind_for_role(mark.role)
    if k is not None:
        return k
    return artist_kind(mark.artists[0]) if mark.artists else None
