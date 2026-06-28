"""The semantic registry + gid assignment + scaffold auto-tagging.

The registry maps a matplotlib ``Figure`` to the :class:`~fluxplot.descriptors.Mark`\\ s tagged on
it, via a ``WeakKeyDictionary`` (no monkeypatching of matplotlib; GC-safe — the registry dies with
the figure). ``resolve_gids`` turns Marks into deterministic SVG ids (``artist.set_gid``);
``autotag_scaffold`` names the axes/legend/title the user never has to tag by hand.
"""
from __future__ import annotations

import weakref

from . import ids as _ids
from .descriptors import GuideTag, Mark

_REGISTRIES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


class Registry:
    def __init__(self) -> None:
        self.marks: list[Mark] = []
        self._overlay_counts: dict[str, int] = {}

    def add(self, mark: Mark) -> Mark:
        self.marks.append(mark)
        return mark

    def next_overlay_index(self, role: str) -> int:
        n = self._overlay_counts.get(role, 0)
        self._overlay_counts[role] = n + 1
        return n


def registry_for(fig) -> Registry:
    reg = _REGISTRIES.get(fig)
    if reg is None:
        reg = Registry()
        _REGISTRIES[fig] = reg
    return reg


def clear(fig) -> None:
    _REGISTRIES.pop(fig, None)


def fig_of(artist):
    """Resolve the Figure that owns an artist (or container)."""
    fig = getattr(artist, "figure", None)
    if fig is not None:
        return fig
    ax = getattr(artist, "axes", None)
    if ax is not None:
        return ax.figure
    raise ValueError(f"cannot resolve a Figure from {artist!r}; tag the artist after it is added")


# ---------------------------------------------------------------------------
# gid assignment
# ---------------------------------------------------------------------------
def resolve_gids(reg: Registry, alloc: "_ids.IdAllocator") -> None:
    """Assign deterministic gids to every Mark's artist(s), in insertion order."""
    for m in reg.marks:
        # reset resolved fields so re-saving the same figure is idempotent
        m.gid = None
        m.member_gids = []
        m.split_use = False
        if m.series is not None:
            _resolve_series_mark(m, alloc)
        else:
            _resolve_overlay_mark(m, alloc)


def _resolve_series_mark(m: Mark, alloc: "_ids.IdAllocator") -> None:
    series = m.series
    if m.role == "point":
        m.gid = alloc.take(_ids.series_id(series, "points"))
        m.artists[0].set_gid(m.gid)
        m.split_use = True
        n = len(m.y) if m.y is not None else 0
        m.member_gids = [alloc.take(_ids.series_id(series, "point", k)) for k in range(n)]
    elif m.role == "bar":
        for k, art in enumerate(m.artists):
            cid = alloc.take(_ids.series_id(series, "bar", k))
            art.set_gid(cid)
            m.member_gids.append(cid)
    else:  # line, area, errorbar, box, or a generic series role
        for k, art in enumerate(m.artists):
            cid = (
                alloc.take(_ids.series_id(series, m.role))
                if k == 0
                else alloc.take(_ids.series_id(series, m.role, k))
            )
            art.set_gid(cid)
            m.member_gids.append(cid)
        m.gid = m.member_gids[0] if m.member_gids else None


def _resolve_overlay_mark(m: Mark, alloc: "_ids.IdAllocator") -> None:
    name = m.name if m.name is not None else str(m.data.get("index", 0))
    m.gid = alloc.take(_ids.join(m.role, _ids.slugify(name)))
    if m.artists:
        m.artists[0].set_gid(m.gid)
    label_artist = m.data.get("label_artist")
    if label_artist is not None:
        label_gid = alloc.take(f"{m.gid}.label")
        label_artist.set_gid(label_gid)
        m.data["label_gid"] = label_gid


# ---------------------------------------------------------------------------
# scaffold auto-tagging
# ---------------------------------------------------------------------------
def autotag_scaffold(ax, alloc: "_ids.IdAllocator") -> list[GuideTag]:
    """Name the axes/title/legend/tick-labels so the user never hand-tags scaffold.

    Returns GuideTags for the manifest + for ``data-role`` injection. Setting a gid on a scaffold
    artist is harmless even if that artist does not emit a wrapping ``<g>`` — post-processing only
    annotates ids that actually appear, and the manifest records the guide regardless (P4).
    """
    guides: list[GuideTag] = []

    for which, mpl_axis in (("x", ax.xaxis), ("y", ax.yaxis)):
        axis_gid = _ids.axis_id(which)
        guides.append(GuideTag(gid=alloc.take(axis_gid), role="axis", axis=which))

        title_gid = alloc.take(_ids.axis_id(which, "title"))
        mpl_axis.label.set_gid(title_gid)
        guides.append(
            GuideTag(gid=title_gid, role="axis-title", axis=which, text=mpl_axis.label.get_text())
        )

        labels = ax.get_xticklabels() if which == "x" else ax.get_yticklabels()
        for k, lbl in enumerate(labels):
            t = lbl.get_text()
            if not t:
                continue
            g = alloc.take(_ids.axis_id(which, "ticklabel", k))
            lbl.set_gid(g)
            guides.append(GuideTag(gid=g, role="tick-label", axis=which, text=t))

    legend = ax.get_legend()
    if legend is not None:
        g = alloc.take("legend")
        legend.set_gid(g)
        guides.append(GuideTag(gid=g, role="legend"))

    title = ax.title
    if title is not None and title.get_text():
        g = alloc.take("title")
        title.set_gid(g)
        guides.append(GuideTag(gid=g, role="title", text=title.get_text()))

    return guides
