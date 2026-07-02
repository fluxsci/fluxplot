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
        # the WHOLE axis as a real <g id="axis.x"> wrapper, so the manifest's axis ref
        # resolves and "hide the entire X axis" targets one element.
        axis_gid = alloc.take(_ids.axis_id(which))
        mpl_axis.set_gid(axis_gid)
        guides.append(GuideTag(gid=axis_gid, role="axis", axis=which))

        title_gid = alloc.take(_ids.axis_id(which, "title"))
        mpl_axis.label.set_gid(title_gid)
        guides.append(
            GuideTag(gid=title_gid, role="axis-title", axis=which, text=mpl_axis.label.get_text())
        )

        labels = ax.get_xticklabels() if which == "x" else ax.get_yticklabels()
        for k, lbl in enumerate(labels):
            t = lbl.get_text()
            if not t or not lbl.get_visible():
                continue
            g = alloc.take(_ids.axis_id(which, "ticklabel", k))
            lbl.set_gid(g)
            guides.append(GuideTag(gid=g, role="tick-label", axis=which, text=t, index=k))

        # tick marks (the little dashes) — keep the enumerate index even when the
        # boundary ticks get culled at render, so ids stay stable & meaningful.
        for k, tick in enumerate(mpl_axis.get_major_ticks()):
            line = getattr(tick, "tick1line", None)
            if line is None or not line.get_visible():
                continue
            g = alloc.take(_ids.axis_id(which, "tick", k))
            line.set_gid(g)
            guides.append(GuideTag(gid=g, role="tick", axis=which, index=k))

        # gridlines
        for k, gl in enumerate(mpl_axis.get_gridlines()):
            if not gl.get_visible():
                continue
            g = alloc.take(_ids.axis_id(which, "gridline", k))
            gl.set_gid(g)
            guides.append(GuideTag(gid=g, role="gridline", axis=which, index=k))

    # spines (bottom/top → x, left/right → y; despined/polar sides are skipped)
    for side in ("bottom", "left", "top", "right"):
        try:
            sp = ax.spines[side]
        except (KeyError, TypeError):
            continue
        if not sp.get_visible():
            continue
        which = "x" if side in ("bottom", "top") else "y"
        g = alloc.take(_ids.axis_id(which, "spine"))
        sp.set_gid(g)
        guides.append(GuideTag(gid=g, role="spine", axis=which, text=side))

    legend = ax.get_legend()
    if legend is not None:
        g = alloc.take("legend")
        legend.set_gid(g)
        guides.append(GuideTag(gid=g, role="legend"))
        # per-entry swatch + label (entry k ↔ the k-th labeled series, in order)
        for k, txt in enumerate(legend.get_texts()):
            lg = alloc.take(_ids.join("legend", "entry", k, "label"))
            txt.set_gid(lg)
            guides.append(GuideTag(gid=lg, role="legend-label", index=k, text=txt.get_text()))
        for k, h in enumerate(getattr(legend, "legend_handles", None) or []):
            try:
                sg = alloc.take(_ids.join("legend", "entry", k, "swatch"))
                h.set_gid(sg)
                guides.append(GuideTag(gid=sg, role="legend-swatch", index=k))
            except Exception:
                pass

    # Titles: the house style writes a LEFT title (matplotlib's ax._left_title), so
    # inspecting only ax.title (center) misses it. Tag every title slot that carries
    # text — left / center / right + the figure suptitle. The get_gid() guard stops
    # the shared suptitle being re-tagged once per axes; alloc.take dedups the rest.
    for t in (
        getattr(ax, "_left_title", None),
        ax.title,
        getattr(ax, "_right_title", None),
        getattr(ax.figure, "_suptitle", None),
    ):
        if t is not None and t.get_text().strip() and not t.get_gid():
            g = alloc.take("figure.title")
            t.set_gid(g)
            guides.append(GuideTag(gid=g, role="title", text=t.get_text()))

    # Free-text sweep: any remaining un-tagged text artist (equation boxes, data /
    # value labels, callouts dropped with raw ax.text / ax.annotate) becomes an
    # addressable annotation, so nothing escapes the scene graph as an anonymous
    # text_N. Artists already tagged (titles above, fp.annotation/fp.tag overlays
    # resolved earlier) carry a gid and are skipped.
    k = 0
    for t in list(ax.texts) + list(ax.figure.texts):
        if not t.get_text().strip() or t.get_gid():
            continue
        g = alloc.take(_ids.join("annotation", k))
        t.set_gid(g)
        guides.append(GuideTag(gid=g, role="annotation", text=t.get_text()))
        k += 1

    # Orphan-artist sweep: the mirror of the free-text sweep for non-text primitives. Raw
    # ax.plot() lines, ax.add_collection()/scatter collections and ax.add_patch() patches that
    # the user never routed through an fp.* helper end up as bare <g id="line2d_N"> — no
    # data-role, absent from the manifest, so Flux can't mask/animate them. Series & overlay
    # artists were gid'd earlier by resolve_gids (which runs before this), so a leftover gid is
    # the exact "already tagged, skip me" signal the text sweep relies on. We assign extra.line.N
    # / extra.collection.N / extra.patch.N and role "extra". The axes' own background patch is
    # mpl scaffolding, not user content — exclude it (spines/ticks/gridlines live in the axis
    # containers, not these lists, so they never appear here).
    _sweep_extra(ax, alloc, guides)

    return guides


def _sweep_extra(ax, alloc: "_ids.IdAllocator", guides: list) -> None:
    background = getattr(ax, "patch", None)
    for kind, artists in (
        ("line", list(ax.lines) + list(ax.figure.lines)),
        ("collection", list(ax.collections)),
        ("patch", list(ax.patches) + list(ax.figure.patches)),
    ):
        n = 0
        for art in artists:
            if art is background:
                continue
            if not art.get_visible() or art.get_gid():
                continue
            g = alloc.take(_ids.join("extra", kind, n))
            art.set_gid(g)
            guides.append(GuideTag(gid=g, role="extra", index=n))
            n += 1
