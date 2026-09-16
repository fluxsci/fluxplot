"""The semantic registry + gid assignment + scaffold auto-tagging.

The registry maps a matplotlib ``Figure`` to the :class:`~fluxplot.descriptors.Mark`\\ s tagged on
it, via a ``WeakKeyDictionary`` (no monkeypatching of matplotlib; GC-safe — the registry dies with
the figure). ``resolve_gids`` turns Marks into deterministic SVG ids (``artist.set_gid``);
``autotag_scaffold`` names the axes/legend/title the user never has to tag by hand.
"""
from __future__ import annotations

import weakref
from dataclasses import replace
from contextlib import contextmanager

from . import ids as _ids
from .descriptors import GuideTag, Mark, artist_kind

_REGISTRIES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


class Registry:
    def __init__(self) -> None:
        self.marks: list[Mark] = []
        self._overlay_counts: dict[str, int] = {}
        self._series_slugs: dict[str, str] = {}

    def add(self, mark: Mark) -> Mark:
        if mark.axes is None and mark.artists:
            mark.axes = getattr(mark.artists[0], "axes", None)
        # Two DIFFERENT series names normalizing to one slug would silently produce
        # order-dependent "-2" ids — a safety net, not durable identity (plan §7). Fail at
        # registration, where the traceback points at the user's own call site.
        if mark.series is not None:
            slug = _ids.series_root(mark.series)
            first = self._series_slugs.setdefault((mark.axes, slug), str(mark.series))
            if first != str(mark.series):
                raise ValueError(
                    f"series name {str(mark.series)!r} collides with {first!r}: both normalize "
                    f"to the id {slug!r} (ids are slugified: lowercase, spaces/underscores "
                    "become '-'). Give each series a stable, distinct name."
                )
        self.marks.append(mark)
        return mark

    def next_overlay_index(self, role: str) -> int:
        n = self._overlay_counts.get(role, 0)
        self._overlay_counts[role] = n + 1
        return n


def registry_for(fig) -> Registry:
    reg = getattr(fig, "_fluxplot_registry", None)
    if reg is None:
        reg = Registry()
        fig._fluxplot_registry = reg
        _REGISTRIES[fig] = weakref.ref(reg)
    return reg


def clear(fig) -> None:
    _REGISTRIES.pop(fig, None)
    if hasattr(fig, "_fluxplot_registry"):
        del fig._fluxplot_registry


def snapshot(fig):
    from .data import refresh
    reg = Registry()
    for original in registry_for(fig).marks:
        mark = replace(original, artists=list(original.artists), data=dict(original.data),
                       member_gids=[], member_indices=[])
        refresh(mark)
        reg.add(mark)
    return reg


@contextmanager
def temporary_gids(fig, reg):
    artists = {id(a): a for a in fig.findobj()}
    for m in reg.marks:
        for a in m.artists:
            artists[id(a)] = a
    before = [(a, a.get_gid()) for a in artists.values()]
    try:
        yield
    finally:
        for a, gid in before:
            a.set_gid(gid)


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
        m.member_indices = []
        m.split_use = False
        if m.series is not None:
            _resolve_series_mark(m, alloc)
        else:
            _resolve_overlay_mark(m, alloc)


def _resolve_series_mark(m: Mark, alloc: "_ids.IdAllocator") -> None:
    series = m.series
    if m.data.get('contour_legacy'):
        m.gid = alloc.take(_ids.series_id(series, m.role))
        for i, art in enumerate(m.artists):
            gid = m.gid + f'.level.{i}'
            art.set_gid(gid)
            m.member_gids.append(gid)
    elif m.role == "point":
        m.gid = alloc.take(_ids.series_id(series, "points"))
        m.artists[0].set_gid(m.gid)
        m.split_use = True
        from .data import point_indices
        m.member_indices = point_indices(m)
        m.member_gids = [alloc.take(_ids.series_id(series, "point", k)) for k in m.member_indices]
    elif m.role == "bar":
        for k, art in enumerate(m.artists):
            cid = alloc.take(_ids.series_id(series, "bar", k))
            art.set_gid(cid)
            m.member_gids.append(cid)
    else:  # line, area, errorbar, box, or a generic series role
        # A series mark may carry an explicit ``name`` to distinguish it from its siblings — several
        # marks of the SAME role under one series (e.g. one collection per region of a surface map).
        # Then the name, not the role, is the meaningful id segment: ``atlas.frontal`` rather
        # than ``atlas.surface-region-2``, so the part is addressable by what it actually is.
        # Roles with a single mark per series (line/area/errorbar/box) never set ``name``, so their
        # ids are unchanged.
        seg = _ids.slugify(m.name) if m.name is not None else m.role
        for k, art in enumerate(m.artists):
            cid = (
                alloc.take(_ids.series_id(series, seg))
                if k == 0
                else alloc.take(_ids.series_id(series, seg, k))
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
def axis_tick_artists(mpl_axis, primary_side=1):
    """Yield visible major/minor tick components with stable, side-aware suffixes.

    Major primary-side IDs retain the original naming. Extra levels/sides gain
    prefixes so adding top/right marks never renumbers the existing bottom/left
    marks. The renderer still determines which boundary components survive.
    """
    for level, ticks in (("major", mpl_axis.get_major_ticks()),
                         ("minor", mpl_axis.get_minor_ticks())):
        prefix = "" if level == "major" else "minor."
        for k, tick in enumerate(ticks):
            if not tick.get_visible():
                continue
            for side in (primary_side, 3 - primary_side):
                side_prefix = "" if side == primary_side else "secondary."
                for part, role, attr in (("tick", "tick", f"tick{side}line"),
                                         ("ticklabel", "tick-label", f"label{side}")):
                    art = getattr(tick, attr, None)
                    if art is None or not art.get_visible():
                        continue
                    if role == "tick-label" and not art.get_text():
                        continue
                    yield f"{prefix}{side_prefix}{part}.{k}", role, k, art
            grid = tick.gridline
            if grid.get_visible():
                yield f"{prefix}gridline.{k}", "gridline", k, grid


def autotag_scaffold(ax, alloc: "_ids.IdAllocator") -> list[GuideTag]:
    """Name the axes/title/legend/tick-labels so the user never hand-tags scaffold.

    Returns GuideTags for the manifest + for ``data-role`` injection. Setting a gid on a scaffold
    artist is harmless even if that artist does not emit a wrapping ``<g>`` — post-processing only
    annotates ids that actually appear, and the manifest records the guide regardless (P4).
    """
    guides: list[GuideTag] = []

    axes = [("x", ax.xaxis), ("y", ax.yaxis)]
    is3d = getattr(ax, "name", None) == "3d"
    if is3d:
        axes.append(("z", ax.zaxis))
    for which, mpl_axis in axes:
        # the WHOLE axis as a real <g id="axis.x"> wrapper, so the manifest's axis ref
        # resolves and "hide the entire X axis" targets one element.
        axis_gid = alloc.take(_ids.axis_id(which))
        # Axes3D draws one Axis through three separate groups with the same
        # gid. Name its actual components and use a virtual organizational axis.
        mpl_axis.set_gid(None if is3d else axis_gid)
        guides.append(GuideTag(gid=axis_gid, role="axis", axis=which, virtual=is3d))
        if is3d:
            for role, art in (("spine", mpl_axis.line), ("background", mpl_axis.pane),
                              ("gridline", mpl_axis.gridlines)):
                gid = alloc.take(_ids.axis_id(which, role))
                art.set_gid(gid)
                guides.append(GuideTag(gid=gid, role=role, axis=which))

        title_gid = alloc.take(_ids.axis_id(which, "title"))
        mpl_axis.label.set_gid(title_gid)
        guides.append(
            GuideTag(gid=title_gid, role="axis-title", axis=which, text=mpl_axis.label.get_text())
        )

        for suffix, role, k, art in axis_tick_artists(mpl_axis):
            g = alloc.take(f"axis.{which}.{suffix}")
            art.set_gid(g)
            guides.append(GuideTag(gid=g, role=role, axis=which, index=k,
                                   text=art.get_text() if role == "tick-label" else None))

    # spines. Rectangular axes key them bottom/left/top/right; polar axes key them
    # polar/start/end/inner (so the old side list silently dropped every polar spine).
    # Axis assignment follows the along-direction convention (a bottom spine runs along
    # x → axis "x"): the outer "polar" circle and the "inner" circle run along theta → x;
    # the "start"/"end" wedge edges run along r → y. Invisible/absent sides are skipped;
    # the spine's own key travels as `text`, exactly like the rectangular sides do.
    if getattr(ax, "name", None) == "polar":
        sides = (("polar", "x"), ("inner", "x"), ("start", "y"), ("end", "y"))
    else:
        sides = (("bottom", "x"), ("left", "y"), ("top", "x"), ("right", "y"))
    for side, which in sides:
        try:
            sp = ax.spines[side]
        except (KeyError, TypeError):
            continue
        if not sp.get_visible():
            continue
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
        # Images (raw ax.imshow) complete the sweep. Beyond consistency this is load-bearing
        # for auto-rasterization: it guarantees every image-emitting artist carries a gid, so
        # an <image> left with matplotlib's generated id is, by construction, one that
        # rasterization produced — which is how raster.reattach identifies them.
        ("image", list(ax.images) + list(ax.figure.images)),
    ):
        n = 0
        for art in artists:
            if art is background:
                continue
            if not art.get_visible() or art.get_gid():
                continue
            g = alloc.take(_ids.join("extra", kind, n))
            art.set_gid(g)
            # "extra" has no static kind (role is heterogeneous) — infer from the artist
            guides.append(GuideTag(gid=g, role="extra", index=n, kind=artist_kind(art)))
            n += 1
