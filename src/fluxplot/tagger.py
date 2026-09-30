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
        # colour-control keys claimed so far (fields.control_key): two same-named colour-mapped
        # series never share a recipe entry
        self._color_keys: set[str] = set()
        # shared colour scales declared with fp.color_scale (fields.SharedScale), by name
        self._scales: dict = {}
        # problems the auto-tagger met that a save should report (SaveResult.warnings), instead
        # of swallowing them
        self.warnings: list[str] = []

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


def autotag_scaffold(ax, alloc: "_ids.IdAllocator", secondary: str | None = None) -> list[GuideTag]:
    """Name the axes/title/legend/tick-labels so the user never hand-tags scaffold.

    Returns GuideTags for the manifest + for ``data-role`` injection. Setting a gid on a scaffold
    artist is harmless even if that artist does not emit a wrapping ``<g>`` — post-processing only
    annotates ids that actually appear, and the manifest records the guide regardless (P4).

    ``secondary="y2"`` / ``"x2"`` tags a twin axes (``ax.twinx()`` / ``twiny()``) as part of its
    primary's panel: only its own value axis (``axis.y2.*``) and that axis' spine are named — the
    shared axis, the background and the titles belong to the primary.
    """
    guides: list[GuideTag] = []

    is3d = getattr(ax, "name", None) == "3d"
    if secondary is not None:
        axes = [(secondary, ax.yaxis if secondary == "y2" else ax.xaxis)]
    else:
        axes = [("x", ax.xaxis), ("y", ax.yaxis)]
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

        # a twin's value axis draws its ticks on the far side (right / top): that side is its primary
        for suffix, role, k, art in axis_tick_artists(mpl_axis, primary_side=2 if secondary else 1):
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
    if secondary == "y2":
        sides = (("right", "y2"),)
    elif secondary == "x2":
        sides = (("top", "x2"),)
    elif getattr(ax, "name", None) == "polar":
        sides = (("polar", "x"), ("inner", "x"), ("start", "y"), ("end", "y"))
    else:
        sides = (("bottom", "x"), ("left", "y"), ("top", "x"), ("right", "y"))
    seen_spines: dict[str, int] = {}
    for side, which in sides:
        try:
            sp = ax.spines[side]
        except (KeyError, TypeError):
            continue
        if not sp.get_visible():
            continue
        # the side is part of the id (axis.x.spine.bottom): a second visible spine no longer
        # depends on collision repair for its name. The id an older fluxplot gave this spine
        # (axis.x.spine, axis.x.spine-2, …) travels as an alias for one minor version.
        k = seen_spines.get(which, 0)
        seen_spines[which] = k + 1
        g = alloc.take(_ids.axis_id(which, "spine") + "." + side)
        sp.set_gid(g)
        legacy = _ids.axis_id(which, "spine") + ("" if k == 0 else f"-{k + 1}")
        guides.append(GuideTag(gid=g, role="spine", axis=which, text=side, data={"alias": legacy}))

    # the grounds: the axes' and the figure's background patches (and a framed legend's box) are
    # parts too — the paints a theme swaps first. The figure patch is shared by every panel; the
    # get_gid() guard tags it once.
    if secondary is None and ax.patch is not None and ax.patch.get_visible():
        g = alloc.take("axes.background")
        ax.patch.set_gid(g)
        guides.append(GuideTag(gid=g, role="background", text="axes"))
    fig_patch = getattr(ax.figure, "patch", None)
    if fig_patch is not None and fig_patch.get_visible() and not fig_patch.get_gid():
        g = alloc.take("figure.background")
        fig_patch.set_gid(g)
        guides.append(GuideTag(gid=g, role="background", text="figure"))

    legend = ax.get_legend()
    if legend is not None:
        g = alloc.take("legend")
        legend.set_gid(g)
        guides.append(GuideTag(gid=g, role="legend"))
        frame = legend.legendPatch
        if legend.get_frame_on() and frame is not None and frame.get_visible():
            fg = alloc.take("legend.background")
            frame.set_gid(fg)
            guides.append(GuideTag(gid=fg, role="background", text="legend"))
        # per-entry swatch + label, each knowing the artist it stands for (legend_sources): the
        # manifest joins entry ↔ series on that artist, not on the label text
        sources = legend_sources(legend, [ax])
        for k, txt in enumerate(legend.get_texts()):
            lg = alloc.take(_ids.join("legend", "entry", k, "label"))
            txt.set_gid(lg)
            guides.append(GuideTag(gid=lg, role="legend-label", index=k, text=txt.get_text(),
                                   data={"_source": sources.get(k)}))
        for k, h in enumerate(getattr(legend, "legend_handles", None) or []):
            try:
                sg = alloc.take(_ids.join("legend", "entry", k, "swatch"))
                h.set_gid(sg)
                guides.append(GuideTag(gid=sg, role="legend-swatch", index=k, data={"_source": sources.get(k)}))
            except Exception as exc:  # a handle that is no Artist: said, not swallowed
                registry_for(ax.figure).warnings.append(
                    f"legend entry {k}: swatch {type(h).__name__} could not be tagged ({exc})")

    # Titles: the house style writes a LEFT title (matplotlib's ax._left_title), so
    # inspecting only ax.title (center) misses it. Tag every title slot that carries
    # text — left / center / right. (The figure's suptitle is figure scope: autotag_figure.)
    titles = () if secondary else (getattr(ax, "_left_title", None), ax.title, getattr(ax, "_right_title", None))
    for t in titles:
        if t is not None and t.get_text().strip() and not t.get_gid():
            g = alloc.take("figure.title")
            t.set_gid(g)
            guides.append(GuideTag(gid=g, role="title", text=t.get_text()))

    # Free-text sweep: any remaining un-tagged text artist (equation boxes, data /
    # value labels, callouts dropped with raw ax.text / ax.annotate) becomes an
    # addressable annotation, so nothing escapes the scene graph as an anonymous
    # text_N. Artists already tagged (titles above, fp.annotation/fp.tag overlays
    # resolved earlier) carry a gid and are skipped. Figure-level text is autotag_figure's.
    k = 0
    for t in list(ax.texts):
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


def autotag_figure(fig, alloc: "_ids.IdAllocator") -> list[GuideTag]:
    """Name the figure-scope artists once, unprefixed: ``fig.suptitle`` → ``figure.title``,
    ``supxlabel`` / ``supylabel`` → ``figure.xlabel`` / ``figure.ylabel``, ``fig.legend()`` →
    ``figure.legend`` (``figure.legend.k`` for more) with ``.entry.k.label`` / ``.swatch``,
    ``fig.text`` → ``figure.annotation.k``, and the figure's own lines / patches / images →
    ``figure.extra.<kind>.k``. Every tag carries ``data["scope"] == "figure"``; they never
    belong to a panel. Runs before the panel scaffolds, so the suptitle claims ``figure.title``.
    """
    guides: list[GuideTag] = []
    scope = {"scope": "figure"}
    sup = getattr(fig, "_suptitle", None)
    if sup is not None and sup.get_text().strip() and not sup.get_gid():
        g = alloc.take("figure.title")
        sup.set_gid(g)
        guides.append(GuideTag(gid=g, role="title", text=sup.get_text(), data={**scope, "slot": "title"}))
    for attr, slot in (("_supxlabel", "xlabel"), ("_supylabel", "ylabel")):
        t = getattr(fig, attr, None)
        if t is not None and t.get_text().strip() and not t.get_gid():
            g = alloc.take("figure." + slot)
            t.set_gid(g)
            guides.append(GuideTag(gid=g, role="title", text=t.get_text(), data={**scope, "slot": slot}))
    for k, legend in enumerate(getattr(fig, "legends", []) or []):
        if not legend.get_visible() or legend.get_gid():
            continue
        g = alloc.take("figure.legend" if k == 0 else f"figure.legend.{k}")
        legend.set_gid(g)
        guides.append(GuideTag(gid=g, role="legend", index=k, data={**scope}))
        frame = legend.legendPatch
        if legend.get_frame_on() and frame is not None and frame.get_visible():
            fg = alloc.take(g + ".background")
            frame.set_gid(fg)
            guides.append(GuideTag(gid=fg, role="background", text=g, data={**scope, "legend": g}))
        sources = legend_sources(legend, fig.axes)
        for e, txt in enumerate(legend.get_texts()):
            lg = alloc.take(f"{g}.entry.{e}.label")
            txt.set_gid(lg)
            guides.append(GuideTag(gid=lg, role="legend-label", index=e, text=txt.get_text(),
                                   data={**scope, "legend": g, "_source": sources.get(e)}))
        for e, h in enumerate(getattr(legend, "legend_handles", None) or []):
            try:
                sg = alloc.take(f"{g}.entry.{e}.swatch")
                h.set_gid(sg)
                guides.append(GuideTag(gid=sg, role="legend-swatch", index=e,
                                       data={**scope, "legend": g, "_source": sources.get(e)}))
            except Exception as exc:
                registry_for(fig).warnings.append(
                    f"figure legend entry {e}: swatch {type(h).__name__} could not be tagged ({exc})")
    k = 0
    for t in list(fig.texts):
        if not t.get_text().strip() or t.get_gid():
            continue
        g = alloc.take(f"figure.annotation.{k}")
        t.set_gid(g)
        guides.append(GuideTag(gid=g, role="annotation", text=t.get_text(), data={**scope}))
        k += 1
    for kind, artists in (("line", list(fig.lines)), ("patch", list(fig.patches)), ("image", list(fig.images))):
        n = 0
        for art in artists:
            if art is fig.patch or not art.get_visible() or art.get_gid():
                continue
            g = alloc.take(f"figure.extra.{kind}.{n}")
            art.set_gid(g)
            guides.append(GuideTag(gid=g, role="extra", index=n, kind=artist_kind(art), data={**scope}))
            n += 1
    return guides


def legend_sources(legend, axes) -> dict:
    """``{entry index: source artist}`` for a legend: the artists its entries stand for.

    ``fp.legend(ax, handles, labels)`` records the handles it was given. A legend made with
    ``ax.legend()`` lists the axes' labelled artists in matplotlib's own order, so when the
    legend's texts still equal those labels one for one, that order is the mapping. Anything else
    (hand-picked handles through raw ``ax.legend(...)``, texts edited afterwards) yields nothing
    rather than a guess.
    """
    texts = [t.get_text() for t in legend.get_texts()]
    explicit = getattr(legend, "_fluxplot_handles", None)
    if explicit is not None:
        return {k: h for k, h in enumerate(explicit) if k < len(texts)}
    handles, labels = [], []
    for ax in axes:
        h, lbl = ax.get_legend_handles_labels()
        handles += h
        labels += lbl
    if handles and labels == texts:
        return dict(enumerate(handles))
    return {}


def _sweep_extra(ax, alloc: "_ids.IdAllocator", guides: list) -> None:
    background = getattr(ax, "patch", None)
    for kind, artists in (
        ("line", list(ax.lines)),
        ("collection", list(ax.collections)),
        ("patch", list(ax.patches)),
        # Images (raw ax.imshow) complete the sweep. Beyond consistency this is load-bearing
        # for auto-rasterization: it guarantees every image-emitting artist carries a gid, so
        # an <image> left with matplotlib's generated id is, by construction, one that
        # rasterization produced — which is how raster.reattach identifies them. (The figure's
        # own lines / patches / images are autotag_figure's.)
        ("image", list(ax.images)),
        # ax.artists holds what add_artist() placed: anchored boxes (an AnchoredSizeBar, an
        # AnchoredText), offset images, arbitrary artists; ax.tables holds table() output.
        ("artist", list(ax.artists)),
        ("table", list(ax.tables)),
    ):
        n = 0
        for art in artists:
            if art is background:
                continue
            if not art.get_visible() or art.get_gid():
                continue
            g = alloc.take(_ids.join("extra", kind, n))
            if kind == "artist" and _is_offsetbox(art):
                # an OffsetBox draws its children without a wrapping group of its own: name the
                # children so they, and not an invisible container, are addressable — the first
                # drawable as the box's id, further drawables as .k, its text as .label
                _tag_offsetbox(art, g, alloc, guides)
                n += 1
                continue
            art.set_gid(g)
            # "extra" has no static kind (role is heterogeneous) — infer from the artist
            guides.append(GuideTag(gid=g, role="extra", index=n,
                                   kind="container" if kind == "table" else artist_kind(art)))
            n += 1


def _is_offsetbox(art) -> bool:
    from matplotlib.offsetbox import OffsetBox
    return isinstance(art, OffsetBox)


def _tag_offsetbox(box, gid: str, alloc: "_ids.IdAllocator", guides: list) -> None:
    from matplotlib.offsetbox import OffsetBox
    from matplotlib.text import Text
    leaves, texts = [], []

    def visit(a):
        if not a.get_visible():
            return
        if isinstance(a, OffsetBox):
            for child in a.get_children():
                visit(child)
            patch = getattr(a, "patch", None)
            if patch is not None and patch.get_visible() and a is box:
                leaves.append(patch)
        elif isinstance(a, Text):
            if a.get_text().strip():
                texts.append(a)
        elif hasattr(a, "set_gid"):
            leaves.append(a)

    visit(box)
    box.set_gid(gid)  # harmless: the box emits no group, and marks it swept
    for k, leaf in enumerate(leaves):
        g = gid if k == 0 else alloc.take(f"{gid}.{k}")
        leaf.set_gid(g)
        guides.append(GuideTag(gid=g, role="extra", index=k if k else None, kind=artist_kind(leaf)))
    for k, txt in enumerate(texts):
        g = alloc.take(f"{gid}.label" if k == 0 else f"{gid}.label.{k}")
        txt.set_gid(g)
        guides.append(GuideTag(gid=g, role="label", index=k if k else None, text=txt.get_text(), kind="text"))
