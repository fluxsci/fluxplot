"""Public API — free functions over real matplotlib artists (spec §11.1).

Free functions (not a wrapped Axes) keep the user in full matplotlib (P2): the convenience helpers
and the ``tag()`` escape hatch are the same shape, and untagged matplotlib keeps working. ``save()``
orchestrates the pipeline: assign gids → draw → auto-tag scaffold → capture coords → render
deterministically → inject ``data-*`` → emit manifest + recipe.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import matplotlib

from . import autotag as _autotag
from . import canonical_json as _cjson
from . import capture as _capture
from . import ids as _ids
from . import manifest as _manifest
from . import postprocess as _postprocess
from . import recipe as _recipe
from . import render as _render
from . import roles as _roles
from . import tagger as _tagger
from .autotag import is_colorbar_axes as _is_colorbar_axes
from .descriptors import Mark
from .version import SPEC_VERSION, __version__


def _list(a):
    return None if a is None else list(a)


# ---------------------------------------------------------------------------
# convenience helpers (auto-tagging) — each returns the real matplotlib artist(s)
# ---------------------------------------------------------------------------
def line(ax, x, y, *, series, marker=None, label=None, **kw):
    """A named line. With ``marker=`` it also draws an addressable per-point group."""
    reg = _tagger.registry_for(ax.figure)
    (ln,) = ax.plot(x, y, label=label, **kw)
    reg.add(Mark(role="line", series=series, kind="line", x=_list(x), y=_list(y), label=label, artists=[ln]))
    if marker:
        (pts,) = ax.plot(x, y, linestyle="none", marker=marker, color=ln.get_color())
        reg.add(Mark(role="point", series=series, kind="line", x=_list(x), y=_list(y), artists=[pts], indexed=True))
        return ln, pts
    return ln


def scatter(ax, x, y, *, series, label=None, **kw):
    reg = _tagger.registry_for(ax.figure)
    coll = ax.scatter(x, y, label=label, **kw)
    reg.add(Mark(role="point", series=series, kind="scatter", x=_list(x), y=_list(y), label=label, artists=[coll], indexed=True))
    return coll


def bar(ax, x, height, *, series, label=None, **kw):
    reg = _tagger.registry_for(ax.figure)
    container = ax.bar(x, height, label=label, **kw)
    reg.add(Mark(role="bar", series=series, kind="bar", x=_list(x), y=_list(height), label=label, artists=list(container.patches), indexed=True))
    return container


def errorbar(ax, x, y, *, series, yerr=None, label=None, **kw):
    reg = _tagger.registry_for(ax.figure)
    container = ax.errorbar(x, y, yerr=yerr, label=label, **kw)
    data_line, caps, barlinecols = container
    # Tag the WHOLE container: the central marker/data line + the cap lines + the
    # error bars. Previously only barlinecols was kept, so the central line and the
    # caps escaped the scene graph entirely.
    artists = ([data_line] if data_line is not None else []) + list(caps) + list(barlinecols)
    reg.add(Mark(role="errorbar", series=series, kind="errorbar", x=_list(x), y=_list(y), label=label, artists=artists, data={"yerr": _list(yerr)}))
    return container


def area(ax, x, y1, y2=0, *, series, label=None, **kw):
    reg = _tagger.registry_for(ax.figure)
    poly = ax.fill_between(x, y1, y2, label=label, **kw)
    reg.add(Mark(role="area", series=series, kind="area", x=_list(x), y=_list(y1), label=label, artists=[poly]))
    return poly


def box(ax, values, *, series, label=None, include_values=False, **kw):
    """One box-and-whisker group as ONE semantic series with addressable statistics.

    Wraps ``Axes.boxplot`` for a single dataset. The documented return-dict pieces — box body,
    whiskers, caps, median, fliers (and mean when shown) — become tagged sub-parts grouped per
    series (``<series>.whiskers``, ``<series>.caps``, …). Multiple groups = multiple calls with
    distinct ``series`` names. Raw sample values are recorded only with ``include_values=True``
    (samples can be large or sensitive). Returns matplotlib's boxplot dict unchanged.
    """
    reg = _tagger.registry_for(ax.figure)
    bp = ax.boxplot(values, **kw)
    if len(bp["boxes"]) != 1:
        raise ValueError(
            "fp.box tags one box per call — pass a single dataset, and call fp.box once per "
            "group with a distinct stable series= name"
        )
    body = bp["boxes"][0]
    if label:
        body.set_label(label)
    data = {"distribution": {"values": [float(v) for v in values]}} if include_values else {}
    reg.add(Mark(role="box", series=series, kind="box", label=label, artists=[body], data=data))
    for role, key in (
        ("whisker", "whiskers"), ("cap", "caps"), ("median", "medians"),
        ("flier", "fliers"), ("mean", "means"),
    ):
        artists = [a for a in bp.get(key, []) if a is not None]
        if artists:  # options that were off create no dead parts
            reg.add(Mark(role=role, series=series, kind="box", artists=artists))
    return bp


def violin(ax, values, *, series, label=None, include_values=False, **kw):
    """One violin as ONE semantic series with addressable statistics.

    Wraps ``Axes.violinplot`` for a single dataset: the body plus whatever the call returned
    (extrema bar, min/max caps, median/mean/quantile lines) each become tagged sub-parts —
    absent options create no dead parts. Multiple groups = multiple calls with distinct
    ``series`` names. Raw sample values are recorded only with ``include_values=True``.
    Returns matplotlib's violinplot dict unchanged.
    """
    reg = _tagger.registry_for(ax.figure)
    vp = ax.violinplot(values, **kw)
    bodies = vp.get("bodies") or []
    if len(bodies) != 1:
        raise ValueError(
            "fp.violin tags one violin per call — pass a single dataset, and call fp.violin "
            "once per group with a distinct stable series= name"
        )
    body = bodies[0]
    if label:
        body.set_label(label)
    data = {"distribution": {"values": [float(v) for v in values]}} if include_values else {}
    reg.add(Mark(role="violin", series=series, kind="violin", label=label, artists=[body], data=data))
    caps = [vp[k] for k in ("cmins", "cmaxes") if vp.get(k) is not None]
    if caps:
        reg.add(Mark(role="cap", series=series, kind="violin", artists=caps))
    for role, key in (
        ("whisker", "cbars"), ("median", "cmedians"), ("mean", "cmeans"), ("segment", "cquantiles"),
    ):
        art = vp.get(key)
        if art is not None:
            reg.add(Mark(role=role, series=series, kind="violin", artists=[art]))
    return vp


def hist(ax, values, *, series, bins=None, label=None, include_values=False, **kw):
    """A histogram as an indexed bar series with its exact distribution recorded.

    Wraps ``Axes.hist`` (single dataset, default bar histtype). The returned patches become
    per-index addressable bars; the manifest additionally carries the exact ``binEdges`` and
    ``counts`` in an additive ``distribution`` payload. Bar heights are never presented as the
    original observations — pass ``include_values=True`` to record the source values (they can
    be large or sensitive). Returns matplotlib's ``(counts, edges, patches)`` unchanged.
    """
    from matplotlib.container import BarContainer

    reg = _tagger.registry_for(ax.figure)
    counts, edges, patches = ax.hist(values, bins=bins, label=label, **kw)
    if not isinstance(patches, BarContainer):
        raise ValueError(
            "fp.hist tags one dataset with the default bar histtype — multiple datasets or "
            "histtype='step' have no exact per-bar contract; call fp.hist once per series"
        )
    edges_f = [float(e) for e in edges]
    counts_f = [float(c) for c in counts]
    centers = [(edges_f[k] + edges_f[k + 1]) / 2.0 for k in range(len(counts_f))]
    dist = {"binEdges": edges_f, "counts": counts_f}
    if include_values:
        dist["values"] = [float(v) for v in values]
    reg.add(
        Mark(role="bar", series=series, kind="bar", x=centers, y=counts_f, label=label,
             artists=list(patches.patches), indexed=True, data={"distribution": dist})
    )
    return counts, edges, patches


# ---------------------------------------------------------------------------
# escape hatch — tag arbitrary raw matplotlib artists
# ---------------------------------------------------------------------------
def tag(artist, *, role, series=None, index=None, name=None, x=None, y=None, **identity):
    """Tag any raw matplotlib artist with a semantic role + identity.

    ``x``/``y`` capture the mark's data coordinates. When omitted, they are read from the
    artist itself for supported types (``Line2D`` data, scatter offsets — the same exact
    adapters save-time promotion uses); otherwise they stay honestly absent rather than
    making an invalid spatial claim.
    """
    reg = _tagger.registry_for(_tagger.fig_of(artist))
    if x is None and y is None:
        x, y = _autotag.extract_xy(artist)
    data = dict(identity)
    if index is not None:
        data["index"] = index
    reg.add(
        Mark(role=_roles.validate(role), series=series, name=name, x=_list(x), y=_list(y),
             artists=[artist], data=data, indexed=index is not None)
    )
    return artist


def tag_points(points, *, series, x=None, y=None):
    """Tag an existing markers ``Line2D`` / ``PathCollection`` as an addressable point group."""
    reg = _tagger.registry_for(_tagger.fig_of(points))
    if x is None and y is None:
        x, y = _autotag.extract_xy(points)  # Line2D data or exact finite scatter offsets
    reg.add(Mark(role="point", series=series, kind="scatter", x=_list(x), y=_list(y), artists=[points], indexed=True))
    return points


def tag_seaborn(ax, *, series=None):
    """Auto-tag the artists a seaborn axes-level plot drew on ``ax`` — one call, done.

    Call it right after the seaborn call (and before raw-matplotlib additions you tag
    yourself). Series names come from ``series=[...]`` if given, else the legend's labels
    (seaborn writes them in hue order), else the y-axis label. Per series it names:

    - data-carrying lines           → role ``line``      (``lineplot`` means, ``kdeplot``, ``regplot`` fits)
    - fill-between bands            → role ``area``      (confidence / error bands)
    - scatter collections           → role ``point``     (``scatterplot`` / ``regplot`` — per-point addressable)
    - bar containers                → role ``bar``       (``barplot`` / ``countplot`` / ``histplot``)
    - vertical error-bar segments   → role ``errorbar``  (joined to their bar by x position)

    Seaborn's empty legend-proxy lines are removed (they draw nothing; the legend keeps its
    own handles). Artists already tagged are skipped, so this composes with the ``fp.*``
    helpers and :func:`tag`. Anything it cannot *confidently* pair with a series name is
    left alone — ``save()``'s orphan sweep still makes it addressable as ``extra.*``.

    Returns ``{series_name: [roles tagged]}`` so you can see exactly what got named.

    Known limits (by construction, not laziness): ``scatterplot(hue=...)`` draws ALL hue
    groups as ONE collection, so per-hue identity is not recoverable from the artists —
    the points become a single per-point-addressable group named after the y-label.
    Composite per-category plots (``boxplot``, ``violinplot``) should be tagged
    explicitly with :func:`tag` (see the box plot example in ``examples/``).
    """
    from matplotlib.collections import PathCollection, PolyCollection
    from matplotlib.container import BarContainer

    reg = _tagger.registry_for(ax.figure)
    already = {id(a) for m in reg.marks for a in m.artists}

    # seaborn appends empty proxy lines used only to build its legend — drop them so the
    # orphan sweep doesn't dutifully tag invisible leftovers.
    for ln in [ln for ln in ax.lines if len(ln.get_xdata()) == 0]:
        ln.remove()

    legend = ax.get_legend()
    if series is not None:
        names = [str(s) for s in series]
    elif legend is not None and legend.get_texts():
        names = [t.get_text() for t in legend.get_texts()]
    else:
        names = [ax.get_ylabel() or "data"]

    tagged: dict = {}

    def _record(name, role):
        tagged.setdefault(str(name), []).append(role)

    def _is_segment(ln):  # a 2-point vertical segment is an error bar, not a data curve
        x = ln.get_xdata()
        return len(x) == 2 and float(x[0]) == float(x[1])

    # data curves (one per hue level) → line
    curves = [ln for ln in ax.lines if id(ln) not in already and len(ln.get_xdata()) and not _is_segment(ln)]
    if curves and len(curves) == len(names):
        for name, ln in zip(names, curves):
            gx, gy = ln.get_data()
            reg.add(Mark(role="line", series=name, kind="line", x=_list(gx), y=_list(gy), artists=[ln]))
            _record(name, "line")

    # fill-between bands (one per hue level) → area
    bands = [c for c in ax.collections if id(c) not in already and isinstance(c, PolyCollection)]
    if bands and len(bands) == len(names):
        for name, band in zip(names, bands):
            reg.add(Mark(role="area", series=name, kind="area", artists=[band]))
            _record(name, "area")

    # scatter collections → point (per-point addressable, with data values from the offsets)
    pts = [
        c for c in ax.collections
        if id(c) not in already and isinstance(c, PathCollection) and not isinstance(c, PolyCollection)
    ]
    if pts and len(pts) == len(names):
        pairs = list(zip(names, pts))
    elif len(pts) == 1:  # scatterplot(hue=) fuses all hues into one collection
        pairs = [(ax.get_ylabel() or "points", pts[0])]
    else:
        pairs = []
    for name, coll in pairs:
        off = coll.get_offsets()
        x, y = [float(v) for v in off[:, 0]], [float(v) for v in off[:, 1]]
        reg.add(Mark(role="point", series=name, kind="scatter", x=x, y=y, artists=[coll], indexed=True))
        _record(name, "point")

    # bar containers (one per hue level) → bar, with bar centers/heights as the data
    bars = [c for c in getattr(ax, "containers", []) if isinstance(c, BarContainer)]
    bar_centers: list = []  # (name, {x center}) for the error-bar join below
    if bars and len(bars) == len(names):
        for name, cont in zip(names, bars):
            patches = [p for p in cont.patches if id(p) not in already]
            if not patches:
                continue
            cx = [float(p.get_x() + p.get_width() / 2.0) for p in patches]
            cy = [float(p.get_height()) for p in patches]
            reg.add(Mark(role="bar", series=name, kind="bar", x=cx, y=cy, artists=patches, indexed=True))
            _record(name, "bar")
            bar_centers.append((name, {round(v, 9) for v in cx}))

    # seaborn draws bar errors as loose 2-point vertical lines — join each to its bar
    # by x position (an exact join on coordinates seaborn itself set, not a guess).
    segs = [ln for ln in ax.lines if id(ln) not in already and len(ln.get_xdata()) and _is_segment(ln)]
    for ln in segs:
        x0 = round(float(ln.get_xdata()[0]), 9)
        for name, centers in bar_centers:
            if x0 in centers:
                reg.add(Mark(role="errorbar", series=name, kind="errorbar", artists=[ln]))
                _record(name, "errorbar")
                break

    return tagged


# ---------------------------------------------------------------------------
# first-class overlays
# ---------------------------------------------------------------------------
def significance_bracket(ax, *, x0, x1, y, label, between=None, p=None, name=None, height=None, **kw):
    """A p-value bracket (spec §5: deliberately first-class — ubiquitous in science)."""
    reg = _tagger.registry_for(ax.figure)
    idx = reg.next_overlay_index("significance-bracket")
    if name is None:
        name = str(idx)
    if height is None:
        if ax.get_yscale() == "log":
            ytop = y * 1.06
        else:
            lo, hi = ax.get_ylim()
            ytop = y + (hi - lo) * 0.03
    else:
        ytop = y + height
    (br,) = ax.plot([x0, x0, x1, x1], [y, ytop, ytop, y], color=kw.pop("color", "black"), linewidth=kw.pop("linewidth", 1.0), **kw)
    txt = ax.text((x0 + x1) / 2.0, ytop, label, ha="center", va="bottom")
    data = {"label": label, "index": idx, "label_artist": txt}
    if between is not None:
        data["between"] = list(between)
    if p is not None:
        data["p"] = p
    reg.add(Mark(role="significance-bracket", series=None, name=name, artists=[br], data=data))
    return br


def reference_line(ax, *, y=None, x=None, name, **kw):
    reg = _tagger.registry_for(ax.figure)
    if y is not None:
        ln = ax.axhline(y, **kw)
    elif x is not None:
        ln = ax.axvline(x, **kw)
    else:
        raise ValueError("reference_line needs x= or y=")
    reg.add(Mark(role="reference-line", series=None, name=name, artists=[ln], data={"x": x, "y": y}))
    return ln


def annotation(ax, *, name, text, **kw):
    reg = _tagger.registry_for(ax.figure)
    art = ax.annotate(text, **kw)
    reg.add(Mark(role="annotation", series=None, name=name, artists=[art], data={"text": text}))
    return art


# ---------------------------------------------------------------------------
# the export
# ---------------------------------------------------------------------------
@dataclass
class SaveResult:
    svg: str
    manifest: str
    recipe: str
    warnings: list = field(default_factory=list)


def _infer_plot_type(reg) -> str:
    kinds = [m.kind for m in reg.marks if m.kind]
    for k in ("line", "scatter", "bar", "errorbar", "area", "box", "violin"):
        if k in kinds:
            return k
    return "plot"


def _validate(manifest_obj, recipe_obj) -> None:
    try:
        import jsonschema
        from importlib.resources import files

        sch = files("fluxplot.schemas")
        mschema = json.loads((sch / "manifest.schema.json").read_text())
        rschema = json.loads((sch / "recipe.schema.json").read_text())
    except (FileNotFoundError, ModuleNotFoundError):
        return
    jsonschema.validate(manifest_obj, mschema)
    jsonschema.validate(recipe_obj, rschema)


def save(fig, path, *, recipe=None, validate=True, _now=None) -> SaveResult:
    """Emit ``<path>.svg`` + ``<path>.fluxplot.json`` + ``<path>.recipe.json`` for ``fig``.

    ``recipe`` controls the provenance sidecar:

    - ``None`` (default) — automatic: the producing script is discovered from the running
      interpreter when that is safe and exact, making the recipe rerunnable with zero ceremony.
    - ``False`` — explicitly suppress discovery (notebooks, generated figures, privacy).
    - ``dict`` — explicit fields (``script``/``command``/``params``/``inputs``) always win;
      an inferred script only fills a missing ``script``.
    """
    base, _ext = os.path.splitext(path)
    plot_name = os.path.basename(base)
    svg_path = base + ".svg"
    manifest_path = base + ".fluxplot.json"
    recipe_path = base + ".recipe.json"
    svg_filename = os.path.basename(svg_path)
    manifest_filename = os.path.basename(manifest_path)

    reg = _tagger.registry_for(fig)
    alloc = _ids.IdAllocator()

    # 0. promote safe labeled raw artists (identity = the user's own public labels; exact
    # artist data only — see autotag.py). Anything ambiguous stays for the extra.* sweep.
    promo_warnings = _autotag.promote_labeled(fig, reg)

    # 1. deterministic gids on the user-tagged marks
    _tagger.resolve_gids(reg, alloc)

    # 2. finalize layout (so ticks/labels exist + transforms are final), then auto-tag scaffold.
    # Colorbar axes (fig.colorbar adds a second Axes) are NOT the plot area: scaffolding them
    # produced duplicate "plot-area" capture entries and collided axis ids (axis.x-2/y-2). Skip
    # them here so the primary plot stays the single, clean plot-area.
    fig.canvas.draw()
    plot_axes = [ax for ax in fig.axes if not _is_colorbar_axes(ax)]
    guides = []
    for ax in plot_axes:
        guides.extend(_tagger.autotag_scaffold(ax, alloc))

    # 3. capture the data↔SVG mapping (after layout is final)
    axes_capture = []
    for ax in plot_axes:
        cap = {"id": "plot-area", "svgId": "plot-area"}
        cap.update(_capture.capture_axes(ax, fig))
        axes_capture.append(cap)

    # 4. render deterministically (hashsalt derived from the plot name)
    svg_bytes = _render.render_svg(fig, hashsalt=plot_name or "fluxplot")

    # 5. inject data-* + canonicalize
    plot_type = _infer_plot_type(reg)
    out_svg, warnings, present = _postprocess.postprocess(svg_bytes, reg, guides, plot_type)
    warnings = promo_warnings + warnings

    # 6. assemble manifest + recipe. Drop scaffold guides matplotlib culled at draw
    # (boundary ticks/gridlines, empty axis titles) so the manifest references only
    # parts that exist in the SVG — keeps the parts tree / group members honest.
    kept_guides = [g for g in guides if g.gid in present]
    man = _manifest.build_manifest(
        fig, reg, kept_guides, axes_capture, plot_type, svg_filename,
        SPEC_VERSION, __version__, matplotlib.__version__, present=present,
    )
    rec = _recipe.build_recipe(
        recipe, plot_name=plot_name, svg_filename=svg_filename,
        manifest_filename=manifest_filename, spec_version=SPEC_VERSION,
        recipe_dir=os.path.dirname(os.path.abspath(svg_path)), now=_now,
    )
    if validate:
        _validate(man, rec)

    # 7. write all three
    out_dir = os.path.dirname(os.path.abspath(svg_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(svg_path, "wb") as f:
        f.write(out_svg)
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(_cjson.dumps(man))
    with open(recipe_path, "w", encoding="utf-8") as f:
        f.write(_cjson.dumps(rec))

    return SaveResult(svg=svg_path, manifest=manifest_path, recipe=recipe_path, warnings=warnings)
