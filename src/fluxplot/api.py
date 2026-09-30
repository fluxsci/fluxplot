"""Public API — free functions over real matplotlib artists (spec §11.1).

Free functions (not a wrapped Axes) keep the user in full matplotlib (P2): the convenience helpers
and the ``tag()`` escape hatch are the same shape, and untagged matplotlib keeps working. ``save()``
orchestrates the pipeline: assign gids → draw → auto-tag scaffold → capture coords → render
deterministically → inject ``data-*`` → emit manifest + recipe.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import stat
import sys
import warnings
from dataclasses import dataclass, field

import matplotlib

from . import autotag as _autotag
from . import canonical_json as _cjson
from . import capture as _capture
from . import ids as _ids
from . import postprocess as _postprocess
from . import raster as _raster
from . import recipe as _recipe
from . import render as _render
from . import roles as _roles
from . import tagger as _tagger
from .descriptors import Mark
from . import data as _data
from . import panels as _panels
from types import SimpleNamespace
import numpy as np
from .version import SPEC_VERSION, __version__


def _list(a):
    return None if a is None else list(np.atleast_1d(a))


def _env_flag(name: str) -> bool:
    """A truthy environment switch: anything but unset/empty/0/false/no/off."""
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")


# ---------------------------------------------------------------------------
# convenience helpers (auto-tagging) — each returns the real matplotlib artist(s)
# ---------------------------------------------------------------------------
def _series_color_override(series):
    """The colour Flux asked this series to take on a rerun: ``__fluxplot__.series[<id>].color``
    (keyed by the series id, i.e. its slug), or ``None``."""
    controls = _recipe.params().get("__fluxplot__") or {}
    per_series = controls.get("series") or {}
    if not isinstance(per_series, dict):
        raise ValueError("FLUX_PARAMS __fluxplot__.series must map series ids to {'color': ...}")
    entry = per_series.get(_ids.series_root(series)) or per_series.get(str(series))
    if isinstance(entry, dict):
        return entry.get("color")
    return entry if isinstance(entry, str) else None


def _series_color(series, kw, *, key="color", auto=True):
    """Apply the series' colour rule to a helper's keywords: a recipe override wins; otherwise a
    colour comes from the category registry when ``fp.colors.categories.auto_series`` is on and
    the call gave none; otherwise the call's own (or matplotlib's cycle) stands."""
    from .colors import categories
    override = _series_color_override(series)
    if override is not None:
        kw.pop("c", None)
        kw[key] = override
    elif auto and categories.auto_series and key not in kw and "c" not in kw:
        kw[key] = categories.get(series)
    return kw


def line(ax, x, y, *, series, marker=None, label=None, **kw):
    """A named line. With ``marker=`` it also draws an addressable per-point group."""
    reg = _tagger.registry_for(ax.figure)
    _series_color(series, kw)
    (ln,) = ax.plot(x, y, label=label, **kw)
    reg.add(Mark(role="line", series=series, kind="line", live_data=True, x=None, y=None, label=label, artists=[ln]))
    if marker:
        # Copy the resolved Line2D style, including aliases/defaults, then disable
        # its stroke. This keeps the established (line, points) return contract.
        from matplotlib.lines import Line2D
        pts = Line2D([], [])
        pts.update_from(ln)
        pts.set_zorder(ln.get_zorder())
        pts.set_markevery(ln.get_markevery())
        pts.set_data(x, y)
        pts.set_linestyle("none")
        pts.set_marker(marker)
        pts.set_label("_nolegend_")
        ax.add_line(pts)
        reg.add(Mark(role="point", series=series, kind="line", live_data=True, x=None, y=None, artists=[pts], indexed=True))
        return ln, pts
    return ln


def _is_value_array(c, x) -> bool:
    """matplotlib's own rule for ``scatter(c=…)``: a 1-D numeric array as long as ``x`` is
    colour-MAPPED (through cmap and norm); anything else is a colour or a list of colours."""
    if c is None or isinstance(c, str):
        return False
    arr = np.asarray(c)
    return arr.ndim == 1 and arr.dtype.kind in "iuf" and arr.size == np.size(x)


def scatter(ax, x, y, *, series, label=None, key=None, scale=None, **kw):
    """A named scatter, every point addressable.

    With ``c=`` an array of values the points are colour-mapped: the scale becomes a recipe colour
    control (named ``key``, default the series), the manifest carries its exact lookup table, each
    point carries its value (``data-value``) and a :func:`colorbar` links to it — so Flux can
    recolour or re-range the points live.
    """
    reg = _tagger.registry_for(ax.figure)
    data: dict = {}
    if not _is_value_array(kw.get("c"), x):
        _series_color(series, kw)
    if _is_value_array(kw.get("c"), x):
        from . import fields as _fields
        from ._fieldmap import resolve_colormap
        ctl = _fields._options(ax, series, key, kw, scale=scale)
        extend = kw.pop("_extend", None)
        if isinstance(kw.get("cmap"), str):
            kw["cmap"] = resolve_colormap(kw["cmap"])
        config = {"kind": "scatter", "controlKey": ctl, "shape": [int(np.size(kw["c"]))]}
        if extend:
            config["extend"] = extend
        data = {"field_config": config}
    coll = ax.scatter(x, y, label=label, **kw)
    if data:
        data["field_artist"] = coll
        data["c"] = _data.values(coll.get_array())
        if scale is not None:
            _fields.join_scale(ax, scale, coll)
    sizes = coll.get_sizes()
    if len(sizes) > 1:
        data["size"] = _data.values(sizes)
    reg.add(Mark(role="point", series=series, kind="scatter", live_data=True, x=None, y=None, label=label,
                 artists=[coll], indexed=True, data=data))
    return coll


def bar(ax, x, height, *, series, label=None, **kw):
    reg = _tagger.registry_for(ax.figure)
    _series_color(series, kw, auto=False)
    container = ax.bar(x, height, label=label, **kw)
    reg.add(Mark(role="bar", series=series, kind="bar", live_data=True, x=None, y=None, label=label, artists=list(container.patches), indexed=True))
    return container


def barh(ax, y, width, *, series, label=None, **kw):
    """A horizontal named bar series; returns the ordinary BarContainer."""
    _series_color(series, kw, auto=False)
    container = ax.barh(y, width, label=label, **kw)
    _tagger.registry_for(ax.figure).add(Mark(
        role="bar", series=series, kind="bar", live_data=True, label=label,
        artists=list(container.patches), indexed=True,
        data={"bar": {"orientation": "horizontal"}}))
    return container


def _err_payload(err, n):
    """An errorbar's ``xerr`` / ``yerr`` broadcast to the N points, and its shape: ``scalar``
    (one value for all), ``symmetric`` (one per point) or ``asymmetric`` (``[lower, upper]``)."""
    if err is None:
        return None, None
    arr = np.asarray(err, dtype=float)
    if arr.ndim == 0:
        return [float(arr)] * n, "scalar"
    if arr.ndim == 1:
        return _data.values(np.broadcast_to(arr, (n,))), "symmetric"
    if arr.ndim == 2 and arr.shape[0] == 2:
        return [_data.values(np.broadcast_to(arr[0], (n,))), _data.values(np.broadcast_to(arr[1], (n,)))], "asymmetric"
    raise ValueError(f"errorbar: an error array must be a scalar, N values or a (2, N) array; got shape {arr.shape}")


def errorbar(ax, x, y, *, series, yerr=None, xerr=None, label=None, **kw):
    """Points with error bars, each part addressable: the data line (``<series>.line``, when the
    format draws one), the markers (``<series>.point.k``, when a marker is set), the caps
    (``<series>.cap``, …) and the bars (``<series>.errorbar``, …). ``uncertainty`` records
    ``xerr`` / ``yerr`` broadcast to the N points with ``errShape`` (scalar / symmetric /
    asymmetric)."""
    from matplotlib.lines import Line2D
    reg = _tagger.registry_for(ax.figure)
    _series_color(series, kw, auto=False)
    container = ax.errorbar(x, y, yerr=yerr, xerr=xerr, label=label, **kw)
    data_line, caps, barlinecols = container
    xs, ys = _data.converted(ax, x, y)
    n = len(xs)
    yv, shape = _err_payload(yerr, n)
    xv, xshape = _err_payload(xerr, n)
    uncertainty = {"xerr": xv, "yerr": yv, "errShape": shape or xshape}
    # the bars carry the series' data (compat: svg.errorbar keeps pointing at them)
    reg.add(Mark(role="errorbar", series=series, kind="errorbar", x=xs, y=ys, label=label,
                 artists=list(barlinecols), data={"uncertainty": uncertainty}))
    if caps:
        reg.add(Mark(role="cap", series=series, kind="errorbar", artists=list(caps)))
    if data_line is not None:
        has_line = str(data_line.get_linestyle()).lower() not in ("none", "", " ")
        has_marker = str(data_line.get_marker()).lower() not in ("none", "", " ")
        if has_marker:
            # markers on their own Line2D (as fp.line does) so each is a <use> with a point id;
            # the data line keeps its stroke and loses the marker
            pts = Line2D([], [])
            pts.update_from(data_line)
            pts.set_zorder(data_line.get_zorder())
            pts.set_data(data_line.get_xdata(orig=False), data_line.get_ydata(orig=False))
            pts.set_linestyle("none")
            pts.set_label("_nolegend_")
            ax.add_line(pts)
            reg.add(Mark(role="point", series=series, kind="errorbar", live_data=True, x=None, y=None,
                         artists=[pts], indexed=True))
            if has_line:
                data_line.set_marker("none")
        if has_line:
            reg.add(Mark(role="line", series=series, kind="errorbar", live_data=True, x=None, y=None,
                         artists=[data_line]))
        elif has_marker:
            data_line.set_visible(False)  # the copy draws the markers; nothing is drawn twice
    return container


def legend(ax, handles=None, labels=None, **kw):
    """``ax.legend`` with the entry → artist mapping kept, so the manifest can say which series
    each legend entry names even when the handles were chosen by hand
    (``fp.legend(ax, [ln], ["Control"])``). Without ``handles`` the axes' own labelled artists
    are used, in matplotlib's order, and the mapping is recovered from them at save."""
    if handles is None and labels is None:
        leg = ax.legend(**kw)
    elif handles is None:
        leg = ax.legend(labels, **kw)
    elif labels is None:
        leg = ax.legend(handles=handles, **kw)
    else:
        leg = ax.legend(handles, labels, **kw)
    leg._fluxplot_handles = list(handles) if handles is not None else None
    return leg


def _broadcast(v, n):
    arr = np.asarray(v, dtype=float)
    return _data.values(np.broadcast_to(arr, (n,)) if arr.ndim == 0 else arr)


def area(ax, x, y1, y2=0, *, series, label=None, **kw):
    """A filled region between ``y1`` and ``y2`` (default the baseline 0) — ``fill_between`` with
    its inputs recorded: the manifest's ``band`` carries ``{x, y1, y2}`` (a scalar ``y2`` is
    broadcast), not the polygon's vertices."""
    reg = _tagger.registry_for(ax.figure)
    _series_color(series, kw, auto=False)
    poly = ax.fill_between(x, y1, y2, label=label, **kw)
    xs = _data.values(x)
    payload = {"x": xs, "y1": _broadcast(y1, len(xs)), "y2": _broadcast(y2, len(xs))}
    reg.add(Mark(role="area", series=series, kind="area", x=None, y=None, label=label, artists=[poly],
                 data={"band": payload}))
    return poly


def band(ax, x, lo, hi, *, series, what="95% CI", label=None, **fill_kw):
    """An uncertainty band around a series' line: ``fill_between(x, lo, hi)`` registered under the
    **same series** as the line, so ``ctl.band`` sits beside ``ctl.line``.

    ``what`` says what the band is (``"95% CI"``, ``"SEM"``, ``"IQR"``, …) and is recorded with
    the inputs in the manifest: ``band = {x, lo, hi, what}``. The band takes the series' line
    colour when one is drawn already (else the call's, else the cycle's), at ``alpha=0.25`` with
    no edge unless the call says otherwise.
    """
    reg = _tagger.registry_for(ax.figure)
    fill_kw.setdefault("alpha", 0.25)
    fill_kw.setdefault("linewidth", 0)
    _series_color(series, fill_kw, auto=False)
    if not any(k in fill_kw for k in ("color", "facecolor", "fc", "c")):
        line = next((m for m in reg.marks if m.series == series and m.role == "line" and m.artists), None)
        if line is not None and hasattr(line.artists[0], "get_color"):
            fill_kw["color"] = line.artists[0].get_color()
    poly = ax.fill_between(x, lo, hi, label=label, **fill_kw)
    xs = _data.values(x)
    payload = {"x": xs, "lo": _broadcast(lo, len(xs)), "hi": _broadcast(hi, len(xs)), "what": str(what)}
    reg.add(Mark(role="area", series=series, name="band", kind="area", x=None, y=None, label=label,
                 artists=[poly], data={"band": payload}))
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
    data = {"distribution": {"values": _data.values(values)}} if include_values else {}
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
    data = {"distribution": {"values": _data.values(values)}} if include_values else {}
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
    _series_color(series, kw, auto=False)
    counts, edges, patches = ax.hist(values, bins=bins, label=label, **kw)
    if not isinstance(patches, BarContainer):
        raise ValueError(
            "fp.hist tags one dataset with the default bar histtype — multiple datasets or "
            "histtype='step' have no exact per-bar contract; call fp.hist once per series"
        )
    edges_f = [float(e) for e in edges]
    counts_f = _data.values(counts)
    centers = [(edges_f[k] + edges_f[k + 1]) / 2.0 for k in range(len(counts_f))]
    dist = {"binEdges": edges_f, "counts": counts_f,
            "normalization": "density" if kw.get("density") else "count",
            "weighted": kw.get("weights") is not None,
            "cumulative": kw.get("cumulative", False),
            "orientation": kw.get("orientation", "vertical")}
    if include_values:
        dist["values"] = _data.values(values)
    reg.add(
        Mark(role="bar", series=series, kind="bar", x=centers, y=counts_f, label=label,
             artists=list(patches.patches), indexed=True, live_data=True,
             data={"distribution": dist, "bar": {"orientation": kw.get("orientation", "vertical")}})
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
    live = x is None and y is None
    if live:
        x, y = _autotag.extract_xy(artist)
    data = dict(identity)
    if index is not None:
        data["index"] = index
    reg.add(
        Mark(role=_roles.validate(role), series=series, name=name, x=_list(x), y=_list(y),
             artists=[artist], data=data, live_data=live, indexed=index is not None)
    )
    return artist


def tag_points(points, *, series, x=None, y=None):
    """Tag an existing markers ``Line2D`` / ``PathCollection`` as an addressable point group."""
    reg = _tagger.registry_for(_tagger.fig_of(points))
    live = x is None and y is None
    if live:
        x, y = _autotag.extract_xy(points)  # Line2D data or exact finite scatter offsets
    reg.add(Mark(role="point", series=series, kind="scatter", live_data=live, x=_list(x), y=_list(y), artists=[points], indexed=True))
    return points


def tag_seaborn(ax, *, series=None, plot=None):
    """Auto-tag the artists a seaborn axes-level plot drew on ``ax`` — one call, done.

    Call it right after the seaborn call (and before raw-matplotlib additions you tag
    yourself). Series names come from ``series=[...]`` if given, else the legend's labels
    for a known ``plot=`` adapter, else the y-axis label. Multi-hue plots require
    ``plot='lineplot'`` (or ``histplot``, ``barplot``, ``kdeplot``, etc.) because
    legend order alone does not identify Seaborn's drawing order. An explicit
    ``series=[...]`` always means artist draw order. Per series it names:

    - data-carrying lines           → role ``line``      (``lineplot`` means, ``kdeplot``, ``regplot`` fits)
    - fill-between bands            → role ``area``      (confidence / error bands)
    - scatter collections           → role ``point``     (``scatterplot`` / ``regplot`` — per-point addressable)
    - bar containers                → role ``bar``       (``barplot`` / ``countplot`` / ``histplot``)
    - capped/horizontal bar errors  → role ``errorbar``  (joined by categorical position)

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

    legend = ax.get_legend()
    legend_texts = {t.get_text() for t in legend.get_texts()} if legend is not None else set()
    # seaborn appends empty proxy lines used only to build its legend — drop them so the orphan
    # sweep doesn't dutifully tag invisible leftovers. Only seaborn's own are removed: an empty
    # line is a proxy when its label is private (``_…``) or is one of the legend's entries; an
    # empty line the user drew with a public label of their own is left alone.
    for ln in [ln for ln in ax.lines if len(ln.get_xdata()) == 0]:
        lbl = str(ln.get_label())
        if lbl.startswith("_") or lbl in legend_texts:
            ln.remove()

    if series is not None:
        names = [str(s) for s in series]
    elif legend is not None and legend.get_texts():
        names = [t.get_text() for t in legend.get_texts()]
    else:
        # a FacetGrid facet has no legend of its own: its title names the facet
        names = [ax.get_ylabel() or ax.get_title() or "panel"]

    # Seaborn deliberately reverses hue iteration for distribution plots.
    # A legend does not encode this provenance. Require the plot kind for a
    # multi-hue adapter instead of inferring identity from color or geometry.
    if plot not in (None, "lineplot", "scatterplot", "barplot", "countplot", "histplot", "kdeplot", "regplot", "heatmap"):
        raise ValueError("unsupported seaborn plot kind")
    tagged: dict = {}
    if plot in ("heatmap", "histplot", "kdeplot"):
        # colour-mapped fields: sns.heatmap's mesh, a bivariate histplot's mesh, a bivariate
        # kdeplot's contour set — each becomes an fp.heatmap / fp.contour-style field mark whose
        # colour scale is recorded and whose colour key links (the seaborn call has already drawn,
        # so recipe colour controls are recorded for the editor but cannot be applied on a rerun)
        _tag_seaborn_fields(ax, reg, already, [str(s) for s in series] if series is not None else None, tagged)
        if plot == "heatmap":
            return tagged
    if len(names) > 1 and series is None and plot is None:
        warnings.warn("tag_seaborn: multi-hue identity needs plot='lineplot', 'histplot', "
                      "'barplot', etc.; ambiguous artists remain addressable as extras", stacklevel=2)
        return {}
    if plot in ("histplot", "kdeplot") and series is None:
        names.reverse()

    def _record(name, role):
        tagged.setdefault(str(name), []).append(role)

    bar_context = plot in ("barplot", "countplot") or bool(ax.containers)

    def _is_segment(ln):
        """A loose uncertainty line seaborn drew for a bar: axis-aligned two-point runs only.

        seaborn draws an error bar as ``[x, x] × [lo, hi]`` and, with ``capsize``, joins the
        two caps to it through NaN breaks (``cap, NaN, bar, NaN, cap``). Every finite run of such
        a line has exactly two points sharing an x (vertical bars) or a y (horizontal bars).
        Anything else — a KDE curve over a histogram, a fit line, a lineplot mean — is a data
        curve, whatever else sits on the axes.
        """
        if not bar_context:
            return False
        x, y = (np.asarray(v, dtype=float) for v in ln.get_data(orig=False))
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            return False
        runs, start = [], None
        for i, ok in enumerate(list(finite) + [False]):
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                runs.append((start, i))
                start = None
        for a, b in runs:
            if b - a != 2:
                return False
            if not (abs(x[a] - x[a + 1]) <= 1e-12 or abs(y[a] - y[a + 1]) <= 1e-12):
                return False
        return True


    from .colors import categories as _categories

    def _pin(name, artist, setter):
        """A hue level pinned in fp.colors.categories (or a recipe series colour) recolours its artist."""
        colour = _series_color_override(name)
        if colour is None and _categories.is_pinned(name):
            colour = _categories.get(name)
        if colour is not None:
            setter(artist, colour)

    # data curves (one per hue level) → line
    curves = [ln for ln in ax.lines if id(ln) not in already and len(ln.get_xdata()) and not _is_segment(ln)]
    if curves and len(curves) == len(names):
        for name, ln in zip(names, curves):
            _pin(name, ln, lambda a, c: a.set_color(c))
            gx, gy = ln.get_data()
            reg.add(Mark(role="line", series=name, kind="line", x=_list(gx), y=_list(gy), live_data=True, label=name, artists=[ln]))
            _record(name, "line")

    # fill-between bands (one per hue level) → area
    bands = [c for c in ax.collections if id(c) not in already and isinstance(c, PolyCollection)]
    if bands and len(bands) == len(names):
        for name, band in zip(names, bands):
            reg.add(Mark(role="area", series=name, kind="area", label=name, artists=[band]))
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
        reg.add(Mark(role="point", series=name, kind="scatter", x=x, y=y, artists=[coll], indexed=True, live_data=True))
        _record(name, "point")

    # bar containers (one per hue level) → bar, with bar centers/heights as the data
    bars = [c for c in getattr(ax, "containers", []) if isinstance(c, BarContainer)]
    bar_centers: list = []  # (name, {x center}) for the error-bar join below
    if bars and len(bars) == len(names):
        for name, cont in zip(names, bars):
            patches = [p for p in cont.patches if id(p) not in already]
            if not patches:
                continue
            for patch in patches:
                _pin(name, patch, lambda a, c: a.set_facecolor(c))
            orientation = getattr(cont, "orientation", "vertical")
            cx, cy, meta = _data.bar_data(patches, orientation)
            reg.add(Mark(role="bar", series=name, kind="bar", x=cx, y=cy, label=name,
                         live_data=True, artists=patches, indexed=True, data={"bar": meta}))
            _record(name, "bar")
            bar_centers.append((name, orientation, {round(v, 9) for v in (cy if orientation == "horizontal" else cx)}))

    # seaborn draws bar errors as loose 2-point vertical lines — join each to its bar
    # by x position (an exact join on coordinates seaborn itself set, not a guess).
    segs = [ln for ln in ax.lines if id(ln) not in already and len(ln.get_xdata()) and _is_segment(ln)]
    for ln in segs:
        matches = []
        for name, orientation, centers in bar_centers:
            positions = np.asarray(ln.get_ydata() if orientation == "horizontal" else ln.get_xdata(), dtype=float)
            center = round(float(np.nanmedian(positions)), 9)
            if center in centers:
                matches.append(name)
        if len(matches) == 1:
            name = matches[0]
            reg.add(Mark(role="errorbar", series=name, kind="errorbar", artists=[ln]))
            _record(name, "errorbar")

    return tagged


def _tag_seaborn_fields(ax, reg, already, names, tagged) -> None:
    from matplotlib.collections import QuadMesh
    from matplotlib.contour import ContourSet

    from . import fields as _fields
    from . import raster as _raster

    mappables = [c for c in ax.collections if id(c) not in already
                 and (isinstance(c, QuadMesh) or (isinstance(c, ContourSet) and c.get_array() is not None))]
    for k, art in enumerate(mappables):
        name = names[k] if names is not None and k < len(names) else ("heatmap" if k == 0 else f"heatmap-{k}")
        key, _legacy = _fields.control_key(ax, name)
        if isinstance(art, QuadMesh):
            arr = np.ma.asarray(art.get_array())
            shape = list(arr.shape) if arr.ndim == 2 else [int(arr.size), 1]
            config = {"kind": "heatmap", "shape": shape, "includeValues": False, "controlKey": key}
            reg.add(Mark(role="x-heatmap", series=name, kind="heatmap", artists=[art],
                         data={"field_config": config, "field_artist": art,
                               "cells": int(arr.size) <= _raster.DEFAULT_THRESHOLD}))
            tagged.setdefault(name, []).append("x-heatmap")
        else:
            config = {"kind": "contourf" if art.filled else "contour", "levels": _data.values(art.levels),
                      "extend": art.extend, "controlKey": key}
            reg.add(Mark(role="x-contourf" if art.filled else "x-contour", series=name, kind=config["kind"],
                         artists=[art], axes=ax,
                         data={"field_config": config, "field_artist": art, "contour_paths": True}))
            tagged.setdefault(name, []).append(config["kind"])
        already.add(id(art))


# ---------------------------------------------------------------------------
# first-class overlays
# ---------------------------------------------------------------------------
def significance_bracket(ax, *, x0, x1, y, label, between=None, p=None, name=None, height=None,
                         color=None, text_kw=None, stats=None, **kw):
    """A p-value bracket (spec §5: deliberately first-class — ubiquitous in science).

    ``color`` paints both the bracket line and its label (default: the theme's text colour,
    so a bracket reads on dark grounds too); ``text_kw`` are extra ``Text`` properties for the
    label (``fontsize``, ``fontweight``, …). ``height`` is the tip height in data units
    (default: 3 % of the y range, or a 6 % step on a log axis). ``stats`` is the provenance of
    the label — the test, statistic, p-values, effect size — recorded on the manifest overlay
    (:func:`fluxplot.brackets` fills it from an ``fp.stats`` row).
    """
    reg = _tagger.registry_for(ax.figure)
    idx = reg.next_overlay_index("significance-bracket")
    if name is None:
        name = str(idx)
    if ax.get_yscale() == "log" and y <= 0:
        raise ValueError(f"significance_bracket: y must be positive on a log axis (got y={y!r})")
    if height is None:
        if ax.get_yscale() == "log":
            ytop = y * 1.06
        else:
            lo, hi = ax.get_ylim()
            ytop = y + (hi - lo) * 0.03
    else:
        ytop = y + height
    themed = color is None
    if themed:
        color = matplotlib.rcParams["text.color"]
    (br,) = ax.plot([x0, x0, x1, x1], [y, ytop, ytop, y], color=color,
                    linewidth=kw.pop("linewidth", 1.0), **kw)
    txt = ax.text((x0 + x1) / 2.0, ytop, label, ha="center", va="bottom", color=color,
                  **(text_kw or {}))
    data = {"label": label, "index": idx, "label_artist": txt}
    if themed:  # drawn in the theme's ink: say so outright (B1), no colour comparison needed
        data["ink"] = {"stroke": "ink"}
        data["ink_label"] = {"fill": "ink"}
    if between is not None:
        data["between"] = list(between)
    if p is not None:
        data["p"] = p
    if stats:
        data["stats"] = dict(stats)
    reg.add(Mark(role="significance-bracket", series=None, name=name, artists=[br], data=data))
    br._fluxplot_label = txt
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
    #: True when FLUXPLOT_ONLY filtered this plot out — nothing was written.
    skipped: bool = False
    #: gids of the layers auto-rasterized on the way out (see ``raster.py``). Empty on an
    #: ordinary save; empty too under ``force_vectors=True``, where the heavy layers are
    #: instead reported through :attr:`warnings`.
    rasterized: list = field(default_factory=list)


def _warn_log_zero_anchors(plot_axes, plot_name: str) -> list:
    """Warn — at generation time, where the fix belongs — when a bar/rect on a
    log-scaled axis extends to data ≤ 0. matplotlib serializes that anchor as a
    huge off-canvas SVG coordinate (−50k…−200k in a ~400-unit canvas): the plot
    renders standalone, but downstream compositors/rasterizers can crash on it
    (flux `validate-plot` now rejects it). The remedy is one line in the plot
    script: anchor at a positive value (barh: left=1, bar: bottom=1)."""
    from matplotlib.patches import Rectangle

    out = []
    for ax in plot_axes:
        logx = ax.get_xscale() == "log"
        logy = ax.get_yscale() == "log"
        if not (logx or logy):
            continue
        bad = 0
        for pt in ax.patches:
            if not isinstance(pt, Rectangle):
                continue
            if logx and min(pt.get_x(), pt.get_x() + pt.get_width()) <= 0:
                bad += 1
            elif logy and min(pt.get_y(), pt.get_y() + pt.get_height()) <= 0:
                bad += 1
        if bad:
            msg = (
                f"{plot_name}: {bad} bar(s)/rect(s) on a log-scaled axis extend to data <= 0; "
                "matplotlib serializes those anchors as huge off-canvas coordinates that can "
                "crash downstream renderers. Anchor at a positive value instead "
                "(barh: left=1, width=count-1; bar: bottom=1, height=count-1)."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            out.append(msg)
    return out


def _infer_plot_type(reg) -> str:
    kinds = [m.kind for m in reg.marks if m.kind]
    for k in ("glowbar", "fluxbox", "hexmatrix", "image", "line", "scatter", "bar", "errorbar", "area", "box", "violin", "heatmap", "contour", "contourf", "surface"):
        if k in kinds:
            return k
    return "plot"


def _validate(manifest_obj, recipe_obj) -> None:
    try:
        import jsonschema
        from importlib.resources import files

        sch = files("fluxplot").joinpath("schemas")
        mschema = json.loads((sch / "manifest.schema.json").read_text())
        rschema = json.loads((sch / "recipe.schema.json").read_text())
    except (FileNotFoundError, ModuleNotFoundError):
        return
    jsonschema.validate(manifest_obj, mschema)
    jsonschema.validate(recipe_obj, rschema)


def _write_staged(files) -> None:
    """Write the sidecar triplet safely: stage EVERY file under a unique temporary name in its
    destination directory (flush + fsync), then commit each with ``os.replace`` in the given
    dependency order. A failure while staging leaves the destination completely untouched, and
    a watcher can never observe a partially written individual file. The three replacements are
    still not globally atomic — the manifest's ``artifact.svgSha256`` is the cross-file commit
    marker a consumer verifies (plan §5). Existing destination permissions are preserved.
    """
    staged: list[tuple[str, str]] = []
    try:
        for path, data in files:
            tmp = f"{path}.{os.getpid()}.staging"
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.chmod(tmp, stat.S_IMODE(os.stat(path).st_mode))
            except OSError:
                pass  # new file → default creation mode
            staged.append((tmp, path))
        for tmp, path in staged:
            os.replace(tmp, path)
    finally:
        for tmp, _ in staged:  # clean whatever a failure left behind (replaced tmps are gone)
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _save(
    fig,
    path,
    *,
    recipe=None,
    validate=True,
    force_vectors=False,
    raster_threshold=None,
    raster_dpi=None,
    theme_vars=False,
    lint="off",
    _now=None,
    _registry=None,
) -> SaveResult:
    """Emit ``<path>.svg`` + ``<path>.fluxplot.json`` + ``<path>.recipe.json`` for ``fig``.

    ``fig`` may also be a 3D scene from :func:`fp.scene3d <fluxplot.scene3d>`: it then writes
    ``<path>.glb`` + ``<path>.fluxplot.json`` (the scene3d manifest) + ``<path>.recipe.json``,
    always at full mesh resolution (``preview_max_faces`` only affects notebook previews).
    ``path`` is a stem or ends in ``.glb``. ``recipe``, ``validate`` and ``FLUXPLOT_ONLY``
    behave as below; the SVG rasterization options do not apply.

    ``recipe`` controls the provenance sidecar:

    - ``None`` (default) — automatic: the producing script is discovered from the running
      interpreter when that is safe and exact, making the recipe rerunnable with zero ceremony.
    - ``False`` — explicitly suppress discovery (notebooks, generated figures, privacy).
    - ``dict`` — explicit fields (``script``/``command``/``params``/``inputs``) always win;
      an inferred script only fills a missing ``script``.

    **Heavy layers are rasterized by default.** An artist that draws more than
    ``raster_threshold`` primitives (default 800) — a ``LineCollection`` of per-edge
    segments, a 10k-point ``scatter`` — would otherwise become that many live SVG nodes and
    make any editor that inlines the plot unusable. Such layers are rendered to a single
    embedded ``<image>`` at ``raster_dpi`` (default 600) while axes, ticks, tick labels,
    legend, annotations and every lighter artist stay fully vector, and ``save`` says on
    stderr exactly what it rasterized. The layers keep their ids, ``data-role``/``data-series``
    and manifest entries, so they remain addressable as a whole; only *per-point* ids are
    unavailable (a rasterized cloud has no per-point nodes). ``SaveResult.rasterized`` lists
    what was rasterized.

    Set ``force_vectors=True`` (or the ``FLUXPLOT_FORCE_VECTORS`` environment variable) to
    keep everything vector — a per-artist ``set_rasterized(False)`` does *not* override the
    safety default, because that is matplotlib's silent factory setting rather than a
    considered choice. Under ``force_vectors`` the heavy layers are still reported, as a
    warning naming them and their node cost.

    **Themes.** The manifest records ``style = {"theme", "tokens"}`` — the scaffold colours in
    force at save (ink, label, tick, axis, grid, plot, paper) and the ``fx.use_*`` theme they
    came from — and every scaffold element painted with one of them carries ``data-ink-fill`` /
    ``data-ink-stroke`` naming the token, so a consumer can restyle the furniture to its own
    theme without touching a data colour. ``theme_vars=True`` additionally rewrites those paints
    to ``var(--fx-<token>, <hex>)``, which a CSS-aware host can drive directly (off by default:
    the fallback form is honoured by browsers and rsvg, still to be checked in Illustrator /
    Inkscape). A rerun with ``FLUX_PARAMS={"__fluxplot__": {"theme": "dark"}}`` makes every
    ``fx.use_*`` call apply that theme instead.

    **Accessibility lint** (``lint="warn"`` / ``"error"``, default ``"off"``): the tagged series'
    colours and the text inks are checked against one another and the ground
    (:mod:`fluxplot.colorcheck` — colour-vision deficiency, greyscale, WCAG contrast). Findings
    land in ``SaveResult.warnings`` and in the manifest's ``quality.color``; ``"error"`` refuses
    the save with them.

    **Targeted reruns** (``FLUXPLOT_ONLY``): a figure-level script that saves
    several plots can be re-run for ONE of them — set ``FLUXPLOT_ONLY`` to a
    comma-separated list of plot names (fnmatch patterns work: ``fig2*``) and
    every non-matching ``save`` becomes a no-op (nothing written, siblings
    untouched on disk). ``flux rerun-plot <recipe> --only`` sets this for you.
    """
    keep_vectors = bool(force_vectors) or _env_flag("FLUXPLOT_FORCE_VECTORS")
    threshold = _raster.DEFAULT_THRESHOLD if raster_threshold is None else int(raster_threshold)
    raster_dpi = _raster.DEFAULT_DPI if raster_dpi is None else int(raster_dpi)

    base, _ext = os.path.splitext(path)
    plot_name = os.path.basename(base)
    svg_path = base + ".svg"
    manifest_path = base + ".fluxplot.json"
    recipe_path = base + ".recipe.json"
    svg_filename = os.path.basename(svg_path)
    manifest_filename = os.path.basename(manifest_path)


    reg = _registry if _registry is not None else _tagger.snapshot(fig)
    alloc = _ids.IdAllocator()
    from . import fields as _fields

    # a value raster (fp.heatmap(value_raster=True)) is named before capture, so the colour scale
    # can point at the sidecar it travels with
    for m in reg.marks:
        if m.data.get('value_raster') is not None and m.data.get('field_config'):
            m.data['value_raster']['filename'] = f"{plot_name}.{m.data['field_config']['controlKey']}.values.json"
        for vr in m.data.get('value_rasters') or []:  # fp.image: one sidecar per channel
            vr['filename'] = f"{plot_name}.{vr['key']}.values.json"
            for rec in m.data.get('color_scales') or []:
                if rec['id'] == vr['key']:
                    rec['valueRaster'] = vr['filename']

    panels = _panels.plan(fig)
    promo_warnings, guides_by_panel, axes_capture, scales_by_panel = [], [], [], []
    registered = {id(m) for m in reg.marks}
    for i, panel in enumerate(panels):
        ax = panel.axes
        # Colorbar marks belong to the plot whose mappable produced their key.
        members = []
        for m in reg.marks:
            owner = getattr(m.axes, "_fluxplot_owner_axes", m.axes)
            if owner is ax or (owner is None and i == 0):
                m._panel_axes = ax
                members.append(m)
        sub = _tagger.Registry()
        for m in members:
            sub.add(m)
        promo_warnings.extend(_autotag.promote_labeled(SimpleNamespace(axes=[ax]), sub))
        for m in sub.marks:
            m._panel_axes = ax
            _data.refresh(m)
            if id(m) not in registered:
                reg.marks.append(m)
                registered.add(id(m))
        scoped = _panels.ScopedAllocator(alloc, panel.prefix)
        _tagger.resolve_gids(sub, scoped)
        ax.set_gid(panel.svg_id)
        guides_by_panel.append(_tagger.autotag_scaffold(ax, scoped) + _fields.colorbar_guides(fig, ax, scoped))
        # raw colour-mapped artists (an imshow, a pcolormesh, a scatter with c=) the sweep just
        # named get an anonymous colour scale: no series, but a linked key and editable colours
        scales_by_panel.append(_fields.anonymous_scales(ax, reg))
        axes_capture.append({"id": "plot-area", "svgId": "plot-area", **_capture.capture_axes(ax, fig)})
    plot_axes = [p.axes for p in panels]
    guides = [g for local in guides_by_panel for g in local]
    geometry_warnings = _warn_log_zero_anchors(plot_axes, plot_name)

    # accessibility lint (B3): findings travel with the plot; "error" refuses it outright
    lint_findings, lint_warnings = [], []
    if lint not in ("off", "warn", "error"):
        raise ValueError("lint must be 'off', 'warn' or 'error'")
    if lint != "off":
        from .colorcheck import check_figure
        lint_findings = [f.as_dict() for f in check_figure(fig)]
        lint_warnings = [f"{plot_name}: colour lint: {f['message']}" for f in lint_findings]
        if lint == "error" and lint_findings:
            raise ValueError("colour lint failed:\n  " + "\n  ".join(f["message"] for f in lint_findings))
        for msg in lint_warnings:
            warnings.warn(msg, UserWarning, stacklevel=3)

    # 3b. auto-rasterize pathologically heavy layers — the safety default. A LineCollection of
    # per-edge segments or a 10k-point scatter becomes one <image> instead of 10^4-10^5 SVG
    # nodes, while axes/ticks/labels/legend stay vector (see raster.py). Planned AFTER the
    # scaffold sweep so untagged-but-heavy artists are named first, and applied around the
    # render only — the user's figure is handed back exactly as they built it.
    heavy = _raster.plan(fig, threshold)
    raster_items = [] if keep_vectors else heavy
    raster_warnings = []
    # 4. render deterministically (hashsalt derived from the plot name)
    with _raster.rasterizing(fig, heavy, force_vectors=keep_vectors), _fields.vector_colorbars(fig):
        svg_bytes = _render.render_svg(
            fig,
            hashsalt=plot_name or "fluxplot",
            dpi=raster_dpi if raster_items else None,
        )

    # 5. inject data-* + canonicalize
    plot_type = _infer_plot_type(reg)
    extra_scales = [sc for local in scales_by_panel for sc in local]
    from . import style as _style
    style_record = _style.theme_record()
    out_svg, post_warnings, present = _postprocess.postprocess(
        svg_bytes, reg, guides, plot_type, raster_items=raster_items, check_ids=validate,
        extra_scales=extra_scales, style_tokens=style_record["tokens"], theme_vars=theme_vars,
    )
    rendered_heavy = [it for it in heavy if it.gid in present]
    if rendered_heavy:
        note = _raster.describe(rendered_heavy, plot_name=plot_name, dpi=raster_dpi,
                                rasterized=not keep_vectors)
        raster_warnings.append(note)
        print(note, file=sys.stderr)
    persistent = _tagger.registry_for(fig)
    tag_warnings, persistent.warnings = list(persistent.warnings), []
    all_warnings = promo_warnings + tag_warnings + geometry_warnings + lint_warnings + raster_warnings + post_warnings

    # 6. assemble manifest + recipe. Drop scaffold guides matplotlib culled at draw
    # (boundary ticks/gridlines, empty axis titles) so the manifest references only
    # parts that exist in the SVG — keeps the parts tree / group members honest.
    kept_guides = [[g for g in local if g.gid in present or (g.virtual and any(other.axis == g.axis and other.gid in present for other in local))]
                   for local in guides_by_panel]
    rasterized_gids = {it.gid for it in raster_items if it.gid and it.gid in present}
    man = _panels.manifest(
        fig, reg, kept_guides, panels, axes_capture, present, rasterized_gids,
        extra_scales_by_panel=scales_by_panel, style=style_record,
        quality={"color": lint_findings} if lint != "off" else None,
        plot_type=plot_type, svg_filename=svg_filename, spec_version=SPEC_VERSION,
        fluxplot_version=__version__, mpl_version=matplotlib.__version__,
        svg_sha256=hashlib.sha256(out_svg).hexdigest(),
    )
    rec = _recipe.build_recipe(
        recipe, plot_name=plot_name, svg_filename=svg_filename,
        manifest_filename=manifest_filename, spec_version=SPEC_VERSION,
        recipe_dir=os.path.dirname(os.path.abspath(svg_path)), now=_now,
    )
    # the complete current state of every colour scale (colorscale.controls_state): what the Flux
    # editor starts from, and what a rerun replays byte for byte
    from .colorscale import controls_state
    controls = {m.data['field']['controlKey']: controls_state(m.data['field'])
                for m in reg.marks if m.data.get('field')}
    for m in reg.marks:  # fp.image: one control per channel
        controls.update(m.data.get('color_controls') or {})
    if style_record["theme"] is not None:  # the theme is a recipe control too (__fluxplot__.theme)
        controls["theme"] = style_record["theme"]
    if controls:
        rec['params'] = {**rec['params'], '__fluxplot__': controls}
    if validate:
        from .integrity import validate_references
        validate_references(man, present)
        _validate(man, rec)

    # 7. stage all three, then commit in dependency order (SVG → manifest → recipe): the
    # manifest checksum is the commit marker a consumer verifies against the SVG it sees.
    out_dir = os.path.dirname(os.path.abspath(svg_path))
    os.makedirs(out_dir, exist_ok=True)
    # value rasters sit between the SVG and the manifest that references them
    value_files = [(os.path.join(out_dir, m.data['value_raster']['filename']),
                    _cjson.dumps(m.data['value_raster']['payload']).encode("utf-8"))
                   for m in reg.marks if m.data.get('value_raster') and m.data['value_raster'].get('payload')]
    value_files += [(os.path.join(out_dir, vr['filename']), _cjson.dumps(vr['payload']).encode("utf-8"))
                    for m in reg.marks for vr in (m.data.get('value_rasters') or [])]
    _write_staged(
        [
            (svg_path, out_svg),
            *value_files,
            (manifest_path, _cjson.dumps(man).encode("utf-8")),
            (recipe_path, _cjson.dumps(rec).encode("utf-8")),
        ]
    )

    return SaveResult(
        svg=svg_path,
        manifest=manifest_path,
        recipe=recipe_path,
        warnings=all_warnings,
        rasterized=sorted(rasterized_gids),
    )


def save(fig, path, *, recipe=None, validate=True, force_vectors=False,
         raster_threshold=None, raster_dpi=None, theme_vars=False, lint="off", _now=None) -> SaveResult:
    from .scene3d import Scene3D
    if isinstance(fig, Scene3D):
        from .scene3d_manifest import save_scene3d
        return save_scene3d(fig, path, recipe=recipe, validate=validate, _now=_now)
    base, _ext = os.path.splitext(path)
    plot_name = os.path.basename(base)
    svg_path, manifest_path, recipe_path = base + '.svg', base + '.fluxplot.json', base + '.recipe.json'
    only = os.environ.get("FLUXPLOT_ONLY", "").strip()
    if only:
        pats = [p.strip() for p in only.split(",") if p.strip()]
        if pats and not any(fnmatch.fnmatchcase(plot_name, p) for p in pats):
            print(f"fluxplot: skipped '{plot_name}' (FLUXPLOT_ONLY={only})", file=sys.stderr)
            return SaveResult(
                svg=svg_path, manifest=manifest_path, recipe=recipe_path,
                warnings=[f"skipped by FLUXPLOT_ONLY={only}"], skipped=True,
            )

    from .fields import resolve_scales
    resolve_scales(fig)  # shared scales take the union of their members' values before layout
    reg = _tagger.snapshot(fig)
    with _tagger.temporary_gids(fig, reg), _render.final_layout(fig):
        return _save(fig, path, recipe=recipe, validate=validate, force_vectors=force_vectors,
                     raster_threshold=raster_threshold, raster_dpi=raster_dpi, theme_vars=theme_vars,
                     lint=lint, _now=_now, _registry=reg)


save.__doc__ = _save.__doc__
