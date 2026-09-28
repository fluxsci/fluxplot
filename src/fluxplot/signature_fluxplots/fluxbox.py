"""``fp.fluxbox`` — the box plot, drawn the FluxPlot way.

A fluxbox is a glowbar whose summary mark is a box plot. Every observation is a dot and, beside each
group, sits a slim box; everything else — the fixed per-unit lanes and colours, the paired
connectors, the names — is the glowbar's own machinery, so a fluxbox and a glowbar of the same table
agree dot for dot and can be swapped for one another. Four statistics are read straight off the box:

* the **interquartile range** (Q1–Q3) — the box, a half-strength wash of the group colour;
* the **median** — a solid line across the box in the group's own hue, deepened (on a dark
  background: lifted) just as far as it takes to stand clearly off the box, so it reads for any
  palette and any theme. That colour is the box plot's one *ink*: the whiskers (and fliers) share it;
* the **mean** — a V-notch cut into both edges of the box (the glowbar's median notch). A mean that
  falls outside the box (a strongly skewed group) keeps its mark: the same two V's, drawn solid in
  the ink, pointing in at the whisker;
* the **whiskers** — plain capless lines in the median's colour, reaching the most extreme
  observations within ``whis`` × IQR of the box (Tukey's rule — exactly ``plt.boxplot``'s
  whiskers), the full range, or a pair of percentiles.

Observations beyond the whiskers are not drawn a second time as fliers — the individual points
already show them. With ``show_individual_points=False`` they are (``show_fliers="auto"``).

Everything drawn is a named part, so a fluxbox round-trips through Flux like any other FluxPlot:

=======================  ===============================================  ===========
part                     default id                                       role
=======================  ===============================================  ===========
box (Q1–Q3)              ``<category>.box``                               ``box``
whiskers                 ``<category>.whiskers``                          ``whisker``
whisker caps (opt-in)    ``<category>.caps``                              ``cap``
median line              ``<category>.median``                            ``median``
mean notch               ``<category>.mean``                              ``mean``
fliers (points hidden)   ``<category>.fliers``                            ``flier``
a unit's point(s)        ``<unit>.points``, ``<unit>.point.<k>``          ``point``
a unit's connector       ``<unit>.line``                                  ``line``
points (no ``units``)    ``<category>.points``, ``<category>.point.<k>``  ``point``
=======================  ===============================================  ===========

Each category series carries a ``fluxbox`` payload in the manifest with the exact statistics drawn
(n, mean, median, sd, sem, q1, q3, iqr, the whisker rule and ends, the outliers, the group colour),
and each unit series carries its identity, exactly as for the glowbar.

Example
-------
>>> import fluxplot as fp, matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(figsize=(1.6, 1.8))
>>> fb = fp.fluxbox(df, x="condition", y="APP/GAPDH", units="subject", ax=ax)
>>> fb.stats["SD"]["whiskerHigh"]
3.47
>>> fp.save(fig, "plots/app_gapdh.svg")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Optional, Sequence, Union

import numpy as np

from .. import tagger as _tagger
from ..descriptors import Mark
from ._colour import perceptual as _perceptual
from .glowbar import (
    _SIDES,
    CONNECT_GREY,
    _cut_colour,
    _draw_units,
    _finish,
    _frame,
    _hex,
    _notch_marker,
    _plain,
)

__all__ = ["fluxbox", "FluxboxResult"]

# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _whisker_rule(whis):
    """Validate ``whis`` into its manifest form: a factor of the IQR, ``"range"`` or ``[lo, hi]``."""
    def number(v):
        return isinstance(v, Real) and not isinstance(v, bool)

    if isinstance(whis, str) and whis == "range" or (number(whis) and np.isinf(whis) and whis > 0):
        return "range"
    if number(whis) and whis >= 0:
        return float(whis)
    if isinstance(whis, (list, tuple, np.ndarray)) and len(whis) == 2 and all(map(number, whis)):
        lo, hi = float(whis[0]), float(whis[1])
        if 0.0 <= lo <= hi <= 100.0:
            return [lo, hi]
    raise ValueError("fluxbox: whis must be a non-negative factor of the IQR, 'range', or a "
                     f"(low, high) pair of percentiles; got {whis!r}")


def _box_stats(vals, rule):
    """One group's box-plot statistics. Quartiles and whiskers follow ``plt.boxplot`` exactly."""
    v = vals[np.isfinite(vals)]
    n = int(v.size)
    if n == 0:
        return None
    mean, median = float(v.mean()), float(np.median(v))
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    sem = sd / float(np.sqrt(n)) if n > 1 else float("nan")
    q1, q3 = (float(q) for q in np.percentile(v, [25, 75]))
    if rule == "range":
        reach_lo, reach_hi = -np.inf, np.inf
    elif isinstance(rule, list):
        reach_lo, reach_hi = np.percentile(v, rule)
    else:
        reach_lo, reach_hi = q1 - rule * (q3 - q1), q3 + rule * (q3 - q1)
    # a whisker ends ON an observation — the most extreme one within reach — and never inside the box
    below, above = v[v >= reach_lo], v[v <= reach_hi]
    w_lo = float(below.min()) if below.size and below.min() <= q1 else q1
    w_hi = float(above.max()) if above.size and above.max() >= q3 else q3
    outliers = sorted(float(o) for o in v[(v < w_lo) | (v > w_hi)])
    return {"n": n, "mean": mean, "median": median, "sd": sd, "sem": sem, "q1": q1, "q3": q3,
            "iqr": q3 - q1, "whis": rule, "whiskerLow": w_lo, "whiskerHigh": w_hi,
            "outliers": outliers}


def _manifest_value(v):
    return [_plain(e) for e in v] if isinstance(v, list) else _plain(v)


def _median_ink(col, box_alpha, ground, contrast):
    """The group colour, deepened — or, on a ground darker than it, lifted — only as far as it takes
    to stand ``contrast`` lightness units (0–100) off the box as rendered over ``ground``."""
    from matplotlib.colors import to_rgba
    rgb, bg = np.array(to_rgba(col)[:3]), np.array(to_rgba(ground)[:3])
    box = _perceptual(box_alpha * rgb + (1 - box_alpha) * bg)[0]
    k = np.linspace(0.0, 1.0, 101)[:, None]
    toward = 0.0 if _perceptual(rgb)[0] <= box else 1.0  # the box is paler than its ink on a light ground
    inks = rgb + (toward - rgb) * k  # the ink → black (or → white), in 1 % steps
    far = np.abs(_perceptual(inks)[:, 0] - box) >= contrast
    return (*inks[int(np.argmax(far)) if far.any() else -1], 1.0)


@dataclass
class FluxboxResult:
    """What :func:`fluxbox` drew — the axes plus everything needed to reuse or annotate it."""

    ax: Any
    #: category values in plotted (left → right) order; category ``i`` sits at x = ``i``
    categories: list
    #: category → the statistics drawn (``n, mean, median, sd, sem, q1, q3, iqr, whis, whiskerLow,
    #: whiskerHigh, outliers, x``)
    stats: dict
    #: category → its group colour (the box; the median / whisker ink is derived from it)
    group_colors: dict
    #: unit (or row index without ``units``) → {category: point colour}
    point_colors: dict
    #: category → series name, and unit → series name (the roots of every part id)
    series: dict
    unit_series: dict
    #: part → matplotlib artists (``box``, ``whiskers``, ``caps``, ``median``, ``mean``, ``fliers``,
    #: ``points``, ``lines``)
    artists: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# the plot
# ---------------------------------------------------------------------------
def fluxbox(
    data=None,
    *,
    x,
    y,
    units=None,
    order: Optional[Sequence] = None,
    unit_order: Optional[Sequence] = None,
    ax=None,
    # the box
    whis: Union[float, str, Sequence[float]] = 1.5,
    show_mean: bool = True,
    show_median: bool = True,
    show_whiskers: bool = True,
    show_caps: bool = False,
    show_fliers: Union[bool, str] = "auto",
    box_width: float = 7.0,
    box_alpha: float = 0.5,
    box_offset: float = 0.42,
    box_side: str = "outer",
    median_line_width: float = 1.4,
    median_color=None,
    median_contrast: float = 30.0,
    mean_notch_depth: float = 1.5,
    mean_notch_height: float = 2.6,
    whisker_width: float = 1.0,
    whisker_color=None,
    cap_size: float = 4.0,
    cap_width: float = 0.8,
    cap_color=None,
    flier_size: float = 8.0,
    cut_color=None,
    # the individual points
    show_individual_points: bool = True,
    point_size: float = 18.0,
    jitter: float = 0.14,
    point_edge="rim",
    point_edge_width: float = 0.5,
    point_fill_alpha: float = 1.0,
    # colour
    palette=None,
    group_color_position: Optional[float] = None,
    group_color=None,
    point_colors="auto",
    shade_range: tuple = (88.0, 22.0),
    interleave_shades: bool = True,
    # paired / repeated measures
    connect_identical_points_across_x_values: bool = False,
    connect_line_width: float = 0.6,
    connect_color=CONNECT_GREY,
    connect_alpha: float = 0.8,
    # identity (the sidecar names)
    series=None,
    unit_series=None,
    label_axes: bool = True,
    zorder: float = 2.0,
) -> FluxboxResult:
    """Draw a fluxbox: the individual points plus a slim box plot with a median line and a mean notch.

    Parameters
    ----------
    data
        A pandas or polars DataFrame, a dict of columns, or ``None`` when ``x``/``y``/``units`` are
        passed as arrays.
    x, y
        Column names (or array-likes): ``x`` is categorical (one box per value), ``y`` numeric. Rows
        whose ``x`` or ``y`` is missing are not drawn.
    units
        Optional identity column (subject, animal, cell…). Each unit gets a fixed lane and colour
        inside every category, derived from the table order (``unit_order``), never from the values
        — so separate plots of different measures from the same table agree, even when a unit is
        missing a value in one of them. Units are also what
        ``connect_identical_points_across_x_values`` joins.
    order, unit_order
        Category order (left → right) and unit order (first lane → last). Default: first appearance
        in the data (numeric categories are sorted). Categories missing from ``order`` are dropped.
    ax
        Target axes (default: the current axes).

    whis
        How far the whiskers reach: a factor of the IQR (default ``1.5`` — Tukey's rule, as
        ``plt.boxplot``), ``"range"`` (the minimum and maximum), or a ``(low, high)`` pair of
        percentiles. Each whisker ends on the most extreme observation within that reach.
    show_mean, show_median, show_whiskers, show_caps
        Toggle the mean notch, the median line, the whiskers and whisker caps (off by default).
    show_fliers
        Mark the observations beyond the whiskers on the box's own axis: ``"auto"`` (only when the
        individual points are hidden, since otherwise they are already drawn — default), ``True``
        or ``False``.
    box_width, box_alpha
        Box width in points (the median line spans exactly this width), and the opacity of its fill
        (default ``0.5``: a wash of the group colour that the solid median line stands out from).
    box_offset, box_side
        Distance (category units) of the box from its category's centre, and its side: ``"outer"``
        (the first category's box to the left, every other to the right, so the boxes frame the
        comparison — default), ``"left"`` or ``"right"``.
    median_line_width, median_color, median_contrast
        Stroke of the median line (points) and its colour. By default it is the group colour,
        deepened — or, where the box renders darker than the group colour (a dark background),
        lifted — only as far as it takes to differ from the box by ``median_contrast`` units of
        perceived lightness (0–100 scale); give ``median_color`` to set it outright. Whichever it is,
        it is also the colour of the whiskers, fliers and a solid (out-of-box) mean mark.
    mean_notch_depth, mean_notch_height
        How far each V-cut of the mean notch reaches into the box, and its height, in points. The
        notch is drawn above the median, so it stays visible when the mean meets the median.
    whisker_width, whisker_color
        Whisker stroke (points) and colour (default: the median's colour).
    cap_size, cap_width, cap_color
        Whisker cap length and stroke (points), and colour (default: the whiskers') — when
        ``show_caps=True``. A whisker of zero length (no observation beyond the box) gets no cap.
    flier_size
        Marker area (points²) of the fliers.
    cut_color
        Colour of the mean notch "cut" (default: the axes background); also the ground the median's
        contrast is judged against.

    show_individual_points
        Draw the individual observations (default ``True``).
    point_size, jitter
        Marker area (points²), and the half-width of the lane spread (category units). Lanes are
        evenly spaced, never random.
    point_edge, point_edge_width
        ``"rim"`` (a deeper shade of each point's own colour, which keeps pale points crisp —
        default), ``"none"`` or any colour; and the edge width in points.
    point_fill_alpha
        Opacity of the points' fill only (default ``1.0``). The rim / edge keeps full opacity, so
        translucent points stay crisply outlined where they overlap.

    palette
        Per-category colour source, given once for every category, as a list (cycled in category
        order) or as a ``{category: spec}`` mapping — any colormap or palette in fluxplot's colour
        library, a matplotlib colormap, a list of colours, or a single colour; exactly as for
        :func:`fluxplot.glowbar`. Default: the ColorBrewer maps ``YlGnBu``, ``YlOrRd``, ``RdPu``,
        ``BuGn``, ``Purples``, ``YlOrBr``.
    group_color_position
        Pin the group colour to a point of an ordered (sequential) source's light → dark ramp
        (0 = palest, 1 = darkest). Default ``None``: 0.75 along a ColorBrewer ramp, else the
        source's most chromatic mid-lightness colour (or a neutral ink for multi-hue sources).
    group_color
        Override the group colour (the box, and the base of the median / whisker ink) outright: one
        colour, or a ``{category: colour}`` mapping. Point colours still come from ``palette``.
    point_colors
        ``"shades"`` (each unit its own shade of its category's map), ``"group"`` (every point in its
        group colour), ``"auto"`` (shades with ``units``, else group — default), one colour for all
        points, or a ``{unit: colour}`` mapping.
    shade_range, interleave_shades
        Lightness of the palest and darkest shade (0 = black, 100 = white), spaced in equal
        perceptual steps; and whether shades are dealt across lanes so neighbours always contrast
        (default ``True``) or run pale → dark in lane order.

    connect_identical_points_across_x_values
        Join each unit's points across the x categories with a line (needs ``units``) — for paired
        or repeated-measures designs. A unit missing from a category breaks its line there instead
        of bridging the gap; several rows of one unit in one category are joined through their mean.
    connect_line_width, connect_color, connect_alpha
        Connector stroke (points), colour (a colour, or ``"unit"`` for each unit's own point colour)
        and opacity. The default is a quiet neutral grey so the lines never compete with the points.

    series, unit_series
        Override the series names — the roots of every part id — per category / per unit, as a
        mapping or a callable. Defaults: the category and unit values (``"SD"`` → ``sd.box``,
        ``"B6_8"`` → ``b6-8.points``).
    label_axes
        Put the category names on the x ticks and the ``y`` column name on the y axis, and set the x
        limits (default ``True``).
    zorder
        Base z-order: connectors sit just below it, the points at it, the box above.

    Returns
    -------
    FluxboxResult
        The axes, category order, statistics, colours, series names and artists.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
    if box_side not in _SIDES:
        raise ValueError(f"fluxbox: box_side must be 'outer', 'left' or 'right'; got {box_side!r}")
    if show_fliers not in (True, False, "auto"):
        raise ValueError(f"fluxbox: show_fliers must be True, False or 'auto'; got {show_fliers!r}")
    if not 0.0 <= box_alpha <= 1.0:
        raise ValueError(f"fluxbox: box_alpha must be in [0, 1]; got {box_alpha!r}")
    rule = _whisker_rule(whis)
    fr = _frame("fluxbox", data, x, y, units, order, unit_order, jitter=jitter, palette=palette,
                group_color=group_color, group_color_position=group_color_position,
                point_colors=point_colors, shade_range=shade_range,
                interleave_shades=interleave_shades, series=series, unit_series=unit_series,
                connect=connect_identical_points_across_x_values, point_fill_alpha=point_fill_alpha)
    reg = _tagger.registry_for(ax.figure)
    artists = {"box": [], "whiskers": [], "caps": [], "median": [], "mean": [], "fliers": [],
               "points": [], "lines": []}
    artists["lines"], artists["points"] = _draw_units(
        fr, ax, reg, connect=connect_identical_points_across_x_values,
        connect_line_width=connect_line_width, connect_color=connect_color,
        connect_alpha=connect_alpha, show_points=show_individual_points, point_size=point_size,
        point_edge=point_edge, point_edge_width=point_edge_width,
        point_fill_alpha=point_fill_alpha, zorder=zorder)
    cut = _cut_colour(ax, cut_color)
    fliers_on = not show_individual_points if show_fliers == "auto" else show_fliers

    # ---- the box ---------------------------------------------------------------------------------------
    notch, notch_ms = _notch_marker(box_width, mean_notch_depth, mean_notch_height)
    stats = {}
    for k, c in enumerate(fr.cats):
        st = _box_stats(fr.ys[fr.rows_in[c]], rule)
        if st is None:
            continue
        s, col = fr.series_of[c], fr.group_colors[c]
        bx = fr.summary_x(k, box_side, box_offset)
        st["x"] = bx
        stats[c] = dict(st, groupColor=col)
        q1, q3, w_lo, w_hi = st["q1"], st["q3"], st["whiskerLow"], st["whiskerHigh"]
        # one ink for every solid part (median, whiskers, caps, fliers); the box alone is the wash
        ink = median_color if median_color is not None else \
            _median_ink(col, box_alpha, cut, median_contrast)
        whisker_ink = whisker_color if whisker_color is not None else ink
        # a whisker (and its cap) only where an observation lies beyond the box on that side
        ends = [(w, q) for w, q in ((w_lo, q1), (w_hi, q3)) if w != q]

        if show_whiskers and ends:
            wx, wy = [], []
            for w, q in ends:
                wx += [bx, bx, np.nan]
                wy += [q, w, np.nan]
            (wh,) = ax.plot(wx[:-1], wy[:-1], color=whisker_ink, lw=whisker_width, solid_capstyle="butt", zorder=zorder + 0.5)
            reg.add(Mark(role="whisker", series=s, name="whiskers", kind="fluxbox", artists=[wh]))
            artists["whiskers"].append(wh)
        if show_caps and ends:
            (caps,) = ax.plot([bx] * len(ends), [w for w, _ in ends], ls="none", marker="_",
                              ms=cap_size, mew=cap_width,
                              color=cap_color if cap_color is not None else whisker_ink,
                              zorder=zorder + 0.5)
            reg.add(Mark(role="cap", series=s, name="caps", kind="fluxbox", artists=[caps]))
            artists["caps"].append(caps)
        if fliers_on and st["outliers"]:
            fl = ax.scatter([bx] * len(st["outliers"]), st["outliers"], s=flier_size, color=[ink],
                            linewidths=0, zorder=zorder + 0.5)
            reg.add(Mark(role="flier", series=s, name="fliers", kind="fluxbox", artists=[fl]))
            artists["fliers"].append(fl)

        # the box: a butt-capped stroke exactly box_width points wide, spanning Q1–Q3 in data units;
        # the exact statistics drawn ride on it → the manifest
        (body,) = ax.plot([bx, bx], [q1, q3], color=col, alpha=box_alpha, lw=box_width,
                          solid_capstyle="butt", zorder=zorder + 0.6)
        payload = {"fluxbox": {"part": "summary", "category": _plain(c), "units": fr.units_name,
                               "groupColor": _hex(col),
                               **{key: _manifest_value(v) for key, v in st.items()}}}
        reg.add(Mark(role="box", series=s, kind="fluxbox", artists=[body], data=payload))
        artists["box"].append(body)

        if show_median:
            # marker '_' spans exactly ms points: the median line is as wide as the box
            (med,) = ax.plot([bx], [st["median"]], ls="none", marker="_", ms=box_width,
                             mew=median_line_width, color=ink, zorder=zorder + 1.5)
            reg.add(Mark(role="median", series=s, kind="fluxbox", artists=[med]))
            artists["median"].append(med)
        if show_mean:
            # inside the box the notch is cut out of it; outside (or on a box of no height) there is
            # nothing to cut, so the same V's are drawn solid in the ink, pointing in at the whisker
            cut_in = q1 < q3 and q1 <= st["mean"] <= q3
            (mean_mk,) = ax.plot([bx], [st["mean"]], ls="none", marker=notch, ms=notch_ms,
                                 mfc=cut if cut_in else ink, mec="none", mew=0, zorder=zorder + 2)
            reg.add(Mark(role="mean", series=s, kind="fluxbox", artists=[mean_mk]))
            artists["mean"].append(mean_mk)

    _finish(fr, ax, label_axes)
    return FluxboxResult(ax=ax, categories=fr.cats, stats=stats, group_colors=fr.group_colors,
                         point_colors=fr.point_colors_by_unit(), series=fr.series_of,
                         unit_series=fr.unit_series_of, artists=artists)
