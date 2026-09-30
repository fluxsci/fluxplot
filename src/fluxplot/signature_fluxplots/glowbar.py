"""``fp.glowbar`` — the glowbar, FluxPlot's first *signature* plot.

A glowbar shows every observation of a categorical comparison as a dot and, beside each group, a
slim bar that *glows*: its ink is densest at the centre of the distribution and thins out toward hard
caps at the interval ends. Three statistics are read straight off that one mark:

* the **interval** (default: mean ± SEM; or the interquartile range, the box of a box plot) — the
  glowing bar and its two caps;
* the **mean** — a heavy line across the bar, lifted off the glow by a thin halo;
* the **median** — a V-notch cut into both edges of the bar.

With ``units=`` (a subject / animal / cell column) every unit keeps a **fixed lane and a fixed
colour**. Both are derived from the table, never from the plotted values, so plots of different
measures made from the same table put "animal 8" at the same spot, in the same colour, in every
panel. The per-unit colours are equal *perceptual* steps of each group's ColorBrewer map (sampling a
map evenly in its parameter crowds its dark end), dealt across lanes so neighbouring lanes always
contrast, and outlined in a deeper shade of themselves so the palest dots stay crisp on white.
``connect_identical_points_across_x_values=True`` joins each unit's points across the x categories
for paired / repeated-measures designs.

Everything drawn is a named part, so a glowbar round-trips through Flux like any other FluxPlot:

=======================  ===============================================  ==========
part                     default id                                       role
=======================  ===============================================  ==========
interval glow            ``<category>.glow``                              ``box``
interval caps            ``<category>.caps``                              ``cap``
mean line                ``<category>.mean``                              ``mean``
median notch             ``<category>.median``                            ``median``
a unit's point(s)        ``<unit>.points``, ``<unit>.point.<k>``          ``point``
a unit's connector       ``<unit>.line``                                  ``line``
points (no ``units``)    ``<category>.points``, ``<category>.point.<k>``  ``point``
=======================  ===============================================  ==========

Each category series also carries a ``glowbar`` payload in the manifest with the exact statistics
drawn (n, mean, median, sd, sem, q1, q3, the interval ends, the glow centre, the group colour), and
each unit series carries its identity (the unit value and the categories it appears in) — a reader
of the ``.fluxplot.json`` never has to reverse-engineer pixels. Series names default to the category
and unit values and can be overridden with ``series=`` / ``unit_series=``.

Example
-------
>>> import fluxplot as fp, matplotlib.pyplot as plt
>>> from fluxplot import style as fx
>>> fx.use_light()
>>> fig, ax = plt.subplots(figsize=(1.6, 1.8))
>>> gb = fp.glowbar(df, x="condition", y="APP/GAPDH", units="subject", ax=ax)
>>> gb.stats["SD"]["median"]
2.839...
>>> fp.save(fig, "plots/app_gapdh.svg")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np

from .. import ids as _ids
from .. import tagger as _tagger
from ..descriptors import Mark
from . import _colour
from ._colour import even_shades, interleaved_order
from ._colour import perceptual as _perceptual  # noqa: F401  (re-exported for tests / callers)

__all__ = ["glowbar", "GlowbarResult", "even_shades", "interleaved_order"]

#: Per-category ColorBrewer maps used when ``palette`` is not given (cycled for more categories).
DEFAULT_MAPS = ("YlGnBu", "YlOrRd", "RdPu", "BuGn", "Purples", "YlOrBr")
#: Flexoki base-300: a quiet neutral for connectors, so paired lines never compete with the points.
#: The default connector colour is the active theme's grid token when a theme is on (a neutral
#: tuned to that ground); this grey is the fallback.
CONNECT_GREY = "#B7B5AC"


def _connect_colour(connect_color, ground=None):
    """The connectors' colour: the caller's, else the theme's grid neutral when it still reads
    against the ground (WCAG ≥ 1.5 — the light theme's grid is too faint on white), else the
    historic Flexoki base-300."""
    if connect_color is not None:
        return connect_color
    from .. import style as _style
    if _style.ACTIVE is not None:
        grid = _style.ACTIVE["tokens"]["grid"]
        if ground is None:
            return grid
        from ..colorcheck import contrast
        from matplotlib.colors import to_hex
        if contrast(grid, to_hex(ground)) >= 1.5:
            return grid
    return CONNECT_GREY


# ---------------------------------------------------------------------------
# data access — pandas, polars, a dict of columns, or bare array-likes
# ---------------------------------------------------------------------------
def _is_missing(v) -> bool:
    if v is None or type(v).__name__ in ("NAType", "NaTType"):  # pandas NA/NaT, without pandas
        return True
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


def _values(col) -> list:
    """A column as a plain Python list (polars/pandas Series, numpy array or any sequence)."""
    for attr in ("to_list", "tolist"):
        if hasattr(col, attr):
            return list(getattr(col, attr)())
    return list(col)


def _column(who, data, key, what):
    """``(values, name)`` for a column given by name (looked up in ``data``) or as an array-like."""
    if key is None:
        return None, None
    if isinstance(key, str):
        if data is None:
            raise TypeError(f"{who}: {what}={key!r} names a column, but no data= was given")
        try:
            col = data[key]
        except Exception as exc:  # KeyError (pandas/dict), ColumnNotFoundError (polars), …
            raise KeyError(f"{who}: {what}={key!r} is not a column of data") from exc
        return _values(col), key
    name = getattr(key, "name", None)
    return _values(key), name if isinstance(name, str) else None


def _floats(who, vals, what) -> np.ndarray:
    out = np.empty(len(vals), dtype=float)
    for i, v in enumerate(vals):
        if _is_missing(v):
            out[i] = np.nan
            continue
        try:
            out[i] = float(v)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{who}: {what} must be numeric; got {v!r}") from exc
    return out


def _plain(v):
    """A JSON-safe scalar for the manifest (numpy scalars → Python, NaN → None, else str)."""
    if _is_missing(v):
        return None
    if hasattr(v, "item"):
        v = v.item()
    return v if isinstance(v, (bool, int, float, str)) else str(v)


def _palette_spec_json(spec):
    """The palette spec as the manifest carries it: a name, or a list of hex colours."""
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        return [_hex(c) for c in spec]
    return str(spec) if isinstance(spec, str) else getattr(spec, "name", str(spec))


def _category_order(xs, order):
    if order is not None:
        return list(order)
    seen = list(dict.fromkeys(v for v in xs if not _is_missing(v)))
    if seen and all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in seen):
        seen.sort()  # numeric categories read in numeric order, like seaborn
    return seen


# ---------------------------------------------------------------------------
# colour: perceptual spaces, light→dark maps, even shades
# ---------------------------------------------------------------------------
def _hex(c):
    from matplotlib.colors import to_hex
    return to_hex(c, keep_alpha=False)


def _cut_colour(ax, cut_color):
    """The colour the halo and the median notch are 'cut' with: the axes background by default."""
    if cut_color is not None:
        return cut_color
    for c in (ax.get_facecolor(), ax.figure.get_facecolor()):
        if c[3] > 0:
            return c
    return (1.0, 1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# statistics + the median-notch marker
# ---------------------------------------------------------------------------
def _stats(vals, interval, center):
    v = vals[np.isfinite(vals)]
    n = int(v.size)
    if n == 0:
        return None
    mean, median = float(v.mean()), float(np.median(v))
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    sem = sd / float(np.sqrt(n)) if n > 1 else float("nan")
    q1, q3 = (float(q) for q in np.percentile(v, [25, 75]))  # linear, as plt.boxplot / fp.box
    if callable(interval):
        low, high = (float(b) for b in interval(v))
        name = getattr(interval, "__name__", "custom")
    elif interval == "iqr":
        low, high, name = q1, q3, "iqr"
    elif interval == "sem":
        low, high, name = mean - sem, mean + sem, "sem"
    elif interval == "sd":
        low, high, name = mean - sd, mean + sd, "sd"
    else:
        raise ValueError(f"glowbar: interval must be 'iqr', 'sem', 'sd' or a callable; got {interval!r}")
    if center == "auto":
        center = "median" if name == "iqr" else "mean"
    if center not in ("mean", "median"):
        raise ValueError(f"glowbar: center must be 'auto', 'mean' or 'median'; got {center!r}")
    return {"n": n, "mean": mean, "median": median, "sd": sd, "sem": sem, "q1": q1, "q3": q3,
            "interval": name, "low": low, "high": high, "center": center,
            "centerValue": mean if center == "mean" else median}


def _notch_marker(bar_w, depth, height, bleed=0.3):
    """Both V-cuts of the median notch as ONE marker path, in points around the bar's centre line.

    Returns ``(path, markersize)``; that markersize makes matplotlib's marker scale exactly 1, so the
    path's units stay points. The bases sit ``bleed`` pt outside the bar edges (no anti-alias seam).
    """
    from matplotlib.path import Path
    e, tip, h = bar_w / 2 + bleed, bar_w / 2 - depth, height / 2
    verts = [(-e, -h), (-tip, 0), (-e, h), (-e, -h), (e, -h), (tip, 0), (e, h), (e, -h)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY] * 2
    return Path(verts, codes), 2 * float(np.abs(verts).max())


def _namer(who, override, what):
    if override is None:
        return lambda v: str(v)
    if callable(override):
        return lambda v: str(override(v))
    if isinstance(override, Mapping):
        return lambda v: str(override.get(v, v))
    raise TypeError(f"{who}: {what} must be a mapping or a callable; got {type(override).__name__}")


def _colour_mode(who, point_colors, has_units):
    from matplotlib.colors import to_rgba
    if isinstance(point_colors, Mapping):
        if not has_units:
            raise ValueError(f"{who}: a {{unit: colour}} point_colors mapping needs units=")
        return "mapping"
    if isinstance(point_colors, str) and point_colors in ("auto", "shades", "group"):
        if point_colors == "auto":
            return "shades" if has_units else "group"
        return point_colors
    to_rgba(point_colors)  # one colour for every point (raises on garbage)
    return "single"


@dataclass
class GlowbarResult:
    """What :func:`glowbar` drew — the axes plus everything needed to reuse or annotate it."""

    ax: Any
    #: category values in plotted (left → right) order; category ``i`` sits at x = ``i``
    categories: list
    #: category → the statistics drawn (``n, mean, median, sd, sem, q1, q3, low, high, center, x``)
    stats: dict
    #: category → its group colour (glow, caps, default mean shade)
    group_colors: dict
    #: unit (or row index without ``units``) → {category: point colour}
    point_colors: dict
    #: category → series name, and unit → series name (the roots of every part id)
    series: dict
    unit_series: dict
    #: part → matplotlib artists (``glow``, ``caps``, ``mean``, ``median``, ``points``, ``lines``)
    artists: dict = field(default_factory=dict)

    @property
    def positions(self) -> dict:
        """Category name → x (category ``i`` sits at ``i``), as :func:`fluxplot.brackets` wants it."""
        return {str(c): float(i) for i, c in enumerate(self.categories)}

    def brackets(self, rows, **kw) -> list:
        """Draw ``fp.stats`` post-hoc rows as stacked significance brackets over these
        categories: :func:`fluxplot.brackets` with this plot's ``positions``."""
        from ..brackets import brackets as _brackets
        return _brackets(self.ax, rows, positions=kw.pop("positions", self.positions), **kw)


# ---------------------------------------------------------------------------
# the scaffold every signature plot shares (fp.glowbar, fp.fluxbox): the table, the unit lanes and
# colours, the names, the points and connectors — everything but the per-category summary mark
# ---------------------------------------------------------------------------
_SIDES = ("outer", "left", "right")


@dataclass
class _Frame:
    """A table resolved into categories, unit lanes, colours and series names."""

    who: str  # the plot's name, for error messages and the manifest payload key
    xs: list
    ys: np.ndarray
    us: Optional[list]
    y_name: Optional[str]
    units_name: Optional[str]
    cats: list
    pos: dict
    rows_in: dict
    plotted: list
    unit_list: list
    lanes_of: dict
    lane_key: dict
    jitter: float
    group_colors: dict
    point_color: dict  # (category, lane key) → colour
    series_of: dict
    unit_series_of: dict
    palette_used: dict = field(default_factory=dict)  # category → the colour spec its shades came from

    def lane_x(self, c, key):
        lanes = self.lanes_of[c]
        spread = np.linspace(-1.0, 1.0, len(lanes)) if len(lanes) > 1 else np.zeros(1)
        return self.pos[c] + self.jitter * float(spread[lanes.index(key)])

    def summary_x(self, k, side, offset):
        """x of category ``k``'s summary mark, ``offset`` to the ``side`` of its points."""
        if side == "outer":  # the first category's summary to the left, every other to the right
            sign = -1 if (k == 0 and len(self.cats) > 1) else 1
        else:
            sign = -1 if side == "left" else 1
        return float(self.pos[self.cats[k]] + sign * offset)

    def point_colors_by_unit(self):
        out: dict = {}
        for (c, key), col in self.point_color.items():
            out.setdefault(key, {})[c] = col
        return out


def _frame(who, data, x, y, units, order, unit_order, *, jitter, palette, group_color,
           group_color_position, point_colors, shade_range, interleave_shades, series, unit_series,
           connect, point_fill_alpha, ground=None) -> _Frame:
    from matplotlib.colors import to_rgba

    xs, _ = _column(who, data, x, "x")
    yv, y_name = _column(who, data, y, "y")
    us, units_name = _column(who, data, units, "units")
    if len(yv) != len(xs) or (us is not None and len(us) != len(xs)):
        raise ValueError(f"{who}: x, y and units must have the same length")
    ys = _floats(who, yv, "y")
    if connect and us is None:
        raise ValueError(f"{who}: connect_identical_points_across_x_values needs units= "
                         "(which points are 'the same' is a unit's identity)")
    colour_mode = _colour_mode(who, point_colors, us is not None)
    if not 0.0 <= point_fill_alpha <= 1.0:
        raise ValueError(f"{who}: point_fill_alpha must be in [0, 1]; got {point_fill_alpha!r}")

    cats = _category_order(xs, order)
    pos = {c: i for i, c in enumerate(cats)}
    rows_in = {c: [i for i, v in enumerate(xs) if not _is_missing(v) and v == c] for c in cats}
    plotted = sorted(i for c in cats for i in rows_in[c])

    # ---- identity: every unit gets a fixed lane per category, from the table — never the values ----
    if us is not None:
        unit_list = (list(unit_order) if unit_order is not None else
                     list(dict.fromkeys(us[i] for i in plotted if not _is_missing(us[i]))))
        lanes_of = {c: [u for u in unit_list if any(us[i] == u for i in rows_in[c])] for c in cats}
        lane_key = {i: us[i] for i in plotted}
    else:
        unit_list = []
        lanes_of = {c: list(rows_in[c]) for c in cats}  # one lane per row, in table order
        lane_key = {i: i for i in plotted}

    # ---- names (the recipe's per-series colours are keyed by the series id) ------------------------
    cat_name, unit_name = _namer(who, series, "series"), _namer(who, unit_series, "unit_series")

    # ---- colours ------------------------------------------------------------------------------------
    from .. import style as _style
    from ..api import _series_color_override
    from ..colors import categories as _categories

    recipe_palette = _style.palette_override()  # a categorical palette Flux asked for on a rerun

    def palette_spec(c, k):
        default = DEFAULT_MAPS[k % len(DEFAULT_MAPS)]
        if palette is None:
            if _categories.is_pinned(c):
                return _categories.get(c)  # a pinned category: a ramp of its own colour
            return _categories.get(c, palette=recipe_palette) if recipe_palette else default
        if isinstance(palette, Mapping):
            return palette.get(c, palette.get(str(c), default))
        if isinstance(palette, (list, tuple)):
            return palette[k % len(palette)]
        return palette

    sources, group_colors, palette_used = {}, {}, {}
    for k, c in enumerate(cats):
        spec = palette_spec(c, k)
        src = _colour.resolve(spec, f"{who} {c}")
        sources[c] = src
        palette_used[c] = spec
        override = group_color.get(c, group_color.get(str(c))) if isinstance(group_color, Mapping) \
            else group_color
        if override is None:
            # a recipe's per-series colour, else a category pinned in fp.colors.categories, else the source
            override = _series_color_override(cat_name(c))
        if override is None and _categories.is_pinned(c):
            override = _categories.get(c)
        group_colors[c] = to_rgba(override) if override is not None else \
            _colour.representative(src, group_color_position, ground=ground)

    point_color = {}  # (category, lane key) → colour
    bounds = _colour.shade_bounds(*shade_range, ground=ground)  # lifted off a dark ground
    for c in cats:
        lanes = lanes_of[c]
        if colour_mode == "shades":
            shades = _colour.shades(sources[c], len(lanes), *bounds)
            # interleaving spreads an ORDERED run of shades; a qualitative palette is already distinct
            spread = interleave_shades and (sources[c].ordered or sources[c].kind == "continuous")
            ranks = interleaved_order(len(lanes)) if spread else range(len(lanes))
            point_color.update({(c, key): shades[r] for key, r in zip(lanes, ranks)})
        elif colour_mode == "group":
            point_color.update({(c, key): group_colors[c] for key in lanes})
        elif colour_mode == "single":
            point_color.update({(c, key): to_rgba(point_colors) for key in lanes})
        else:  # mapping
            point_color.update({(c, key): to_rgba(point_colors.get(key, group_colors[c])) for key in lanes})

    # ---- names --------------------------------------------------------------------------------------
    series_of = {c: cat_name(c) for c in cats}
    unit_series_of = {u: unit_name(u) for u in unit_list}
    clash = ({_ids.series_root(s) for s in series_of.values()}
             & {_ids.series_root(s) for s in unit_series_of.values()})
    if clash:
        raise ValueError(f"{who}: a category and a unit would share the series id(s) {sorted(clash)}; "
                         "pass series= or unit_series= to rename one of them")

    return _Frame(who=who, xs=xs, ys=ys, us=us, y_name=y_name, units_name=units_name, cats=cats,
                  pos=pos, rows_in=rows_in, plotted=plotted, unit_list=unit_list, lanes_of=lanes_of,
                  lane_key=lane_key, jitter=jitter, group_colors=group_colors,
                  point_color=point_color, series_of=series_of, unit_series_of=unit_series_of,
                  palette_used=palette_used)


def _draw_units(fr: _Frame, ax, reg, *, connect, connect_line_width, connect_color, connect_alpha,
                show_points, point_size, point_edge, point_edge_width, point_fill_alpha, zorder,
                ground=None):
    """The connectors (just under ``zorder``) and the individual points (at it) → ``(lines, points)``."""
    from matplotlib.colors import to_rgba

    xs, ys, us = fr.xs, fr.ys, fr.us
    lines, points = [], []
    connect_color = _connect_colour(connect_color, ground)
    edge = (lambda col: _colour.rim(col, ground)) if point_edge == "rim" else (lambda col: point_edge)

    if connect:
        for u in fr.unit_list:
            px, py = [], []
            for c in fr.cats:
                vals = [ys[i] for i in fr.rows_in[c] if us[i] == u and np.isfinite(ys[i])]
                px.append(fr.lane_x(c, u) if vals else np.nan)
                py.append(float(np.mean(vals)) if vals else np.nan)
            finite = np.isfinite(py)
            if not np.any(finite[:-1] & finite[1:]):
                continue  # nothing adjacent to join
            first = next(c for c, f in zip(fr.cats, finite) if f)
            colour = fr.point_color[(first, u)] if connect_color == "unit" else connect_color
            (ln,) = ax.plot(px, py, color=colour, alpha=connect_alpha, lw=connect_line_width,
                            solid_capstyle="round", zorder=zorder - 0.5)
            reg.add(Mark(role="line", series=fr.unit_series_of[u], kind="line", live_data=True,
                         artists=[ln]))
            lines.append(ln)

    if show_points:
        if us is not None:  # one series per unit: the same animal is the same part in every plot
            groups = [(fr.unit_series_of[u], [i for i in fr.plotted if us[i] == u], u)
                      for u in fr.unit_list]
        else:
            groups = [(fr.series_of[c], fr.rows_in[c], None) for c in fr.cats]
        for name, idx, u in groups:
            keep = [i for i in idx if np.isfinite(ys[i])]
            if not keep:
                continue
            cols = [fr.point_color[(xs[i], fr.lane_key[i])] for i in keep]
            coll = ax.scatter([fr.lane_x(xs[i], fr.lane_key[i]) for i in keep],
                              [float(ys[i]) for i in keep],
                              s=point_size, edgecolors=[edge(col) for col in cols],
                              facecolors=[(*to_rgba(col)[:3], to_rgba(col)[3] * point_fill_alpha)
                                          for col in cols],
                              linewidths=point_edge_width, zorder=zorder)
            payload = {}
            if u is not None:
                payload = {fr.who: {
                    "part": "unit", "units": fr.units_name, "unit": _plain(u),
                    "categories": [_plain(xs[i]) for i in keep],
                    "colors": [_hex(col) for col in cols]}}
            reg.add(Mark(role="point", series=name, kind="scatter", live_data=True, artists=[coll],
                         indexed=True, data=payload))
            points.append(coll)
    return lines, points


def _finish(fr: _Frame, ax, label_axes):
    if label_axes:
        ax.set_xticks(range(len(fr.cats)), [str(c) for c in fr.cats])
        ax.set_xlim(-0.75, len(fr.cats) - 0.25)
        ax.margins(y=0.08)
        ax.tick_params(axis="x", length=0)
        if fr.y_name:
            ax.set_ylabel(fr.y_name)
    ax.autoscale_view()


# ---------------------------------------------------------------------------
# the plot
# ---------------------------------------------------------------------------
def glowbar(
    data=None,
    *,
    x,
    y,
    units=None,
    order: Optional[Sequence] = None,
    unit_order: Optional[Sequence] = None,
    ax=None,
    # the summary bar
    interval: Union[str, Callable] = "sem",
    center: str = "auto",
    show_mean: bool = True,
    show_median: bool = True,
    show_caps: bool = True,
    bar_width: float = 5.0,
    bar_offset: float = 0.42,
    bar_side: str = "outer",
    glow_steps: int = 28,
    glow_alpha: float = 0.06,
    mean_line_width: float = 1.2,
    mean_color=None,
    mean_halo_width: float = 1.5,
    median_notch_depth: float = 1.3,
    median_notch_height: float = 2.2,
    cap_width: float = 0.8,
    cap_color=None,
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
    connect_color=None,
    connect_alpha: float = 0.8,
    # identity (the sidecar names)
    series=None,
    unit_series=None,
    label_axes: bool = True,
    zorder: float = 2.0,
) -> GlowbarResult:
    """Draw a glowbar: the individual points plus a glowing interval bar with mean line and median notch.

    Parameters
    ----------
    data
        A pandas or polars DataFrame, a dict of columns, or ``None`` when ``x``/``y``/``units`` are
        passed as arrays.
    x, y
        Column names (or array-likes): ``x`` is categorical (one bar per value), ``y`` numeric. Rows
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

    interval
        What the glowing bar and its caps span: ``"sem"`` (mean ± SEM — default), ``"iqr"`` (Q1–Q3,
        the box of a box plot), ``"sd"`` (mean ± SD), or a callable ``values -> (low, high)``.
    center
        Where the glow is densest: ``"auto"`` (the median for ``"iqr"``, else the mean), ``"mean"``
        or ``"median"``. The glow fades from there toward each cap independently, so a skewed group
        glows asymmetrically.
    show_mean, show_median, show_caps
        Toggle the mean line, the median notch and the interval caps.
    bar_width
        Bar width in points; the mean line and the caps span exactly this width.
    bar_offset, bar_side
        Distance (category units) of the bar from its category's centre, and its side: ``"outer"``
        (the first category's bar to the left, every other to the right, so the bars frame the
        comparison — default), ``"left"`` or ``"right"``.
    glow_steps, glow_alpha
        The glow is ``glow_steps`` nested segments of opacity ``glow_alpha``; their overlap builds the
        gradient (peak opacity ≈ ``1 - (1 - glow_alpha) ** glow_steps``).
    mean_line_width, mean_color, mean_halo_width
        Stroke of the mean line (points), its colour (default: the group colour deepened — or, on a
        dark ground, lifted — just far enough to stand off the glow) and the width of the
        background-coloured halo that lifts it off the glow (``0`` disables it).
    median_notch_depth, median_notch_height
        How far each V-cut reaches into the bar, and its height, in points. The notch is drawn above
        the mean line, so it stays visible when the median meets the mean.
    cap_width, cap_color
        Stroke and colour (default: the group colour) of the interval caps.
    cut_color
        Colour of the notch and halo "cuts" (default: the axes background).

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
        order) or as a ``{category: spec}`` mapping. A spec can be anything in fluxplot's colour
        library — a colormap from ``fp.colors.maps`` (``"cmasher.emerald"``, ``"crameri.batlow"``,
        ``"tol.sunset"``, a bare ``"YlGnBu"``, ``_r`` reversed), a palette from
        ``fp.colors.palettes`` (``"brewer.Set2"``, ``"tol.bright"``, ``"flexoki.blue"``), any
        matplotlib colormap or ``Colormap``, a list of colours, or a single colour. The glowbar
        picks as many distinct point colours as each group needs plus one solid group colour:
        sequential sources give equal perceptual steps (oriented light → dark, whatever their
        source direction); diverging / cyclic maps are sampled evenly along their usable (not too
        pale) stretches; qualitative palettes keep their own order. Default: the ColorBrewer
        maps ``YlGnBu``, ``YlOrRd``, ``RdPu``, ``BuGn``, ``Purples``, ``YlOrBr``. (Because a palette list
        like ``["#123", "#456"]`` means one spec per category, wrap a hand-made palette for a
        single category in a mapping: ``{"SD": ["#123", "#456", "#789"]}``.)
    group_color_position
        Pin the group colour to a point of an ordered (sequential) source's light → dark ramp
        (0 = palest, 1 = darkest). Default ``None`` chooses: 0.75 along a ColorBrewer ramp; for any
        other source its most chromatic mid-lightness colour when its hues agree (e.g. cmasher
        ``emerald``), or a neutral ink when they spread around the wheel (diverging, rainbow,
        qualitative) — no single hue honestly stands for those.
    group_color
        Override the group colour (glow, caps, mean shade) outright: one colour, or a
        ``{category: colour}`` mapping. Point colours still come from ``palette``.
    point_colors
        ``"shades"`` (each unit its own shade of its category's map), ``"group"`` (every point in its
        group colour), ``"auto"`` (shades with ``units``, else group — default), one colour for all
        points, or a ``{unit: colour}`` mapping.
    shade_range, interleave_shades
        Lightness of the palest and darkest shade (0 = black, 100 = white), spaced in equal
        perceptual steps; and whether shades are dealt across lanes so neighbours always contrast
        (default ``True``) or run pale → dark in lane order. On a dark ground the dark bound is
        lifted to stay at least 20 lightness units above the ground.

    connect_identical_points_across_x_values
        Join each unit's points across the x categories with a line (needs ``units``) — for paired
        or repeated-measures designs. A unit missing from a category breaks its line there instead
        of bridging the gap; several rows of one unit in one category are joined through their mean.
    connect_line_width, connect_color, connect_alpha
        Connector stroke (points), colour (a colour, or ``"unit"`` for each unit's own point colour)
        and opacity. The default is a quiet neutral — the active theme's grid colour, else Flexoki
        base-300 — so the lines never compete with the points.

    series, unit_series
        Override the series names — the roots of every part id — per category / per unit, as a
        mapping or a callable. Defaults: the category and unit values (``"SD"`` → ``sd.glow``,
        ``"B6_8"`` → ``b6-8.points``).
    label_axes
        Put the category names on the x ticks and the ``y`` column name on the y axis, and set the x
        limits (default ``True``).
    zorder
        Base z-order: connectors sit just below it, the points at it, the bar above.

    Returns
    -------
    GlowbarResult
        The axes, category order, statistics, colours, series names and artists.
    """
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba

    if ax is None:
        ax = plt.gca()
    if bar_side not in _SIDES:
        raise ValueError(f"glowbar: bar_side must be 'outer', 'left' or 'right'; got {bar_side!r}")
    cut = _cut_colour(ax, cut_color)  # the ground every ink is judged against
    fr = _frame("glowbar", data, x, y, units, order, unit_order, jitter=jitter, palette=palette,
                group_color=group_color, group_color_position=group_color_position,
                point_colors=point_colors, shade_range=shade_range,
                interleave_shades=interleave_shades, series=series, unit_series=unit_series,
                connect=connect_identical_points_across_x_values, point_fill_alpha=point_fill_alpha,
                ground=cut)
    reg = _tagger.registry_for(ax.figure)
    artists = {"glow": [], "caps": [], "mean": [], "median": [], "points": [], "lines": []}
    artists["lines"], artists["points"] = _draw_units(
        fr, ax, reg, connect=connect_identical_points_across_x_values,
        connect_line_width=connect_line_width, connect_color=connect_color,
        connect_alpha=connect_alpha, show_points=show_individual_points, point_size=point_size,
        point_edge=point_edge, point_edge_width=point_edge_width,
        point_fill_alpha=point_fill_alpha, zorder=zorder, ground=cut)

    # ---- the glowing bar -------------------------------------------------------------------------------
    notch, notch_ms = _notch_marker(bar_width, median_notch_depth, median_notch_height)
    stats = {}
    for k, c in enumerate(fr.cats):
        st = _stats(fr.ys[fr.rows_in[c]], interval, center)
        if st is None:
            continue
        s, col = fr.series_of[c], fr.group_colors[c]
        bx = fr.summary_x(k, bar_side, bar_offset)
        st["x"] = bx
        stats[c] = dict(st, groupColor=col)
        lo, hi = st["low"], st["high"]
        drawable = bool(np.isfinite(lo) and np.isfinite(hi) and hi >= lo)
        # the exact statistics drawn ride on the category's first summary part → the manifest
        payload = {"glowbar": {"part": "summary", "category": _plain(c), "units": fr.units_name,
                               "groupColor": _hex(col), "palette": _palette_spec_json(fr.palette_used.get(c)),
                               **{key: _plain(v) for key, v in st.items()}}}
        if drawable:
            mid = min(max(st["centerValue"], lo), hi)  # glow centre, clamped into the interval
            t = np.linspace(1.0, 0.0, glow_steps, endpoint=False)  # nested: overlap builds the gradient
            glow = LineCollection([[(bx, mid - (mid - lo) * ti), (bx, mid + (hi - mid) * ti)] for ti in t],
                                  colors=[(*to_rgba(col)[:3], glow_alpha)], linewidths=bar_width,
                                  capstyle="butt", zorder=zorder + 0.5)
            ax.add_collection(glow)
            reg.add(Mark(role="box", series=s, name="glow", kind="glowbar", artists=[glow], data=payload))
            artists["glow"].append(glow)
            payload = {}
            if show_caps:
                (caps,) = ax.plot([bx, bx], [lo, hi], ls="none", marker="_", ms=bar_width, mew=cap_width,
                                  color=cap_color if cap_color is not None else col, zorder=zorder + 0.6)
                reg.add(Mark(role="cap", series=s, name="caps", kind="glowbar", artists=[caps]))
                artists["caps"].append(caps)
        if show_mean:
            # marker '_' spans exactly ms points: the mean line is as wide as the bar
            effects = ([pe.Stroke(linewidth=mean_halo_width, foreground=cut), pe.Normal()]
                       if mean_halo_width else None)
            # the mean's default ink stands 30 lightness units off the glow's peak over the ground:
            # deepened on a light ground, lifted on a dark one (the fluxbox's median rule)
            peak = 1.0 - (1.0 - glow_alpha) ** glow_steps
            (mean_ln,) = ax.plot([bx], [st["mean"]], marker="_", ms=bar_width, mew=mean_line_width,
                                 color=mean_color if mean_color is not None else _colour.median_ink(col, peak, cut, 30.0),
                                 zorder=zorder + 1.5, path_effects=effects)
            reg.add(Mark(role="mean", series=s, kind="glowbar", artists=[mean_ln], data=payload))
            artists["mean"].append(mean_ln)
            payload = {}
        if show_median and drawable:
            (med,) = ax.plot([bx], [st["median"]], ls="none", marker=notch, ms=notch_ms, mfc=cut,
                             mec="none", mew=0, zorder=zorder + 2)
            reg.add(Mark(role="median", series=s, kind="glowbar", artists=[med], data=payload))
            artists["median"].append(med)

    _finish(fr, ax, label_axes)
    return GlowbarResult(ax=ax, categories=fr.cats, stats=stats, group_colors=fr.group_colors,
                         point_colors=fr.point_colors_by_unit(), series=fr.series_of,
                         unit_series=fr.unit_series_of, artists=artists)
