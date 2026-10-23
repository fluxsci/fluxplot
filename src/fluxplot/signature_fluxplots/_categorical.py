"""Shared categorical data, colours, lanes and connectors for signature plots."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

from .. import ids as _ids
from ..descriptors import Mark
from . import _colour
from ._colour import interleaved_order

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
    """Effective ground used to judge colour contrast."""
    if cut_color is not None:
        return cut_color
    for c in (ax.get_facecolor(), ax.figure.get_facecolor()):
        if c[3] > 0:
            return c
    return (1.0, 1.0, 1.0, 1.0)


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


# Shared categorical layout.
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
                ground=None, orientation="vertical"):
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
            (ln,) = ax.plot(*( (py, px) if orientation == "horizontal" else (px, py)), color=colour, alpha=connect_alpha, lw=connect_line_width,
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
            px = [fr.lane_x(xs[i], fr.lane_key[i]) for i in keep]
            py = [float(ys[i]) for i in keep]
            coll = ax.scatter(*((py, px) if orientation == "horizontal" else (px, py)),
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


def _finish(fr: _Frame, ax, label_axes, orientation="vertical"):
    if label_axes:
        horizontal = orientation == "horizontal"
        (ax.set_yticks if horizontal else ax.set_xticks)(range(len(fr.cats)), [str(c) for c in fr.cats])
        (ax.set_ylim if horizontal else ax.set_xlim)(-0.75, len(fr.cats) - 0.25)
        ax.margins(**{"x" if horizontal else "y": 0.08})
        ax.tick_params(axis="y" if horizontal else "x", length=0)
        if fr.y_name:
            (ax.set_xlabel if horizontal else ax.set_ylabel)(fr.y_name)
    ax.autoscale_view()
