"""A mean circle with separately addressable lower and upper uncertainty arms.

Geometry is resolved in display space at draw time: the circle stays round and its
notches remain true transparent cuts under resizing, log scales and axis inversion.
The observations, colours and paired connectors share fluxbox's categorical layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D, IdentityTransform

from .. import tagger as _tagger
from ..descriptors import Mark
from ._categorical import (_cut_colour, _draw_units, _finish, _frame, _hex,
                           _palette_spec_json, _plain)

__all__ = ["meanmark", "MeanmarkResult"]


def _statistics(values, interval, confidence):
    v = values[np.isfinite(values)]
    if not len(v):
        return None
    n, mean = len(v), float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else None
    sem = sd / np.sqrt(n) if sd is not None else None
    name = getattr(interval, "__name__", "custom") if callable(interval) else interval
    if callable(interval):
        try:
            low, high = map(float, interval(v.copy()))
        except (TypeError, ValueError) as exc:
            raise ValueError("meanmark: interval callable must return (low, high) bounds") from exc
        if not np.isfinite([low, high]).all() or not low <= mean <= high:
            raise ValueError("meanmark: interval bounds must be finite and contain the mean")
    elif interval == "range":
        low, high = float(v.min()), float(v.max())
    elif sd is None:
        low = high = None  # a single observation cannot estimate sampling uncertainty
    else:
        half = sem if interval == "sem" else sd
        if interval == "ci":
            from scipy.stats import t
            half = float(t.ppf((1 + confidence) / 2, n - 1)) * sem
        low, high = mean - half, mean + half
    return dict(n=int(n), mean=mean, meanKind="arithmetic", sd=sd, sem=sem,
                interval=name, low=low, high=high,
                confidence=confidence if interval == "ci" else None)


def _circle_path(radius, depth, height):
    if depth == 0 or height == 0:
        return Path.unit_circle().transformed(Affine2D().scale(radius))
    # Remove the two equatorial wedges from the outline itself. The apex of each
    # V is on the exact mean, never a background-coloured overlay.
    theta = np.degrees(np.arcsin(height / (2 * radius)))
    upper = Path.arc(theta, 180 - theta)
    lower = Path.arc(180 + theta, 360 - theta)
    verts = list(upper.vertices * radius)
    codes = list(upper.codes)
    verts += [(-radius + depth, 0), tuple(lower.vertices[0] * radius)]
    codes += [Path.LINETO, Path.LINETO]
    verts += list(lower.vertices[1:] * radius)
    codes += list(lower.codes[1:])
    verts += [(radius - depth, 0), verts[0], verts[0]]
    codes += [Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    return Path(verts, codes)


class _MeanPart(PathPatch):
    """Point-sized geometry anchored to data, recalculated for every renderer."""

    def __init__(self, ax, position, mean, orientation, *, shape=None, bound=None,
                 radius=0, cap_size=None, **kwargs):
        self._ax, self.position, self.mean = ax, position, mean
        self.orientation, self.shape, self.bound = orientation, shape, bound
        self.radius, self.cap_size = radius, cap_size
        super().__init__(Path([[0, 0]], [Path.MOVETO]), transform=IdentityTransform(), **kwargs)

    def get_path(self):
        horizontal = self.orientation == "horizontal"
        xy = (self.mean, self.position) if horizontal else (self.position, self.mean)
        center = self._ax.transData.transform(xy)
        pt = self._ax.figure.dpi / 72
        if self.shape is not None:
            v = self.shape.vertices * pt
            if horizontal:
                v = v[:, ::-1]
            return Path(v + center, self.shape.codes)
        xy_end = (self.bound, self.position) if horizontal else (self.position, self.bound)
        end = self._ax.transData.transform(xy_end)
        axis = 0 if horizontal else 1
        delta = end[axis] - center[axis]
        if self.cap_size is not None:
            a, b = end.copy(), end.copy()
            a[1-axis] -= self.cap_size * pt / 2
            b[1-axis] += self.cap_size * pt / 2
            radius = self.radius * pt
            if abs(delta) < radius:
                # Keep even zero-width intervals from painting across a hollow
                # circle. A cap inside the disk retains only its exterior ends.
                cut = np.sqrt(radius**2 - delta**2)
                if self.cap_size * pt / 2 <= cut:
                    return Path([end], [Path.MOVETO])
                inner_a, inner_b = end.copy(), end.copy()
                inner_a[1-axis] -= cut
                inner_b[1-axis] += cut
                return Path([a, inner_a, inner_b, b],
                            [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO])
        else:
            # Arms meet the rim, so hollow or translucent circles never reveal a
            # line running through their centre. Short intervals stay inside it.
            a, b = center.copy(), end.copy()
            a[axis] += np.sign(delta) * min(abs(delta), self.radius * pt)
        return Path([a, b], [Path.MOVETO, Path.LINETO])


@dataclass
class MeanmarkResult:
    """Axes, exact statistics, identities, colours and independently editable artists."""

    ax: Any
    categories: list
    stats: dict
    group_colors: dict
    point_colors: dict
    series: dict
    unit_series: dict
    artists: dict = field(default_factory=dict)
    orientation: str = "vertical"

    @property
    def positions(self):
        """Category centres for significance brackets (vertical plots)."""
        return {str(c): float(i) for i, c in enumerate(self.categories)}

    def brackets(self, rows, **kw):
        from ..brackets import brackets
        if self.orientation != "vertical":
            raise ValueError("meanmark: significance brackets currently require a vertical plot")
        return brackets(self.ax, rows, positions=kw.pop("positions", self.positions), **kw)


def meanmark(
    data=None, *, x, y, units=None,
    order: Optional[Sequence] = None, unit_order: Optional[Sequence] = None, ax=None,
    interval: Union[str, Callable] = "ci", confidence: float = 0.95,
    orientation: str = "vertical", value_scale: Optional[str] = None,
    mark_size: float = 4.5, mark_edge_width: float = 1.5, mark_color=None,
    mark_fill_color="auto", mark_fill_alpha: float = 1.0, mark_fill_mix: float = 0.25,
    mark_offset: Optional[float] = None, mark_side: str = "right",
    show_notch: bool = False, notch_depth: float = 1.2, notch_height: float = 2.0,
    show_error: bool = True, error_width: float = 1.7, error_color=None,
    show_caps: bool = True, cap_size: float = 3.5, cap_width: float = 1.2, cap_color=None,
    show_individual_points: bool = True, point_size: float = 13.0, jitter: float = 0.06,
    point_edge="rim", point_edge_width: float = 0.2, point_fill_alpha: float = 0.45,
    palette=None, group_color_position: Optional[float] = None, group_color=None,
    point_colors="group", shade_range: tuple = (88.0, 22.0), interleave_shades: bool = True,
    connect_identical_points_across_x_values: Union[bool, str] = "auto",
    connect_line_width: float = 0.5, connect_color="#5f5f5f", connect_alpha: float = 0.2,
    series=None, unit_series=None, label_axes: bool = True, zorder: float = 2.0,
) -> MeanmarkResult:
    """Draw an arithmetic mean circle and uncertainty, optionally beside individual points.

    ``data``, ``x`` (categories), ``y`` (numeric values), ``units``, category/unit
    ordering, all point/colour/connector controls and series naming work as in
    :func:`fluxplot.fluxbox`. Missing/nonfinite values are excluded. ``units``
    identifies paired points; statistics use observations, without aggregating
    repeated rows per unit. Connectors break at missing categories.

    interval : {'sem', 'sd', 'ci', 'range'} or callable
        Mean ± sample SEM, mean ± sample SD, a two-sided Student-t
        confidence interval (default), the observed range, or ``values -> (low, high)``.
        Custom bounds are absolute data values, must be finite and contain the
        arithmetic mean. For n=1, SEM/SD/CI are unavailable: only the circle is
        drawn and bounds are null. Empty categories keep their tick but no mark.
    confidence : float
        Confidence level for ``interval='ci'`` (default .95).
    orientation : {'vertical', 'horizontal'}
        Direction of uncertainty. ``x`` remains categorical and ``y`` numeric
        in either orientation; a horizontal plot displays values on the x axis.
    value_scale : {None, 'linear', 'log'}
        Value-axis scale, or the existing scale. Statistics always use original
        data units, including on log axes. Log plots require positive observations
        and positive interval bounds; invalid bounds raise instead of clipping.
    mark_size, mark_edge_width : float
        Circle diameter and outline width in points, independent of axis limits.
    mark_color, mark_fill_color, mark_fill_alpha, mark_fill_mix
        Outline defaults to the group colour. Fill ``'auto'`` mixes that outline
        with the axes ground (fraction ``mark_fill_mix``, default .25), giving an
        opaque tinted interior. ``'stroke'`` makes a solid dot; ``'none'`` makes
        a hollow circle. Any matplotlib colour is accepted. Fill alpha affects
        only the interior, leaving the outline opaque.
    mark_offset, mark_side
        Offset in category units, default .25 with individual points or connectors,
        otherwise zero. Side defaults to ``'right'``; alternatives are
        ``'left'`` or ``'outer'`` (first group left, others right); for horizontal plots these mean below/above.
    show_notch, notch_depth, notch_height
        Cut two genuine V-shaped notches into the circle at the exact mean.
        Depth and full height are in points; depth must be less than the radius
        and height less than the diameter. Nothing paints over the cutouts.
    show_error, error_width, error_color
        Draw lower/upper arms independently. Width is in points; colour defaults
        to the outline. Arms end at exact bounds and start at the circle rim;
        intervals shorter than its radius are hidden inside the circle.
    show_caps, cap_size, cap_width, cap_color
        Optional endpoint caps, on by default. Length and width are in points;
        colour defaults to the arms. Caps are separate editable parts.

    connect_identical_points_across_x_values : {'auto', True, False}
        Default 'auto' connects paired observations when units is supplied.
        False disables connectors; True requires units. Lines default to .5 pt,
        colour #5f5f5f and alpha .2. Points default to group colours, 13 pt² area,
        .06 jitter, .2 pt rims and .45 fill alpha.

    Returns
    -------
    MeanmarkResult
        ``artists`` maps ``mean``, ``error_low``, ``error_high``, ``cap_low``,
        ``cap_high``, ``points`` and ``lines`` to artist lists. The saved SVG has
        stable category IDs ``.mean``, ``.error-low``, ``.error-high`` and optional
        ``.cap-low`` / ``.cap-high``. Every part is a separate manifest component;
        the mean is a shape (fill and stroke editable), arms and caps are lines.
        ``stats`` records the arithmetic mean and exact bounds, sample n/SD/SEM,
        orientation, scale and summary position. ``positions`` and ``brackets``
        support vertical significance annotations just as in fluxbox.
    """
    import matplotlib.pyplot as plt

    if orientation not in ("vertical", "horizontal"):
        raise ValueError("meanmark: orientation must be 'vertical' or 'horizontal'")
    if not callable(interval) and interval not in ("sem", "sd", "ci", "range"):
        raise ValueError("meanmark: interval must be 'sem', 'sd', 'ci', 'range' or a callable")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("meanmark: confidence must be between 0 and 1")
    if mark_side not in ("outer", "left", "right"):
        raise ValueError("meanmark: mark_side must be 'outer', 'left' or 'right'")
    for name, value in dict(mark_size=mark_size, mark_edge_width=mark_edge_width,
                            error_width=error_width, cap_size=cap_size, cap_width=cap_width,
                            notch_depth=notch_depth, notch_height=notch_height,
                            point_size=point_size, jitter=jitter, point_edge_width=point_edge_width,
                            connect_line_width=connect_line_width).items():
        if not np.isfinite(value) or value < 0 or (name == "mark_size" and value == 0):
            raise ValueError(f"meanmark: {name} must be finite and {'positive' if name == 'mark_size' else 'nonnegative'}")
    for name, value in dict(mark_fill_alpha=mark_fill_alpha, mark_fill_mix=mark_fill_mix,
                            connect_alpha=connect_alpha, point_fill_alpha=point_fill_alpha).items():
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"meanmark: {name} must be in [0, 1]")
    if show_notch and (notch_depth >= mark_size / 2 or notch_height >= mark_size):
        raise ValueError("meanmark: notch_depth must be less than the radius and notch_height less than mark_size")
    if show_notch and notch_depth > 0 and notch_height > 0:
        shoulder = np.sqrt((mark_size / 2)**2 - (notch_height / 2)**2)
        if mark_size / 2 - notch_depth >= shoulder:
            raise ValueError("meanmark: notch_depth is too shallow for notch_height to form an inward V")
    if connect_identical_points_across_x_values == "auto":
        connect_identical_points_across_x_values = units is not None
    elif not isinstance(connect_identical_points_across_x_values, (bool, np.bool_)):
        raise ValueError("meanmark: connect_identical_points_across_x_values must be 'auto', True or False")
    if mark_offset is None:
        mark_offset = 0.25 if show_individual_points or connect_identical_points_across_x_values else 0.0
    if not np.isfinite(mark_offset) or mark_offset < 0:
        raise ValueError("meanmark: mark_offset must be finite and nonnegative")
    ax = plt.gca() if ax is None else ax
    horizontal = orientation == "horizontal"
    axis = ax.xaxis if horizontal else ax.yaxis
    scale = axis.get_scale() if value_scale is None else value_scale
    if scale not in ("linear", "log"):
        raise ValueError("meanmark: value_scale must be 'linear' or 'log'")
    ground = _cut_colour(ax, None)
    fr = _frame("meanmark", data, x, y, units, order, unit_order, jitter=jitter, palette=palette,
                group_color=group_color, group_color_position=group_color_position,
                point_colors=point_colors, shade_range=shade_range,
                interleave_shades=interleave_shades, series=series, unit_series=unit_series,
                connect=connect_identical_points_across_x_values, point_fill_alpha=point_fill_alpha,
                ground=ground)
    # Validate every interval before drawing anything, including user callbacks.
    stats = {}
    for k, c in enumerate(fr.cats):
        vals = fr.ys[fr.rows_in[c]]
        st = _statistics(vals, interval, confidence)
        if st is None:
            continue
        if scale == "log" and (np.any(vals[np.isfinite(vals)] <= 0) or
                               any(st[key] is not None and st[key] <= 0 for key in ("low", "high"))):
            raise ValueError(f"meanmark: log scale requires positive observations and bounds in {c!r}")
        pos = fr.summary_x(k, mark_side, mark_offset)
        st.update(categoryPosition=pos, orientation=orientation, scale=scale,
                  x=st["mean"] if horizontal else pos, y=pos if horizontal else st["mean"],
                  groupColor=_hex(fr.group_colors[c]))
        stats[c] = st
    (ax.set_xscale if horizontal else ax.set_yscale)(scale)
    reg = _tagger.registry_for(ax.figure)
    artists = {key: [] for key in ("mean", "error_low", "error_high", "cap_low", "cap_high", "points", "lines")}
    artists["lines"], artists["points"] = _draw_units(
        fr, ax, reg, connect=connect_identical_points_across_x_values,
        connect_line_width=connect_line_width, connect_color=connect_color,
        connect_alpha=connect_alpha, show_points=show_individual_points, point_size=point_size,
        point_edge=point_edge, point_edge_width=point_edge_width,
        point_fill_alpha=point_fill_alpha, zorder=zorder, ground=ground, orientation=orientation)
    shape = _circle_path(mark_size / 2, notch_depth if show_notch else 0, notch_height)
    for c, st in stats.items():
        pos, mean = st["categoryPosition"], st["mean"]
        ink = to_rgba(fr.group_colors[c] if mark_color is None else mark_color)
        if isinstance(mark_fill_color, str) and mark_fill_color == "auto":
            fill = (*((1 - mark_fill_mix) * np.array(to_rgba(ground)[:3]) +
                      mark_fill_mix * np.array(ink[:3])), mark_fill_alpha)
        elif isinstance(mark_fill_color, str) and mark_fill_color == "stroke":
            fill = (*ink[:3], ink[3] * mark_fill_alpha)
        else:
            raw = to_rgba(mark_fill_color)
            fill = (*raw[:3], raw[3] * mark_fill_alpha)
        err_ink = ink if error_color is None else error_color
        payload = {"meanmark": {"part": "summary", "category": _plain(c), "units": fr.units_name,
                                "palette": _palette_spec_json(fr.palette_used.get(c)), **st}}

        def register(key, role, **kw):
            art = _MeanPart(ax, pos, mean, orientation, **kw)
            ax.add_patch(art)
            reg.add(Mark(role=role, series=fr.series_of[c], name=key.replace("_", "-"),
                         kind="meanmark", artists=[art], data=payload if key == "mean" else {}))
            artists[key].append(art)

        for side in ("low", "high"):
            bound = st[side]
            if show_error and bound is not None:
                register("error_" + side, "errorbar", bound=bound, radius=mark_size / 2,
                         facecolor="none", edgecolor=err_ink, linewidth=error_width,
                         capstyle="butt", zorder=zorder + 0.5)
                if show_caps:
                    register("cap_" + side, "cap", bound=bound, cap_size=cap_size, radius=mark_size / 2,
                             facecolor="none", edgecolor=err_ink if cap_color is None else cap_color,
                             linewidth=cap_width, capstyle="butt", zorder=zorder + 0.5)
        register("mean", "mean", shape=shape, facecolor=fill, edgecolor=ink,
                 linewidth=mark_edge_width, joinstyle="round", zorder=zorder + 1)
        bounds = [mean] + ([st["low"], st["high"]] if show_error and st["low"] is not None else [])
        ax.update_datalim([(v, pos) if horizontal else (pos, v) for v in bounds])
    _finish(fr, ax, label_axes, orientation)
    # Include intentionally large summary offsets when labelling the categorical axis.
    if label_axes and stats:
        lo = min(-0.75, min(s["categoryPosition"] for s in stats.values()) - .25)
        hi = max(len(fr.cats) - .25, max(s["categoryPosition"] for s in stats.values()) + .25)
        (ax.set_ylim if horizontal else ax.set_xlim)(lo, hi)
    return MeanmarkResult(ax, fr.cats, stats, fr.group_colors, fr.point_colors_by_unit(),
                          fr.series_of, fr.unit_series_of, artists, orientation)
