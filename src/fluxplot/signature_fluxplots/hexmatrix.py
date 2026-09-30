"""``fp.hexmatrix`` — hexagonal binning and hex-lattice maps, every hexagon a named part.

A hexmatrix tiles the plane with regular hexagons and colours each one by a value. It covers three
jobs with one mark:

* **density** of a point cloud (``x``, ``y``) — how many observations fall in each hexagon, or a
  density / probability / percent of them, optionally weighted (the classic hexbin);
* a **2D gradient** of a third variable (``C``) — the mean / median / sum / std / any reduction of
  ``C`` over the observations in each hexagon;
* a **hex matrix** (``matrix=``) — a 2D array drawn on an offset hex lattice (a SOM component plane,
  a hexagonal detector, a tiled spatial map), one hexagon per entry.

Everything shares one lattice: hexagons are addressed by ``(row, col)`` — ``row`` counts up the y
axis, ``col`` along x — so a hexagon has the same name in every plot made with the same extent and
grid, whatever the data. Hexagons are regular *on the page*: with ``aspect="auto"`` the axes' box
aspect is locked so a later layout pass cannot squash them; ``aspect="equal"`` bins in true data
units (spatial coordinates). Log axes bin in log space (``xscale="log"``), exactly as they display.

=======================  ===============================================  ===========
part                     default id                                       role
=======================  ===============================================  ===========
all the hexagons         ``<series>.hexes``                               ``x-hexbin``
one hexagon              ``<series>.hex.<row>.<col>``                     ``x-hex``
sparse / raw points      ``<series>-points.points``, ``….point.<k>``      ``point``
identity line            ``reference-line.identity``                      ``reference-line``
marginal histograms      ``<series>-x.bar.<k>`` / ``<series>-y.bar.<k>``  ``bar``
colour key               ``colorbar.color``                               ``colorbar``
=======================  ===============================================  ===========

Each hexagon also carries ``data-row``, ``data-column``, ``data-x``, ``data-y`` (its centre, in data
units), ``data-count`` and ``data-value`` in the SVG. The series' ``hexmatrix`` payload in the
manifest records the lattice (orientation, radius, extent, scales), the statistic and every bin drawn;
its ``field`` payload records the colormap and normalisation, and — like ``fp.heatmap`` — the
colormap and colour limits are recipe controls, so Flux's *Color scales* editor can change them and
regenerate.

Example
-------
>>> import fluxplot as fp, matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(figsize=(3, 3))
>>> hm = fp.hexmatrix(df, x="wake_rate", y="nrem_rate", ax=ax, xscale="log", yscale="log",
...                   norm="log", identity_line=True, colorbar_label="Synapses per hexbin")
>>> hm.bins["count"].max()
41
>>> fp.save(fig, "plots/rates.svg")
"""
from __future__ import annotations

import colorsys
import warnings
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from .. import ids as _ids
from .. import tagger as _tagger
from ..descriptors import Mark
from .glowbar import _column, _floats, _hex, _plain

__all__ = ["hexmatrix", "HexMatrixResult"]

SQRT3 = float(np.sqrt(3.0))
#: prefix of the single-colour ramps built from ``color=`` — recorded as the colormap's name so a
#: Flux regeneration (which replays the recorded name) rebuilds exactly the same map
MONO_PREFIX = "hexmatrix.mono:"
#: the same ramp turned for a dark ground: deep (near the ground) → the colour → pale
MONO_DARK_PREFIX = "hexmatrix.mono-dark:"
_STATS = ("count", "density", "probability", "percent")
_NORMS = ("linear", "log", "sqrt")


# ---------------------------------------------------------------------------
# the lattice
# ---------------------------------------------------------------------------
def _nearest(a, b, R):
    """Nearest centre of a pointy-top lattice of circumradius ``R`` → ``(row along b, col along a)``.

    The lattice is two offset rectangular lattices (even and odd rows); a point belongs to whichever
    centre is nearer, which is exactly the hexagon (the lattice's Voronoi cell) containing it. Ties go
    to the even row, deterministically.
    """
    dx, dy = SQRT3 * R, 1.5 * R
    i1, j1 = np.round(a / dx), np.round(b / (2 * dy))
    i2, j2 = np.round((a - dx / 2) / dx), np.round((b - dy) / (2 * dy))
    d1 = (a - i1 * dx) ** 2 + (b - 2 * j1 * dy) ** 2
    d2 = (a - i2 * dx - dx / 2) ** 2 + (b - (2 * j2 + 1) * dy) ** 2
    odd = d2 < d1
    return np.where(odd, 2 * j2 + 1, 2 * j1).astype(int), np.where(odd, i2, i1).astype(int)


def _centre(row, col, R):
    """Centre of pointy-top lattice cell ``(row, col)`` → ``(a, b)``."""
    row, col = np.asarray(row), np.asarray(col)
    return col * SQRT3 * R + (row % 2) * SQRT3 * R / 2, row * 1.5 * R


class _Lattice:
    """Maps data ↔ the unit plane the hexagons are regular in, for either orientation.

    ``u`` runs along x, ``v`` along y. A flat-top lattice is the pointy-top one with ``u``/``v``
    swapped, so ``row`` (the y index) and ``col`` (the x index) keep their meaning in both.
    """

    def __init__(self, *, R, orientation, x0, sx, y0, sy, xscale, yscale):
        self.R, self.orientation = R, orientation
        self.x0, self.sx, self.y0, self.sy = x0, sx, y0, sy  # u = (X - x0) / sx, on scaled X
        self.xscale, self.yscale = xscale, yscale

    @staticmethod
    def fwd(v, scale):
        return np.log10(v) if scale == "log" else np.asarray(v, dtype=float)

    @staticmethod
    def inv(v, scale):
        return 10.0 ** v if scale == "log" else v

    def to_unit(self, x, y):
        return ((self.fwd(x, self.xscale) - self.x0) / self.sx,
                (self.fwd(y, self.yscale) - self.y0) / self.sy)

    def to_data(self, u, v):
        return (self.inv(np.asarray(u) * self.sx + self.x0, self.xscale),
                self.inv(np.asarray(v) * self.sy + self.y0, self.yscale))

    def index(self, u, v):
        if self.orientation == "pointy":
            return _nearest(u, v, self.R)
        c, r = _nearest(v, u, self.R)
        return r, c

    def centre(self, row, col):
        if self.orientation == "pointy":
            return _centre(row, col, self.R)
        b, a = _centre(col, row, self.R)
        return a, b

    def polygons(self, row, col, shrink=1.0):
        """Hexagon vertices in data coordinates, one ``(6, 2)`` array per cell."""
        u, v = self.centre(row, col)
        start = 90.0 if self.orientation == "pointy" else 0.0
        ang = np.deg2rad(start + 60.0 * np.arange(6))
        du, dv = self.R * shrink * np.cos(ang), self.R * shrink * np.sin(ang)
        x, y = self.to_data(u[:, None] + du[None, :], v[:, None] + dv[None, :])
        return np.stack([x, y], axis=-1)

    def area(self):
        """Area of one hexagon in (scaled) data units²."""
        return 1.5 * SQRT3 * self.R ** 2 * self.sx * self.sy


# ---------------------------------------------------------------------------
# colour
# ---------------------------------------------------------------------------
def _mono_cmap(color, dark=False):
    """A single-hue ramp at the colour's own hue and saturation: pale → ``color`` → deep (seaborn's
    jointplot ramp, lightness 95 % → 12 %) on a light ground; on a dark ground (``dark=True``) the
    ramp is turned — deep (near the ground) → pale — so the fullest hexagons are the ones that
    stand out."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    h, _l, s = colorsys.rgb_to_hls(*to_rgb(color))
    lums = np.linspace(0.22, 0.95, 12) if dark else np.linspace(0.95, 0.12, 12)
    ramp = [colorsys.hls_to_rgb(h, lum, s) for lum in lums]
    return LinearSegmentedColormap.from_list((MONO_DARK_PREFIX if dark else MONO_PREFIX) + _hex(color), ramp)


def _resolve_cmap(spec):
    from .._fieldmap import resolve_colormap
    if isinstance(spec, str) and spec.startswith(MONO_DARK_PREFIX):
        return _mono_cmap(spec[len(MONO_DARK_PREFIX):], dark=True)
    if isinstance(spec, str) and spec.startswith(MONO_PREFIX):
        return _mono_cmap(spec[len(MONO_PREFIX):])
    return resolve_colormap(spec)


def _ground_is_dark(ax) -> bool:
    from ._colour import is_dark
    for c in (ax.get_facecolor(), ax.figure.get_facecolor()):
        if c[3] > 0:
            return is_dark(c)
    return False


def _auto_limits(vals, norm, robust, center):
    """Colour limits from the drawn values: min/max (positive only for log), robust percentiles,
    and symmetric about ``center`` when one is given."""
    v = vals[np.isfinite(vals)]
    if norm == "log":
        v = v[v > 0]
    if not v.size:
        return (1.0, 10.0) if norm == "log" else (0.0, 1.0)
    if robust:
        lo_p, hi_p = (2.0, 98.0) if robust is True else robust
        lo, hi = (float(q) for q in np.percentile(v, [lo_p, hi_p]))
    else:
        lo, hi = float(v.min()), float(v.max())
    if center is not None:
        half = max(abs(lo - center), abs(hi - center)) or 1.0
        lo, hi = center - half, center + half
    if lo == hi:
        lo, hi = (lo / 10.0, hi * 10.0) if norm == "log" else (lo - 0.5, hi + 0.5)
    return lo, hi


def _make_norm(norm, center, vmin, vmax, gamma):
    from matplotlib import colors as mcolors
    if center is not None and norm != "linear":
        # a TwoSlopeNorm IS the scale; it cannot also be log / sqrt / a caller's Normalize
        raise ValueError("hexmatrix: center= needs a linear norm")
    if isinstance(norm, mcolors.Normalize):
        return norm
    if center is not None:
        return mcolors.TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
    if norm == "log":
        return mcolors.LogNorm(vmin=vmin, vmax=vmax)
    if norm == "sqrt":
        return mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _plain_log_ticks(axis, lo, hi, subs=(1.0,)):
    """Label a log axis with plain numbers (0.01, 0.1, 1, 10 — not 10⁻²) while it spans no more
    than ~10⁻⁴…10⁵, where plain numbers stay short."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    if not (lo > 0 and hi > 0 and lo >= 1e-4 and hi <= 1e5):
        return
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _reducer(reduce):
    if callable(reduce):
        return reduce, getattr(reduce, "__name__", "custom")
    table = {
        "mean": np.mean, "median": np.median, "sum": np.sum, "min": np.min, "max": np.max,
        "std": lambda v: float(np.std(v, ddof=1)) if v.size > 1 else float("nan"),
        "count": lambda v: float(v.size),
    }
    if reduce not in table:
        raise ValueError(f"hexmatrix: reduce must be one of {sorted(table)} or a callable; got {reduce!r}")
    return table[reduce], reduce


def _bin_points(lat, u, v, C, weights, reduce):
    """Group points by hexagon → ``(rows, cols, counts, weighted, value, inverse)``; bins in
    ``(row, col)`` order."""
    rows, cols = lat.index(u, v)
    keys, inv = np.unique(np.stack([rows, cols], axis=1), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    counts = np.bincount(inv, minlength=len(keys)).astype(float)
    weighted = counts if weights is None else np.bincount(inv, weights=weights, minlength=len(keys))
    value = None
    if C is not None:
        fn, name = _reducer(reduce)
        if name in ("mean", "sum") and weights is None:
            s = np.bincount(inv, weights=C, minlength=len(keys))
            value = s / counts if name == "mean" else s
        else:
            order = np.argsort(inv, kind="stable")
            groups = np.split(C[order], np.cumsum(counts.astype(int))[:-1])
            value = np.array([float(fn(g)) for g in groups])
    return keys[:, 0], keys[:, 1], counts, weighted, value, inv


def _lattice_cells(lat, u_lo, u_hi, v_lo, v_hi):
    """Every lattice cell whose centre lies inside the unit-space box (for ``mincnt=0``)."""
    R = lat.R
    span = int(np.ceil(max(u_hi - u_lo, v_hi - v_lo) / R)) + 4
    r, c = np.meshgrid(np.arange(-2, span), np.arange(-2, span), indexing="ij")
    r, c = r.ravel(), c.ravel()
    u, v = lat.centre(r, c)
    eps = 1e-9 * R
    keep = (u >= u_lo - eps) & (u <= u_hi + eps) & (v >= v_lo - eps) & (v <= v_hi + eps)
    return r[keep], c[keep]


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------
@dataclass
class HexMatrixResult:
    """What :func:`hexmatrix` drew — the axes, the hexagons and everything needed to reuse them."""

    ax: Any
    #: the hexagons (a matplotlib PolyCollection; also the colorbar mappable)
    hexes: Any
    #: one entry per drawn hexagon, in ``(row, col)`` order: ``row``, ``col``, ``x``, ``y`` (centre,
    #: data units), ``count`` (observations; ``None`` for a matrix) and ``value`` (what is coloured)
    bins: dict
    #: the colormap and normalisation actually used (after any Flux recipe override)
    cmap: Any
    norm: Any
    #: the series name — the root of every part id
    series: str
    #: the recipe key under which Flux edits the colormap and limits
    control_key: str
    colorbar: Any = None
    #: ``{"x": ax_top, "y": ax_right}`` when marginals were drawn
    marginal_axes: dict = field(default_factory=dict)
    #: the raw points drawn (``show_points`` / ``sparse``), a PathCollection or ``None``
    points: Any = None
    #: every part → its matplotlib artist(s)
    artists: dict = field(default_factory=dict)

    def hex_id(self, row, col) -> str:
        """The SVG id of hexagon ``(row, col)`` (before any panel prefix)."""
        return f"{_ids.series_root(self.series)}.hex.{int(row)}.{int(col)}"

    def lookup(self, x, y):
        """``(row, col)`` of the hexagon containing data point ``(x, y)``."""
        lat = self.artists["lattice"]
        u, v = lat.to_unit(np.atleast_1d(x), np.atleast_1d(y))
        r, c = lat.index(u, v)
        return int(r[0]), int(c[0])


# ---------------------------------------------------------------------------
# the plot
# ---------------------------------------------------------------------------
def hexmatrix(
    data=None,
    *,
    x=None,
    y=None,
    C=None,
    matrix=None,
    ax=None,
    # the lattice
    gridsize: int = 30,
    binwidth: Optional[float] = None,
    extent: Optional[Sequence[float]] = None,
    orientation: str = "pointy",
    aspect: Union[str, float] = "auto",
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    origin: str = "upper",
    # what is coloured
    stat: str = "count",
    reduce: Union[str, Callable] = "mean",
    weights=None,
    mincnt: int = 1,
    # the colour scale
    cmap=None,
    color=None,
    norm: Any = "linear",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    robust: Union[bool, Sequence[float]] = False,
    center: Optional[float] = None,
    gamma: float = 0.5,
    # the hexagons
    gap: float = 0.0,
    edgecolor="face",
    linewidth: Optional[float] = None,
    alpha: Optional[float] = None,
    # points
    sparse: Optional[int] = None,
    show_points: bool = False,
    point_size: float = 4.0,
    point_color=None,
    # furniture
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    colorbar_size: float = 0.05,
    colorbar_pad: float = 0.04,
    marginals: Union[bool, str] = False,
    marginal_size: float = 0.22,
    marginal_pad: float = 0.03,
    marginal_bins: Optional[int] = None,
    marginal_color=None,
    identity_line: Union[bool, dict] = False,
    # identity
    series: Optional[str] = None,
    key: Optional[str] = None,
    scale: Optional[str] = None,
    alpha_by=None,
    alpha_range=(0.25, 1.0),
    alpha_norm: str = "linear",
    vector_limit: Optional[int] = 5000,
    label_axes: bool = True,
    zorder: float = 2.0,
) -> HexMatrixResult:
    """Draw a hexmatrix: hexagonal bins of a point cloud, or a 2D array on a hex lattice.

    Parameters
    ----------
    data
        A pandas or polars DataFrame, a dict of columns, or ``None`` when ``x``/``y``/``C`` are
        passed as arrays.
    x, y
        Column names (or array-likes) of the point coordinates. Rows with a missing / non-finite
        coordinate (or a non-positive one on a log axis) are dropped and counted in the manifest.
    C
        Optional third variable (column name or array). Each hexagon is coloured by ``reduce`` of the
        ``C`` values that fall in it instead of by a count — a 2D gradient / mean map.
    matrix
        Instead of points: a 2D array drawn one hexagon per entry on an offset lattice (row ``r`` of
        the array is lattice row ``r``; odd rows are shifted half a hexagon). ``NaN`` entries are left
        empty. Excludes ``x``/``y``/``C``.
    ax
        Target axes (default: the current axes).

    gridsize, binwidth
        Hexagon size: ``gridsize`` hexagons across the x extent (default ``30``), or ``binwidth``, the
        centre-to-centre distance of neighbouring hexagons in (scaled) x data units — e.g. ``0.1`` for
        0.1 mm bins of spatial data, or ``0.1`` decades on a log axis. ``binwidth`` wins.
    extent
        ``(xmin, xmax, ymin, ymax)`` of the binned region in data units (default: the data range).
        Points outside are dropped. Fix it to make hexagon names (and colours) comparable across
        plots of different data.
    orientation
        ``"pointy"`` (pointy-top hexagons in horizontal rows — default) or ``"flat"``.
    aspect
        ``"auto"`` (default): hexagons are regular on the page for the axes' current box shape, whose
        aspect is then locked (``Axes.set_box_aspect``) so a later layout pass cannot distort them.
        ``"equal"`` (or a number, the y-per-x data ratio): hexagons are regular in data units and the
        axes' data aspect is set — right for spatial coordinates.
    xscale, yscale
        ``"linear"`` or ``"log"`` (default: the axes' current scale). Log axes bin in log space, so the
        hexagons are regular as displayed.
    origin
        Matrix mode only: ``"upper"`` (row 0 at the top, like ``imshow`` — default) or ``"lower"``.

    stat
        Without ``C``: what a hexagon's colour counts — ``"count"`` (default), ``"density"`` (count per
        unit (scaled) data area, integrating to 1), ``"probability"`` (fraction of all points) or
        ``"percent"``.
    reduce
        With ``C``: ``"mean"`` (default), ``"median"``, ``"sum"``, ``"min"``, ``"max"``, ``"std"``,
        ``"count"`` or any callable taking a 1D array.
    weights
        Optional per-point weights (column name or array) for the count statistics.
    mincnt
        Minimum number of observations for a hexagon to be drawn (default ``1``: empty hexagons are
        left empty). ``0`` draws every hexagon of the extent, empty ones at a count of zero.

    cmap, color
        The colormap: any matplotlib or fluxplot colormap name (``"viridis"``, ``"emerald"``,
        ``"crameri.batlow"``) or a Colormap; default: the style's image colormap. Or ``color``: one
        colour, from which a pale → colour → deep single-hue ramp is built.
    norm
        How values map to colour: ``"linear"`` (default), ``"log"``, ``"sqrt"`` (a power law,
        ``gamma``), or any matplotlib ``Normalize`` instance (``SymLogNorm``, ``BoundaryNorm``, …).
    vmin, vmax
        Colour limits (default: the range of the drawn values). Flux can override them (and ``cmap``)
        through the recipe's colour controls.
    robust
        Derive default limits from the 2nd–98th percentile (``True``) or a ``(low, high)`` percentile
        pair instead of the extremes, so a few outlying hexagons do not wash out the rest.
    center
        A value to centre a diverging map on (``TwoSlopeNorm``); default limits become symmetric.
    gamma
        Exponent of ``norm="sqrt"`` (default ``0.5``).

    gap
        Fraction of each hexagon trimmed away to leave a gap between neighbours (``0`` — seamless —
        to ``<1``).
    edgecolor, linewidth
        Hexagon outline: ``"face"`` (the fill colour — default; closes antialiasing seams), ``"none"``
        or a colour; and its width in points (default ``0.25`` when seamless, ``0`` with a gap).
    alpha
        Hexagon opacity.

    sparse
        Draw hexagons only where at least ``sparse`` observations fall, and the individual points
        elsewhere — a density map whose sparse fringe stays honest.
    show_points
        Overlay every observation as a small point.
    point_size, point_color
        Marker area (points²) and colour of those points (default: the colour of the lowest value).

    colorbar, colorbar_label
        Attach a named colour key (default ``True``) and its label (default: the statistic, e.g.
        ``"Count per hexbin"``, or ``"mean <C>"``).
    colorbar_size, colorbar_pad
        Its width and gap, as fractions of the plot's width.
    marginals
        ``True`` / ``"hist"``: histograms of ``x`` and ``y`` along the top and right (points mode).
    marginal_size, marginal_pad
        Their depth and gap, as fractions of the plot's size.
    marginal_bins, marginal_color
        Number of marginal bins across the x extent (default: ``gridsize``; y gets the same bin
        width) and their colour (default: a mid tone of the colormap, or ``color``).
    identity_line
        Draw the ``y = x`` line across the plot (``True``, or a dict of ``Line2D`` properties).

    series
        The series name — the root of every part id (default ``"hexbin"``).
    key
        The recipe key of the colour controls (default: the series).
    scale
        Join a shared colour scale declared with :func:`fluxplot.color_scale` (its map, norm and
        union limits; one key for every panel).
    alpha_by, alpha_range, alpha_norm
        Value × confidence: wash out hexagons by a second variable. ``"count"`` (observations
        per hexagon), a column of ``data`` or a per-point array (its mean per hexagon), a
        per-hexagon array (one value per drawn hexagon), or a matrix of the ``matrix`` shape.
        Alpha runs over ``alpha_range`` with the value (``alpha_norm="log"`` for p-values); the
        manifest records ``colorScales[].alpha`` and every hexagon carries ``data-alpha-value``.
    vector_limit
        Hexagons stay individually addressable up to this many (default 5000; ``None`` for the
        save's generic threshold), where a generic collection would be rasterized at 800.
    label_axes
        Label the axes with the ``x`` / ``y`` column names (default ``True``).
    zorder
        z-order of the hexagons; points and the identity line sit above.

    Returns
    -------
    HexMatrixResult
        The axes, the hexagons, the bins drawn, the colour scale, the colorbar and the marginal axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from ..fields import _options
    from ..fields import colorbar as _colorbar

    who = "hexmatrix"
    if ax is None:
        ax = plt.gca()
    if orientation not in ("pointy", "flat"):
        raise ValueError(f"{who}: orientation must be 'pointy' or 'flat'; got {orientation!r}")
    if stat not in _STATS:
        raise ValueError(f"{who}: stat must be one of {_STATS}; got {stat!r}")
    if not (isinstance(norm, str) and norm in _NORMS) and not hasattr(norm, "autoscale_None"):
        raise ValueError(f"{who}: norm must be one of {_NORMS} or a matplotlib Normalize; got {norm!r}")
    if not 0.0 <= gap < 1.0:
        raise ValueError(f"{who}: gap must be in [0, 1); got {gap!r}")
    if cmap is not None and color is not None:
        raise ValueError(f"{who}: give cmap or color, not both")
    if marginals not in (False, True, "hist"):
        raise ValueError(f"{who}: marginals must be False, True or 'hist'; got {marginals!r}")
    series = str(series) if series is not None else "hexbin"
    matrix_mode = matrix is not None
    if matrix_mode and any(v is not None for v in (x, y, C, weights)):
        raise ValueError(f"{who}: matrix= excludes x, y, C and weights")
    if not matrix_mode and (x is None or y is None):
        raise ValueError(f"{who}: give x and y (points), or matrix= (a 2D array)")
    if matrix_mode and marginals:
        raise ValueError(f"{who}: marginals need points (x, y), not a matrix")

    xscale = xscale or ax.get_xscale()
    yscale = yscale or ax.get_yscale()
    for name, sc in (("xscale", xscale), ("yscale", yscale)):
        if sc not in ("linear", "log"):
            raise ValueError(f"{who}: {name} must be 'linear' or 'log'; got {sc!r}")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    fwd = _Lattice.fwd

    # ---- the lattice ------------------------------------------------------------------------------------
    info: dict = {}
    if matrix_mode:
        M = np.ma.masked_invalid(np.ma.asarray(matrix, dtype=float))
        if M.ndim != 2 or not M.size:
            raise ValueError(f"{who}: matrix must be a nonempty 2D array")
        if origin not in ("upper", "lower"):
            raise ValueError(f"{who}: origin must be 'upper' or 'lower'; got {origin!r}")
        nr, nc = M.shape
        # unit radius, data units = unit units; flip rows for origin="upper" without renaming them
        lat = _Lattice(R=1.0, orientation=orientation, x0=0.0, sx=1.0,
                       y0=0.0, sy=1.0, xscale="linear", yscale="linear")
        rr, cc = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
        rows, cols = rr.ravel(), cc.ravel()
        vals = M.filled(np.nan).ravel()
        keep = np.isfinite(vals)
        rows, cols, value = rows[keep], cols[keep], vals[keep]
        counts = None
        alpha_src = None
        if alpha_by is not None:
            if isinstance(alpha_by, str):
                raise ValueError(f"{who}: alpha_by must be a matrix of shape {M.shape} in matrix mode")
            A = np.ma.filled(np.ma.asarray(alpha_by, dtype=float), np.nan)
            if A.shape != M.shape:
                raise ValueError(f"{who}: alpha_by must have the matrix shape {M.shape}, got {A.shape}")
            alpha_src = A.ravel()[keep]
        u, v = lat.centre(rr.ravel(), cc.ravel())
        if origin == "upper":
            lat.sy, lat.y0 = -1.0, float(v.max())  # v → -(Y - y0): row 0 at the top
        xs_all, ys_all = lat.to_data(u, v)
        pad = 1.0
        xlim = (float(xs_all.min()) - pad, float(xs_all.max()) + pad)
        ylim = (float(ys_all.min()) - pad, float(ys_all.max()) + pad)
        ax.set_aspect("equal")
        n_used = dropped = None
        info.update(mode="matrix", shape=[int(nr), int(nc)], origin=origin)
        x_name = y_name = c_name = None
    else:
        xv, x_name = _column(who, data, x, "x")
        yv, y_name = _column(who, data, y, "y")
        cv, c_name = _column(who, data, C, "C")
        wv, _w = _column(who, data, weights, "weights")
        xs = _floats(who, xv, "x")
        ys = _floats(who, yv, "y")
        n = len(xs)
        if len(ys) != n:
            raise ValueError(f"{who}: x and y have different lengths ({n} vs {len(ys)})")
        cs = _floats(who, cv, "C") if cv is not None else None
        ws = _floats(who, wv, "weights") if wv is not None else None
        for arr, what in ((cs, "C"), (ws, "weights")):
            if arr is not None and len(arr) != n:
                raise ValueError(f"{who}: {what} has {len(arr)} values for {n} points")
        ok = np.isfinite(xs) & np.isfinite(ys)
        if xscale == "log":
            ok &= xs > 0
        if yscale == "log":
            ok &= ys > 0
        if cs is not None:
            ok &= np.isfinite(cs)
        if ws is not None:
            ok &= np.isfinite(ws)
        if extent is not None:
            if len(extent) != 4:
                raise ValueError(f"{who}: extent must be (xmin, xmax, ymin, ymax)")
            e = [float(v) for v in extent]
            ok &= (xs >= e[0]) & (xs <= e[1]) & (ys >= e[2]) & (ys <= e[3])
        if not ok.any():
            raise ValueError(f"{who}: no finite points to bin")
        dropped = int(n - ok.sum())
        xs, ys = xs[ok], ys[ok]
        cs = cs[ok] if cs is not None else None
        ws = ws[ok] if ws is not None else None
        n_used = int(xs.size)
        X, Y = fwd(xs, xscale), fwd(ys, yscale)
        if extent is not None:
            X0, X1 = fwd(e[0], xscale), fwd(e[1], xscale)
            Y0, Y1 = fwd(e[2], yscale), fwd(e[3], yscale)
        else:
            X0, X1, Y0, Y1 = float(X.min()), float(X.max()), float(Y.min()), float(Y.max())
        if X1 <= X0:
            X0, X1 = X0 - 0.5, X1 + 0.5
        if Y1 <= Y0:
            Y0, Y1 = Y0 - 0.5, Y1 + 0.5
        across = SQRT3 if orientation == "pointy" else 1.5  # x spacing of hexagon centres, per R
        if isinstance(gridsize, bool) or not isinstance(gridsize, (int, np.integer)) or gridsize < 1:
            raise ValueError(f"{who}: gridsize must be a positive integer; got {gridsize!r}")
        if binwidth is not None and not (isinstance(binwidth, Real) and binwidth > 0):
            raise ValueError(f"{who}: binwidth must be a positive number; got {binwidth!r}")
        if aspect == "auto":
            # unit plane = the axes box: u ∈ [0, 1] across the x limits, v ∈ [0, ρ] up the y limits,
            # with one circumradius of padding all round so no hexagon is clipped
            shared = len(ax.get_shared_x_axes().get_siblings(ax)) > 1 or len(ax.get_shared_y_axes().get_siblings(ax)) > 1
            rho = ax.get_box_aspect()
            if rho is None:
                pos = ax.get_position()
                fw, fh = ax.figure.get_size_inches()
                rho = (pos.height * fh) / (pos.width * fw)
            rho = float(rho)
            if binwidth is not None:
                k = binwidth / SQRT3 / (X1 - X0)
                R = k / (1 + 2 * k)
            else:
                R = 1.0 / (gridsize * across + 2.0)
            if rho <= 2 * R:
                raise ValueError(f"{who}: the axes are too flat for hexagons this large; raise gridsize")
            sx, sy = (X1 - X0) / (1 - 2 * R), (Y1 - Y0) / (rho - 2 * R)
            lat = _Lattice(R=R, orientation=orientation, x0=X0 - R * sx, sx=sx,
                           y0=Y0 - R * sy, sy=sy, xscale=xscale, yscale=yscale)
            if shared:
                # locking the box aspect of an axes that shares x or y would distort its siblings;
                # the hexagons are regular now, but a later layout pass may squash them
                warnings.warn(f"{who}: aspect='auto' on axes sharing x or y leaves the box aspect unlocked; "
                              "hexagons may be distorted by a later layout pass (pass aspect='equal' or a number)",
                              stacklevel=2)
            else:
                ax.set_box_aspect(rho)
            info.update(boxAspect=rho)
        else:
            a = 1.0 if aspect == "equal" else aspect
            if isinstance(a, bool) or not isinstance(a, Real) or a <= 0:
                raise ValueError(f"{who}: aspect must be 'auto', 'equal' or a positive number; got {aspect!r}")
            a = float(a)
            R = (binwidth / SQRT3) if binwidth is not None else (X1 - X0) / (gridsize * across)
            lat = _Lattice(R=R, orientation=orientation, x0=X0 - R, sx=1.0,
                           y0=Y0 - R * a, sy=a, xscale=xscale, yscale=yscale)
            ax.set_aspect(a)
        xlim = tuple(float(v) for v in lat.inv(np.array([X0 - lat.R * lat.sx, X1 + lat.R * lat.sx]), xscale))
        ylim = tuple(float(v) for v in lat.inv(np.array([Y0 - lat.R * lat.sy, Y1 + lat.R * lat.sy]), yscale))

        u, v = lat.to_unit(xs, ys)
        rows, cols, counts, weighted, value, inv = _bin_points(lat, u, v, cs, ws, reduce)
        per_point = counts[inv]  # observations sharing each point's hexagon
        # the alpha variable per hexagon: the count, or the mean of a per-point variable
        alpha_src = None
        if alpha_by is not None:
            if isinstance(alpha_by, str) and alpha_by == "count":
                alpha_src = counts.copy()
            else:
                av, _ = _column(who, data, alpha_by, "alpha_by") if isinstance(alpha_by, str) else (list(alpha_by), None)
                av = np.asarray(_floats(who, av, "alpha_by"), dtype=float)
                if av.size == n and n != len(rows):  # one value per point: the mean per hexagon
                    av = av[ok]
                    ok = np.isfinite(av)
                    sums = np.bincount(inv[ok], weights=av[ok], minlength=len(counts))
                    nn = np.bincount(inv[ok], minlength=len(counts))
                    alpha_src = np.where(nn > 0, sums / np.maximum(nn, 1), np.nan)
                elif av.size == len(rows):
                    alpha_src = av  # one value per hexagon, in bin order (rows then cols)
                else:
                    raise ValueError(f"{who}: alpha_by must be 'count', a column / array with one value per point "
                                     f"({n}) or one per hexagon ({len(rows)}); got {av.size}")
        if mincnt == 0:
            er, ec = _lattice_cells(lat, lat.R, (X1 - X0) / lat.sx + lat.R,
                                    lat.R, (Y1 - Y0) / lat.sy + lat.R)
            have = set(zip(rows.tolist(), cols.tolist()))
            extra = [(r, c) for r, c in zip(er.tolist(), ec.tolist()) if (r, c) not in have]
            if extra:
                er, ec = np.array(extra, dtype=int).T
                rows = np.r_[rows, er]
                cols = np.r_[cols, ec]
                counts = np.r_[counts, np.zeros(len(er))]
                weighted = np.r_[weighted, np.zeros(len(er))]
                if value is not None:
                    value = np.r_[value, np.full(len(er), np.nan)]
                if alpha_src is not None:
                    alpha_src = np.r_[alpha_src, np.full(len(er), np.nan)]
                order = np.lexsort((cols, rows))
                rows, cols, counts, weighted = rows[order], cols[order], counts[order], weighted[order]
                value = value[order] if value is not None else None
                alpha_src = alpha_src[order] if alpha_src is not None else None
        if value is None:
            total = float(weighted.sum()) or 1.0
            value = {"count": weighted, "probability": weighted / total, "percent": 100 * weighted / total,
                     "density": weighted / (total * lat.area())}[stat]
        # which hexagons are drawn; the points of the rest (sparse) are drawn as points
        drawn = counts >= max(int(mincnt), 0)
        sparse_pts = None
        if sparse is not None:
            drawn &= counts >= int(sparse)
            sparse_pts = per_point < int(sparse)
        rows, cols, counts, value = rows[drawn], cols[drawn], counts[drawn], np.asarray(value)[drawn]
        alpha_src = alpha_src[drawn] if alpha_src is not None else None
        info.update(mode="points", n=n_used, dropped=dropped, stat=None if cs is not None else stat,
                    reduce=None if cs is None else _reducer(reduce)[1],
                    weighted=ws is not None, mincnt=int(mincnt),
                    sparse=None if sparse is None else int(sparse),
                    extent=[_plain(float(v)) for v in (lat.inv(X0, xscale), lat.inv(X1, xscale),
                                                       lat.inv(Y0, yscale), lat.inv(Y1, yscale))],
                    hexArea=float(lat.area()))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if not matrix_mode:
        for axis, sc, lims in ((ax.xaxis, xscale, xlim), (ax.yaxis, yscale, ylim)):
            if sc == "log":
                _plain_log_ticks(axis, *lims)

    # ---- the colour scale ------------------------------------------------------------------------------
    dark_ground = _ground_is_dark(ax)
    base_cmap = _mono_cmap(color, dark=dark_ground) if color is not None else _resolve_cmap(cmap)
    norm_kind = norm if isinstance(norm, str) else None
    lo, hi = _auto_limits(np.asarray(value, dtype=float), norm_kind, robust, center)
    if hasattr(norm, "autoscale_None"):
        lo = norm.vmin if norm.vmin is not None else lo
        hi = norm.vmax if norm.vmax is not None else hi
    opts = {"cmap": base_cmap.name, "vmin": vmin if vmin is not None else lo,
            "vmax": vmax if vmax is not None else hi,
            "norm": _make_norm(norm, center, None, None, gamma)}
    control_key = _options(ax, series, key, opts, resolve=_resolve_cmap, scale=scale)  # any Flux recipe override
    cb_extend = opts.pop("_extend", None)
    the_cmap = base_cmap if opts["cmap"] == base_cmap.name else _resolve_cmap(opts["cmap"])
    the_norm = opts["norm"]
    if "vmin" in opts:  # _options moved the limits onto the (copied) Normalize itself
        the_norm.vmin, the_norm.vmax = opts["vmin"], opts["vmax"]
    if scale is not None and the_norm.vmin is None:  # a shared scale's limits are resolved at save
        the_norm.vmin, the_norm.vmax = lo, hi
    if center is not None and not (the_norm.vmin < center < the_norm.vmax):
        raise ValueError(f"{who}: center must lie between vmin and vmax "
                         f"(center={center!r}, vmin={the_norm.vmin!r}, vmax={the_norm.vmax!r})")

    # ---- the hexagons ----------------------------------------------------------------------------------
    polys = lat.polygons(rows, cols, shrink=1.0 - gap)
    lw = linewidth if linewidth is not None else (0.25 if gap == 0 else 0.0)
    hexes = PolyCollection(list(polys), array=np.asarray(value, dtype=float), cmap=the_cmap,
                           norm=the_norm, edgecolors=edgecolor, linewidths=lw, alpha=alpha,
                           zorder=zorder)
    ax.add_collection(hexes, autolim=False)
    if scale is not None:
        from ..fields import join_scale
        join_scale(ax, scale, hexes, np.asarray(value, dtype=float))
    alpha_data: dict = {}
    if alpha_src is not None:
        from ..fields import apply_alpha
        source = alpha_by if isinstance(alpha_by, str) else "alpha_by"
        apply_alpha(hexes, alpha_data, alpha_src, alpha_range=alpha_range, alpha_norm=alpha_norm, source=source)
    cx, cy = lat.to_data(*lat.centre(rows, cols))
    bins = {"row": rows.astype(int), "col": cols.astype(int), "x": np.asarray(cx, dtype=float),
            "y": np.asarray(cy, dtype=float), "count": counts, "value": np.asarray(value, dtype=float)}
    names = [f"{int(r)}.{int(c)}" for r, c in zip(rows, cols)]
    boxes = [(float(np.min(v[:, 0])), float(np.max(v[:, 0])), float(np.min(v[:, 1])), float(np.max(v[:, 1])))
             for v in (np.asarray(poly, dtype=float) for poly in polys)]  # each hexagon's data-space box
    attrs = [{"data_row": int(r), "data_column": int(c), "data_key": f"{int(r)}.{int(c)}", "data_x": _plain(float(px)),
              "data_y": _plain(float(py)), "data_count": None if counts is None else _plain(float(k)),
              "data_value": _plain(float(val)),
              "data_x0": _plain(b[0]), "data_x1": _plain(b[1]), "data_y0": _plain(b[2]), "data_y1": _plain(b[3])}
             for r, c, px, py, k, val, b in zip(rows, cols, cx, cy,
                                                 counts if counts is not None else [None] * len(rows), value, boxes)]
    if alpha_src is not None:
        for a, av in zip(attrs, alpha_src):
            a["data_alpha_value"] = None if not np.isfinite(av) else _plain(float(av))
    payload = {"orientation": orientation, "hexRadius": float(lat.R),
               "aspect": aspect if isinstance(aspect, str) else float(aspect),
               "scale": {"x": xscale if not matrix_mode else "linear",
                         "y": yscale if not matrix_mode else "linear"},
               "gridsize": None if (matrix_mode or binwidth is not None) else int(gridsize),
               "binwidth": None if binwidth is None else float(binwidth),
               "valueLabel": None, "nBins": int(len(rows)), **info,
               "bins": [{"row": int(r), "col": int(c), "x": _plain(float(px)), "y": _plain(float(py)),
                         "count": None if counts is None else _plain(float(k)),
                         "value": _plain(float(val))}
                        for r, c, px, py, k, val in zip(rows, cols, cx, cy,
                                                         counts if counts is not None else [None] * len(rows),
                                                         value)]}
    reg = _tagger.registry_for(ax.figure)
    field_config = {"kind": "hexbin", "controlKey": control_key, "shape": [int(len(rows))]}
    if cb_extend:
        field_config["extend"] = cb_extend
    mark_data = {"field_config": field_config, "field_artist": hexes, "field_resolve": _resolve_cmap,
                 "field_names": names, "field_member_prefix": "hex",
                 "field_member_role": "x-hex", "field_attrs": attrs, "hexmatrix": payload, **alpha_data}
    if vector_limit is not None:
        mark_data["raster_threshold"] = int(vector_limit)  # hexagons stay addressable up to this many
    reg.add(Mark(role="x-hexbin", series=series, name="hexes", kind="hexmatrix", artists=[hexes], data=mark_data))
    artists = {"hexes": hexes, "lattice": lat}

    # ---- points ---------------------------------------------------------------------------------------
    points = None
    if not matrix_mode and (show_points or sparse is not None):
        sel = np.ones(n_used, bool) if show_points else sparse_pts
        if sel.any():
            pc = point_color if point_color is not None else the_cmap(0.0)
            points = ax.scatter(xs[sel], ys[sel], s=point_size, color=[pc], linewidths=0,
                                zorder=zorder + 0.5)
            reg.add(Mark(role="point", series=f"{series}-points", kind="scatter", live_data=True,
                         artists=[points], indexed=True))
            artists["points"] = points

    # ---- identity line --------------------------------------------------------------------------------
    if identity_line:
        lo_d, hi_d = max(xlim[0], ylim[0]), min(xlim[1], ylim[1])
        if lo_d < hi_d:
            props = {"color": "0.7" if dark_ground else "0.35", "lw": 0.8, "ls": (0, (4, 3)), "zorder": zorder + 1}
            if isinstance(identity_line, dict):
                props.update(identity_line)
            (line,) = ax.plot([lo_d, hi_d], [lo_d, hi_d], **props)
            reg.add(Mark(role="reference-line", name="identity", artists=[line],
                         data={"x": None, "y": None, "slope": 1.0, "intercept": 0.0}))
            artists["identity"] = line

    # ---- labels ---------------------------------------------------------------------------------------
    if label_axes and not matrix_mode:
        if x_name and not ax.get_xlabel():
            ax.set_xlabel(x_name)
        if y_name and not ax.get_ylabel():
            ax.set_ylabel(y_name)
    if matrix_mode:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    if colorbar_label is None:
        if matrix_mode:
            colorbar_label = ""
        elif cs is not None:
            colorbar_label = f"{info['reduce']} {c_name}" if c_name else info["reduce"]
        else:
            colorbar_label = {"count": "Count per hexbin", "density": "Density",
                              "probability": "Probability", "percent": "Percent"}[stat]
    payload["valueLabel"] = colorbar_label or None
    field_config["label"] = colorbar_label or None

    # ---- marginals ------------------------------------------------------------------------------------
    marginal_axes = {}
    right_edge = 1.0
    if marginals:
        from ..api import hist as _hist
        from ..panels import panel as _panel
        mcol = marginal_color if marginal_color is not None else (
            color if color is not None else the_cmap(0.6))
        nb = int(marginal_bins or gridsize)
        width = (X1 - X0) / nb
        ex = lat.inv(np.linspace(X0, X1, nb + 1), xscale)
        ey = lat.inv(np.arange(Y0, Y1 + width, width) if Y1 > Y0 else np.array([Y0, Y1]), yscale)
        top = ax.inset_axes([0, 1 + marginal_pad, 1, marginal_size], sharex=ax)
        side = ax.inset_axes([1 + marginal_pad, 0, marginal_size, 1], sharey=ax)
        edge = ax.get_facecolor()
        _hist(top, xs, series=f"{series}-x", bins=ex, color=mcol, edgecolor=edge, linewidth=0.4)
        _hist(side, ys, series=f"{series}-y", bins=ey, color=mcol, edgecolor=edge, linewidth=0.4,
              orientation="horizontal")
        for m_ax, base in ((top, "bottom"), (side, "left")):
            for name_, sp in m_ax.spines.items():
                sp.set_visible(name_ == base)
            m_ax.tick_params(axis="both", which="both", left=base == "left", bottom=base == "bottom",
                             labelleft=False, labelbottom=False, top=False, right=False)
            m_ax.set_xlabel("")
            m_ax.set_ylabel("")
            m_ax.patch.set_alpha(0)
        top.set_yticks([])
        side.set_xticks([])
        _panel(top, f"{series} x-marginal")
        _panel(side, f"{series} y-marginal")
        marginal_axes = {"x": top, "y": side}
        artists["marginals"] = marginal_axes
        right_edge = 1 + marginal_pad + marginal_size

    # ---- colour key -----------------------------------------------------------------------------------
    cb = None
    if colorbar:
        cax = ax.inset_axes([right_edge + colorbar_pad, 0, colorbar_size, 1])
        cb = _colorbar(hexes, name="color", ax=ax, cax=cax)
        if colorbar_label:
            cb.set_label(colorbar_label)
        if type(the_norm).__name__ == "LogNorm" and the_norm.vmin and the_norm.vmax:
            decades = np.log10(the_norm.vmax / the_norm.vmin)
            _plain_log_ticks(cax.yaxis, the_norm.vmin, the_norm.vmax,
                             subs=(1.0, 3.0) if decades <= 3 else (1.0,))
        artists["colorbar"] = cb

    return HexMatrixResult(ax=ax, hexes=hexes, bins=bins, cmap=the_cmap, norm=the_norm,
                           series=series, control_key=control_key, colorbar=cb,
                           marginal_axes=marginal_axes, points=points, artists=artists)
