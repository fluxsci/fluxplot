"""``fp.regression`` and ``fp.kde``: fitted curves drawn as addressable series with the fit recorded.

A regression is three parts of one series — the points (``<series>.point.k``), the fit line
(``<series>.fit``) and its confidence band (``<series>.band``) — and a ``regression`` payload in
the manifest: the coefficients, R², the p-value, n and the confidence level, so a caption or a
consumer can state the fit exactly as drawn. A KDE is a line (and optionally its fill) with the
evaluation grid and density recorded, so it can be re-drawn without the raw values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from . import data as _data
from . import tagger as _tagger
from .descriptors import Mark

__all__ = ["regression", "kde", "RegressionResult", "KdeResult", "lowess", "REGRESSION_KINDS"]

REGRESSION_KINDS = ("linear", "poly", "lowess")
_BOOT_SEED = 0


@dataclass
class RegressionResult:
    ax: Any
    #: the fit ``Line2D``, the band ``PolyCollection`` (or ``None``) and the points (or ``None``)
    line: Any
    band: Any
    points: Any
    #: the evaluation grid and the fitted curve with its confidence bounds
    grid: np.ndarray = field(repr=False, default=None)
    fit: np.ndarray = field(repr=False, default=None)
    lower: np.ndarray = field(repr=False, default=None)
    upper: np.ndarray = field(repr=False, default=None)
    #: ``{kind, degree, coefficients, r2, p, n, ci, …}`` as the manifest records it
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KdeResult:
    ax: Any
    line: Any
    area: Any
    grid: np.ndarray = field(repr=False, default=None)
    density: np.ndarray = field(repr=False, default=None)
    stats: Dict[str, Any] = field(default_factory=dict)


def _clean_xy(x, y, who):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"{who}: x and y have different lengths ({x.size} vs {y.size})")
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        raise ValueError(f"{who} needs at least 3 finite (x, y) pairs, got {int(ok.sum())}")
    return x[ok], y[ok]


def lowess(x, y, grid, frac=2 / 3, it=3):
    """Cleveland's (1979) locally weighted regression: at each grid point a linear fit weighted by
    the tricube of distance to the nearest ``frac`` of the data, with ``it`` robustifying
    iterations that down-weight large residuals (bisquare). Returns the smooth on ``grid``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n = x.size
    k = max(int(np.ceil(frac * n)), 2)
    robust = np.ones(n)

    def fit_at(x0, weights_extra):
        d = np.abs(x - x0)
        h = np.sort(d)[min(k, n) - 1] or 1e-12
        w = np.clip(1 - (d / h) ** 3, 0, 1) ** 3 * weights_extra
        sw = w.sum()
        if sw <= 0:
            return np.nan
        xm = (w * x).sum() / sw
        ym = (w * y).sum() / sw
        var = (w * (x - xm) ** 2).sum()
        slope = (w * (x - xm) * (y - ym)).sum() / var if var > 0 else 0.0
        return ym + slope * (x0 - xm)

    for _ in range(max(it, 0)):
        fitted = np.array([fit_at(xi, robust) for xi in x])
        resid = y - fitted
        s = np.median(np.abs(resid)) or 1e-12
        robust = np.clip(1 - (resid / (6 * s)) ** 2, 0, 1) ** 2
    return np.array([fit_at(g, robust) for g in grid])


def _poly_ci(x, y, grid, degree, ci):
    """Polynomial least squares with the confidence band of the mean response."""
    from scipy import stats as _sp
    n = x.size
    dof = n - (degree + 1)
    if dof < 1:
        raise ValueError(f"regression: degree {degree} needs more than {degree + 1} points (n={n})")
    coef, cov = np.polyfit(x, y, degree, cov=True)
    fit_grid = np.polyval(coef, grid)
    fitted = np.polyval(coef, x)
    design = np.vander(grid, degree + 1)  # highest power first, as polyfit's coefficients
    se = np.sqrt(np.einsum("ij,jk,ik->i", design, cov, design))
    t = _sp.t.ppf(0.5 + ci / 2, dof)
    ss_res = float(((y - fitted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if degree == 1:
        p = float(_sp.linregress(x, y).pvalue)
    else:
        f = ((ss_tot - ss_res) / degree) / (ss_res / dof) if ss_res > 0 else float("inf")
        p = float(_sp.f.sf(f, degree, dof))
    return coef, fit_grid, fit_grid - t * se, fit_grid + t * se, r2, p, dof


def regression(ax, x, y, *, series, kind="linear", ci=0.95, degree=1, frac=2 / 3, points=True, n_grid=100,
               n_boot=200, label=None, color=None, band_kw=None, point_kw=None, **line_kw) -> RegressionResult:
    """Fit ``y ~ x`` and draw the fit line, its confidence band and (by default) the points as one
    series, recording the fit in the manifest.

    Parameters
    ----------
    kind
        ``"linear"`` (a straight line), ``"poly"`` (a polynomial of ``degree``) or ``"lowess"``
        (Cleveland's locally weighted regression with span ``frac``).
    ci
        The confidence level of the band: for linear / poly the t-interval of the mean response;
        for lowess a seeded percentile bootstrap of the smooth (``n_boot`` resamples).
    points
        Draw the observations as ``<series>.point.k`` too.
    color
        The series colour (line, band and points); default the cycle's next colour.
    band_kw, point_kw, **line_kw
        Extra keywords for the band (``fill_between``), the points (``scatter``) and the fit line.

    Returns
    -------
    RegressionResult
        With ``stats = {kind, degree, coefficients, r2, p, n, ci, frac?}`` (``coefficients`` in
        ``numpy.polyfit`` order, highest power first; ``None`` for lowess, whose ``p`` is ``None``).
    """
    from .api import _series_color, band as _band, scatter as _scatter
    if kind not in REGRESSION_KINDS:
        raise ValueError(f"regression: kind must be one of {REGRESSION_KINDS}, got {kind!r}")
    if not 0 < ci < 1:
        raise ValueError(f"regression: ci must lie in (0, 1), got {ci!r}")
    xs, ys = _clean_xy(x, y, "regression")
    grid = np.linspace(xs.min(), xs.max(), int(n_grid))
    deg = 1 if kind == "linear" else int(degree)
    if kind in ("linear", "poly"):
        coef, fit, lo, hi, r2, p, dof = _poly_ci(xs, ys, grid, deg, ci)
        stats = {"kind": kind, "degree": deg, "coefficients": [float(c) for c in coef], "r2": float(r2), "p": p,
                 "n": int(xs.size), "ci": float(ci), "dof": int(dof)}
    else:
        fit = lowess(xs, ys, grid, frac=frac)
        rng = np.random.default_rng(_BOOT_SEED)
        draws = []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, xs.size, xs.size)
            draws.append(lowess(xs[idx], ys[idx], grid, frac=frac))
        draws = np.array(draws)
        lo, hi = np.nanpercentile(draws, [50 * (1 - ci), 50 * (1 + ci)], axis=0)
        fitted = np.interp(xs, grid, fit)
        ss_res = float(((ys - fitted) ** 2).sum())
        ss_tot = float(((ys - ys.mean()) ** 2).sum())
        stats = {"kind": "lowess", "degree": None, "coefficients": None,
                 "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), "p": None, "n": int(xs.size),
                 "ci": float(ci), "frac": float(frac), "bootstrap": int(n_boot)}

    kw = dict(line_kw)
    if color is not None:
        kw["color"] = color
    _series_color(series, kw)
    the_color = kw.get("color")
    reg = _tagger.registry_for(ax.figure)
    # the fit is registered first, so the series is a "regression" (its first mark names its kind);
    # drawn first too, it still sits above the points (a line's zorder beats a collection's)
    (ln,) = ax.plot(grid, fit, label=label, **kw)
    if the_color is None:
        the_color = ln.get_color()
    payload = {**stats, "grid": _data.values(grid), "fit": _data.values(fit)}
    reg.add(Mark(role="line", series=series, name="fit", kind="regression", x=_data.values(grid), y=_data.values(fit),
                 label=label, artists=[ln], data={"regression": payload}))
    pts = None
    if points:
        pkw = dict(point_kw or {})
        pkw.setdefault("s", 14)
        pkw.setdefault("alpha", 0.6)
        pkw.setdefault("color", the_color)
        pts = _scatter(ax, xs, ys, series=series, **pkw)
    bkw = dict(band_kw or {})
    bkw.setdefault("color", the_color)
    band = _band(ax, grid, lo, hi, series=series, what=f"{ci:.0%} CI", **bkw)
    return RegressionResult(ax=ax, line=ln, band=band, points=pts, grid=grid, fit=fit, lower=np.asarray(lo),
                            upper=np.asarray(hi), stats=stats)


def kde(ax, values, *, series, bw="scott", fill=False, n_grid=200, cut=3.0, label=None, color=None,
        fill_kw=None, **line_kw) -> KdeResult:
    """A Gaussian kernel density estimate drawn as a line (``<series>.line``; ``fill=True`` adds
    ``<series>.fill``), recording the grid, the density and the bandwidth in the manifest.

    ``bw`` is scipy's ``bw_method`` (``"scott"``, ``"silverman"``, a number or a callable);
    the grid runs ``cut`` bandwidths past the data on either side.
    """
    from scipy import stats as _sp
    from .api import _series_color
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2:
        raise ValueError(f"kde needs at least 2 finite values, got {v.size}")
    if v.std(ddof=1) == 0:
        raise ValueError("kde: the values have zero variance")
    estimator = _sp.gaussian_kde(v, bw_method=bw)
    bandwidth = float(estimator.factor * v.std(ddof=1))
    grid = np.linspace(v.min() - cut * bandwidth, v.max() + cut * bandwidth, int(n_grid))
    density = estimator(grid)
    kw = dict(line_kw)
    if color is not None:
        kw["color"] = color
    _series_color(series, kw)
    (ln,) = ax.plot(grid, density, label=label, **kw)
    reg = _tagger.registry_for(ax.figure)
    payload = {"grid": _data.values(grid), "density": _data.values(density), "bandwidth": bandwidth,
               "method": bw if isinstance(bw, str) else ("callable" if callable(bw) else float(bw)), "n": int(v.size)}
    reg.add(Mark(role="line", series=series, kind="kde", live_data=True, x=None, y=None, label=label, artists=[ln],
                 data={"kde": payload}))
    area = None
    if fill:
        fkw = dict(fill_kw or {})
        fkw.setdefault("alpha", 0.25)
        fkw.setdefault("linewidth", 0)
        fkw.setdefault("color", ln.get_color())
        area = ax.fill_between(grid, 0.0, density, **fkw)
        reg.add(Mark(role="area", series=series, name="fill", kind="kde", artists=[area],
                     data={"band": {"x": _data.values(grid), "y1": _data.values(density), "y2": [0.0] * int(n_grid)}}))
    return KdeResult(ax=ax, line=ln, area=area, grid=grid, density=density, stats=payload)
