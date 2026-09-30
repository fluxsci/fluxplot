"""D3 — fp.regression and fp.kde."""
import json
import re

import matplotlib
import numpy as np
import pytest
from scipy import stats as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fluxplot as fp  # noqa: E402
from fluxplot.fits import lowess  # noqa: E402

RNG = np.random.default_rng(1)
X = np.linspace(0, 10, 40)
Y = 2.0 * X + 1.0 + RNG.normal(0, 1.5, 40)


def _save(fig, tmp_path, name="r.svg"):
    res = fp.save(fig, str(tmp_path / name), recipe=False)
    return json.loads(open(res.manifest).read()), set(re.findall(r'id="([^"]+)"', open(res.svg).read()))


def test_linear_regression_matches_scipy_and_draws_three_parts(tmp_path):
    fig, ax = plt.subplots()
    r = fp.regression(ax, X, Y, series="fit", label="OLS")
    lr = sp.linregress(X, Y)
    assert r.stats["kind"] == "linear" and r.stats["degree"] == 1 and r.stats["n"] == 40 and r.stats["ci"] == 0.95
    assert r.stats["coefficients"] == pytest.approx([lr.slope, lr.intercept])
    assert r.stats["r2"] == pytest.approx(lr.rvalue**2) and r.stats["p"] == pytest.approx(lr.pvalue)
    # the band is the t-interval of the mean response: narrowest at the mean x, symmetric about the fit
    half = (r.upper - r.lower) / 2
    assert np.argmin(half) == np.argmin(np.abs(r.grid - X.mean()))
    assert np.allclose((r.upper + r.lower) / 2, r.fit)
    # the textbook width: t * s * sqrt(1/n + (x0 - mean)^2 / Sxx) at every grid point
    resid = Y - (lr.slope * X + lr.intercept)
    s_err = np.sqrt((resid**2).sum() / 38)
    t = sp.t.ppf(0.975, 38)
    sxx = ((X - X.mean()) ** 2).sum()
    expected = t * s_err * np.sqrt(1 / 40 + (r.grid - X.mean()) ** 2 / sxx)
    assert np.allclose(half, expected, rtol=1e-6)
    man, ids = _save(fig, tmp_path)
    assert {"fit.fit", "fit.band", "fit.points", "fit.point.0", "fit.point.39"} <= ids
    (s,) = man["series"]
    assert s["kind"] == "regression" and s["regression"]["coefficients"] == pytest.approx([lr.slope, lr.intercept])
    assert s["band"]["what"] == "95% CI" and len(s["regression"]["grid"]) == 100 and s["label"] == "OLS"
    assert man["plotType"] == "regression"
    # colours agree across the three parts
    assert matplotlib.colors.to_hex(r.band.get_facecolor()[0][:3]) == matplotlib.colors.to_hex(r.line.get_color())


def test_poly_and_lowess_and_options(tmp_path):
    y2 = 0.5 * X**2 - 3 * X + RNG.normal(0, 2, 40)
    fig, ax = plt.subplots()
    r = fp.regression(ax, X, y2, series="q", kind="poly", degree=2, points=False, ci=0.9)
    assert r.points is None and r.stats["degree"] == 2 and len(r.stats["coefficients"]) == 3
    assert r.stats["coefficients"] == pytest.approx(list(np.polyfit(X, y2, 2)))
    assert 0 < r.stats["p"] < 1e-6 and 0.9 < r.stats["r2"] <= 1
    man, ids = _save(fig, tmp_path, "p.svg")
    assert "q.points" not in ids and "q.fit" in ids and man["series"][0]["band"]["what"] == "90% CI"
    yl = np.sin(X) + RNG.normal(0, 0.2, 40)
    fig, ax = plt.subplots()
    r = fp.regression(ax, X, yl, series="s", kind="lowess", frac=0.4, n_boot=50)
    assert r.stats["kind"] == "lowess" and r.stats["coefficients"] is None and r.stats["p"] is None
    assert r.stats["frac"] == 0.4 and r.stats["bootstrap"] == 50 and 0.5 < r.stats["r2"] <= 1
    assert np.all(r.lower <= r.fit + 1e-9) and np.all(r.upper >= r.fit - 1e-9)
    # a seeded bootstrap: the same call gives the same band
    fig, ax = plt.subplots()
    r2 = fp.regression(ax, X, yl, series="s", kind="lowess", frac=0.4, n_boot=50)
    assert np.array_equal(r.lower, r2.lower)
    # lowess reproduces a line exactly and smooths noise
    assert np.allclose(lowess(X, 2 * X + 1, X), 2 * X + 1)
    with pytest.raises(ValueError, match="kind must be"):
        fp.regression(ax, X, Y, series="bad", kind="spline")
    with pytest.raises(ValueError, match="at least 3"):
        fp.regression(ax, [0, 1], [0, 1], series="bad")
    with pytest.raises(ValueError, match="degree 5 needs"):
        fp.regression(ax, [0, 1, 2, 3], [0, 1, 2, 3], series="bad", kind="poly", degree=5)


def test_kde_matches_scipy_and_records_the_grid(tmp_path):
    vals = RNG.normal(3, 1, 200)
    fig, ax = plt.subplots()
    k = fp.kde(ax, vals, series="d", fill=True)
    est = sp.gaussian_kde(vals)
    assert k.stats["bandwidth"] == pytest.approx(est.factor * vals.std(ddof=1)) and k.stats["method"] == "scott"
    assert np.allclose(k.density, est(k.grid)) and len(k.grid) == 200 and k.stats["n"] == 200
    assert k.grid[0] == pytest.approx(vals.min() - 3 * k.stats["bandwidth"])
    assert np.trapezoid(k.density, k.grid) == pytest.approx(1.0, abs=0.02)
    man, ids = _save(fig, tmp_path, "k.svg")
    assert {"d.line", "d.fill"} <= ids
    (s,) = man["series"]
    assert s["kind"] == "kde" and s["kde"]["density"] == pytest.approx(list(k.density)) and s["band"]["y2"] == [0.0] * 200
    fig, ax = plt.subplots()
    k2 = fp.kde(ax, vals, series="d", bw=0.5)
    assert k2.area is None and k2.stats["method"] == 0.5 and k2.stats["bandwidth"] == pytest.approx(0.5 * vals.std(ddof=1))
    with pytest.raises(ValueError, match="zero variance"):
        fp.kde(ax, [1.0, 1.0, 1.0], series="bad")
