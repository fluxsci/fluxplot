"""Tests for two independent (unpaired) samples.

Each function returns one row as a dict keyed by :data:`REPORT_COLUMNS`, so a set of comparisons
drops straight into a DataFrame (``pl.DataFrame([fp.stats.welch_hedges(a, b)])``) and from there
into a plot's ``_stats`` dissection as a CSV.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy import optimize as _opt
from scipy import stats as _sp

from ._common import REPORT_COLUMNS, as_sample, check_alternative, report_row

__all__ = ["welch_hedges", "mann_whitney_cliff", "REPORT_COLUMNS"]

_Z = _sp.norm.ppf(0.975)


def hedges_unpooled(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Hedges' g standardised by the non-pooled SD ``sqrt((var_a + var_b) / 2)`` with its Bonett
    (2008) 95% CI: ``(g, lo, hi)``. Shared by :func:`welch_hedges` and the Games–Howell post hoc."""
    n1, n2 = a.size, b.size
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    s = np.sqrt((v1 + v2) / 2)
    if s == 0:
        raise ValueError("both samples have zero variance; the effect size is undefined")
    d = (a.mean() - b.mean()) / s
    j = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
    se = np.sqrt(d**2 * (v1**2 / (n1 - 1) + v2**2 / (n2 - 1)) / (8 * s**4)
                 + (v1 / (n1 - 1) + v2 / (n2 - 1)) / s**2)
    return float(j * d), float(d - _Z * se), float(d + _Z * se)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Cliff's delta ``P(a > b) - P(a < b)`` with Newcombe's Method 5 95% CI: ``(delta, lo, hi)``.
    Shared by :func:`mann_whitney_cliff` and Dunn's post hoc."""
    delta = float(np.sign(a[:, None] - b[None, :]).mean())  # mean of sign(a_i - b_j) over all pairs
    lo, hi = _newcombe_auc_bounds((delta + 1) / 2, a.size, b.size)
    return delta, 2 * lo - 1, 2 * hi - 1


def welch_hedges(a: Any, b: Any, *, alternative: str = "two-sided", names=("a", "b")) -> Dict[str, Any]:
    """Welch's t-test of ``a`` vs ``b`` with Hedges' g (non-pooled SD) and its 95% CI.

    For independent samples compared on their arithmetic means. The effect size is standardized by
    the non-pooled SD ``sqrt((var_a + var_b) / 2)``, so it stays meaningful when the group variances
    differ. The estimate carries the small-sample bias correction
    ``J = 1 - 3 / (4 (n_a + n_b - 2) - 1)``. The 95% CI is Bonett's (2008) unequal-variance interval
    for the population value; it is not multiplied by ``J``, because the interval targets the
    parameter itself and ``J`` would pull it off target (simulated coverage is 95–97% as is).

    Signs follow ``a - b``: a positive statistic and effect size mean ``a`` has the larger mean.

    Parameters
    ----------
    a, b
        The two samples (any 1-D array-like of numbers). Each needs at least 2 finite values.
    alternative
        ``"two-sided"`` (default), ``"less"`` (``a`` has the smaller mean) or ``"greater"``. The
        p-value follows it; the effect-size CI is always the two-sided 95% interval.
    names
        The names of the two groups, recorded in the row's ``groups`` (``("a", "b")`` by default).

    Returns
    -------
    dict
        One reporting row keyed by :data:`REPORT_COLUMNS`: the test name, the t statistic, the
        p-value, the Welch–Satterthwaite degrees of freedom, the effect-size method, its value and
        its 95% CI formatted as ``"[low, high]"`` (2 decimals) and as numbers, the sample sizes,
        the group names and the alternative.

    Example
    -------
    >>> row = fp.stats.welch_hedges(sd_values, sleep_values, names=("SD", "sleep"))
    >>> pl.DataFrame([row]).write_csv("plots/_dissections/app/_stats/welch_ttest.csv")
    """
    a = as_sample(a, "a")
    b = as_sample(b, "b")
    g, lo, hi = hedges_unpooled(a, b)
    t = _sp.ttest_ind(a, b, equal_var=False, alternative=check_alternative(alternative))
    return report_row("Welch's t-test", t.statistic, t.pvalue, t.df, "Hedges' g (non-pooled SD)",
                      g, lo, hi, n=(a.size, b.size), groups=names, alternative=alternative)


def _newcombe_auc_bounds(theta: float, m: int, n: int) -> Tuple[float, float]:
    """Newcombe's (2006) Method 5 95% interval for ``theta = P(a > b) + P(a = b) / 2``.

    A score-type interval: the bounds are the ``t`` solving ``(t - theta)^2 = z^2 V(t)``, where
    ``V`` is the Hanley–McNeil variance of the Mann–Whitney estimate with both sample sizes
    replaced by ``N* = (m + n) / 2 - 1``. Unlike a Wald interval it stays inside ``[0, 1]`` and
    does not collapse when the groups do not overlap (``theta`` = 0 or 1).
    """
    n_star = (m + n) / 2 - 1

    def gap(t: float) -> float:
        var = t * (1 - t) * (1 + (n_star - 1) * ((1 - t) / (2 - t) + t / (1 + t))) / (m * n)
        return (t - theta) ** 2 - _Z**2 * var

    # gap(0) = theta^2 > 0 and gap(1) = (1 - theta)^2 > 0 while gap(theta) < 0, so each bound is
    # the one root on its side; at theta = 0 or 1 the bound on that side is the boundary itself
    # (gap vanishes there), so search just inside it.
    eps = 1e-12
    lo = 0.0 if theta <= 0 else _opt.brentq(gap, 0.0, min(theta, 1 - eps), xtol=1e-14)
    hi = 1.0 if theta >= 1 else _opt.brentq(gap, max(theta, eps), 1.0, xtol=1e-14)
    return lo, hi


def mann_whitney_cliff(a: Any, b: Any, *, alternative: str = "two-sided", names=("a", "b")) -> Dict[str, Any]:
    """Mann–Whitney U test of ``a`` vs ``b`` with Cliff's delta and its 95% CI.

    For independent samples compared by rank (ordered categories, or a rank-based comparison chosen
    before looking at the results). The p-value is scipy's two-sided ``mannwhitneyu``
    (``method="auto"``: exact for small samples without ties, otherwise the normal approximation
    with tie and continuity corrections).

    Cliff's delta is ``P(a > b) - P(a < b)`` over all ``n_a * n_b`` cross-group pairs (ties count
    as neither), which equals ``2 U_a / (n_a n_b) - 1``. Its 95% CI is Newcombe's (2006) Method 5
    score interval for ``theta = (delta + 1) / 2``, mapped back to the delta scale. It stays inside
    ``[-1, 1]`` and, unlike Cliff's own (1996) interval, does not collapse to a point when the
    groups do not overlap at all (``delta = ±1``) — common with small samples.

    Signs follow ``a - b``: a positive delta means values in ``a`` tend to be larger.

    Parameters
    ----------
    a, b
        The two samples (any 1-D array-like of numbers). Each needs at least 2 finite values.
    alternative
        ``"two-sided"`` (default), ``"less"`` or ``"greater"`` (the direction of ``a`` relative to
        ``b``). The CI is always two-sided.
    names
        The names of the two groups, recorded in the row's ``groups``.

    Returns
    -------
    dict
        One reporting row keyed by :data:`REPORT_COLUMNS`. The statistic is ``U`` for ``a`` (the
        number of pairs with ``a > b``, ties counting one half); ``dof`` is ``None`` (a rank test
        has no degrees of freedom).
    """
    a = as_sample(a, "a")
    b = as_sample(b, "b")
    u = _sp.mannwhitneyu(a, b, alternative=check_alternative(alternative), method="auto")
    delta, lo, hi = cliffs_delta(a, b)
    return report_row("Mann–Whitney U test", u.statistic, u.pvalue, None, "Cliff's delta", delta,
                      lo, hi, n=(a.size, b.size), groups=names, alternative=alternative)
