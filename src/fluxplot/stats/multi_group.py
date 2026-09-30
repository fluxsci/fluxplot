"""Tests for three or more groups — the omnibus question ("do the groups differ at all?") and the
post-hoc pairwise comparisons that follow it — plus :func:`pairwise`, which runs any two-group
test of this package over every pair of a family and corrects the p-values.

Each omnibus test returns one row keyed by :data:`REPORT_COLUMNS` (``groups`` lists every group,
``n_total`` their combined size, ``dof`` / ``dof_error`` the numerator and denominator dof of an F
test); each post-hoc test returns one row per pair (``groups = [a, b]``, signs follow ``a - b``),
ready for :func:`fluxplot.brackets`, which draws them over a plot and records which test each star
came from.

Repeated-measures designs take a long table (``table``, a pandas / polars DataFrame or a dict of
columns) with a ``subject`` column, a ``within`` (condition) column and a ``dv`` column; every
subject must have every condition exactly once.
"""
from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import optimize as _opt
from scipy import stats as _sp

from ._common import REPORT_COLUMNS, as_sample, bh, holm, report_row
from .two_group import cliffs_delta, hedges_unpooled

__all__ = [
    "anova_oneway", "welch_anova", "kruskal_epsilon", "rm_anova", "friedman_kendall",
    "tukey_hsd", "games_howell", "dunn", "pairwise", "REPORT_COLUMNS",
]

_Z = _sp.norm.ppf(0.975)
_BOOT_SEED = 0


# ---------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------
def _groups(groups: Sequence[Any], names: Optional[Sequence[str]], who: str) -> Tuple[List[np.ndarray], List[str]]:
    if len(groups) == 1 and isinstance(groups[0], dict):  # a {name: sample} mapping
        names = list(groups[0]) if names is None else list(names)
        groups = list(groups[0].values())
    if len(groups) < 2:
        raise ValueError(f"{who} needs at least 2 groups, got {len(groups)}")
    if names is None:
        names = [f"group{i + 1}" for i in range(len(groups))]
    names = [str(n) for n in names]
    if len(names) != len(groups):
        raise ValueError(f"{who}: names has {len(names)} entries for {len(groups)} groups")
    return [as_sample(g, n) for g, n in zip(groups, names)], names


def _column(table: Any, name: str) -> list:
    try:
        col = table[name]
    except Exception as exc:  # KeyError (pandas/dict), ColumnNotFoundError (polars), …
        raise KeyError(f"{name!r} is not a column of the table") from exc
    for attr in ("to_list", "tolist"):
        if hasattr(col, attr):
            return list(getattr(col, attr)())
    return list(col)


def _matrix(table: Any, subject: str, within: str, dv: str, who: str) -> Tuple[np.ndarray, List[str], List[Any]]:
    """The complete ``subjects × conditions`` matrix of a long table: conditions in order of first
    appearance, subjects likewise; every cell filled exactly once."""
    subs, conds, vals = _column(table, subject), _column(table, within), _column(table, dv)
    if not (len(subs) == len(conds) == len(vals)):
        raise ValueError(f"{who}: the columns differ in length")
    sub_order = list(dict.fromkeys(subs))
    cond_order = list(dict.fromkeys(conds))
    n, k = len(sub_order), len(cond_order)
    if k < 2:
        raise ValueError(f"{who} needs at least 2 conditions in {within!r}, got {k}")
    if n < 2:
        raise ValueError(f"{who} needs at least 2 subjects in {subject!r}, got {n}")
    m = np.full((n, k), np.nan)
    si, ci = {s: i for i, s in enumerate(sub_order)}, {c: j for j, c in enumerate(cond_order)}
    seen = set()
    for s, c, v in zip(subs, conds, vals):
        cell = (si[s], ci[c])
        if cell in seen:
            raise ValueError(f"{who}: subject {s!r} has more than one row for {within}={c!r}")
        seen.add(cell)
        m[cell] = float(v) if v is not None else np.nan
    if not np.all(np.isfinite(m)):
        missing = [(sub_order[i], cond_order[j]) for i, j in zip(*np.where(~np.isfinite(m)))]
        raise ValueError(f"{who}: incomplete or non-finite cells for {missing[:3]}{'…' if len(missing) > 3 else ''}; "
                         "every subject needs a finite value for every condition")
    return m, [str(c) for c in cond_order], sub_order


# ---------------------------------------------------------------------------------------------
# noncentral-F confidence intervals for variance-explained effect sizes (Steiger 2004)
# ---------------------------------------------------------------------------------------------
def _ncf_ncp_for(f_obs: float, df1: float, df2: float, target: float) -> float:
    """The noncentrality ``lam >= 0`` with ``ncf.cdf(f_obs, df1, df2, lam) == target``, or 0 when
    even the central distribution puts less than ``target`` below ``f_obs``."""
    f = lambda lam: _sp.ncf.cdf(f_obs, df1, df2, lam) - target  # noqa: E731
    if f(0.0) <= 0:
        return 0.0
    hi = max(1.0, f_obs * df1)
    while f(hi) > 0:
        hi *= 2
    return _opt.brentq(f, 0.0, hi, xtol=1e-10)


def _variance_explained_ci(f_obs: float, df1: float, df2: float) -> Tuple[float, float]:
    """95% CI for the population proportion of variance explained (``eta^2`` / ``omega^2`` /
    partial ``eta^2`` estimate the same parameter) from the noncentral F distribution of the
    observed F: the bounds are ``lam / (lam + df1 + df2 + 1)`` for the noncentralities bracketing
    ``f_obs`` at the 2.5% and 97.5% points (Steiger 2004; Kelley 2007). The lower bound is 0 when
    ``p >= 0.025``."""
    lo = _ncf_ncp_for(f_obs, df1, df2, 0.975)
    hi = _ncf_ncp_for(f_obs, df1, df2, 0.025)
    n_eff = df1 + df2 + 1
    return lo / (lo + n_eff), hi / (hi + n_eff)


def _bootstrap_ci(statistic: Callable[[np.random.Generator], float], n_boot: int) -> Tuple[float, float]:
    """A seeded percentile-bootstrap 95% interval (deterministic across runs)."""
    rng = np.random.default_rng(_BOOT_SEED)
    draws = np.array([statistic(rng) for _ in range(int(n_boot))])
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------------------------
# omnibus tests, independent groups
# ---------------------------------------------------------------------------------------------
def _sums_of_squares(groups: List[np.ndarray]):
    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_b = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    return ss_b, ss_w, allv.size


def anova_oneway(*groups: Any, names: Optional[Sequence[str]] = None, effect: str = "eta2") -> Dict[str, Any]:
    """Classic one-way ANOVA (equal variances assumed) with η² or ω² and its 95% CI.

    ``F = MS_between / MS_within`` on ``(k - 1, N - k)`` dof (scipy's ``f_oneway``). The effect
    size is η² = SS_between / SS_total (``effect="eta2"``, the default) or the less biased
    ω² = (SS_between − (k − 1) MS_within) / (SS_total + MS_within) (``effect="omega2"``). Both
    estimate the population proportion of variance explained, so both get the same 95% CI: the
    noncentral-F interval of Steiger (2004), whose lower bound is 0 whenever ``p >= 0.025``.

    Parameters
    ----------
    *groups
        Two or more samples, or a single ``{name: sample}`` dict.
    names
        Group names (recorded in ``groups``); default ``group1, group2, …``.
    effect
        ``"eta2"`` or ``"omega2"``.
    """
    gs, names = _groups(groups, names, "anova_oneway")
    if effect not in ("eta2", "omega2"):
        raise ValueError(f"effect must be 'eta2' or 'omega2', got {effect!r}")
    k = len(gs)
    ss_b, ss_w, n = _sums_of_squares(gs)
    df1, df2 = k - 1, n - k
    if ss_w == 0:
        raise ValueError("every group has zero within-group variance; F is undefined")
    res = _sp.f_oneway(*gs)
    ms_w = ss_w / df2
    eta2 = ss_b / (ss_b + ss_w)
    omega2 = (ss_b - df1 * ms_w) / (ss_b + ss_w + ms_w)
    lo, hi = _variance_explained_ci(res.statistic, df1, df2)
    value, method = (eta2, "Eta squared") if effect == "eta2" else (omega2, "Omega squared")
    return report_row("One-way ANOVA", res.statistic, res.pvalue, df1, method, value, lo, hi,
                      n=[g.size for g in gs], groups=names, dof_error=df2)


def welch_anova(*groups: Any, names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Welch's ANOVA (unequal variances) with ω² and its 95% CI.

    Welch's (1951) F weights each group by ``n_i / var_i``; the denominator dof are
    ``(k² − 1) / (3 Σ (1 − w_i / W)² / (n_i − 1))``. The effect size is
    ω² = (k − 1)(F − 1) / ((k − 1)(F − 1) + N) (Kirk's formula; identical to the classic ω² when
    the variances are equal), with the noncentral-F 95% CI on Welch's dof.
    """
    gs, names = _groups(groups, names, "welch_anova")
    k = len(gs)
    n = np.array([g.size for g in gs], dtype=float)
    m = np.array([g.mean() for g in gs])
    v = np.array([g.var(ddof=1) for g in gs])
    if np.any(v == 0):
        raise ValueError("a group has zero variance; Welch's weights are undefined")
    w = n / v
    big_w = w.sum()
    grand = (w * m).sum() / big_w
    lam = ((1 - w / big_w) ** 2 / (n - 1)).sum()
    f_stat = ((w * (m - grand) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k**2 - 1) * lam)
    df1 = k - 1
    df2 = (k**2 - 1) / (3 * lam)
    p = _sp.f.sf(f_stat, df1, df2)
    total = int(n.sum())
    omega2 = df1 * (f_stat - 1) / (df1 * (f_stat - 1) + total)
    lo, hi = _variance_explained_ci(f_stat, df1, df2)
    return report_row("Welch's ANOVA", f_stat, p, df1, "Omega squared", omega2, lo, hi,
                      n=[g.size for g in gs], groups=names, dof_error=df2)


def kruskal_epsilon(*groups: Any, names: Optional[Sequence[str]] = None, n_boot: int = 2000) -> Dict[str, Any]:
    """Kruskal–Wallis H test with the ε² effect size and a bootstrap 95% CI.

    ``H`` is scipy's tie-corrected statistic on ``k − 1`` dof. ε² = H / (N − 1) (Kelley 1935; the
    rank analogue of η², in ``[0, 1]``). Its 95% CI is a seeded percentile bootstrap of ε² over
    ``n_boot`` within-group resamples — deterministic across runs.
    """
    gs, names = _groups(groups, names, "kruskal_epsilon")
    res = _sp.kruskal(*gs)
    total = sum(g.size for g in gs)
    eps2 = res.statistic / (total - 1)

    def draw(rng):
        boot = [rng.choice(g, g.size, replace=True) for g in gs]
        try:
            return _sp.kruskal(*boot).statistic / (total - 1)
        except ValueError:  # every resampled value identical
            return np.nan

    lo, hi = _bootstrap_ci(draw, n_boot)
    return report_row("Kruskal–Wallis H test", res.statistic, res.pvalue, len(gs) - 1,
                      "Epsilon squared", eps2, lo, hi, n=[g.size for g in gs], groups=names)


# ---------------------------------------------------------------------------------------------
# omnibus tests, repeated measures
# ---------------------------------------------------------------------------------------------
def rm_anova(table: Any, subject: str, within: str, dv: str) -> Dict[str, Any]:
    """One-way repeated-measures ANOVA with the Greenhouse–Geisser correction and partial η².

    The within-subject decomposition ``SS_total = SS_subjects + SS_conditions + SS_error`` gives
    ``F = MS_conditions / MS_error`` on ``(k − 1, (k − 1)(n − 1))`` dof. Sphericity is not assumed:
    both dof are multiplied by the Greenhouse–Geisser ε (from the double-centred covariance matrix
    of the conditions, clamped to ``[1 / (k − 1), 1]``) before the p-value is taken, and the row
    reports those corrected dof. Partial η² = SS_conditions / (SS_conditions + SS_error), with the
    noncentral-F 95% CI on the corrected dof.

    ``groups`` lists the conditions; ``n_total`` is the number of subjects.
    """
    m, conds, subs = _matrix(table, subject, within, dv, "rm_anova")
    n, k = m.shape
    grand = m.mean()
    ss_total = ((m - grand) ** 2).sum()
    ss_subj = k * ((m.mean(axis=1) - grand) ** 2).sum()
    ss_cond = n * ((m.mean(axis=0) - grand) ** 2).sum()
    ss_err = ss_total - ss_subj - ss_cond
    df1, df2 = k - 1, (k - 1) * (n - 1)
    if ss_err <= 0:
        raise ValueError("rm_anova: the error sum of squares is zero; F is undefined")
    f_stat = (ss_cond / df1) / (ss_err / df2)
    # Greenhouse–Geisser epsilon from the double-centred covariance matrix
    cov = np.cov(m, rowvar=False, ddof=1)
    centre = np.eye(k) - np.ones((k, k)) / k
    dc = centre @ cov @ centre
    eps = np.trace(dc) ** 2 / ((k - 1) * (dc**2).sum())
    eps = float(min(1.0, max(eps, 1.0 / (k - 1))))
    gdf1, gdf2 = eps * df1, eps * df2
    p = _sp.f.sf(f_stat, gdf1, gdf2)
    eta_p = ss_cond / (ss_cond + ss_err)
    lo, hi = _variance_explained_ci(f_stat, gdf1, gdf2)
    return report_row("Repeated-measures ANOVA (Greenhouse–Geisser)", f_stat, p, gdf1,
                      "Partial eta squared", eta_p, lo, hi, n=[n] * k, n_total=n, groups=conds, dof_error=gdf2)


def friedman_kendall(table: Any, subject: str, within: str, dv: str, n_boot: int = 2000) -> Dict[str, Any]:
    """Friedman's test with Kendall's W and a bootstrap 95% CI.

    scipy's ``friedmanchisquare`` on the ``subjects × conditions`` matrix gives the χ² statistic on
    ``k − 1`` dof. Kendall's W = χ² / (n (k − 1)) is the agreement between subjects about the
    ordering of the conditions, in ``[0, 1]``. Its 95% CI is a seeded percentile bootstrap over
    subjects (rows resampled with replacement, ``n_boot`` draws).

    ``groups`` lists the conditions; ``n_total`` is the number of subjects.
    """
    m, conds, subs = _matrix(table, subject, within, dv, "friedman_kendall")
    n, k = m.shape
    if k < 3:
        raise ValueError("friedman_kendall needs at least 3 conditions (use a paired test for 2)")
    res = _sp.friedmanchisquare(*m.T)
    w = res.statistic / (n * (k - 1))

    def draw(rng):
        rows = m[rng.integers(0, n, n)]
        try:
            return _sp.friedmanchisquare(*rows.T).statistic / (n * (k - 1))
        except ValueError:
            return np.nan

    lo, hi = _bootstrap_ci(draw, n_boot)
    return report_row("Friedman test", res.statistic, res.pvalue, k - 1, "Kendall's W", w, lo, hi,
                      n=[n] * k, n_total=n, groups=conds)


# ---------------------------------------------------------------------------------------------
# post-hoc pairwise comparisons
# ---------------------------------------------------------------------------------------------
def _hedges_pooled(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Hedges' g with the pooled SD (equal variances) and its large-sample 95% CI (Hedges & Olkin
    1985): ``(g, lo, hi)``."""
    n1, n2 = a.size, b.size
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if sp == 0:
        raise ValueError("both samples have zero variance; the effect size is undefined")
    d = (a.mean() - b.mean()) / sp
    j = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
    g = j * d
    se = np.sqrt((n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2)))
    return float(g), float(g - _Z * se), float(g + _Z * se)


def _pairs(k: int) -> List[Tuple[int, int]]:
    return list(itertools.combinations(range(k), 2))


def tukey_hsd(*groups: Any, names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Tukey's honestly-significant-difference test over every pair, with Hedges' g (pooled SD).

    scipy's ``tukey_hsd`` gives each pair's mean difference ``a − b`` and its p-value from the
    studentized range on ``N − k`` dof. That p-value already controls the family-wise error rate,
    so ``p_corrected_holm`` and ``p_corrected_bh`` equal it: do **not** correct these rows again.
    The effect size is Hedges' g with the pooled SD (Tukey assumes equal variances), with the
    Hedges–Olkin large-sample 95% CI.
    """
    gs, names = _groups(groups, names, "tukey_hsd")
    res = _sp.tukey_hsd(*gs)
    total = sum(g.size for g in gs)
    rows = []
    for i, j in _pairs(len(gs)):
        g, lo, hi = _hedges_pooled(gs[i], gs[j])
        rows.append(report_row("Tukey HSD", res.statistic[i, j], res.pvalue[i, j], total - len(gs),
                               "Hedges' g (pooled SD)", g, lo, hi, n=(gs[i].size, gs[j].size),
                               groups=(names[i], names[j])))
    return rows


def games_howell(*groups: Any, names: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """The Games–Howell test over every pair (unequal variances and sizes), with Hedges' g
    (non-pooled SD).

    Each pair's ``t = (mean_a − mean_b) / sqrt(var_a / n_a + var_b / n_b)`` on Welch–Satterthwaite
    dof is referred to the studentized range distribution of ``k`` groups
    (``p = P(Q_{k, df} > |t| sqrt 2)``), which controls the family-wise error rate: as for
    :func:`tukey_hsd`, the corrected columns equal ``p-value``. The effect size and its Bonett CI
    are those of :func:`fluxplot.stats.welch_hedges`.
    """
    gs, names = _groups(groups, names, "games_howell")
    k = len(gs)
    rows = []
    for i, j in _pairs(k):
        a, b = gs[i], gs[j]
        va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
        if va + vb == 0:
            raise ValueError(f"groups {names[i]!r} and {names[j]!r} have zero variance")
        t = (a.mean() - b.mean()) / np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
        p = _sp.studentized_range.sf(abs(t) * np.sqrt(2), k, df)
        g, lo, hi = hedges_unpooled(a, b)
        rows.append(report_row("Games–Howell test", t, p, df, "Hedges' g (non-pooled SD)", g, lo, hi,
                               n=(a.size, b.size), groups=(names[i], names[j])))
    return rows


def dunn(*groups: Any, names: Optional[Sequence[str]] = None, adjust: Union[str, Sequence[str]] = "holm") -> List[Dict[str, Any]]:
    """Dunn's (1964) rank-sum test over every pair after Kruskal–Wallis, with Cliff's delta.

    The pooled ranks give each pair ``z = (R̄_a − R̄_b) / sqrt((N (N + 1) / 12 − T) (1 / n_a + 1 / n_b))``
    with the tie term ``T = Σ (t³ − t) / (12 (N − 1))``; ``p-value`` is the raw two-sided normal
    p, and ``adjust`` (``"holm"``, ``"bh"``, or both) fills the matching corrected column across
    the pairs. The effect size is Cliff's delta with Newcombe's 95% CI, as in
    :func:`fluxplot.stats.mann_whitney_cliff`.
    """
    gs, names = _groups(groups, names, "dunn")
    pooled = np.concatenate(gs)
    ranks = _sp.rankdata(pooled)
    total = pooled.size
    _, counts = np.unique(pooled, return_counts=True)
    tie_term = float(((counts**3 - counts).sum()) / (12 * (total - 1)))
    bounds = np.cumsum([0] + [g.size for g in gs])
    mean_ranks = [ranks[bounds[i]:bounds[i + 1]].mean() for i in range(len(gs))]
    rows = []
    for i, j in _pairs(len(gs)):
        a, b = gs[i], gs[j]
        se = np.sqrt((total * (total + 1) / 12 - tie_term) * (1 / a.size + 1 / b.size))
        z = (mean_ranks[i] - mean_ranks[j]) / se
        p = 2 * _sp.norm.sf(abs(z))
        delta, lo, hi = cliffs_delta(a, b)
        rows.append(report_row("Dunn's test", z, p, None, "Cliff's delta", delta, lo, hi,
                               n=(a.size, b.size), groups=(names[i], names[j])))
    return _adjust(rows, adjust)


def _adjust(rows: List[dict], adjust: Union[str, Sequence[str], None]) -> List[dict]:
    if adjust is None:
        return rows
    kinds = (adjust,) if isinstance(adjust, str) else tuple(adjust)
    for kind in kinds:
        if kind == "holm":
            rows = holm(rows)
        elif kind == "bh":
            rows = bh(rows)
        else:
            raise ValueError(f"adjust must be 'holm', 'bh' or a sequence of them, got {kind!r}")
    return rows


def pairwise(test: Callable[..., dict], groups: Dict[str, Any], pairs: Optional[Iterable[Tuple[str, str]]] = None,
             adjust: Union[str, Sequence[str], None] = "holm") -> List[Dict[str, Any]]:
    """Run a two-group test of this package over pairs of named groups and correct the p-values.

    >>> rows = fp.stats.pairwise(fp.stats.welch_hedges, {"ctl": ctl, "drug": drug, "sham": sham})
    >>> fp.brackets(ax, rows, positions=gb.positions)

    Parameters
    ----------
    test
        ``welch_hedges``, ``mann_whitney_cliff``, ``paired_t_hedges`` or ``wilcoxon_rank_biserial``
        (any callable ``test(a, b, names=(a_name, b_name)) -> row``).
    groups
        ``{name: sample}``; the order fixes the default pairs and the sign convention (``a − b``
        with ``a`` the earlier group).
    pairs
        The ``(a, b)`` name pairs to compare; default every pair in ``groups`` order.
    adjust
        ``"holm"`` (default), ``"bh"``, a sequence of both, or ``None``: the correction(s) filled
        in across the family (``p_corrected_holm`` / ``p_corrected_bh``).
    """
    names = list(groups)
    if pairs is None:
        pairs = list(itertools.combinations(names, 2))
    rows = []
    for a, b in pairs:
        for name in (a, b):
            if name not in groups:
                raise KeyError(f"pairwise: {name!r} is not one of the groups {names}")
        rows.append(test(groups[a], groups[b], names=(a, b)))
    return _adjust(rows, adjust)
