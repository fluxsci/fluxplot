"""fp.stats — two-group tests returning a reporting row.

What this file pins down: every row carries exactly the reporting columns; tests match scipy;
effect sizes match their textbook definitions; each CI satisfies the equation that defines it and
stays non-degenerate at the boundaries (complete separation is common at n = 6); signs follow
``a - b``; and bad input fails loudly instead of producing a silently wrong row.
"""
import numpy as np
import pytest
from scipy import stats as sp

import fluxplot as fp
from fluxplot.stats.paired import _nct_ncp_for, _rank_biserial_moments
from fluxplot.stats.two_group import _newcombe_auc_bounds

SD = [2.76512194, 2.87610159, 2.08456236, 3.46759285, 2.80182563, 3.32866851]
SLEEP = [1.47117588, 2.37839778, 1.89789517, 2.29746906, 1.64774307, 1.82273808]
Z = sp.norm.ppf(0.975)

ALL = [fp.stats.welch_hedges, fp.stats.mann_whitney_cliff,
       fp.stats.paired_t_hedges, fp.stats.wilcoxon_rank_biserial]


def _ci(row):
    return tuple(float(v) for v in row["effect_size_95_CI"].strip("[]").split(","))


@pytest.mark.parametrize("fn", ALL)
def test_row_has_exactly_the_report_columns(fn):
    assert tuple(fn(SD, SLEEP)) == fp.stats.REPORT_COLUMNS


@pytest.mark.parametrize("fn", ALL)
def test_sign_follows_a_minus_b(fn):
    fwd, rev = fn(SD, SLEEP), fn(SLEEP, SD)
    assert rev["effect_size_value"] == pytest.approx(-fwd["effect_size_value"])
    lo, hi = _ci(fwd)
    assert _ci(rev) == (-hi, -lo)


@pytest.mark.parametrize("fn", ALL)
def test_estimate_lies_inside_its_ci(fn):
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.normal(0.7, 1, 7), rng.normal(0, 1, 7)
        row = fn(a, b)
        lo, hi = _ci(row)
        assert lo - 0.005 <= row["effect_size_value"] <= hi + 0.005


# --- Welch's t-test + Hedges' g ------------------------------------------------------------------

def test_welch_matches_scipy_and_definitions():
    row = fp.stats.welch_hedges(SD, SLEEP)
    ref = sp.ttest_ind(SD, SLEEP, equal_var=False)
    assert row["sig_test_used"] == "Welch's t-test"
    assert row["test_statistic_value"] == pytest.approx(ref.statistic)
    assert row["p-value"] == pytest.approx(ref.pvalue)
    assert row["dof"] == pytest.approx(ref.df)
    a, b = np.array(SD), np.array(SLEEP)
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    j = 1 - 3 / (4 * 10 - 1)
    assert row["effect_size_method"] == "Hedges' g (non-pooled SD)"
    assert row["effect_size_value"] == pytest.approx(j * (a.mean() - b.mean()) / s)
    assert row["effect_size_value"] == pytest.approx(2.0853, abs=1e-4)
    assert row["effect_size_95_CI"] == "[0.64, 3.87]"  # Bonett CI for delta, not J-scaled


# --- Mann–Whitney U + Cliff's delta --------------------------------------------------------------

def test_mann_whitney_matches_scipy_and_delta_identity():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.5, 1, 7), rng.normal(0, 1, 9)
    row = fp.stats.mann_whitney_cliff(a, b)
    ref = sp.mannwhitneyu(a, b, alternative="two-sided")
    assert row["sig_test_used"] == "Mann–Whitney U test"
    assert row["test_statistic_value"] == ref.statistic
    assert row["p-value"] == pytest.approx(ref.pvalue)
    assert row["dof"] is None
    assert row["effect_size_method"] == "Cliff's delta"
    assert row["effect_size_value"] == pytest.approx(2 * ref.statistic / (7 * 9) - 1)


def test_cliffs_delta_counts_ties_as_neither():
    a, b = [1, 2, 2, 3], [2, 3, 3, 5]
    # pairs (a>b, a<b) counted by hand: 1 win (3>2), 11 losses, 4 ties of 16
    assert fp.stats.mann_whitney_cliff(a, b)["effect_size_value"] == pytest.approx((1 - 11) / 16)


def test_newcombe_bounds_solve_the_score_equation():
    m, n, theta = 6, 6, 0.8
    n_star = (m + n) / 2 - 1
    lo, hi = _newcombe_auc_bounds(theta, m, n)
    for t in (lo, hi):
        var = t * (1 - t) * (1 + (n_star - 1) * ((1 - t) / (2 - t) + t / (1 + t))) / (m * n)
        assert (t - theta) ** 2 == pytest.approx(Z**2 * var, abs=1e-10)
    assert lo < theta < hi


def test_complete_separation_gives_a_real_interval():
    row = fp.stats.mann_whitney_cliff([10, 11, 12, 13, 14, 15], [0, 1, 2, 3, 4, 5])
    assert row["effect_size_value"] == 1
    lo, hi = _ci(row)
    assert 0 < lo < 1 and hi == 1


# --- paired t-test + Hedges' g_z -----------------------------------------------------------------

def test_paired_t_matches_scipy_and_definitions():
    row = fp.stats.paired_t_hedges(SD, SLEEP)
    ref = sp.ttest_rel(SD, SLEEP)
    assert row["sig_test_used"] == "Paired t-test"
    assert row["test_statistic_value"] == pytest.approx(ref.statistic)
    assert row["p-value"] == pytest.approx(ref.pvalue)
    assert row["dof"] == 5
    diff = np.array(SD) - np.array(SLEEP)
    j = 1 - 3 / (4 * 5 - 1)
    assert row["effect_size_method"] == "Hedges' g_z"
    assert row["effect_size_value"] == pytest.approx(j * diff.mean() / diff.std(ddof=1))


def test_paired_ci_is_the_noncentral_t_interval():
    t_obs, df, n = 4.2, 5, 6
    lo, hi = _nct_ncp_for(t_obs, df, 0.975), _nct_ncp_for(t_obs, df, 0.025)
    assert sp.nct.cdf(t_obs, df, lo) == pytest.approx(0.975, abs=1e-10)
    assert sp.nct.cdf(t_obs, df, hi) == pytest.approx(0.025, abs=1e-10)
    diff = np.array(SD) - np.array(SLEEP)
    t = diff.mean() / diff.std(ddof=1) * np.sqrt(n)
    expected = (_nct_ncp_for(t, df, 0.975) / np.sqrt(n), _nct_ncp_for(t, df, 0.025) / np.sqrt(n))
    got = _ci(fp.stats.paired_t_hedges(SD, SLEEP))
    assert got == pytest.approx(expected, abs=0.005)  # not J-scaled


def test_paired_t_null_ci_is_symmetric():
    lo, hi = _nct_ncp_for(0.0, 9, 0.975), _nct_ncp_for(0.0, 9, 0.025)
    assert lo == pytest.approx(-hi)


# --- Wilcoxon signed-rank + matched-pairs rank-biserial ------------------------------------------

def test_wilcoxon_matches_scipy_and_kerby_r():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0.4, 1, 9), rng.normal(0, 1, 9)
    row = fp.stats.wilcoxon_rank_biserial(a, b)
    ref = sp.wilcoxon(a, b)
    total = 9 * 10 / 2
    w_plus = row["test_statistic_value"]
    assert row["sig_test_used"] == "Wilcoxon signed-rank test"
    assert min(w_plus, total - w_plus) == ref.statistic  # scipy reports min(W+, W-)
    assert row["p-value"] == pytest.approx(ref.pvalue)
    assert row["dof"] is None
    assert row["effect_size_method"] == "Matched-pairs rank-biserial correlation"
    assert row["effect_size_value"] == pytest.approx(2 * w_plus / total - 1)


def test_wilcoxon_drops_zero_differences():
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b = [1.0, 1.5, 3.5, 3.0, 4.0, 4.5]  # one zero difference
    row = fp.stats.wilcoxon_rank_biserial(a, b)
    same = fp.stats.wilcoxon_rank_biserial(a[1:], b[1:])
    assert row["effect_size_value"] == same["effect_size_value"]
    assert row["effect_size_95_CI"] == same["effect_size_95_CI"]


def test_rank_biserial_moments():
    # at delta = 0 the variance is the classical null n(n+1)(2n+1)/24 on the W+ scale
    for n in (2, 3, 6, 20):
        mean, var = _rank_biserial_moments(0.0, n)
        assert mean == pytest.approx(0.0, abs=1e-12)
        assert var == pytest.approx(2 * (2 * n + 1) / (3 * n * (n + 1)))
    # off the null, against simulation
    rng = np.random.default_rng(3)
    n, delta = 8, 0.7
    d = rng.normal(delta, 1, (20000, n))
    ranks = sp.rankdata(np.abs(d), axis=1)
    r = 2 * (ranks * (d > 0)).sum(axis=1) / (n * (n + 1) / 2) - 1
    mean, var = _rank_biserial_moments(delta, n)
    assert r.mean() == pytest.approx(mean, abs=0.005)
    assert r.var() == pytest.approx(var, rel=0.03)


def test_all_same_sign_gives_a_real_interval():
    row = fp.stats.wilcoxon_rank_biserial(SD, SLEEP)  # every SD value exceeds its SLEEP partner
    assert row["effect_size_value"] == 1
    lo, hi = _ci(row)
    assert 0 < lo < 1 and hi == 1


# --- input validation ----------------------------------------------------------------------------

@pytest.mark.parametrize("fn", ALL)
@pytest.mark.parametrize("a, b, match", [
    ([1.0], [1.0], "at least 2"),
    ([1.0, np.nan], [1.0, 2.0], "NaN"),
])
def test_bad_input_raises(fn, a, b, match):
    with pytest.raises(ValueError, match=match):
        fn(a, b)


def test_degenerate_inputs_raise():
    with pytest.raises(ValueError, match="zero variance"):
        fp.stats.welch_hedges([1.0, 1.0], [2.0, 2.0])
    with pytest.raises(ValueError, match="zero variance"):
        fp.stats.paired_t_hedges([2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="every paired difference is zero"):
        fp.stats.wilcoxon_rank_biserial([1.0, 2.0], [1.0, 2.0])


@pytest.mark.parametrize("fn", [fp.stats.paired_t_hedges, fp.stats.wilcoxon_rank_biserial])
def test_paired_needs_equal_lengths(fn):
    with pytest.raises(ValueError, match="equal length"):
        fn([1.0, 2.0, 3.0], [1.0, 2.0])


# --- Holm correction across a family --------------------------------------------------------------

@pytest.mark.parametrize("fn", ALL)
def test_single_row_holm_equals_raw_p(fn):
    row = fn(SD, SLEEP)
    assert row["p_corrected_holm"] == row["p-value"]
    assert fp.stats.holm([row]) == [row]


def test_holm_adjusted_matches_the_step_down_definition():
    # Holm (1979): sorted p × (m, m-1, …, 1), running max, capped at 1 (values as statsmodels gives)
    p = [0.0034, 0.1245, 0.1355, 0.0061]
    assert fp.stats.holm_adjusted(p) == pytest.approx([0.0136, 0.2490, 0.2490, 0.0183], abs=1e-4)
    assert fp.stats.holm_adjusted([0.04, 0.03]) == pytest.approx([0.06, 0.06])  # running max
    assert fp.stats.holm_adjusted([0.6, 0.9]) == pytest.approx([1.0, 1.0])
    assert fp.stats.holm_adjusted([]).size == 0


def test_holm_is_monotone_never_below_raw_and_at_most_bonferroni():
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = rng.uniform(size=rng.integers(1, 9))
        q = fp.stats.holm_adjusted(p)
        assert np.all(q >= p) and np.all(q <= np.minimum(1, p * p.size))
        assert np.all(np.diff(q[np.argsort(p)]) >= 0)


def test_holm_fills_rows_keeps_order_and_extra_keys_and_does_not_mutate():
    measures = {"app": (SD, SLEEP), "tau": (SLEEP, SD), "flat": (SD, [x + 0.01 for x in SD])}
    rows = [dict(measure=k, **fp.stats.welch_hedges(a, b)) for k, (a, b) in measures.items()]
    before = [dict(r) for r in rows]
    out = fp.stats.holm(rows)
    assert rows == before
    assert [r["measure"] for r in out] == list(measures)
    assert [r["p_corrected_holm"] for r in out] == pytest.approx(
        list(fp.stats.holm_adjusted([r["p-value"] for r in rows])))
    assert all(tuple(k for k in r if k != "measure") == fp.stats.REPORT_COLUMNS for r in out)


def test_holm_rejects_rows_without_a_p_value():
    with pytest.raises(ValueError, match="no 'p-value' key"):
        fp.stats.holm([fp.stats.welch_hedges(SD, SLEEP), {"measure": "x"}])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fp.stats.holm_adjusted([0.5, 1.5])
