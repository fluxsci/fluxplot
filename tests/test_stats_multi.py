"""fp.stats — k-group omnibus tests, post-hoc pairs, BH, and the E2 row columns.

Reference numbers are hard-coded from pingouin 0.7.0 / scikit-posthocs 0.17.0 / scipy on the
McClave & Dietrich hair-colour / pain-threshold data and Field's Bushtucker repeated-measures data.
"""
import numpy as np
import pytest
from scipy import stats as sp

import fluxplot as fp

HAIR = {"Light Blond": [62, 60, 71, 55, 48], "Dark Blond": [63, 57, 52, 41, 43],
        "Light Brunette": [42, 50, 41, 37], "Dark Brunette": [32, 39, 51, 30, 35]}
BUSH = {"stick": [8, 9, 6, 5, 8, 7, 10, 12], "testicle": [7, 5, 2, 3, 4, 5, 2, 6],
        "eye": [1, 2, 3, 1, 5, 6, 7, 8], "grub": [6, 5, 8, 9, 8, 7, 2, 1]}
LONG = {"subj": [s for c in BUSH for s in range(8)], "food": [c for c in BUSH for _ in range(8)],
        "dv": [v for c in BUSH for v in BUSH[c]]}
SD = [2.76512194, 2.87610159, 2.08456236, 3.46759285, 2.80182563, 3.32866851]
SLEEP = [1.47117588, 2.37839778, 1.89789517, 2.29746906, 1.64774307, 1.82273808]

OMNIBUS = [fp.stats.anova_oneway, fp.stats.welch_anova, fp.stats.kruskal_epsilon]
POSTHOC = [fp.stats.tukey_hsd, fp.stats.games_howell, fp.stats.dunn]


def _row(fn):
    out = fn(*HAIR.values(), names=list(HAIR))
    return out if isinstance(out, dict) else out[0]


# --- the E2 columns ---------------------------------------------------------------------------

def test_report_columns_were_appended_not_reordered():
    assert fp.stats.REPORT_COLUMNS[:8] == (
        "sig_test_used", "test_statistic_value", "p-value", "p_corrected_holm", "dof",
        "effect_size_method", "effect_size_value", "effect_size_95_CI")
    assert set(fp.stats.REPORT_COLUMNS[8:]) >= {"effect_size_ci_low", "effect_size_ci_high", "n_a", "n_b",
                                                "groups", "alternative", "p_corrected_bh"}


def test_two_group_rows_carry_sizes_groups_alternative_and_numeric_ci():
    row = fp.stats.welch_hedges(SD, SLEEP, names=("SD", "sleep"))
    assert tuple(row) == fp.stats.REPORT_COLUMNS
    assert (row["n_a"], row["n_b"], row["n_total"]) == (6, 6, 12)
    assert row["groups"] == ["SD", "sleep"] and row["alternative"] == "two-sided"
    assert row["effect_size_95_CI"] == f"[{row['effect_size_ci_low']:.2f}, {row['effect_size_ci_high']:.2f}]"
    assert row["p_corrected_bh"] == row["p-value"] and row["dof_error"] is None
    paired = fp.stats.paired_t_hedges(SD, SLEEP)
    assert (paired["n_a"], paired["n_total"], paired["groups"]) == (6, 6, ["a", "b"])


@pytest.mark.parametrize("fn, ref", [
    (fp.stats.welch_hedges, lambda alt: sp.ttest_ind(SD, SLEEP, equal_var=False, alternative=alt).pvalue),
    (fp.stats.mann_whitney_cliff, lambda alt: sp.mannwhitneyu(SD, SLEEP, alternative=alt).pvalue),
    (fp.stats.paired_t_hedges, lambda alt: sp.ttest_rel(SD, SLEEP, alternative=alt).pvalue),
    (fp.stats.wilcoxon_rank_biserial, lambda alt: sp.wilcoxon(SD, SLEEP, alternative=alt).pvalue),
])
def test_alternative_reaches_scipy_and_leaves_the_ci_two_sided(fn, ref):
    two = fn(SD, SLEEP)
    for alt in ("less", "greater"):
        row = fn(SD, SLEEP, alternative=alt)
        assert row["p-value"] == pytest.approx(ref(alt)) and row["alternative"] == alt
        assert row["effect_size_95_CI"] == two["effect_size_95_CI"]
    with pytest.raises(ValueError, match="alternative"):
        fn(SD, SLEEP, alternative="both")


# --- Benjamini–Hochberg and NaN pass-through -------------------------------------------------------

def test_bh_matches_the_step_up_definition():
    p = [0.0034, 0.1245, 0.1355, 0.0061]
    # statsmodels multipletests(method="fdr_bh")
    assert fp.stats.bh_adjusted(p) == pytest.approx([0.0122, 0.1355, 0.1355, 0.0122], abs=1e-4)
    assert fp.stats.bh_adjusted([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])  # running min
    assert fp.stats.bh_adjusted([0.9]) == pytest.approx([0.9])
    rng = np.random.default_rng(3)
    for _ in range(50):
        p = rng.uniform(size=rng.integers(1, 9))
        q = fp.stats.bh_adjusted(p)
        h = fp.stats.holm_adjusted(p)
        assert np.all(q >= p) and np.all(q <= 1) and np.all(q <= h + 1e-12)  # BH never stricter than Holm
        assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)


def test_bh_fills_rows_and_nan_passes_through():
    rows = [fp.stats.welch_hedges(SD, SLEEP), fp.stats.welch_hedges(SLEEP, SD), fp.stats.welch_hedges(SD, [x + 0.01 for x in SD])]
    out = fp.stats.bh(rows)
    assert [r["p_corrected_bh"] for r in out] == pytest.approx(list(fp.stats.bh_adjusted([r["p-value"] for r in rows])))
    assert all(r["p_corrected_holm"] == r["p-value"] for r in out)  # untouched by bh()
    q = fp.stats.holm_adjusted([0.01, np.nan, 0.04])
    assert np.isnan(q[1]) and q[[0, 2]] == pytest.approx([0.02, 0.04])  # a family of two
    q = fp.stats.bh_adjusted([np.nan, 0.02, 0.04])
    assert np.isnan(q[0]) and q[1:] == pytest.approx([0.04, 0.04])
    assert np.all(np.isnan(fp.stats.bh_adjusted([np.nan])))


# --- omnibus tests -----------------------------------------------------------------------------

def test_anova_matches_pingouin():
    row = fp.stats.anova_oneway(*HAIR.values(), names=list(HAIR))
    assert row["sig_test_used"] == "One-way ANOVA"
    assert row["test_statistic_value"] == pytest.approx(6.791407, abs=1e-5)
    assert row["p-value"] == pytest.approx(0.004114, abs=1e-5)
    assert (row["dof"], row["dof_error"]) == (3, 15)
    assert row["effect_size_method"] == "Eta squared" and row["effect_size_value"] == pytest.approx(0.575962, abs=1e-5)
    assert row["groups"] == list(HAIR) and row["n_total"] == 19 and row["n_a"] is None
    omega = fp.stats.anova_oneway(*HAIR.values(), effect="omega2")
    ss_b, ss_w = 0.575962, 1 - 0.575962  # proportions: omega2 = (eta2 - df1/df2 (1-eta2)) / (1 + (1-eta2)/df2)
    assert omega["effect_size_value"] == pytest.approx((ss_b - 3 / 15 * ss_w) / (1 + ss_w / 15), abs=1e-5)
    assert omega["effect_size_95_CI"] == row["effect_size_95_CI"]  # the same population parameter
    assert omega["groups"] == ["group1", "group2", "group3", "group4"]


def test_variance_explained_ci_inverts_the_noncentral_f():
    from fluxplot.stats.multi_group import _variance_explained_ci
    f_obs, df1, df2 = 6.791407, 3, 15
    lo, hi = _variance_explained_ci(f_obs, df1, df2)
    n = df1 + df2 + 1
    for bound, target in ((lo, 0.975), (hi, 0.025)):
        lam = bound * n / (1 - bound)
        assert sp.ncf.cdf(f_obs, df1, df2, lam) == pytest.approx(target, abs=1e-6)
    assert lo < 0.575962 < hi
    # a null-ish F: the lower bound is exactly 0
    assert _variance_explained_ci(1.0, 3, 15)[0] == 0.0


def test_welch_anova_matches_pingouin():
    row = fp.stats.welch_anova(*HAIR.values())
    assert row["test_statistic_value"] == pytest.approx(5.890115, abs=1e-5)
    assert row["dof"] == 3 and row["dof_error"] == pytest.approx(8.329841, abs=1e-5)
    assert row["p-value"] == pytest.approx(0.018813, abs=1e-5)
    assert row["effect_size_method"] == "Omega squared"
    f, n = 5.890115, 19
    assert row["effect_size_value"] == pytest.approx(3 * (f - 1) / (3 * (f - 1) + n), abs=1e-5)
    # equal variances: Welch's omega2 formula equals the classic one
    eq = {k: v for k, v in HAIR.items()}
    classic = fp.stats.anova_oneway(*eq.values(), effect="omega2")["effect_size_value"]
    f_classic = fp.stats.anova_oneway(*eq.values())["test_statistic_value"]
    assert 3 * (f_classic - 1) / (3 * (f_classic - 1) + 19) == pytest.approx(classic)


def test_kruskal_matches_scipy_and_epsilon_definition():
    row = fp.stats.kruskal_epsilon(*HAIR.values(), n_boot=200)
    assert row["test_statistic_value"] == pytest.approx(10.58863, abs=1e-5)
    assert row["p-value"] == pytest.approx(0.014172, abs=1e-5)
    assert row["dof"] == 3 and row["dof_error"] is None
    assert row["effect_size_value"] == pytest.approx(10.58863 / 18, abs=1e-5)
    assert row["effect_size_ci_low"] <= row["effect_size_value"] <= row["effect_size_ci_high"]
    assert 0 <= row["effect_size_ci_low"] and row["effect_size_ci_high"] <= 1
    assert fp.stats.kruskal_epsilon(*HAIR.values(), n_boot=200) == row  # seeded: deterministic


def test_rm_anova_matches_pingouin_with_greenhouse_geisser():
    row = fp.stats.rm_anova(LONG, "subj", "food", "dv")
    assert row["test_statistic_value"] == pytest.approx(3.793806, abs=1e-5)
    eps = 0.532846
    assert row["dof"] == pytest.approx(3 * eps, abs=1e-4) and row["dof_error"] == pytest.approx(21 * eps, abs=1e-4)
    assert row["p-value"] == pytest.approx(0.062584, abs=1e-5)  # the GG-corrected p
    assert row["effect_size_method"] == "Partial eta squared"
    assert row["effect_size_value"] == pytest.approx(0.35148, abs=1e-5)
    assert row["groups"] == list(BUSH) and row["n_total"] == 8
    # a pandas frame is accepted too
    pd = pytest.importorskip("pandas")
    assert fp.stats.rm_anova(pd.DataFrame(LONG), "subj", "food", "dv") == row


def test_friedman_matches_pingouin():
    row = fp.stats.friedman_kendall(LONG, "subj", "food", "dv", n_boot=200)
    assert row["test_statistic_value"] == pytest.approx(11.526316, abs=1e-5)
    assert row["p-value"] == pytest.approx(0.009195, abs=1e-5)
    assert row["effect_size_method"] == "Kendall's W"
    assert row["effect_size_value"] == pytest.approx(0.480263, abs=1e-5)
    assert row["dof"] == 3 and row["groups"] == list(BUSH) and row["n_total"] == 8
    assert 0 <= row["effect_size_ci_low"] <= row["effect_size_value"] <= row["effect_size_ci_high"] <= 1


def test_long_table_validation():
    bad = {k: list(v) for k, v in LONG.items()}
    bad["dv"][0] = None
    with pytest.raises(ValueError, match="incomplete"):
        fp.stats.rm_anova(bad, "subj", "food", "dv")
    dup = {k: v + [v[0]] for k, v in LONG.items()}
    with pytest.raises(ValueError, match="more than one row"):
        fp.stats.rm_anova(dup, "subj", "food", "dv")
    with pytest.raises(KeyError, match="not a column"):
        fp.stats.rm_anova(LONG, "subj", "condition", "dv")
    two = {k: [x for x, c in zip(v, LONG["food"]) if c in ("stick", "eye")] for k, v in LONG.items()}
    with pytest.raises(ValueError, match="at least 3 conditions"):
        fp.stats.friedman_kendall(two, "subj", "food", "dv")


@pytest.mark.parametrize("fn", OMNIBUS)
def test_omnibus_rows_have_the_report_columns(fn):
    row = fn(*HAIR.values(), names=list(HAIR)) if fn is not fp.stats.kruskal_epsilon else fn(*HAIR.values(), names=list(HAIR), n_boot=50)
    assert tuple(row) == fp.stats.REPORT_COLUMNS
    assert row["n_a"] is None and row["n_b"] is None and row["n_total"] == 19
    with pytest.raises(ValueError, match="at least 2 groups"):
        fn([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="names has"):
        fn([1.0, 2.0], [3.0, 4.0], names=["only one"])


def test_groups_accept_a_dict():
    assert fp.stats.anova_oneway(HAIR) == fp.stats.anova_oneway(*HAIR.values(), names=list(HAIR))


# --- post-hoc tests ------------------------------------------------------------------------------

def _pair(rows, a, b):
    return next(r for r in rows if r["groups"] == [a, b])


def test_tukey_matches_pingouin():
    rows = fp.stats.tukey_hsd(*HAIR.values(), names=list(HAIR))
    assert len(rows) == 6 and all(tuple(r) == fp.stats.REPORT_COLUMNS for r in rows)
    r = _pair(rows, "Dark Blond", "Dark Brunette")
    assert r["test_statistic_value"] == pytest.approx(13.8) and r["p-value"] == pytest.approx(0.074068, abs=1e-5)
    assert r["effect_size_value"] == pytest.approx(1.413596, abs=1e-5) and r["effect_size_method"] == "Hedges' g (pooled SD)"
    assert r["p_corrected_holm"] == r["p_corrected_bh"] == r["p-value"]  # already family-wise
    assert (r["n_a"], r["n_b"], r["dof"]) == (5, 5, 15)
    r = _pair(rows, "Light Blond", "Light Brunette")
    assert r["test_statistic_value"] == pytest.approx(16.7) and r["p-value"] == pytest.approx(0.036647, abs=1e-5)
    assert r["effect_size_value"] == pytest.approx(2.015280, abs=1e-5)
    assert r["effect_size_ci_low"] < r["effect_size_value"] < r["effect_size_ci_high"]


def test_games_howell_matches_pingouin():
    rows = fp.stats.games_howell(*HAIR.values(), names=list(HAIR))
    r = _pair(rows, "Dark Blond", "Dark Brunette")
    assert r["test_statistic_value"] == pytest.approx(2.474565, abs=1e-5)
    assert r["dof"] == pytest.approx(7.906609, abs=1e-5) and r["p-value"] == pytest.approx(0.140085, abs=1e-5)
    r = _pair(rows, "Light Blond", "Dark Brunette")  # pairs follow the groups' order; signs a - b
    assert r["p-value"] == pytest.approx(0.014769, abs=1e-5) and r["test_statistic_value"] == pytest.approx(4.090697, abs=1e-5)
    # the effect size is welch_hedges' (non-pooled SD, Bonett CI)
    w = fp.stats.welch_hedges(HAIR["Light Blond"], HAIR["Dark Brunette"])
    assert r["effect_size_value"] == w["effect_size_value"] and r["effect_size_95_CI"] == w["effect_size_95_CI"]


def test_dunn_matches_scikit_posthocs():
    raw = fp.stats.dunn(*HAIR.values(), names=list(HAIR), adjust=None)
    assert _pair(raw, "Dark Blond", "Dark Brunette")["p-value"] == pytest.approx(0.030429, abs=1e-5)
    assert _pair(raw, "Light Blond", "Dark Brunette")["p-value"] == pytest.approx(0.002886, abs=1e-5)
    assert _pair(raw, "Light Blond", "Light Brunette")["p-value"] == pytest.approx(0.038098, abs=1e-5)
    assert _pair(raw, "Dark Blond", "Light Brunette")["p-value"] == pytest.approx(0.191812, abs=1e-5)
    holm = fp.stats.dunn(*HAIR.values(), names=list(HAIR))  # adjust="holm"
    assert _pair(holm, "Dark Blond", "Dark Brunette")["p_corrected_holm"] == pytest.approx(0.152144, abs=1e-5)
    assert _pair(holm, "Light Blond", "Dark Brunette")["p_corrected_holm"] == pytest.approx(0.017315, abs=1e-5)
    assert _pair(holm, "Light Blond", "Light Brunette")["p_corrected_holm"] == pytest.approx(0.152390, abs=1e-5)
    both = fp.stats.dunn(*HAIR.values(), names=list(HAIR), adjust=("holm", "bh"))
    assert [r["p_corrected_bh"] for r in both] == pytest.approx(list(fp.stats.bh_adjusted([r["p-value"] for r in raw])))
    # effect: Cliff's delta as mann_whitney_cliff reports it; z sign follows a - b
    mw = fp.stats.mann_whitney_cliff(HAIR["Light Blond"], HAIR["Dark Brunette"])
    r = _pair(raw, "Light Blond", "Dark Brunette")
    assert r["effect_size_value"] == mw["effect_size_value"] and r["test_statistic_value"] > 0
    assert r["dof"] is None
    with pytest.raises(ValueError, match="adjust must be"):
        fp.stats.dunn(*HAIR.values(), adjust="bonferroni")


def test_dunn_tie_correction_reduces_to_kruskal_variance():
    # with ties, the variance term shrinks by sum(t^3 - t) / (12 (N - 1)); without ties it is N(N+1)/12
    from fluxplot.stats.multi_group import dunn as _dunn
    a, b, c = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]
    row = _pair(_dunn(a, b, c, names="abc", adjust=None), "a", "b")
    n = 9
    z = (2.0 - 5.0) / np.sqrt(n * (n + 1) / 12 * (2 / 3))  # mean ranks 2 and 5
    assert row["test_statistic_value"] == pytest.approx(z)


def test_pairwise_runs_a_two_group_test_over_named_pairs_and_corrects():
    rows = fp.stats.pairwise(fp.stats.welch_hedges, HAIR)
    assert [r["groups"] for r in rows] == [list(p) for p in __import__("itertools").combinations(HAIR, 2)]
    direct = fp.stats.welch_hedges(HAIR["Light Blond"], HAIR["Dark Blond"], names=("Light Blond", "Dark Blond"))
    assert {k: v for k, v in rows[0].items() if k != "p_corrected_holm"} == {k: v for k, v in direct.items() if k != "p_corrected_holm"}
    assert [r["p_corrected_holm"] for r in rows] == pytest.approx(list(fp.stats.holm_adjusted([r["p-value"] for r in rows])))
    sub = fp.stats.pairwise(fp.stats.mann_whitney_cliff, HAIR, pairs=[("Light Blond", "Dark Brunette")], adjust="bh")
    assert len(sub) == 1 and sub[0]["p_corrected_bh"] == sub[0]["p-value"]
    with pytest.raises(KeyError, match="not one of the groups"):
        fp.stats.pairwise(fp.stats.welch_hedges, HAIR, pairs=[("Light Blond", "Red")])
    paired = fp.stats.pairwise(fp.stats.paired_t_hedges, BUSH, adjust=("holm", "bh"))
    assert len(paired) == 6 and all(r["n_total"] == 8 for r in paired)


@pytest.mark.parametrize("fn", POSTHOC)
def test_posthoc_rows_are_one_per_pair_with_the_report_columns(fn):
    rows = fn(*HAIR.values(), names=list(HAIR))
    assert len(rows) == 6 and all(tuple(r) == fp.stats.REPORT_COLUMNS for r in rows)
    assert all(r["n_total"] == r["n_a"] + r["n_b"] for r in rows)
    rev = fn(*reversed(list(HAIR.values())), names=list(reversed(list(HAIR))))
    fwd_pair = _pair(rows, "Light Blond", "Dark Brunette")
    rev_pair = _pair(rev, "Dark Brunette", "Light Blond")
    assert rev_pair["effect_size_value"] == pytest.approx(-fwd_pair["effect_size_value"])
    assert rev_pair["p-value"] == pytest.approx(fwd_pair["p-value"])
