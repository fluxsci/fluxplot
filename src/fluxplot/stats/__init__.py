"""Statistics behind the plots — tests that return rows ready for a plot's ``_stats`` dissection.

Accessed as ``fp.stats``. Independent (unpaired) samples:

* :func:`welch_hedges` — Welch's t-test + Hedges' g (non-pooled SD), for comparing means.
* :func:`mann_whitney_cliff` — Mann–Whitney U test + Cliff's delta, for a rank-based comparison.

Paired samples (``a[i]`` matched with ``b[i]``):

* :func:`paired_t_hedges` — paired t-test + Hedges' g_z, for the mean paired difference.
* :func:`wilcoxon_rank_biserial` — Wilcoxon signed-rank test + matched-pairs rank-biserial
  correlation, for a rank-based comparison of paired differences.

Every function takes ``(a, b)``, orients signs as ``a - b``, reports a 95% CI for the effect size,
and returns one reporting row keyed by :data:`REPORT_COLUMNS` (``dof`` is ``None`` for rank tests).

Three or more groups (:mod:`fluxplot.stats.multi_group`): the omnibus tests
:func:`anova_oneway`, :func:`welch_anova`, :func:`kruskal_epsilon`, :func:`rm_anova` and
:func:`friedman_kendall` (one row each), the post-hoc tests :func:`tukey_hsd`, :func:`games_howell`
and :func:`dunn` (one row per pair), and :func:`pairwise`, which runs any two-group test over the
pairs of a ``{name: sample}`` family and corrects the p-values. Post-hoc rows drop straight into
:func:`fluxplot.brackets`.

Multiple comparisons: each row's ``p_corrected_holm`` / ``p_corrected_bh`` equals its ``p-value``
until the rows of a family are passed through :func:`holm` (family-wise error, Holm step-down) or
:func:`bh` (false discovery rate, Benjamini–Hochberg); :func:`holm_adjusted` / :func:`bh_adjusted`
do the same for a bare array of p-values.
"""
from ._common import REPORT_COLUMNS, bh, bh_adjusted, holm, holm_adjusted
from .multi_group import (
    anova_oneway,
    dunn,
    friedman_kendall,
    games_howell,
    kruskal_epsilon,
    pairwise,
    rm_anova,
    tukey_hsd,
    welch_anova,
)
from .paired import paired_t_hedges, wilcoxon_rank_biserial
from .two_group import mann_whitney_cliff, welch_hedges

__all__ = [
    "welch_hedges",
    "mann_whitney_cliff",
    "paired_t_hedges",
    "wilcoxon_rank_biserial",
    "anova_oneway",
    "welch_anova",
    "kruskal_epsilon",
    "rm_anova",
    "friedman_kendall",
    "tukey_hsd",
    "games_howell",
    "dunn",
    "pairwise",
    "holm",
    "holm_adjusted",
    "bh",
    "bh_adjusted",
    "REPORT_COLUMNS",
]
