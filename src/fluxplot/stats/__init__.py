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

Multiple comparisons: each row's ``p_corrected_holm`` equals its ``p-value`` until the rows of a
family are passed through :func:`holm`, which fills in Holm step-down adjusted p-values across them
(:func:`holm_adjusted` does the same for a bare array of p-values).
"""
from ._common import REPORT_COLUMNS, holm, holm_adjusted
from .paired import paired_t_hedges, wilcoxon_rank_biserial
from .two_group import mann_whitney_cliff, welch_hedges

__all__ = [
    "welch_hedges",
    "mann_whitney_cliff",
    "paired_t_hedges",
    "wilcoxon_rank_biserial",
    "holm",
    "holm_adjusted",
    "REPORT_COLUMNS",
]
