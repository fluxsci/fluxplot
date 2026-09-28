"""Shared pieces of the ``fp.stats`` tests: the reporting-row contract and input validation."""
from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np

REPORT_COLUMNS = (
    "sig_test_used",
    "test_statistic_value",
    "p-value",
    "p_corrected_holm",
    "dof",
    "effect_size_method",
    "effect_size_value",
    "effect_size_95_CI",
)


def report_row(test: str, statistic: float, p: float, dof: Optional[float], method: str,
               effect: float, lo: float, hi: float) -> dict:
    """One reporting row keyed by :data:`REPORT_COLUMNS`; ``dof=None`` for rank tests.

    ``p_corrected_holm`` starts out equal to ``p-value``: a row on its own is a family of one, for
    which the Holm correction is the identity. Pass the rows of a family through :func:`holm` to
    fill it in across them.
    """
    return {
        "sig_test_used": test,
        "test_statistic_value": float(statistic),
        "p-value": float(p),
        "p_corrected_holm": float(p),
        "dof": None if dof is None else float(dof),
        "effect_size_method": method,
        "effect_size_value": float(effect),
        "effect_size_95_CI": f"[{lo:.2f}, {hi:.2f}]",
    }


def as_sample(x: Any, name: str) -> np.ndarray:
    """A 1-D float array of at least 2 finite values, or a ``ValueError``."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError(f"{name} needs at least 2 observations, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values; drop them before testing")
    return arr


def as_pairs(a: Any, b: Any) -> tuple:
    """Two equal-length samples of matched observations (``a[i]`` pairs with ``b[i]``)."""
    a, b = as_sample(a, "a"), as_sample(b, "b")
    if a.size != b.size:
        raise ValueError(f"paired samples must have equal length, got {a.size} and {b.size}")
    return a, b


def holm_adjusted(p: Any) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values for a family of ``m`` raw two-sided p-values.

    The smallest p-value is multiplied by ``m``, the next by ``m - 1``, and so on down to ``1``;
    running maxima keep the adjusted values monotone in the raw ones and each is capped at 1.
    Rejecting every adjusted value below ``alpha`` controls the family-wise error rate at ``alpha``
    under any dependence between the tests, and is never less powerful than Bonferroni.
    """
    p = np.asarray(p, dtype=float).ravel()
    if p.size == 0:
        return p
    if not np.all((p >= 0) & (p <= 1)):
        raise ValueError("p-values must lie in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    stepped = p[order] * np.arange(m, 0, -1)
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(np.maximum.accumulate(stepped), 1.0)
    return adjusted


def holm(rows: Iterable[dict]) -> List[dict]:
    """Fill ``p_corrected_holm`` across a family of reporting rows.

    Holm's step-down correction needs every p-value in the family (each one's multiplier depends
    on its rank among the others), so it cannot be applied inside a single test call. Collect the
    rows of the comparisons that form one family — for instance every measure tested on the same
    animals in one figure — and pass them here once:

    >>> rows = [dict(measure=m, **fp.stats.welch_hedges(sd[m], sleep[m])) for m in measures]
    >>> rows = fp.stats.holm(rows)          # p_corrected_holm now spans the whole family
    >>> pl.DataFrame(rows)

    Parameters
    ----------
    rows
        Reporting rows (dicts with a ``"p-value"`` key, as returned by any ``fp.stats`` test).
        Extra keys such as a measure name are kept.

    Returns
    -------
    list of dict
        Copies of the rows in the same order, with ``p_corrected_holm`` set to the Holm-adjusted
        p-value across the family. A single row comes back unchanged (Holm of one is the identity).
    """
    rows = [dict(r) for r in rows]
    missing = [i for i, r in enumerate(rows) if "p-value" not in r]
    if missing:
        raise ValueError(f"rows {missing} have no 'p-value' key; pass fp.stats reporting rows")
    adjusted = holm_adjusted([r["p-value"] for r in rows])
    for r, q in zip(rows, adjusted):
        r["p_corrected_holm"] = float(q)
    return rows
