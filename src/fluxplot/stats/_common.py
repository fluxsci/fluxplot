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
    # appended (never reordered): the CI as numbers, the sample sizes, the groups compared, the
    # alternative hypothesis, the Benjamini–Hochberg column and the error dof of F tests
    "effect_size_ci_low",
    "effect_size_ci_high",
    "n_a",
    "n_b",
    "n_total",
    "groups",
    "alternative",
    "p_corrected_bh",
    "dof_error",
)

ALTERNATIVES = ("two-sided", "less", "greater")


def check_alternative(alternative: str) -> str:
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}, got {alternative!r}")
    return alternative


def report_row(test: str, statistic: float, p: float, dof: Optional[float], method: str,
               effect: float, lo: float, hi: float, *, n=None, groups=None,
               alternative: str = "two-sided", dof_error: Optional[float] = None,
               n_total: Optional[int] = None) -> dict:
    """One reporting row keyed by :data:`REPORT_COLUMNS`; ``dof=None`` for rank tests.

    ``p_corrected_holm`` and ``p_corrected_bh`` start out equal to ``p-value``: a row on its own is
    a family of one, for which both corrections are the identity. Pass the rows of a family through
    :func:`holm` / :func:`bh` to fill them in across the family.

    ``n`` is the group sizes (``(n_a, n_b)`` for a two-group row, one size per group for an
    omnibus row; ``n_total`` is their sum unless given — paired designs pass the number of
    subjects). ``groups`` names the groups compared, in the order the sign convention (``a - b``)
    refers to. ``dof_error`` is the denominator dof of an F test
    (``dof`` is then its numerator).
    """
    sizes = [int(v) for v in (n if n is not None else ())]
    two = len(sizes) == 2
    return {
        "sig_test_used": test,
        "test_statistic_value": float(statistic),
        "p-value": float(p),
        "p_corrected_holm": float(p),
        "dof": None if dof is None else float(dof),
        "effect_size_method": method,
        "effect_size_value": float(effect),
        "effect_size_95_CI": f"[{lo:.2f}, {hi:.2f}]",
        "effect_size_ci_low": float(lo),
        "effect_size_ci_high": float(hi),
        "n_a": sizes[0] if two else None,
        "n_b": sizes[1] if two else None,
        "n_total": int(n_total) if n_total is not None else (sum(sizes) if sizes else None),
        "groups": None if groups is None else [str(g) for g in groups],
        "alternative": check_alternative(alternative),
        "p_corrected_bh": float(p),
        "dof_error": None if dof_error is None else float(dof_error),
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
    keep = _checked(p)
    m = int(keep.sum())
    adjusted = np.full(p.size, np.nan)
    if m == 0:
        return adjusted
    q = p[keep]
    order = np.argsort(q, kind="stable")
    stepped = q[order] * np.arange(m, 0, -1)
    out = np.empty(m)
    out[order] = np.minimum(np.maximum.accumulate(stepped), 1.0)
    adjusted[keep] = out
    return adjusted


def _checked(p: np.ndarray) -> np.ndarray:
    """The mask of the p-values that take part in a correction: NaN passes through untouched (a
    test that could not be run leaves its slot empty instead of aborting the whole family)."""
    keep = ~np.isnan(p)
    if not np.all((p[keep] >= 0) & (p[keep] <= 1)):
        raise ValueError("p-values must lie in [0, 1]")
    return keep


def bh_adjusted(p: Any) -> np.ndarray:
    """Benjamini–Hochberg (1995) step-up adjusted p-values (q-values) for a family of ``m`` raw
    p-values.

    The largest p-value keeps its value, the next is multiplied by ``m / (m - 1)``, and so on down
    to ``m / 1`` for the smallest; running minima from the top keep the adjusted values monotone
    and each is capped at 1. Rejecting every adjusted value below ``alpha`` controls the false
    discovery rate at ``alpha`` for independent or positively dependent tests — the right control
    for a screen of many measures, where Holm's family-wise guarantee is needlessly strict. NaN
    passes through.
    """
    p = np.asarray(p, dtype=float).ravel()
    keep = _checked(p)
    m = int(keep.sum())
    adjusted = np.full(p.size, np.nan)
    if m == 0:
        return adjusted
    q = p[keep]
    order = np.argsort(q, kind="stable")
    stepped = q[order] * (m / np.arange(1, m + 1))  # the largest p keeps its exact value
    out = np.empty(m)
    out[order] = np.minimum(np.minimum.accumulate(stepped[::-1])[::-1], 1.0)
    adjusted[keep] = out
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
    return _fill(rows, "p_corrected_holm", holm_adjusted)


def bh(rows: Iterable[dict]) -> List[dict]:
    """Fill ``p_corrected_bh`` across a family of reporting rows with Benjamini–Hochberg adjusted
    p-values (see :func:`bh_adjusted`); the counterpart of :func:`holm` for false-discovery-rate
    control. Copies the rows, keeps their order and extra keys."""
    return _fill(rows, "p_corrected_bh", bh_adjusted)


def _fill(rows: Iterable[dict], column: str, adjust) -> List[dict]:
    rows = [dict(r) for r in rows]
    missing = [i for i, r in enumerate(rows) if "p-value" not in r]
    if missing:
        raise ValueError(f"rows {missing} have no 'p-value' key; pass fp.stats reporting rows")
    adjusted = adjust([r["p-value"] for r in rows])
    for r, q in zip(rows, adjusted):
        r[column] = float(q)
    return rows
