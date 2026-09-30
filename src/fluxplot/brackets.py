"""``fp.brackets``: significance brackets drawn from ``fp.stats`` rows, auto-stacked, with the test
behind every star recorded in the manifest.

>>> rows = fp.stats.pairwise(fp.stats.welch_hedges, {"ctl": ctl, "drug": drug, "sham": sham})
>>> gb = fp.glowbar(data=df, x="group", y="value", ax=ax)
>>> gb.brackets(rows)                      # or fp.brackets(ax, rows, positions=gb.positions)

Each post-hoc row names its pair in ``groups``; ``positions`` maps a group name to its x. Brackets
are placed shortest first, each one step above the data it spans and above every bracket it
overlaps in x, so no bracket ever crosses another or sits on the points. The manifest overlay of
each bracket carries ``stats`` — test, statistic, raw and corrected p, correction, effect size and
its CI, sizes — so the figure states exactly which test each star came from.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["brackets", "stars", "format_p", "THRESHOLDS"]

#: the default star thresholds: ``p < 0.001`` → ``***``, ``p < 0.01`` → ``**``, ``p < 0.05`` → ``*``
THRESHOLDS = ((0.001, "***"), (0.01, "**"), (0.05, "*"))

_CORRECTION_OF = {"p_corrected_holm": "holm", "p_corrected_bh": "bh", "p-value": "none"}


def stars(p: float, thresholds: Sequence[Tuple[float, str]] = THRESHOLDS, ns: str = "ns") -> str:
    """The star label of a p-value: the first threshold it falls under, else ``ns``."""
    if p is None or not math.isfinite(p):
        return ns
    for cut, label in sorted(thresholds, key=lambda t: t[0]):
        if p < cut:
            return label
    return ns


def format_p(p: float, floor: float = 0.001) -> str:
    """``p < 0.001`` below ``floor``, else ``p = 0.003`` (three decimals)."""
    if p is None or not math.isfinite(p):
        return "p = n/a"
    if p < floor:
        return f"p < {floor:g}"
    return f"p = {p:.3f}"


def _positions_from_ticks(ax) -> Dict[str, float]:
    return {lbl.get_text(): float(tick) for tick, lbl in zip(ax.get_xticks(), ax.get_xticklabels())
            if lbl.get_text()}


def _data_top(ax, x_lo: float, x_hi: float) -> Optional[float]:
    """The highest data y drawn between ``x_lo`` and ``x_hi`` (inclusive) on ``ax``, from lines,
    collections (offsets or path vertices) and patches — the things a bracket must clear."""
    from matplotlib.collections import Collection
    from matplotlib.patches import Patch
    top = -np.inf

    def take(x, y):
        nonlocal top
        x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
        keep = np.isfinite(x) & np.isfinite(y) & (x >= x_lo) & (x <= x_hi)
        if keep.any():
            top = max(top, float(y[keep].max()))

    for ln in ax.lines:
        if ln.get_transform() is ax.transData or ln.get_transform().contains_branch(ax.transData):
            take(ln.get_xdata(orig=False), ln.get_ydata(orig=False))
    for coll in ax.collections:
        if not isinstance(coll, Collection):
            continue
        offsets = coll.get_offsets()
        if offsets is not None and len(offsets) and coll.get_offset_transform() is ax.transData:
            off = np.ma.filled(np.ma.asarray(offsets, dtype=float), np.nan)
            take(off[:, 0], off[:, 1])
        elif coll.get_transform() is ax.transData or coll.get_transform().contains_branch(ax.transData):
            for path in coll.get_paths():
                v = path.vertices
                take(v[:, 0], v[:, 1])
    for patch in ax.patches:
        if not isinstance(patch, Patch):
            continue
        try:
            v = patch.get_path().transformed(patch.get_patch_transform()).vertices
        except Exception:  # a patch with no data-space geometry
            continue
        take(v[:, 0], v[:, 1])
    return None if not np.isfinite(top) else top


def _stats_payload(row: dict, p_column: str) -> dict:
    lo, hi = row.get("effect_size_ci_low"), row.get("effect_size_ci_high")
    if lo is None and isinstance(row.get("effect_size_95_CI"), str):
        try:
            lo, hi = (float(v) for v in row["effect_size_95_CI"].strip("[]").split(","))
        except ValueError:
            lo = hi = None
    n = [row.get("n_a"), row.get("n_b")] if row.get("n_a") is not None else row.get("n_total")
    out = {
        "test": row.get("sig_test_used"),
        "statistic": row.get("test_statistic_value"),
        "p": row.get("p-value"),
        "pCorrected": row.get(p_column),
        "correction": _CORRECTION_OF.get(p_column, p_column),
        "effectSizeMethod": row.get("effect_size_method"),
        "effectSize": row.get("effect_size_value"),
        "ciLow": lo,
        "ciHigh": hi,
        "n": n,
    }
    if row.get("dof") is not None:
        out["dof"] = row["dof"]
    if row.get("alternative"):
        out["alternative"] = row["alternative"]
    return {k: v for k, v in out.items() if v is not None}


def brackets(ax, rows: Iterable[dict], *, positions: Optional[Dict[Any, float]] = None,
             pairs: Optional[Iterable[Tuple[Any, Any]]] = None,
             label: Union[str, Callable[[dict], str]] = "stars", p_column: str = "p_corrected_holm",
             thresholds: Sequence[Tuple[float, str]] = THRESHOLDS, ns: bool = True,
             top: Optional[float] = None, step: float = 0.06, tip: float = 0.02, **line_kw) -> list:
    """Draw one significance bracket per ``fp.stats`` row, stacked so none overlap.

    Parameters
    ----------
    ax
        The axes the compared groups are plotted on.
    rows
        Reporting rows whose ``groups`` is the pair ``[a, b]`` (``fp.stats.pairwise``,
        ``tukey_hsd``, ``games_howell``, ``dunn``; or any two-group row given ``names=``).
    positions
        ``{group name: x}``. Default: the axes' x tick labels (a glowbar / fluxbox / categorical
        axis), which must name every group; ``GlowbarResult.positions`` and
        ``FluxboxResult.positions`` give it directly.
    pairs
        Draw only these ``(a, b)`` pairs (either order), in this order of priority when several
        rows name the same pair. Default: every row.
    label
        ``"stars"`` (``***``/``**``/``*``/``ns`` by ``thresholds``), ``"p"`` (``p = 0.003`` /
        ``p < 0.001``), ``"both"`` (stars over the p), or a callable ``row -> str``.
    p_column
        The row column the label and the stacking read: ``"p_corrected_holm"`` (default),
        ``"p_corrected_bh"`` or ``"p-value"``.
    thresholds
        ``((p, stars), …)``: the first threshold ``p`` falls under gives its label.
    ns
        Draw a bracket labelled ``ns`` for non-significant pairs (default). ``False`` omits them.
    top
        Start the stack at this y instead of above the data each bracket spans.
    step, tip
        Vertical step between brackets and the tip height, as fractions of the y range (on a log
        axis, of the range in decades).
    **line_kw
        Passed to :func:`fluxplot.significance_bracket` (``color``, ``linewidth``, ``text_kw``, …).

    Returns
    -------
    list
        The bracket line artists, in drawing order (shortest span first). Each carries the
        manifest ``stats`` payload of its row.
    """
    from .api import significance_bracket
    rows = list(rows)
    from_ticks = positions is None
    if from_ticks:
        positions = _positions_from_ticks(ax)
    pos = {str(k): float(v) for k, v in positions.items()}
    wanted = None if pairs is None else [(str(a), str(b)) for a, b in pairs]

    items = []
    for row in rows:
        groups = row.get("groups")
        if not groups or len(groups) != 2:
            raise ValueError("brackets: every row needs groups=[a, b]; run the test with names=(a, b) or use fp.stats.pairwise")
        a, b = (str(g) for g in groups)
        if wanted is not None and (a, b) not in wanted and (b, a) not in wanted:
            continue
        for g in (a, b):
            if g not in pos:
                if from_ticks:
                    raise ValueError(f"brackets: positions= is needed — the x tick labels do not name group {g!r}")
                raise KeyError(f"brackets: group {g!r} has no position; positions has {sorted(pos)}")
        if p_column not in row:
            raise KeyError(f"brackets: rows have no {p_column!r} column (have {sorted(row)})")
        p = row[p_column]
        significant = p is not None and math.isfinite(p) and p < max(t for t, _ in thresholds)
        if not significant and not ns:
            continue
        items.append((a, b, row, p))
    if wanted is not None:
        order = {pair: i for i, pair in enumerate(wanted)}
        items.sort(key=lambda it: min(order.get((it[0], it[1]), len(order)), order.get((it[1], it[0]), len(order))))
    # shortest span first: a long bracket then rises above the short ones it covers
    items.sort(key=lambda it: abs(pos[it[0]] - pos[it[1]]))

    log = ax.get_yscale() == "log"
    y0, y1 = ax.get_ylim()
    if log:
        span = abs(math.log10(y1) - math.log10(y0)) or 1.0
        rise = lambda y, frac: y * 10 ** (frac * span)  # noqa: E731
    else:
        span = abs(y1 - y0) or 1.0
        rise = lambda y, frac: y + frac * span  # noqa: E731
    xs = sorted(set(pos.values()))
    half = (min(np.diff(xs)) / 2.0) if len(xs) > 1 else 0.5

    placed: List[Tuple[float, float, float]] = []  # (x_lo, x_hi, y of the bracket base)
    out = []
    for a, b, row, p in items:
        x_lo, x_hi = sorted((pos[a], pos[b]))
        if top is not None:
            base = float(top)
        else:
            data_top = _data_top(ax, x_lo - half, x_hi + half)
            base = rise(data_top if data_top is not None else y0, step)
        for px_lo, px_hi, py in placed:
            if x_lo <= px_hi and px_lo <= x_hi:  # x-spans overlap (a shared category counts)
                base = max(base, rise(py, step))
        if callable(label):
            text = str(label(row))
        elif label == "stars":
            text = stars(p, thresholds)
        elif label == "p":
            text = format_p(p)
        elif label == "both":
            text = f"{stars(p, thresholds)}\n{format_p(p)}"
        else:
            raise ValueError(f"brackets: label must be 'stars', 'p', 'both' or a callable, got {label!r}")
        height = rise(base, tip) - base
        br = significance_bracket(ax, x0=pos[a], x1=pos[b], y=base, label=text, between=(a, b), p=p,
                                  name=f"{a}-{b}", height=height, stats=_stats_payload(row, p_column),
                                  **line_kw)
        placed.append((x_lo, x_hi, base))
        out.append(br)
    if out:  # make room: the top bracket must lie inside the axes
        highest = max(rise(py, tip) for _, _, py in placed)
        top_tip = rise(highest, step / 2)
        if top_tip > y1:
            ax.set_ylim(y0, top_tip)
    return out
