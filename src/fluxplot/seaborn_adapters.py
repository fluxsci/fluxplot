"""Exact identity for seaborn's categorical plots (:func:`fluxplot.tag_seaborn`).

seaborn draws a categorical plot as anonymous artists in a fixed order: one ``BoxPlotContainer``
per hue level, one violin body per (category, hue) pair, one strip / swarm collection per
category with the hue levels mixed inside it, one mean line plus its error lines per hue level.
Given the plot kind — and, for the row-wise splits, the frame it was drawn from — those artists
can be named exactly: the hue levels in seaborn's own ``categorical_order``, the categories in
tick order, and a mixed collection's points by the frame rows they came from
(``Mark.data["point_subset"]``). Nothing is inferred from colour or geometry.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from .descriptors import Mark

CATEGORICAL_KINDS = ("boxplot", "violinplot", "stripplot", "swarmplot", "pointplot")


def categorical_order(values) -> list:
    """seaborn's category order: a Categorical's categories, else sorted numbers, else the order
    of appearance (``seaborn._core.rules.categorical_order`` when seaborn is importable)."""
    try:
        from seaborn._core.rules import categorical_order as _order
        import pandas as pd
        return list(_order(pd.Series(values)))
    except Exception:
        pass
    seen = list(dict.fromkeys(v for v in list(values) if not _missing(v)))
    try:
        if all(isinstance(v, (int, float, np.number)) for v in seen):
            return sorted(seen)
    except TypeError:
        pass
    return seen


def _missing(v) -> bool:
    if v is None:
        return True
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


def _column(data, key) -> list:
    if key is None:
        return []
    try:
        col = data[key]
    except Exception as exc:
        raise KeyError(f"tag_seaborn: {key!r} is not a column of data") from exc
    for attr in ("to_list", "tolist"):
        if hasattr(col, attr):
            return list(getattr(col, attr)())
    return list(col)


def _is_numeric(values) -> bool:
    try:
        return all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in values if not _missing(v))
    except TypeError:
        return False


class Frame:
    """What the frame says about a categorical plot: the categorical variable and its order, the
    hue levels and, per row, the (category, hue) it belongs to."""

    def __init__(self, data, x, y, hue, order=None, hue_order=None, orient=None):
        xs, ys = _column(data, x), _column(data, y)
        if orient is None:
            if x is not None and y is not None:
                orient = "h" if _is_numeric(xs) and not _is_numeric(ys) else "v"
            else:
                orient = "v" if x is not None else "h"
        self.orient = orient
        self.cat_var, self.val_var = (x, y) if orient == "v" else (y, x)
        cats = _column(data, self.cat_var)
        vals = _column(data, self.val_var)
        self.hue_var = hue
        hues = _column(data, hue) if hue is not None else None
        self.categories = list(order) if order is not None else categorical_order(cats)
        self.hues = (list(hue_order) if hue_order is not None else categorical_order(hues)) if hue is not None else None
        n = len(cats) if cats else len(vals)
        # rows seaborn keeps, in frame order: the value, the category and the hue all present
        self.rows: List[int] = []
        self.cat_of: Dict[int, Any] = {}
        self.hue_of: Dict[int, Any] = {}
        for i in range(n):
            c = cats[i] if cats else None
            v = vals[i] if vals else None
            h = hues[i] if hues is not None else None
            if (cats and _missing(c)) or (vals and _missing(v)) or (hues is not None and _missing(h)):
                continue
            if cats and c not in self.categories:
                continue
            if hues is not None and h not in self.hues:
                continue
            self.rows.append(i)
            self.cat_of[i] = c
            self.hue_of[i] = h

    def rows_in(self, category) -> List[int]:
        return [i for i in self.rows if self.cat_of.get(i) == category]

    @property
    def n_hues(self) -> int:
        return len(self.hues) if self.hues else 1


def _names(values) -> List[str]:
    return [str(v) for v in values]


def _untagged(artists, already):
    return [a for a in artists if id(a) not in already and a.get_visible()]


def tag_categorical(ax, plot: str, *, frame: Optional[Frame], categories, hues, reg, already, tagged, pin) -> None:
    """Tag one of seaborn's categorical plot kinds. ``categories`` / ``hues`` are the levels in
    seaborn's order (from the frame, or from the tick labels / legend when there is no frame)."""
    cats = _names(categories)
    hue_names = _names(hues) if hues else None
    if plot == "boxplot":
        _tag_boxes(ax, cats, hue_names, reg, already, tagged, pin)
    elif plot == "violinplot":
        _tag_violins(ax, cats, hue_names, reg, already, tagged, pin)
    elif plot in ("stripplot", "swarmplot"):
        _tag_strips(ax, cats, hue_names, frame, categories, reg, already, tagged, pin)
    elif plot == "pointplot":
        _tag_points(ax, cats, hue_names, reg, already, tagged, pin)


def _record(tagged, name, role):
    tagged.setdefault(str(name), []).append(role)


# ---------------------------------------------------------------------------------------------
# box plots: one BoxPlotContainer per hue level, k boxes each
# ---------------------------------------------------------------------------------------------
def _tag_boxes(ax, cats, hues, reg, already, tagged, pin):
    containers = [c for c in getattr(ax, "containers", []) if type(c).__name__ == "BoxPlotContainer"]
    containers = [c for c in containers if any(id(b) not in already for b in c.boxes)]
    if not containers:
        return
    if hues is not None and len(containers) != len(hues):
        warnings.warn(f"tag_seaborn: {len(containers)} box containers for {len(hues)} hue levels; boxes left as extras", stacklevel=3)
        return
    if hues is None and len(containers) != 1:
        warnings.warn("tag_seaborn: several box containers without hue=; boxes left as extras", stacklevel=3)
        return
    for j, cont in enumerate(containers):
        boxes = list(cont.boxes)
        if len(boxes) != len(cats):
            warnings.warn(f"tag_seaborn: {len(boxes)} boxes for {len(cats)} categories; boxes left as extras", stacklevel=3)
            return
        for i, cat in enumerate(cats):
            series = hues[j] if hues is not None else cat
            name = cat if hues is not None else None
            pin(series, boxes[i], lambda a, c: a.set_facecolor(c))
            reg.add(Mark(role="box", series=series, name=name, kind="box", label=series, artists=[boxes[i]]))
            _record(tagged, series, "box")
            parts = (("whisker", list(cont.whiskers)[2 * i:2 * i + 2]), ("cap", list(cont.caps)[2 * i:2 * i + 2]),
                     ("median", list(cont.medians)[i:i + 1]), ("flier", list(cont.fliers)[i:i + 1]),
                     ("mean", list(getattr(cont, "means", []))[i:i + 1]))
            for role, arts in parts:
                arts = [a for a in arts if a is not None]
                if not arts:
                    continue
                part_name = f"{cat}-{role}" if hues is not None else None
                reg.add(Mark(role=role, series=series, name=part_name, kind="box", artists=arts))
                _record(tagged, series, role)


# ---------------------------------------------------------------------------------------------
# violins: one body (PolyCollection) per (category, hue), category-major; inner lines follow
# ---------------------------------------------------------------------------------------------
def _tag_violins(ax, cats, hues, reg, already, tagged, pin):
    from matplotlib.collections import PolyCollection
    bodies = _untagged([c for c in ax.collections if isinstance(c, PolyCollection)], already)
    n_h = len(hues) if hues else 1
    expected = len(cats) * n_h
    if len(bodies) != expected:
        warnings.warn(f"tag_seaborn: {len(bodies)} violin bodies for {len(cats)} categories × {n_h} hue levels; left as extras", stacklevel=3)
        return
    lines = _untagged([ln for ln in ax.lines if len(ln.get_xdata())], already)
    per = len(lines) // expected if expected and len(lines) % expected == 0 else 0
    # seaborn iterates the categorical variable outermost, then hue: (a, p), (a, q), (b, p), …
    for k, body in enumerate(bodies):
        cat, hue = cats[k // n_h], (hues[k % n_h] if hues else None)
        series = hue if hues is not None else cat
        name = cat if hues is not None else None
        pin(series, body, lambda a, c: a.set_facecolor(c))
        reg.add(Mark(role="violin", series=series, name=name, kind="violin", label=series, artists=[body]))
        _record(tagged, series, "violin")
        if per:
            inner = lines[k * per:(k + 1) * per]
            reg.add(Mark(role="segment", series=series, name=f"{cat}-inner" if hues is not None else "inner",
                         kind="violin", artists=inner))
            _record(tagged, series, "segment")


# ---------------------------------------------------------------------------------------------
# strips / swarms: one PathCollection per category (hue levels mixed inside, split by rows) or
# per (category, hue) when dodged
# ---------------------------------------------------------------------------------------------
def _tag_strips(ax, cats, hues, frame, categories, reg, already, tagged, pin):
    from matplotlib.collections import PathCollection, PolyCollection
    colls = _untagged([c for c in ax.collections if isinstance(c, PathCollection) and not isinstance(c, PolyCollection)], already)
    n_h = len(hues) if hues else 1
    if len(colls) == len(cats) * n_h and n_h > 1:
        # dodged: one collection per (category, hue), category-major
        for k, coll in enumerate(colls):
            cat, hue = cats[k // n_h], hues[k % n_h]
            _point_mark(reg, coll, series=hue, name=cat, tagged=tagged, pin=pin)
        return
    if len(colls) != len(cats):
        warnings.warn(f"tag_seaborn: {len(colls)} point collections for {len(cats)} categories; left as extras", stacklevel=3)
        return
    for i, (cat, coll) in enumerate(zip(cats, colls)):
        if hues is None:
            _point_mark(reg, coll, series=cat, name=None, tagged=tagged, pin=pin)
            continue
        if frame is None:
            warnings.warn("tag_seaborn: hue levels inside one collection need data= to split by row; tagging per category", stacklevel=3)
            _point_mark(reg, coll, series=cat, name=None, tagged=tagged, pin=pin)
            continue
        rows = frame.rows_in(categories[i])
        off = coll.get_offsets()
        if len(rows) != len(off):
            warnings.warn(f"tag_seaborn: category {cat!r} draws {len(off)} points but the frame has {len(rows)} rows; tagging per category", stacklevel=3)
            _point_mark(reg, coll, series=cat, name=None, tagged=tagged, pin=pin)
            continue
        x, y = [float(v) for v in off[:, 0]], [float(v) for v in off[:, 1]]
        for hue, level in zip(hues, frame.hues):
            subset = [k for k, r in enumerate(rows) if frame.hue_of[r] == level]
            if not subset:
                continue
            reg.add(Mark(role="point", series=hue, name=cat, kind="scatter", x=x, y=y, artists=[coll], indexed=True,
                         data={"point_subset": subset}))
            _record(tagged, hue, "point")


def _point_mark(reg, coll, *, series, name, tagged, pin):
    pin(series, coll, lambda a, c: a.set_facecolor(c))
    off = coll.get_offsets()
    x, y = [float(v) for v in off[:, 0]], [float(v) for v in off[:, 1]]
    reg.add(Mark(role="point", series=series, name=name, kind="scatter", x=x, y=y, artists=[coll], indexed=True, live_data=True))
    _record(tagged, series, "point")


def split_scatter(reg, coll, frame: Frame, tagged) -> bool:
    """``scatterplot(hue=)``: one collection for every hue level. Register one point mark per
    level owning the rows of that level (``point_subset``); the collection stays one artist."""
    off = coll.get_offsets()
    if len(frame.rows) != len(off):
        warnings.warn(f"tag_seaborn: the scatter draws {len(off)} points but the frame keeps {len(frame.rows)} rows; not split by hue", stacklevel=3)
        return False
    x, y = [float(v) for v in off[:, 0]], [float(v) for v in off[:, 1]]
    for level in frame.hues:
        subset = [k for k, r in enumerate(frame.rows) if frame.hue_of[r] == level]
        if not subset:
            continue
        reg.add(Mark(role="point", series=str(level), kind="scatter", x=x, y=y, artists=[coll], indexed=True,
                     data={"point_subset": subset}))
        _record(tagged, level, "point")
    return True


# ---------------------------------------------------------------------------------------------
# point plots: per hue level a mean line (with markers) followed by its error lines
# ---------------------------------------------------------------------------------------------
def _tag_points(ax, cats, hues, reg, already, tagged, pin):
    from matplotlib.lines import Line2D
    lines = _untagged([ln for ln in ax.lines if len(ln.get_xdata())], already)
    n_h = len(hues) if hues else 1
    if not lines or len(lines) % n_h:
        warnings.warn(f"tag_seaborn: {len(lines)} lines for {n_h} hue levels in a pointplot; left as extras", stacklevel=3)
        return
    per = len(lines) // n_h
    for j in range(n_h):
        chunk = lines[j * per:(j + 1) * per]
        mean = chunk[0]
        series = hues[j] if hues else (ax.get_ylabel() or "means")
        pin(series, mean, lambda a, c: (a.set_color(c), a.set_markerfacecolor(c), a.set_markeredgecolor(c)))
        has_line = str(mean.get_linestyle()).lower() not in ("none", "", " ")
        has_marker = str(mean.get_marker()).lower() not in ("none", "", " ")
        if has_marker:
            pts = Line2D([], [])
            pts.update_from(mean)
            pts.set_zorder(mean.get_zorder())
            pts.set_data(mean.get_xdata(orig=False), mean.get_ydata(orig=False))
            pts.set_linestyle("none")
            pts.set_label("_nolegend_")
            ax.add_line(pts)
            reg.add(Mark(role="point", series=series, kind="line", live_data=True, x=None, y=None, artists=[pts], indexed=True))
            _record(tagged, series, "point")
            if has_line:
                mean.set_marker("none")
            else:
                mean.set_visible(False)
        if has_line:
            gx, gy = mean.get_data()
            reg.add(Mark(role="line", series=series, kind="line", x=[float(v) for v in gx], y=[float(v) for v in gy],
                         live_data=True, label=series, artists=[mean]))
            _record(tagged, series, "line")
        errs = chunk[1:]
        if errs:
            for ln in errs:
                pin(series, ln, lambda a, c: a.set_color(c))
            reg.add(Mark(role="errorbar", series=series, kind="errorbar", artists=errs))
            _record(tagged, series, "errorbar")


def tick_categories(ax, orient: str = "v") -> List[str]:
    """The categories seaborn wrote as tick labels (when no frame was given)."""
    axis = ax.xaxis if orient == "v" else ax.yaxis
    return [t.get_text() for t in axis.get_ticklabels() if t.get_text()]
