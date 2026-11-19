"""Tables for layers (SPEC 6): equal-length columns of numbers, strings, booleans or nulls.

Accepts a polars or pandas DataFrame, a dict of columns, or a list of row dicts, without importing
either DataFrame library.
"""
from __future__ import annotations

import math
import numbers

import numpy as np

__all__ = ["columns_of", "json_value", "json_safe", "is_integer_column", "key_text"]


def json_value(v):
    """One table cell as a JSON value: numbers, strings, booleans or None (NaN and infinities
    become None; numpy scalars become Python ones; anything else becomes its text)."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, str):
        return v
    if isinstance(v, np.generic):
        return json_value(v.item())
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", "replace")
    # pandas.NA / NaT / polars nulls compare oddly; treat anything that says it is missing as null
    try:
        if v != v:  # noqa: PLR0124  (NaN-like)
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def json_safe(obj):
    """A JSON-safe deep copy of ``meta``-like data (dicts, lists, numpy scalars and arrays)."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    return json_value(obj)


def _column_values(col):
    for attr in ("to_list", "tolist"):
        fn = getattr(col, attr, None)
        if callable(fn):
            return list(fn())
    return list(col)


def columns_of(table, what="table") -> dict:
    """``{column name: [JSON values]}`` from a DataFrame (polars / pandas), a dict of columns or a
    list of row dicts; raises when column lengths differ."""
    if table is None:
        return {}
    if isinstance(table, dict):
        cols = {str(k): _column_values(v) if not isinstance(v, (str, bytes)) else [v] for k, v in table.items()}
    elif isinstance(table, (list, tuple)) and all(isinstance(r, dict) for r in table):
        names = list(dict.fromkeys(str(k) for r in table for k in r))
        cols = {n: [r.get(n) for r in table] for n in names}
    elif hasattr(table, "columns") and hasattr(table, "__getitem__"):
        cols = {str(c): _column_values(table[c]) for c in list(table.columns)}
    else:
        raise TypeError(f"{what} must be a polars / pandas DataFrame, a dict of columns or a list of row dicts, "
                        f"not {type(table).__name__}")
    lengths = {len(v) for v in cols.values()}
    if len(lengths) > 1:
        raise ValueError(f"{what}: columns must have equal lengths, got {sorted(lengths)}")
    return {k: [json_value(x) for x in v] for k, v in cols.items()}


def is_integer_column(values) -> bool:
    """Every cell is an integer (bools excluded; floats with integral values count)."""
    if not values:
        return False
    for v in values:
        if isinstance(v, bool) or v is None:
            return False
        if isinstance(v, int):
            continue
        if isinstance(v, float) and v.is_integer():
            continue
        return False
    return True


def is_string_column(values) -> bool:
    return bool(values) and all(isinstance(v, str) for v in values)


def key_text(v) -> str:
    """The ``data-key`` text of a table value, as a JSON reader would print it: ``3`` for 3 and
    3.0, ``true`` for True."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e21:
        return str(int(v))
    if isinstance(v, float):
        return repr(v)
    return str(v)
