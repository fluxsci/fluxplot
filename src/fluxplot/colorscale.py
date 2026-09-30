"""Portable colour scales — the exact value → colour law of every colour-mapped mark.

A heatmap, a hexmatrix, a filled contour or a colour-mapped scatter paints each element with
``cmap(norm(value))``. matplotlib's rendering of that law is two tables and a handful of formulas:
a *lookup table* of ``N`` colours indexed by the normalised value, three extra colours for values
below, above and missing, and a *norm* that maps data to ``[0, 1]``. This module records both,
completely and portably (hex strings and plain numbers — no matplotlib name a consumer would have
to resolve), so anything that reads the manifest can recolour the plot exactly as matplotlib did,
change its limits or its map, and redraw its colour key, without running Python.

The record is the ``colorScales[]`` entry of the manifest::

    {"id": "rates",                       # == the recipe's colour-control key; unique per figure
     "kind": "continuous",                # continuous | binned (a BoundaryNorm) | categorical
     "colormap": {"name": "viridis", "source": "matplotlib", "N": 256, "lut": ["#440154ff", …],
                  "under": "#440154ff", "over": "#fde725ff", "bad": "#00000000", "discrete": false},
     "norm": {"kind": "log", "vmin": 1.0, "vmax": 53.0, "clip": false, "base": 10, "extend": "neither"},
     "mappables": ["rates.hexes"],        # svg ids of every group this scale colours
     "colorbars": ["colorbar.color"],     # the colour keys drawing it
     "label": "Synapses per hexbin",
     "recolor": "live",                   # live | raster | regenerate
     "editable": {"cmap": true, "limits": true, "normKinds": ["linear", "log", "power", "symlog"],
                  "center": false}}

**The lookup rule** (``Colormap.__call__``): a normalised ``x`` maps to ``lut[trunc(x * N)]``;
``x == 1`` maps to the last entry; ``x < 0`` to ``under``, ``x * N >= N`` (``x > 1``) to ``over``,
NaN or masked to ``bad``. A ``boundary`` norm yields an *index* instead of a fraction: it is used
directly, ``-1`` meaning ``under`` and ``N`` meaning ``over``. :func:`apply` is the reference
implementation of the whole law — every consumer (Flux's ``colorscale.ts``) must agree with it
hex for hex, and ``tests/fixtures/colorscale_vectors.json`` carries the vectors both sides check.

The same module owns the colour controls a recipe carries (``recipe.params.__fluxplot__[id]``):
:func:`apply_override` turns an edit — a map name or a LUT, limits, a norm kind, ``extend`` —
into the matplotlib objects a helper draws with, and :func:`controls_state` writes the complete
current state back, so an editor always starts from the real values.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.colors import to_hex

#: matplotlib norm class → the portable kind name.
NORM_KINDS = {
    "Normalize": "linear",
    "LogNorm": "log",
    "SymLogNorm": "symlog",
    "PowerNorm": "power",
    "TwoSlopeNorm": "twoslope",
    "CenteredNorm": "centered",
    "BoundaryNorm": "boundary",
    "NoNorm": "none",
}
#: The kinds a consumer may switch a continuous scale to (each needs only limits, or one number).
SWITCHABLE_KINDS = ("linear", "log", "power", "symlog")
#: Above this many entries a lookup table is resampled (and flagged ``approximate``).
LUT_MAX = 1024

_HUGE = 1e250  # matplotlib's stand-in level for a filled contour's extend bands


def _hex(rgba) -> str:
    return to_hex(rgba, keep_alpha=True)


def _num(v):
    return None if v is None else float(v)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
def colormap_source(cmap) -> str:
    """Where a colormap comes from: ``matplotlib`` | ``fluxplot`` | ``crameri`` | ``tol`` |
    ``cmasher`` | ``custom`` (an unnamed or unregistered map)."""
    import matplotlib as mpl

    from .colors import maps
    name = getattr(cmap, "name", None) or ""
    base = name[:-2] if name.endswith("_r") else name
    if base in getattr(mpl.colormaps, "_builtin_cmaps", ()):
        return "matplotlib"
    try:
        collection = maps.info(name).get("collection")
    except Exception:
        return "custom"
    if collection == "mpl":
        return "matplotlib"
    if collection == "flexoki":
        return "fluxplot" if base in maps._custom else "custom"
    return collection or "custom"


def colormap_record(cmap) -> dict:
    """The exact lookup table matplotlib uses, portable: ``{name, source, N, lut, under, over, bad,
    discrete}`` (+ ``approximate: true`` when a >1024-entry table was resampled)."""
    from .colors import DISCRETE_MAX
    approximate = False
    if cmap.N > LUT_MAX:
        cmap, approximate = cmap.resampled(LUT_MAX), True
    lut = [_hex(c) for c in cmap(np.arange(cmap.N))]  # integer indexing reads the table itself
    out = {
        "name": cmap.name,
        "source": colormap_source(cmap),
        "N": int(cmap.N),
        "lut": lut,
        "under": _hex(cmap.get_under()),
        "over": _hex(cmap.get_over()),
        "bad": _hex(cmap.get_bad()),
        "discrete": bool(isinstance(cmap, mcolors.ListedColormap) and cmap.N <= DISCRETE_MAX),
    }
    if approximate:
        out["approximate"] = True
    return out


def norm_record(norm, *, extend="neither") -> dict:
    """``{kind, vmin, vmax, clip, extend, …}`` with the kind's own parameters (``vcenter``,
    ``halfrange``, ``gamma``, ``linthresh``, ``linscale``, ``base``, ``boundaries``, ``ncolors``).
    An unknown ``Normalize`` subclass is recorded as ``custom`` with its class name."""
    kind = NORM_KINDS.get(type(norm).__name__)
    out = {"kind": kind or "custom", "vmin": _num(norm.vmin), "vmax": _num(norm.vmax),
           "clip": bool(norm.clip), "extend": extend}
    if kind is None:
        out["className"] = type(norm).__name__
    if isinstance(norm, mcolors.TwoSlopeNorm):
        out["vcenter"] = float(norm.vcenter)
    elif isinstance(norm, mcolors.CenteredNorm):
        out["vcenter"] = float(norm.vcenter)
        out["halfrange"] = _num(norm.halfrange)
    elif isinstance(norm, mcolors.PowerNorm):
        out["gamma"] = float(norm.gamma)
    elif isinstance(norm, mcolors.SymLogNorm):
        scale = norm._scale
        out["linthresh"] = float(scale.linthresh)
        out["linscale"] = float(scale.linscale)
        out["base"] = float(scale.base)
    elif isinstance(norm, mcolors.LogNorm):
        out["base"] = float(getattr(norm._scale, "base", 10.0))
    elif isinstance(norm, mcolors.BoundaryNorm):
        out["boundaries"] = [float(b) for b in norm.boundaries]
        out["ncolors"] = int(norm.Ncmap)
        out["extend"] = norm.extend
    return out


def scale_record(scale_id, mappable, *, label=None, extend=None, kind=None, recolor="live") -> dict:
    """One manifest ``colorScales[]`` entry for a matplotlib ``ScalarMappable``. ``mappables`` and
    ``colorbars`` are filled in when the manifest is assembled (they are svg ids)."""
    norm = mappable.norm
    if extend is None:
        cb = getattr(mappable, "colorbar", None)
        extend = getattr(cb, "extend", None) or getattr(mappable, "extend", None) or "neither"
    nr = norm_record(norm, extend=extend)
    if kind is None:
        kind = "binned" if nr["kind"] == "boundary" else "continuous"
    switchable = kind == "continuous" and nr["kind"] in SWITCHABLE_KINDS + ("twoslope", "centered")
    kinds = list(SWITCHABLE_KINDS) if switchable else []
    if switchable and nr["kind"] not in kinds:
        kinds.append(nr["kind"])
    return {
        "id": str(scale_id),
        "kind": kind,
        "colormap": colormap_record(mappable.get_cmap()),
        "norm": nr,
        "mappables": [],
        "colorbars": [],
        "label": label,
        "recolor": recolor,
        "editable": {
            "cmap": True,
            "limits": nr["kind"] not in ("boundary", "custom", "none"),
            "normKinds": kinds,
            "center": nr["kind"] in ("twoslope", "centered"),
        },
    }


# ---------------------------------------------------------------------------
# the reference implementation of the law
# ---------------------------------------------------------------------------
def _log(values, base):
    if base == 10:
        return np.log10(values)
    if base == 2:
        return np.log2(values)
    if base == np.e:
        return np.log(values)
    return np.log(values) / np.log(base)


def normalize(norm: dict, values) -> np.ma.MaskedArray:
    """Port of each matplotlib norm's ``__call__`` for a ``norm`` record: floats in (about)
    ``[0, 1]`` for the continuous kinds, colour *indices* for ``boundary`` / ``none``. Masked
    entries are the ``bad`` values (missing data, non-positive on a log scale)."""
    v = np.ma.masked_invalid(np.ma.asarray(values, dtype=float))
    kind, vmin, vmax = norm["kind"], norm.get("vmin"), norm.get("vmax")
    clip = bool(norm.get("clip"))
    if kind in ("linear", "centered", "power", "log", "symlog"):
        if vmin is None or vmax is None:
            raise ValueError(f"colour scale: a {kind} norm needs vmin and vmax")
        if vmin > vmax:
            raise ValueError("colour scale: vmin must not exceed vmax")
        if vmin == vmax:
            return np.ma.zeros(v.shape) + np.ma.masked_array(np.zeros(v.shape), mask=np.ma.getmaskarray(v))
        if clip:
            v = np.ma.clip(v, vmin, vmax)
        if kind in ("linear", "centered"):
            return (v - vmin) / (vmax - vmin)
        if kind == "power":
            r = (v - vmin) / (vmax - vmin)
            pos = np.ma.filled(r > 0, False)
            r[pos] = np.power(r[pos], norm["gamma"])
            return r
        if kind == "log":
            base = norm.get("base", 10.0)
            t = np.ma.masked_invalid(_log(np.ma.filled(np.ma.masked_less_equal(v, 0), np.nan), base))
            t_lo, t_hi = _log(np.array([vmin, vmax], dtype=float), base)
            return (t - t_lo) / (t_hi - t_lo)
        if kind == "symlog":
            base, linthresh, linscale = norm.get("base", 10.0), norm["linthresh"], norm.get("linscale", 1.0)
            adj = linscale / (1.0 - 1.0 / base)

            def trf(a):
                a = np.asarray(a, dtype=float)
                abs_a = np.abs(a)
                with np.errstate(divide="ignore", invalid="ignore"):
                    out = np.sign(a) * linthresh * (adj - np.log(linthresh) / np.log(base) + np.log(abs_a) / np.log(base))
                inside = abs_a <= linthresh
                out[inside] = a[inside] * adj
                return out
            t = np.ma.masked_array(trf(np.ma.filled(v, 0.0)), mask=np.ma.getmaskarray(v))
            t_lo, t_hi = trf(np.array([vmin, vmax]))
            return (t - t_lo) / (t_hi - t_lo)
    if kind == "twoslope":
        vcenter = norm["vcenter"]
        if not vmin <= vcenter <= vmax:
            raise ValueError("colour scale: vmin, vcenter, vmax must increase monotonically")
        return np.ma.masked_array(np.interp(np.ma.filled(v, vmin), [vmin, vcenter, vmax], [0, 0.5, 1],
                                            left=-np.inf, right=np.inf), mask=np.ma.getmaskarray(v))
    if kind == "boundary":
        b = np.asarray(norm["boundaries"], dtype=float)
        ncolors, extend = int(norm["ncolors"]), norm.get("extend", "neither")
        lo, hi = float(b[0]), float(b[-1])
        offset = 1 if extend in ("min", "both") else 0
        n_regions = len(b) - 1 + offset + (1 if extend in ("max", "both") else 0)
        xx = np.ma.filled(v, hi + 1)
        if clip:
            xx = np.clip(xx, lo, hi)
            max_col = ncolors - 1
        else:
            max_col = ncolors
        iret = np.digitize(xx, b) - 1 + offset
        if ncolors > n_regions:
            if n_regions == 1:
                iret[iret == 0] = (ncolors - 1) // 2
            else:
                iret = (ncolors - 1) / (n_regions - 1) * iret
        iret = iret.astype(np.int16)
        iret[xx < lo] = -1
        iret[xx >= hi] = max_col
        return np.ma.masked_array(iret, mask=np.ma.getmaskarray(v))
    if kind == "none":
        return v
    raise ValueError(f"colour scale: cannot evaluate a {kind!r} norm")


def lookup(colormap: dict, x) -> list:
    """Colour of each normalised value (or index) per the lookup rule, as ``#rrggbbaa``."""
    lut, n = colormap["lut"], int(colormap["N"])
    xa = np.ma.asarray(x)
    # matplotlib masks NaN (and masked input) as bad; an infinity is merely far below / above
    bad = np.ma.getmaskarray(xa) | np.isnan(np.ma.filled(xa.astype(float), 0.0))
    data = np.ma.filled(xa, 0.0)
    if np.asarray(data).dtype.kind == "f":
        scaled = np.asarray(data, dtype=float) * n
        scaled = np.where(scaled == n, n - 1, scaled)
        under = scaled < 0
        over = scaled >= n
        with np.errstate(invalid="ignore"):
            idx = np.trunc(np.where(np.isfinite(scaled), scaled, 0)).astype(int)
    else:
        idx = np.asarray(data, dtype=int)
        under = idx < 0
        over = idx >= n
    idx = np.clip(idx, 0, n - 1)
    out = []
    for i, u, o, b in zip(idx.ravel(), under.ravel(), over.ravel(), bad.ravel()):
        out.append(colormap["bad"] if b else colormap["under"] if u else colormap["over"] if o else lut[int(i)])
    return out


def apply(record: dict, values) -> list:
    """Reference implementation: the colour (``#rrggbbaa``) matplotlib paints each value with."""
    return lookup(record["colormap"], normalize(record["norm"], values))


# ---------------------------------------------------------------------------
# the recipe's colour controls
# ---------------------------------------------------------------------------
def _lut_name(lut) -> str:
    return "custom:" + hashlib.sha1(json.dumps(list(lut)).encode()).hexdigest()[:10]


def colormap_from_spec(spec, key, resolve):
    """A ``Colormap`` from a control's ``cmap``: a name (through ``resolve``), a ``Colormap``, or a
    ``{"lut": [...], "under"?, "over"?, "bad"?}`` table."""
    if isinstance(spec, mcolors.Colormap):
        return spec
    if isinstance(spec, dict):
        lut = spec.get("lut")
        if not lut:
            raise ValueError(f"colour control {key!r}: a colormap table needs a non-empty 'lut'")
        cm = mcolors.ListedColormap(list(lut), name=spec.get("name") or _lut_name(lut))
        if spec.get("under"):
            cm.set_under(spec["under"])
        if spec.get("over"):
            cm.set_over(spec["over"])
        if spec.get("bad"):
            cm.set_bad(spec["bad"])
        return cm
    if isinstance(spec, str):
        try:
            return resolve(spec)
        except (ValueError, KeyError):
            raise ValueError(
                f"colour control {key!r}: unknown colormap {spec!r}; use a matplotlib name, "
                "a fluxplot map (fp.colors.maps.collections()), a registered custom map or a LUT"
            ) from None
    raise ValueError(f"colour control {key!r}: cmap must be a name or a {{'lut': [...]}} table")


def _same_colormap(a, b) -> bool:
    if a is None or b is None:
        return False
    if a is b:
        return True
    if a.N != b.N:
        return False
    return (np.array_equal(a(np.arange(a.N)), b(np.arange(b.N)))
            and np.array_equal(a.get_under(), b.get_under()) and np.array_equal(a.get_over(), b.get_over())
            and np.array_equal(a.get_bad(), b.get_bad()))


def _as_colormap(spec, resolve):
    if spec is None or isinstance(spec, mcolors.Colormap):
        return spec
    if isinstance(spec, str):
        try:
            return resolve(spec)
        except (ValueError, KeyError):
            return None
    return None


def make_norm(spec: dict, vmin, vmax, key) -> mcolors.Normalize:
    """A matplotlib norm from a control's ``norm`` record and the carried limits, validated."""
    kind = spec.get("kind", "linear")
    if kind == "log":
        if vmin is not None and vmin <= 0:
            raise ValueError(f"colour control {key!r}: a log norm needs vmin > 0 (got {vmin!r})")
        return mcolors.LogNorm(vmin=vmin, vmax=vmax)
    if kind == "symlog":
        linthresh = spec.get("linthresh", 1.0)
        if linthresh <= 0:
            raise ValueError(f"colour control {key!r}: symlog linthresh must be positive")
        return mcolors.SymLogNorm(linthresh, linscale=spec.get("linscale", 1.0), vmin=vmin, vmax=vmax,
                                  base=spec.get("base", 10))
    if kind == "power":
        return mcolors.PowerNorm(spec.get("gamma", 1.0), vmin=vmin, vmax=vmax)
    if kind == "twoslope":
        vcenter = spec.get("vcenter", 0.0)
        if vmin is not None and not vmin < vcenter:
            raise ValueError(f"colour control {key!r}: twoslope needs vmin < vcenter (vmin={vmin!r}, vcenter={vcenter!r})")
        if vmax is not None and not vcenter < vmax:
            raise ValueError(f"colour control {key!r}: twoslope needs vcenter < vmax (vcenter={vcenter!r}, vmax={vmax!r})")
        return mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    if kind == "centered":
        halfrange = spec.get("halfrange")
        if halfrange is None and vmin is not None and vmax is not None:
            halfrange = max(abs(spec.get("vcenter", 0.0) - vmin), abs(vmax - spec.get("vcenter", 0.0)))
        return mcolors.CenteredNorm(vcenter=spec.get("vcenter", 0.0), halfrange=halfrange)
    if kind == "linear":
        return mcolors.Normalize(vmin=vmin, vmax=vmax)
    raise ValueError(f"colour control {key!r}: cannot build a {kind!r} norm; "
                     f"use one of {', '.join(SWITCHABLE_KINDS + ('twoslope', 'centered'))}")


def _norm_kind_of(norm) -> str | None:
    if isinstance(norm, str):
        return {"linear": "linear", "log": "log", "symlog": "symlog", "logit": None}.get(norm)
    if isinstance(norm, mcolors.Normalize):
        return NORM_KINDS.get(type(norm).__name__)
    return "linear" if norm is None else None


def _norm_params_equal(spec: dict, norm) -> bool:
    """True when the record's parameters match the script's norm object (limits aside)."""
    if not isinstance(norm, mcolors.Normalize):
        return spec.get("kind", "linear") == _norm_kind_of(norm)
    rec = norm_record(norm)
    if rec["kind"] != spec.get("kind", "linear"):
        return False
    for k in ("vcenter", "gamma", "linthresh", "linscale", "base"):
        if k in spec and not np.isclose(float(spec[k]), float(rec.get(k, np.nan))):
            return False
    return True


def apply_override(kwargs: dict, override: dict, key: str, *, resolve) -> None:
    """Apply one recipe colour control (v1 ``{cmap, vmin, vmax}`` or v2, see the module doc) to a
    helper's colour keywords, in place, keeping the script's own objects wherever the control
    merely restates them.

    ``kwargs['cmap']`` becomes a ``Colormap`` when it changes; ``kwargs['norm']`` a norm object
    when the kind or a parameter changes; ``vmin`` / ``vmax`` are set outright (the caller moves
    them onto the norm); ``kwargs['_extend']`` carries ``extend`` for the colour key.
    """
    wanted = override.get("cmap")
    if wanted is not None:
        script = _as_colormap(kwargs.get("cmap"), resolve)
        target = colormap_from_spec(wanted, key, resolve)
        if not (_same_colormap(script, target) or (isinstance(wanted, str) and isinstance(kwargs.get("cmap"), str)
                                                   and wanted == kwargs["cmap"])):
            kwargs["cmap"] = target
    if override.get("reversed"):
        cm = _as_colormap(kwargs.get("cmap"), resolve)
        if cm is None:
            from ._fieldmap import resolve_colormap
            cm = resolve_colormap(None)
        kwargs["cmap"] = cm.reversed()
    for option in ("vmin", "vmax"):
        if option in override:
            kwargs[option] = override[option]
    spec = override.get("norm")
    if isinstance(spec, dict) and spec:
        current = kwargs.get("norm")
        if not _norm_params_equal(spec, current):
            vmin = kwargs.pop("vmin", None) if "vmin" in kwargs else getattr(current, "vmin", None)
            vmax = kwargs.pop("vmax", None) if "vmax" in kwargs else getattr(current, "vmax", None)
            kwargs["norm"] = make_norm(spec, vmin, vmax, key)
        elif isinstance(current, mcolors.Normalize) and spec.get("kind") == "log":
            vmin = kwargs.get("vmin", current.vmin)
            if vmin is not None and vmin <= 0:
                raise ValueError(f"colour control {key!r}: a log norm needs vmin > 0 (got {vmin!r})")
    if "extend" in override:
        if override["extend"] not in ("neither", "min", "max", "both"):
            raise ValueError(f"colour control {key!r}: extend must be neither, min, max or both")
        kwargs["_extend"] = override["extend"]


def cmap_spec(cmap, resolve) -> object:
    """How a recipe names this map: its name when ``resolve`` rebuilds the identical map from it,
    else the portable ``{"lut", "under", "over", "bad"}`` table."""
    rebuilt = _as_colormap(getattr(cmap, "name", None), resolve)
    if rebuilt is not None and _same_colormap(rebuilt, cmap):
        return cmap.name
    rec = colormap_record(cmap)
    return {k: rec[k] for k in ("lut", "under", "over", "bad")}


def controls_state(field: dict) -> dict:
    """The complete v2 colour control for a captured field: what an editor starts from and what a
    rerun replays."""
    nr = field["normalization"]  # its kind is the matplotlib class name (kept as an alias)
    norm = {"kind": NORM_KINDS.get(nr["kind"], "custom")}
    for k in ("vcenter", "gamma", "linthresh", "linscale"):
        if nr.get(k) is not None:
            norm[k] = nr[k]
    out = {"cmap": field.get("cmapSpec", field["cmap"]), "vmin": nr.get("vmin"), "vmax": nr.get("vmax"),
           "norm": norm}
    if field.get("extend"):
        out["extend"] = field["extend"]
    return out
