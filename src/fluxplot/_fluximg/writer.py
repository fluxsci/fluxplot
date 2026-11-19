"""Write fluximg/1 bundles: :func:`fluximg` (an image you add layers to, then ``save``) and :func:`imsave`.

A bundle is a directory ``<stem>.fluximg/`` holding the pixels as a sharded Zarr v3 array, its
layers, and ``fluximg.json`` (written last). The pixels are streamed one ``(y, x)`` plane at a time
in two passes (learn the image, then store it), so a multi-gigabyte stack never has a second
full copy in memory.
"""
from __future__ import annotations

import concurrent.futures
import fnmatch
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from . import _encoding as enc
from . import _zarr
from . import _zstd
from ._outlines import label_outlines
from ._png import encode_png
from ._tables import columns_of, is_integer_column, is_string_column, json_safe, key_text

__all__ = ["fluximg", "imsave", "FluxImage", "SaveReport", "canonical_dims", "SPEC", "SUFFIX", "MANIFEST"]

SPEC = "fluximg/1"
SUFFIX = ".fluximg"
MANIFEST = "fluximg.json"
CORE_DIMS = ("t", "c", "z", "y", "x")
#: a plane larger than this on either side is stored as tiles and gets overview levels
TILE_THRESHOLD = 1024
TILE_SIZE = 512
#: overview levels stop once the longer side is at most this
OVERVIEW_FLOOR = 512
FIRST_OVERVIEW = 4
MAX_SHARD_BYTES = 64 * 1024 * 1024
DEFAULT_ZSTD_LEVEL = 5
PREVIEW_SIDE = 512
REGION_PLANES = 16
MAX_AUTO_REGIONS = 64
#: channel colours of a multi-channel image without ``channels`` colours (the reference writer's)
DEFAULT_COLOURS = ("#00ff00", "#ff00ff", "#00ffff", "#ffff00", "#ff8000", "#0080ff")
RGB_COLOURS = ("#ff0000", "#00ff00", "#0000ff")
LABEL_COLOUR = "#ffd400"
INTENSITY_COLOUR = "#ff00ff"
VECTOR_COLOUR = "#00e5ff"
ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_STORED_INTS = ("uint8", "int8", "uint16", "int16", "uint32", "int32")
_SPATIAL = ("y", "x", "z")


def canonical_dims(dims) -> list:
    """SPEC 3.1: extra dimensions first (in the order given), then ``t, c, z, y, x``."""
    dims = list(dims)
    return [d for d in dims if d not in CORE_DIMS] + [d for d in CORE_DIMS if d in dims]


def _bad(message):
    raise ValueError(f"fluximg: {message}")


def _finite(v, what):
    try:
        f = float(v)
    except (TypeError, ValueError):
        _bad(f"{what} must be a number, got {v!r}")
    if not math.isfinite(f):
        _bad(f"{what} must be finite, got {v!r}")
    return f


def _is_dataarray(obj) -> bool:
    return not isinstance(obj, np.ndarray) and all(hasattr(obj, a) for a in ("dims", "values", "coords"))


def _slug(text) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    if not s or not s[0].isalpha():
        s = "r_" + s if s else "region"
    return s


# ---------------------------------------------------------------------------------------------
# colours and LUTs
# ---------------------------------------------------------------------------------------------
def _hex(colour, what):
    from matplotlib import colors as mcolors
    if not mcolors.is_color_like(colour):
        _bad(f"{what}: {colour!r} is not a colour")
    return mcolors.to_hex(colour, keep_alpha=False).lower()


def _colour_or_lut(spec, what):
    """``(color, lut)`` from a LUT spec: a colour name / hex is a colour (the black-to-colour ramp),
    anything else a colour-map name, resolved like ``fp.image`` does (``images.resolve_lut``)."""
    from ..images import _MONO_PREFIX, resolve_lut
    if spec is None:
        return None, None
    try:
        cmap = resolve_lut(spec)
    except (ValueError, TypeError, KeyError):
        _bad(f"{what}: {spec!r} is neither a colour nor a colour-map name")
    name = str(cmap.name)
    if name.startswith(_MONO_PREFIX):
        return name[len(_MONO_PREFIX):], None
    return None, name


def _range(value, what):
    if value is None:
        return None
    try:
        lo, hi = value
    except (TypeError, ValueError):
        _bad(f"{what} must be (black, white), got {value!r}")
    lo, hi = _finite(lo, what), _finite(hi, what)
    if not hi > lo:
        _bad(f"{what} must have white > black, got {value!r}")
    return [lo, hi]


# ---------------------------------------------------------------------------------------------
# input arrays
# ---------------------------------------------------------------------------------------------
def _unpack(data, what="data"):
    """``(array, dims or None, coords {dim: (values, units)}, attrs units)`` from a numpy array or
    anything with ``.dims`` / ``.values`` / ``.coords`` (an xarray DataArray)."""
    if _is_dataarray(data):
        arr = np.asarray(data.values)
        dims = tuple(str(d) for d in data.dims)
        coords = {}
        for d in data.dims:
            if d in data.coords:
                c = data.coords[d]
                units = (getattr(c, "attrs", None) or {}).get("units")
                coords[str(d)] = (np.asarray(c.values), None if units is None else str(units))
        units = (getattr(data, "attrs", None) or {}).get("units")
        return arr, dims, coords, None if units is None else str(units)
    if isinstance(data, np.ma.MaskedArray):
        return data, None, {}, None
    arr = np.asarray(data)
    if arr.dtype == object:
        _bad(f"{what} must be a numeric array, not dtype object")
    return arr, None, {}, None


def _work_dtype(dtype, what):
    """The dtype a plane is handed to the encoder in (before any integer narrowing)."""
    dt = np.dtype(dtype)
    if dt.kind == "b":
        return np.dtype(np.uint8)
    if dt.kind == "f":
        return np.dtype(np.float32) if dt.itemsize <= 4 else np.dtype(np.float64)
    if dt.kind in "iu":
        return dt.newbyteorder("=")
    _bad(f"{what} must be integers, booleans or floats, not {dt}")


def _narrow_int(lo, hi, what):
    """The narrowest stored integer type holding ``lo..hi``."""
    for name in _STORED_INTS:
        info = np.iinfo(name)
        if info.min <= lo and hi <= info.max:
            return np.dtype(name)
    _bad(f"{what}: integer values {lo}..{hi} do not fit a 32-bit integer; convert to float64 first")


def _axis_from_values(values, unit, d, starts=None):
    """An axis entry from coordinate values: evenly spaced → ``scale`` / ``offset``; strings →
    ``labels``; otherwise explicit ``coords``. datetime64 / timedelta64 become seconds."""
    v = np.asarray(values)
    if v.ndim != 1:
        _bad(f"coords for {d!r} must be one-dimensional")
    if v.dtype.kind in "USO" and not (v.dtype.kind == "O" and all(isinstance(x, (int, float)) for x in v.tolist())):
        return {"unit": None, "scale": 1.0, "offset": 0.0, "labels": [str(x) for x in v.tolist()]}
    if v.dtype.kind == "M":
        if v.size and starts is not None:
            starts[d] = str(np.datetime_as_string(v[0]))   # the absolute start, kept in provenance.meta
        v = (v - v[0]) / np.timedelta64(1, "s") if v.size else v.astype(float)
        unit = unit or "s"
    elif v.dtype.kind == "m":
        v = v / np.timedelta64(1, "s")
        unit = unit or "s"
    v = v.astype(np.float64)
    if not np.isfinite(v).all():
        _bad(f"coords for {d!r} must be finite")
    n = v.size
    if n >= 2:
        step = (v[-1] - v[0]) / (n - 1)
        if step != 0 and np.all(np.abs(np.diff(v) - step) <= 1e-9 * abs(step)):
            return {"unit": unit, "scale": float(step), "offset": float(v[0])}
    if d in ("x", "y"):
        if n == 1:
            return {"unit": unit, "scale": 1.0, "offset": float(v[0])}
        _bad(f"axis {d!r} must be evenly spaced (x and y are pixel grids)")
    return {"unit": unit, "coords": [float(x) for x in v]}


class _Plane:
    """Hands out ``(y, x)`` planes of an array already viewed in canonical order."""

    def __init__(self, arr, work_dtype, nodata_float):
        self.arr = arr
        self.work = work_dtype
        self.nodata_float = nodata_float

    def __call__(self, lead):
        v = self.arr[lead] if lead else self.arr
        if isinstance(v, np.ma.MaskedArray):
            v = v.filled(np.nan if self.work.kind == "f" else 0)
        if self.nodata_float is not None:
            p = np.array(v, dtype=self.work, copy=True, order="C")
            p[p == self.nodata_float] = np.nan
            return p
        return np.ascontiguousarray(v, dtype=self.work)


# ---------------------------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------------------------
def _size(n):
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000.0


def _wrap(head, text, width=100):
    import textwrap
    return textwrap.wrap(text, width=width, initial_indent=head, subsequent_indent=" " * len(head),
                         break_on_hyphens=False) or [head.rstrip()]


@dataclass
class SaveReport:
    """What :meth:`FluxImage.save` / :func:`imsave` wrote. ``str(report)`` is the printed summary."""

    #: the bundle directory (``<stem>.fluximg``)
    path: str
    name: str
    dims: List[str]
    shape: List[int]
    #: the dtype the pixels were given in, and the dtype stored on disk
    dtype: str
    stored_dtype: str
    #: ``"exact"`` or ``"linear16"``
    encoding: str
    #: why the pixels were stored that way, in words
    reason: str
    #: bytes of the bundle on disk, and of the pixels uncompressed in their input dtype
    bytes: int
    raw_bytes: int
    #: ``[(id, kind), …]`` in manifest order
    layers: List[tuple]
    #: region names
    regions: List[str]
    #: the manifest as written
    manifest: Dict[str, Any] = field(repr=False, default_factory=dict)
    #: the recipe written beside the bundle, if any
    recipe: Optional[str] = None
    #: per-layer notes (encodings of intensity layers, borrowed layers)
    notes: List[str] = field(default_factory=list)
    seconds: float = 0.0
    #: True when ``FLUXPLOT_ONLY`` filtered this image out: nothing was written
    skipped: bool = False

    def __str__(self):
        if self.skipped:
            return f"fluximg: skipped {self.path} ({self.reason})"
        shape = " × ".join(f"{d}={n}" for d, n in zip(self.dims, self.shape))
        stored = self.encoding if self.stored_dtype == self.dtype else f"{self.encoding} ({self.stored_dtype})"
        ratio = f", {100 * self.bytes / self.raw_bytes:.0f}% of" if self.raw_bytes else " vs"
        lines = [f"fluximg  {self.path}",
                 f"  pixels   {shape}, {self.dtype} → stored {stored}",
                 *_wrap("  why      ", self.reason),
                 f"  size     {_size(self.bytes)} on disk{ratio} the {_size(self.raw_bytes)} of raw pixels"]
        if self.layers:
            lines += _wrap("  layers   ", ", ".join(f"{i} ({k})" for i, k in self.layers))
        for note in self.notes:
            lines += _wrap("           ", note)
        if self.regions:
            lines.append("  regions  " + ", ".join(self.regions))
        if self.recipe:
            lines.append(f"  recipe   {self.recipe}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# the image
# ---------------------------------------------------------------------------------------------
class FluxImage:
    """An N-dimensional image waiting to be saved as a fluximg bundle; build it with :func:`fluximg`.

    Add layers on the same ``(y, x)`` pixel grid with :meth:`add_image`, :meth:`add_labels` and
    :meth:`add_shapes`, name rectangles with :meth:`add_region`, then :meth:`save`.
    """

    def __init__(self, data, dims=None, *, pixel_size=None, unit="µm", channels=None, coords=None, dt=None,
                 t0=None, nodata=None, transform=None, meta=None, name=None, lossless=False, photometric=None,
                 default_index=None):
        arr, da_dims, da_coords, da_units = _unpack(data)
        if dims is None:
            dims = da_dims
        if dims is None:
            if arr.ndim == 2:
                dims = ("y", "x")
            elif photometric == "rgb" and arr.ndim == 3 and 3 in (arr.shape[0], arr.shape[-1]):
                dims = ("y", "x", "c") if arr.shape[-1] == 3 else ("c", "y", "x")
            else:
                _bad(f"data has {arr.ndim} dimensions (shape {arr.shape}); say what they are with dims=, "
                     "e.g. dims=('c', 'y', 'x') or ('t', 'c', 'y', 'x')")
        if isinstance(dims, str):
            dims = (dims,)
        dims = [str(d) for d in dims]
        if len(dims) != arr.ndim:
            _bad(f"dims={tuple(dims)} names {len(dims)} dimensions but data has {arr.ndim} (shape {arr.shape})")
        if len(set(dims)) != len(dims):
            _bad(f"dims={tuple(dims)} repeats a name")
        for d in dims:
            if not ID_RE.match(d):
                _bad(f"dimension name {d!r} must be lower-case letters, digits and _ (starting with a letter)")
        if "y" not in dims or "x" not in dims:
            _bad(f"dims={tuple(dims)} must include 'y' and 'x'")
        if any(s == 0 for s in arr.shape):
            _bad(f"data is empty (shape {arr.shape})")
        if photometric not in (None, "rgb"):
            _bad(f"photometric must be None or 'rgb', got {photometric!r}")
        if da_dims is not None and len(da_dims) == len(dims):
            # explicit dims rename a DataArray's dimensions position by position
            da_coords = {dims[da_dims.index(k)]: v for k, v in da_coords.items() if k in da_dims}

        self.dims = canonical_dims(dims)
        self._array = arr.transpose([dims.index(d) for d in self.dims]) if dims != self.dims else arr
        self.shape = [int(s) for s in self._array.shape]
        self.input_dtype = np.dtype(arr.dtype)
        self._work = _work_dtype(self.input_dtype, "data")
        self.H, self.W = self.shape[-2], self.shape[-1]
        self.C = self.shape[self.dims.index("c")] if "c" in self.dims else 1
        self.lossless = bool(lossless)
        self.photometric = photometric
        if photometric == "rgb" and self.C != 3:
            _bad(f"photometric='rgb' needs exactly three channels, got {self.C}")
        self.name = None if name is None else str(name)

        # ---- no data -------------------------------------------------------------------------
        self.nodata = None
        self._nodata_float = None
        if nodata is not None:
            if self._work.kind == "f":
                f = float(nodata)
                self._nodata_float = None if math.isnan(f) else f
            else:
                f = float(nodata)
                if not f.is_integer():
                    _bad(f"nodata={nodata!r} must be an integer for {self.input_dtype} data")
                self.nodata = int(f)
        if isinstance(self._array, np.ma.MaskedArray) and self._work.kind != "f":
            if self.nodata is None:
                _bad("a masked integer array needs nodata= (the stored value that means 'no data')")
            self._array = self._array.filled(self.nodata)

        # ---- axes ----------------------------------------------------------------------------
        coords = dict(coords or {})
        for d in coords:
            if d not in self.dims or d == "c":
                _bad(f"coords names {d!r}, which is not a dimension with positions ({', '.join(x for x in self.dims if x != 'c')})")
        if (dt is not None or t0 is not None) and "t" not in self.dims:
            _bad("dt= and t0= describe a 't' dimension, but dims has none")
        size = None
        if pixel_size is not None:
            ps = [pixel_size, pixel_size] if np.ndim(pixel_size) == 0 else list(pixel_size)
            if len(ps) != 2:
                _bad("pixel_size must be one number or (y, x)")
            size = [_finite(p, "pixel_size") for p in ps]
            if min(size) <= 0:
                _bad("pixel_size must be positive")
        unit = None if unit is None else str(unit)
        self._starts = {}
        self.axes = {}
        for d in self.dims:
            if d == "c":
                continue
            n = self.shape[self.dims.index(d)]
            if d in coords:
                vals = np.asarray(coords[d])
                if vals.shape != (n,):
                    _bad(f"coords[{d!r}] has {vals.size} values for {n} positions")
                axis = _axis_from_values(vals, unit if d in _SPATIAL else ("s" if d == "t" else None), d, self._starts)
            elif d in da_coords:
                vals, cunit = da_coords[d]
                if cunit is None and d in _SPATIAL:
                    cunit = da_units if da_units is not None else unit
                axis = _axis_from_values(vals, cunit, d, self._starts)
            else:
                axis = {"unit": None, "scale": 1.0, "offset": 0.0}
            if d in ("y", "x") and size is not None:
                axis = {"unit": unit, "scale": size[0] if d == "y" else size[1], "offset": axis.get("offset", 0.0)}
            if d == "t" and (dt is not None or t0 is not None):
                if "coords" in axis and dt is None:
                    _bad("t0= needs dt= when the times are not evenly spaced")
                step = _finite(dt, "dt") if dt is not None else axis.get("scale", 1.0)
                if step == 0:
                    _bad("dt must not be 0")
                axis = {"unit": "s", "scale": step, "offset": _finite(t0, "t0") if t0 is not None else 0.0}
            self.axes[d] = axis

        # ---- channels ------------------------------------------------------------------------
        if channels is None and "c" in da_coords:
            labels = da_coords["c"][0]
            channels = [str(x) for x in np.asarray(labels).tolist()]
        if channels is not None:
            if isinstance(channels, (str, dict)):
                channels = [channels]
            channels = list(channels)
            if len(channels) != self.C:
                _bad(f"channels= has {len(channels)} entries for {self.C} channel(s)")
        self._channels = []
        for i in range(self.C):
            given = channels[i] if channels is not None else None
            if given is None:
                spec = {}
            elif isinstance(given, dict):
                unknown = set(given) - {"name", "color", "colour", "lut", "range"}
                if unknown:
                    _bad(f"channel {i}: unknown keys {sorted(unknown)} (use name, color, lut, range)")
                spec = dict(given)
            else:
                spec = {"name": str(given)}
            color = spec.get("color", spec.get("colour"))
            lut = spec.get("lut")
            if color is not None and lut is not None:
                _bad(f"channel {i}: give color or lut, not both")
            if color is not None:
                color, lut = _hex(color, f"channel {i} color"), None
            elif lut is not None:
                color, lut = _colour_or_lut(lut, f"channel {i} lut")
            self._channels.append({"name": None if spec.get("name") is None else str(spec["name"]), "color": color,
                                   "lut": lut, "range": _range(spec.get("range"), f"channel {i} range")})

        # ---- world, meta ---------------------------------------------------------------------
        self.world = None
        if transform is not None:
            world = dict(transform) if isinstance(transform, dict) else {"matrix": transform, "unit": unit}
            m = np.asarray(world.get("matrix"), dtype=float)
            if m.shape != (4, 4) or not np.isfinite(m).all():
                _bad("transform must be a finite 4 x 4 matrix")
            world["matrix"] = m.tolist()
            self.world = json_safe(world)
        if meta is not None and not isinstance(meta, dict):
            _bad("meta must be a dict")
        starts = {f"{d}_start": iso for d, iso in self._starts.items()}
        self.meta = None if meta is None and not starts else json_safe({**starts, **(meta or {})})

        # ---- where the image opens ---------------------------------------------------------------
        self.default_index = None
        if default_index is not None:
            if not isinstance(default_index, dict):
                _bad("default_index must be a dict such as {'z': 10}")
            index = {}
            for d, i in default_index.items():
                if d not in self.dims[:-2] or d == "c":
                    movable = [x for x in self.dims[:-2] if x != "c"]
                    _bad(f"default_index names {d!r}; it can name {', '.join(movable) or 'no dimension of this image'}")
                n = self.shape[self.dims.index(d)]
                if isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= int(i) < n:
                    _bad(f"default_index[{d!r}] must be an integer 0..{n - 1}, got {i!r}")
                index[d] = int(i)
            self.default_index = index
        self._layers: List[dict] = []
        self._regions: List[dict] = []

    # ---- display ---------------------------------------------------------------------------
    def __repr__(self):
        shape = " × ".join(f"{d}={n}" for d, n in zip(self.dims, self.shape))
        extra = [f"{len(self._layers)} layer(s)"] if self._layers else []
        if self._regions:
            extra.append(f"{len(self._regions)} region(s)")
        return f"FluxImage({shape}, {self.input_dtype}{''.join(', ' + e for e in extra)})"

    @property
    def _physical(self):
        """True when the pixel size is known (y / x have a unit)."""
        return self.axes["x"].get("unit") is not None and self.axes["y"].get("unit") is not None

    def _plane2d(self, data, what):
        """A 2-D (y, x) array on this image's grid (a DataArray is transposed by its dims)."""
        arr, ddims, _coords, _units = _unpack(data, what)
        if ddims is not None and set(ddims) == {"y", "x"} and list(ddims) != ["y", "x"]:
            arr = arr.T
        if isinstance(arr, np.ma.MaskedArray):
            arr = arr.filled(np.nan) if arr.dtype.kind == "f" else arr.filled(0)
        if arr.ndim != 2:
            _bad(f"{what} must be a 2-D (y, x) array, got shape {arr.shape}")
        if arr.shape != (self.H, self.W):
            hint = " (it looks transposed: pass data.T)" if arr.shape == (self.W, self.H) else ""
            _bad(f"{what} has shape {arr.shape} but the image's (y, x) grid is {(self.H, self.W)}{hint}")
        return arr

    def _take_layer_id(self, id, what):
        if not isinstance(id, str) or not ID_RE.match(id):
            _bad(f"{what} id {id!r} must be lower-case letters, digits and _ (starting with a letter)")
        if any(L["id"] == id for L in self._layers):
            _bad(f"there is already a layer {id!r}")
        return id

    # ---- layers ----------------------------------------------------------------------------
    def add_image(self, id, data, *, name=None, lut=None, range=None, opacity=1.0, visible=False, lossless=False):
        """Add an intensity layer: another image on the same ``(y, x)`` grid.

        ``lut`` is a colour (the black-to-colour ramp, default magenta) or a colour-map name;
        ``range`` the display window (default the median to the 99.8th percentile); ``opacity`` a
        number 0..1 or ``"value"`` (opacity follows the normalized value). Float data is stored at
        16 bits when the guard allows it, unless ``lossless=True``. Returns the image.
        """
        self._take_layer_id(id, "layer")
        arr = self._plane2d(data, f"layer {id!r}")
        _work_dtype(arr.dtype, f"layer {id!r}")
        if opacity != "value":
            o = _finite(opacity, f"layer {id!r} opacity")
            if not 0 <= o <= 1:
                _bad(f"layer {id!r}: opacity must be 0..1 or 'value', got {opacity!r}")
            opacity = o
        color, lut_name = _colour_or_lut(lut, f"layer {id!r} lut") if lut is not None else (INTENSITY_COLOUR, None)
        self._layers.append({"kind": "intensity", "id": id, "name": str(name) if name is not None else id, "data": arr,
                             "color": color, "lut": lut_name, "range": _range(range, f"layer {id!r} range"),
                             "opacity": opacity, "visible": bool(visible), "lossless": bool(lossless)})
        return self

    def add_labels(self, id, labels, *, background=0, table=None, key=None, weights=None, name=None, color=None,
                   visible=True):
        """Add a label layer: objects as an integer map, with exact outlines and a table.

        ``background`` is the value meaning "no object" (stored as 0); every other value must be a
        positive integer, the object's label. ``table`` (a polars / pandas DataFrame or a dict of
        columns) has one row per object: its label column is ``"label"``, else the ``key`` column
        when that is integer-valued (a ``label`` copy is added). Without a table one is made with
        ``label``, ``pixels``, ``cx``, ``cy``. ``key`` names the column that names an object.
        ``weights`` (a float map) becomes the intensity layer ``<id>_weights``, stored losslessly.
        Returns the image.
        """
        self._take_layer_id(id, "layer")
        arr = self._plane2d(labels, f"labels {id!r}")
        if arr.dtype.kind == "f":
            if not np.isfinite(arr).all() or not np.array_equal(arr, np.round(arr)):
                _bad(f"labels {id!r} must be integers")
        elif arr.dtype.kind not in "iub":
            _bad(f"labels {id!r} must be integers, not {arr.dtype}")
        bg = _finite(background, "background")
        is_bg = arr == bg
        rest = arr[~is_bg]
        if rest.size and rest.min() <= 0:
            bad_value = rest[rest <= 0][0].item()
            _bad(f"labels {id!r}: every value but background={background!r} must be a positive label; found {bad_value!r}")
        top = int(rest.max()) if rest.size else 0
        if top > 0xFFFFFFFF:
            _bad(f"labels {id!r}: label {top} does not fit uint32")
        ldtype = np.uint16 if top <= 0xFFFF else np.uint32
        lab = np.where(is_bg, 0, arr).astype(ldtype)
        present = np.unique(lab[lab > 0]).astype(np.int64)

        cols = columns_of(table, f"labels {id!r} table") if table is not None else None
        if cols is None:
            if key not in (None, "label"):
                _bad(f"labels {id!r}: key={key!r} names a table column, but no table= was given")
            key = "label"
        else:
            if key is not None and key not in cols:
                _bad(f"labels {id!r}: key={key!r} is not a column of the table ({', '.join(cols)})")
            if "label" in cols:
                if not is_integer_column(cols["label"]):
                    _bad(f"labels {id!r}: the table's 'label' column must hold integers")
            elif key is not None and is_integer_column(cols[key]):
                cols["label"] = [int(v) for v in cols[key]]
            else:
                _bad(f"labels {id!r}: the table needs an integer 'label' column, or key= naming an integer column "
                     "whose values are the label values")
            cols["label"] = [int(v) for v in cols["label"]]
            key = key or "label"
            if len(set(cols["label"])) != len(cols["label"]):
                _bad(f"labels {id!r}: the table's label column repeats a label")
            rows = set(cols["label"])
            missing = [int(v) for v in present if int(v) not in rows]
            if missing:
                _bad(f"labels {id!r}: {len(missing)} label(s) in the map have no table row, e.g. {missing[:5]}")
            key_of = dict(zip(cols["label"], cols[key]))
            names = [key_of[int(v)] for v in present]
            if any(n is None for n in names):
                _bad(f"labels {id!r}: an object in the map has an empty {key!r} in the table")
            texts = [key_text(n) for n in names]
            if len(set(texts)) != len(texts):
                _bad(f"labels {id!r}: two objects share the same {key!r}; the key column must name each object once")
        wid = None
        warr = None
        if weights is not None:
            wid = self._take_layer_id(f"{id}_weights", "weights layer")
            warr = self._plane2d(weights, f"labels {id!r} weights")
            if warr.dtype.kind not in "fiub":
                _bad(f"labels {id!r}: weights must be numeric")
        colour = _hex(color, f"labels {id!r} color") if color is not None else LABEL_COLOUR
        if warr is not None:
            self._layers.append({"kind": "intensity", "id": wid, "name": f"{name or id} weights", "data": warr,
                                 "color": colour, "lut": None, "range": None, "opacity": 1.0, "visible": False,
                                 "lossless": True})
        self._layers.append({"kind": "labels", "id": id, "name": str(name) if name is not None else id, "data": lab,
                             "table": cols, "key": key, "weights": wid, "color": colour, "visible": bool(visible)})
        return self

    def add_shapes(self, id, *, lines=None, circles=None, polygons=None, coords="xy", diameter=None, radius=None,
                   table=None, key=None, name=None, color=None, visible=True):
        """Add a vector layer of lines, circles and polygons in image-pixel coordinates.

        ``lines`` / ``polygons``: a list of ``(N, 2)`` arrays, a dict key → array, or a list of
        dicts with ``"vertices"`` plus other fields (those become table columns). ``circles``: a
        DataFrame / dict of columns / list of dicts (centre columns ``center_row``/``center_col``,
        ``row``/``col``, ``cy``/``cx`` or ``y``/``x``) or an ``(N, 2)`` array of centres.
        ``coords="row_col"`` means array indices (an integer vertex is a pixel centre: +0.5, and
        swapped to x, y); ``"xy"`` SVG pixel coordinates as given. Centre columns named ``row`` /
        ``col`` are array indices whatever ``coords`` says. ``diameter`` / ``radius`` (one number,
        one per circle, or a column name) are in the image's physical unit when its pixel size is
        known, else in pixels. ``key`` names the identifying field; for a circle table it is found
        automatically when exactly one string column is not a centre column. Returns the image.
        """
        self._take_layer_id(id, "layer")
        if coords not in ("xy", "row_col"):
            _bad(f"add_shapes: coords must be 'xy' or 'row_col', got {coords!r}")
        if lines is None and circles is None and polygons is None:
            _bad("add_shapes needs lines=, circles= or polygons=")
        shapes = []      # (key value, svg tag, attributes, fields)
        explicit_key = key
        for kind, items in (("polyline", lines), ("polygon", polygons)):
            if items is not None:
                shapes.extend(self._paths(id, kind, items, coords, explicit_key))
        if circles is not None:
            shapes.extend(self._circles(id, circles, coords, diameter, radius, explicit_key))
        elif diameter is not None or radius is not None:
            _bad("add_shapes: diameter= / radius= only apply to circles=")
        texts = [key_text(s[0]) for s in shapes]
        if len(set(texts)) != len(texts):
            dup = next(t for t in texts if texts.count(t) > 1)
            _bad(f"layer {id!r}: shape key {dup!r} is used twice; keys must be unique in a layer")
        key_col = None
        cols = None
        if explicit_key is not None and table is None and not any(s[3] for s in shapes):
            _bad(f"layer {id!r}: key={explicit_key!r} names a field, but the shapes carry no fields; "
                 "pass dicts / a table, or table=")
        if table is not None:
            cols = columns_of(table, f"layer {id!r} table")
            key_col = explicit_key or next((s[4] for s in shapes if s[4]), None)
            if key_col is None or key_col not in cols:
                _bad(f"layer {id!r}: with table= give key= naming its column that holds the shape keys")
            have = {key_text(v) for v in cols[key_col]}
            missing = [t for t in texts if t not in have]
            if missing:
                _bad(f"layer {id!r}: shape key(s) {missing[:5]} have no row in the table's {key_col!r} column")
        elif any(s[3] for s in shapes):
            key_col = next((s[4] for s in shapes if s[4]), None) or "key"
            names = list(dict.fromkeys(k for s in shapes for k in s[3]))
            if key_col not in names:
                names.insert(0, key_col)
            cols = {n: [] for n in names}
            for s in shapes:
                row = dict(s[3])
                row.setdefault(key_col, s[0])
                for n in names:
                    cols[n].append(row.get(n))
            cols = columns_of(cols, f"layer {id!r} table")
        self._layers.append({"kind": "vector", "id": id, "name": str(name) if name is not None else id,
                             "shapes": [(t, s[1], s[2]) for t, s in zip(texts, shapes)], "table": cols,
                             "key": key_col, "color": _hex(color, f"layer {id!r} color") if color is not None
                             else VECTOR_COLOUR, "visible": bool(visible)})
        return self

    def _xy(self, pts, coords, what):
        a = np.asarray(pts, dtype=float)
        if a.ndim != 2 or a.shape[1] != 2:
            _bad(f"{what} must be an (N, 2) array of vertices, got shape {a.shape}")
        if not np.isfinite(a).all():
            _bad(f"{what} has non-finite vertices")
        if coords == "row_col":
            return np.column_stack([a[:, 1] + 0.5, a[:, 0] + 0.5])
        return a

    def _paths(self, lid, tag, items, coords, key):
        need = 2 if tag == "polyline" else 3
        what = "lines" if tag == "polyline" else "polygons"
        if isinstance(items, dict):
            entries = [(k, v, {}, None) for k, v in items.items()]
        else:
            entries = []
            rows = list(items)
            dict_rows = [r for r in rows if isinstance(r, dict)]
            if dict_rows and len(dict_rows) != len(rows):
                _bad(f"{what}: mix of dicts and arrays; use one form")
            kcol = key
            if dict_rows:
                for r in rows:
                    if "vertices" not in r:
                        _bad(f"{what}: every dict needs a 'vertices' entry")
                fields = [k for k in dict.fromkeys(k for r in rows for k in r) if k != "vertices"]
                if kcol is None:
                    strs = [f for f in fields if is_string_column([r.get(f) for r in rows])]
                    if len(strs) == 1:
                        kcol = strs[0]
                    elif fields:
                        _bad(f"{what}: say which field identifies a shape with key= (fields: {', '.join(fields)})")
                elif kcol not in fields:
                    _bad(f"{what}: key={kcol!r} is not a field of the dicts ({', '.join(fields)})")
                for i, r in enumerate(rows):
                    extra = {str(k): v for k, v in r.items() if k != "vertices"}
                    kval = extra.get(kcol, i) if kcol else i
                    if kval is None:
                        _bad(f"{what}: shape {i} has no {kcol!r}")
                    entries.append((kval, r["vertices"], extra, kcol))
            else:
                entries = [(i, v, {}, None) for i, v in enumerate(rows)]
        out = []
        for kval, verts, extra, kcol in entries:
            xy = self._xy(verts, coords, f"{what}[{kval!r}]")
            if len(xy) < need:
                _bad(f"{what}[{kval!r}] needs at least {need} vertices")
            pts = " ".join(f"{_num(x)},{_num(y)}" for x, y in xy)
            out.append((kval, tag, {"points": pts}, extra, kcol))
        return out

    def _circles(self, lid, circles, coords, diameter, radius, key):
        if diameter is not None and radius is not None:
            _bad("circles: give diameter= or radius=, not both")
        cols = None
        if isinstance(circles, np.ndarray) or (isinstance(circles, (list, tuple)) and circles
                                               and not isinstance(circles[0], dict)):
            centres = self._xy(circles, coords, "circles")
            keys = list(range(len(centres)))
            rows = [{} for _ in keys]
            kcol = None
        else:
            cols = columns_of(circles, "circles")
            pairs = (("center_row", "center_col", True), ("row", "col", True), ("cy", "cx", False), ("y", "x", False))
            found = next(((r, c, idx) for r, c, idx in pairs if r in cols and c in cols), None)
            if found is None:
                _bad(f"circles: no centre columns found among {', '.join(cols)} (expected center_row/center_col, "
                     "row/col, cy/cx or y/x)")
            rcol, ccol, indices = found
            yy = np.asarray([np.nan if v is None else v for v in cols[rcol]], dtype=float)
            xx = np.asarray([np.nan if v is None else v for v in cols[ccol]], dtype=float)
            if not (np.isfinite(yy).all() and np.isfinite(xx).all()):
                _bad("circles: centre coordinates must be finite")
            if indices or coords == "row_col":
                yy, xx = yy + 0.5, xx + 0.5
            centres = np.column_stack([xx, yy])
            kcol = key
            if kcol is None:
                strs = [c for c in cols if c not in (rcol, ccol) and is_string_column(cols[c])]
                if len(strs) != 1:
                    _bad("circles: say which column identifies a circle with key= "
                         f"({'string columns: ' + ', '.join(strs) if strs else 'no string column'})")
                kcol = strs[0]
            elif kcol not in cols:
                _bad(f"circles: key={kcol!r} is not a column ({', '.join(cols)})")
            keys = cols[kcol]
            if any(k is None for k in keys):
                _bad(f"circles: column {kcol!r} has empty keys")
            rows = [{c: cols[c][i] for c in cols} for i in range(len(keys))]
        n = len(centres)
        size = diameter if diameter is not None else radius
        if size is None:
            _bad("circles need diameter= or radius=")
        if isinstance(size, str):
            if cols is None or size not in cols:
                _bad(f"circles: {size!r} is not a column")
            size = cols[size]
        sizes = np.broadcast_to(np.asarray(size, dtype=float), (n,)) if np.ndim(size) == 0 else np.asarray(size, float)
        if sizes.shape != (n,) or not np.isfinite(sizes).all() or (sizes <= 0).any():
            _bad(f"circles: {'diameter' if diameter is not None else 'radius'} must be positive, one value or one per circle")
        r = sizes / 2 if diameter is not None else sizes
        if self._physical:
            ry, rx = r / self.axes["y"]["scale"], r / self.axes["x"]["scale"]
        else:
            ry = rx = r
        out = []
        for i in range(n):
            x, y = centres[i]
            if np.isclose(rx[i], ry[i], rtol=1e-12, atol=0):
                attrs = {"cx": _num(x), "cy": _num(y), "r": _num(rx[i])}
                tag = "circle"
            else:
                attrs = {"cx": _num(x), "cy": _num(y), "rx": _num(rx[i]), "ry": _num(ry[i])}
                tag = "ellipse"
            out.append((keys[i], tag, attrs, rows[i], kcol))
        return out

    def add_region(self, name, rect, *, id=None):
        """Name a rectangle worth looking at: ``rect = (x, y, width, height)`` in level-0 pixels,
        inside the image. ``id`` defaults to a slug of ``name``. Returns the image."""
        try:
            x, y, w, h = (_finite(v, "region rect") for v in rect)
        except TypeError:
            _bad(f"region rect must be (x, y, width, height), got {rect!r}")
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > self.W or y + h > self.H:
            _bad(f"region {name!r}: rect {rect!r} must lie inside the {self.W} x {self.H} image")
        if id is None:
            base = rid = _slug(name)
            k = 2
            while any(r["id"] == rid for r in self._regions):
                rid, k = f"{base}_{k}", k + 1
        else:
            rid = str(id)
            if not ID_RE.match(rid):
                _bad(f"region id {rid!r} must be lower-case letters, digits and _ (starting with a letter)")
            if any(r["id"] == rid for r in self._regions):
                _bad(f"there is already a region {rid!r}")
        self._regions.append({"id": rid, "name": str(name), "rect": {"x": _num_v(x), "y": _num_v(y),
                                                                    "width": _num_v(w), "height": _num_v(h)}})
        return self

    # ---- saving ----------------------------------------------------------------------------
    def save(self, path, *, layers=None, recipe=None, regions="auto", zstd_level=DEFAULT_ZSTD_LEVEL) -> SaveReport:
        """Write the bundle ``<path>.fluximg`` (``.fluximg`` is appended when missing).

        An existing bundle is replaced safely: everything is written to a sibling temporary
        directory, ``fluximg.json`` last, and then swapped in. Also writes ``preview.png``.

        Parameters
        ----------
        layers
            A bundle path (or a list of them) whose layers are copied into this one, each recording
            ``from``; refused when the pixel grids differ.
        recipe
            Writes ``<stem>.recipe.json`` beside the bundle: ``True`` for fp.save's automatic
            script discovery, or a dict with the same meaning as for ``fp.save``. ``None`` / ``False``
            write none.
        regions
            ``"auto"``: when the image has missing data, the connected valid areas (of up to 16
            evenly spaced planes) become regions ``ROI 1…``, left to right (none when there are more
            than 64). ``None`` / ``False``: only regions added with :meth:`add_region`.
        zstd_level
            The zstd compression level (default 5).

        Returns
        -------
        SaveReport
            Printed as a short summary; its ``manifest`` is what was written.
        """
        return _save(self, path, layers=layers, recipe=recipe, regions=regions, zstd_level=zstd_level)


def _num(v) -> str:
    """A coordinate in SVG text: at most 4 decimals, no exponent, no trailing zeros."""
    s = f"{round(float(v), 4):.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _num_v(v):
    f = float(v)
    return int(f) if f.is_integer() else f


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------------------------
# pixels
# ---------------------------------------------------------------------------------------------
def _inner(shape):
    nd = len(shape)
    H, W = shape[-2], shape[-1]
    tiled = max(H, W) > TILE_THRESHOLD
    return [1] * (nd - 2) + [min(TILE_SIZE, H) if tiled else H, min(TILE_SIZE, W) if tiled else W]


def _factors(H, W):
    factors = [1]
    if max(H, W) > TILE_THRESHOLD:
        f = FIRST_OVERVIEW
        while True:
            factors.append(f)
            if max(-(-H // f), -(-W // f)) <= OVERVIEW_FLOOR:
                break
            f *= 2
    return factors


def _tiles(stored, ih, iw, fill):
    H, W = stored.shape
    if ih == H and iw == W:
        yield 0, 0, stored
        return
    for ty in range(-(-H // ih)):
        for tx in range(-(-W // iw)):
            block = stored[ty * ih:(ty + 1) * ih, tx * iw:(tx + 1) * iw]
            if block.shape != (ih, iw):
                chunk = np.full((ih, iw), fill, dtype=stored.dtype)
                chunk[:block.shape[0], :block.shape[1]] = block
                block = chunk
            yield ty, tx, block


def _store(values, encoding, channel, stored_dtype):
    """One plane's stored samples under an encoding."""
    if encoding["kind"] == "linear16":
        p = encoding["channels"][min(channel, len(encoding["channels"]) - 1)]
        return enc.encode_linear16(values, p["scale"], p["offset"])
    if values.dtype != stored_dtype:
        return values.astype(stored_dtype)
    return values


def _stored_dtype_fill(work, encoding, nodata, accs, what):
    if encoding["kind"] == "linear16":
        return np.dtype(np.uint16), 0
    if work.kind == "f":
        return work, math.nan
    if work.name in _STORED_INTS:
        dtype = work
    else:
        lo = min((a.min for a in accs if math.isfinite(a.min)), default=0)
        hi = max((a.max for a in accs if math.isfinite(a.max)), default=0)
        if nodata is not None:
            lo, hi = min(lo, nodata), max(hi, nodata)
        dtype = _narrow_int(int(lo), int(hi), what)
    if nodata is not None:
        info = np.iinfo(dtype)
        if not info.min <= nodata <= info.max:
            _bad(f"nodata={nodata} does not fit the stored type {dtype.name}")
    return dtype, (nodata if nodata is not None else 0)


def _overview(plane, f, work, encoding, channel, stored_dtype, nodata):
    values = plane.astype(np.float64)
    if work.kind != "f" and nodata is not None:
        values[plane == nodata] = np.nan
    m = enc.block_mean(values, f)
    if encoding["kind"] == "linear16":
        return _store(m, encoding, channel, stored_dtype)
    if work.kind == "f":
        return m.astype(stored_dtype)
    filled = np.where(np.isnan(m), nodata if nodata is not None else 0, enc.round_half_up(m))
    return filled.astype(stored_dtype)


def _opening_index(img, d):
    """Where Flux opens along ``d``: ``default_index``, else the middle plane in z, else 0."""
    if d == "c":
        return 0
    if img.default_index and d in img.default_index:
        return img.default_index[d]
    return img.shape[img.dims.index(d)] // 2 if d == "z" else 0


def _even_planes(total, k):
    k = min(k, total)
    if k <= 1:
        return {0}
    return {int(round(i * (total - 1) / (k - 1))) for i in range(k)}


def _write_pixels(root, img, *, level, codec, pool, max_shard_bytes):
    dims, shape = img.dims, img.shape
    nd = len(dims)
    lead_shape = tuple(shape[:-2])
    H, W, C = img.H, img.W, img.C
    ci = dims.index("c") if "c" in dims else -1
    work = img._work
    is_float = work.kind == "f"
    nodata = img.nodata
    source = _Plane(img._array, work, img._nodata_float)
    total = int(np.prod(lead_shape)) if lead_shape else 1
    per_channel = total // C
    sample_stride = max(1, math.ceil(per_channel * H * W / enc.SAMPLE_TARGET))
    diff_every = max(1, per_channel // enc.DIFF_PLANES)
    diff_stride = max(1, math.ceil(min(per_channel, enc.DIFF_PLANES) * H * W / enc.DIFF_TARGET))
    accs = [enc.ChannelAcc(sample_stride, diff_stride) for _ in range(C)]
    profile_dims = [d for d in dims[:-2] if d != "c"]
    prof_sum = {d: np.zeros((C, shape[dims.index(d)])) for d in profile_dims}
    prof_cnt = {d: np.zeros((C, shape[dims.index(d)])) for d in profile_dims}
    region_planes = _even_planes(total, REGION_PLANES)
    union = None
    missing_seen = False
    preview = {}
    opening = [_opening_index(img, d) for d in dims[:-2]]
    seen = [0] * C

    # ---- pass 1: learn the image --------------------------------------------------------------
    for flat, lead in enumerate(np.ndindex(*lead_shape)):
        c = lead[ci] if ci >= 0 else 0
        plane = source(lead)
        if is_float:
            mask = np.isfinite(plane)
        else:
            mask = None if nodata is None else plane != nodata
        vals = plane.reshape(-1) if mask is None else plane[mask]
        s, n = accs[c].learn(vals)
        if is_float and seen[c] % diff_every == 0:
            accs[c].learn_diffs(plane)
        seen[c] += 1
        for d in profile_dims:
            i = lead[dims.index(d)]
            prof_sum[d][c, i] += s
            prof_cnt[d][c, i] += n
        if mask is not None and n < H * W:
            missing_seen = True
        if flat in region_planes:
            m = np.ones((H, W), dtype=bool) if mask is None else mask
            union = m.copy() if union is None else (union | m)
        if all(lead[k] == opening[k] for k in range(nd - 2) if k != ci) and c not in preview:
            preview[c] = _preview_values(plane, mask, work)

    names = [ch["name"] or (("red", "green", "blue")[i] if img.photometric == "rgb" else "image" if C == 1
                            else f"channel {i + 1}") for i, ch in enumerate(img._channels)]
    encoding, note = enc.choose_encoding(accs, work, img.lossless, what="pixels", channel_names=names)
    if encoding.get("sourceDtype") != img.input_dtype.name:
        encoding["sourceDtype"] = img.input_dtype.name
    stored_dtype, fill = _stored_dtype_fill(work, encoding, nodata, accs, "data")

    # ---- pass 2: store -----------------------------------------------------------------------
    factors = _factors(H, W)
    writers = []
    for i, f in enumerate(factors):
        lshape = list(lead_shape) + [-(-H // f), -(-W // f)]
        writers.append(_zarr.ArrayWriter(root, f"pixels/{i}", lshape, stored_dtype, _inner(lshape), fill, dims,
                                         level=level, max_shard_bytes=max_shard_bytes, codec=codec, pool=pool))
    try:
        for lead in np.ndindex(*lead_shape):
            c = lead[ci] if ci >= 0 else 0
            plane = source(lead)
            for li, (f, w) in enumerate(zip(factors, writers)):
                if li == 0:
                    stored = _store(plane, encoding, c, stored_dtype)
                else:
                    stored = _overview(plane, f, work, encoding, c, stored_dtype, nodata)
                for ty, tx, chunk in _tiles(stored, w.inner[-2], w.inner[-1], fill):
                    w.add(lead + (ty, tx), chunk)
        levels = []
        nbytes = files = 0
        for i, (f, w) in enumerate(zip(factors, writers)):
            w.finish()
            levels.append({"path": str(i), "shape": w.shape, "downsample": f})
            nbytes += w.bytes
            files += w.files
    except BaseException:
        for w in writers:
            w.abort()
        raise

    # ---- channels, profiles ------------------------------------------------------------------
    stats = [enc.finish_stats(a) for a in accs]
    channels = []
    rgb_range = None
    if img.photometric == "rgb":
        if stored_dtype.name in ("uint8", "uint16") and encoding["kind"] == "exact":
            rgb_range = [0.0, float(np.iinfo(stored_dtype).max)]
        else:
            lo = min(enc.default_range(s)[0] for s in stats)
            hi = max(enc.default_range(s)[1] for s in stats)
            rgb_range = [lo, hi]
    for i, (given, st) in enumerate(zip(img._channels, stats)):
        color, lut = given["color"], given["lut"]
        if color is None and lut is None:
            if img.photometric == "rgb":
                color = RGB_COLOURS[i]
            elif C == 1:
                lut = "gray"
            else:
                color = DEFAULT_COLOURS[i % len(DEFAULT_COLOURS)]
        channels.append({"name": names[i], "color": color, "lut": lut,
                         "range": given["range"] or rgb_range or enc.default_range(st), "stats": st})
    profiles = {}
    for d in profile_dims:
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = prof_sum[d] / prof_cnt[d]
        profiles[d] = {"mean": [[None if prof_cnt[d][c, i] == 0 else float(mean[c, i]) for i in range(mean.shape[1])]
                                for c in range(C)]}
    pixels = {"path": "pixels", "dims": list(dims), "shape": list(shape), "dtype": stored_dtype.name,
              "levels": levels, "encoding": encoding}
    if not is_float and nodata is not None:
        pixels["nodata"] = nodata
    pixels["ome"] = all(d in CORE_DIMS for d in dims)
    return {"pixels": pixels, "channels": channels, "profiles": profiles, "note": note, "factors": factors,
            "union": union if missing_seen else None, "preview": preview, "bytes": nbytes, "files": files}


def _ome_attrs(name, dims, axes, factors, channels, encoding):
    def unit(d):
        u = (axes.get(d) or {}).get("unit")
        return _zarr.OME_UNITS.get(u) if u else None

    ome_axes = []
    for d in dims:
        kind = "time" if d == "t" else "channel" if d == "c" else "space"
        entry = {"name": d, "type": kind}
        if kind != "channel" and unit(d):
            entry["unit"] = unit(d)
        ome_axes.append(entry)
    scale0 = [1.0 if d == "c" else float(axes[d].get("scale", 1.0)) for d in dims]
    datasets = [{"path": str(i), "coordinateTransformations": [
        {"type": "scale", "scale": [s * (f if d in ("x", "y") else 1) for s, d in zip(scale0, dims)]}]}
        for i, f in enumerate(factors)]
    ome = {"version": "0.5", "multiscales": [{"name": name, "axes": ome_axes, "datasets": datasets}]}
    if "c" in dims:
        def stored(v, c):
            if encoding["kind"] == "linear16":
                p = encoding["channels"][c]
                return 1 + (v - p["offset"]) / p["scale"]
            return v
        ome["omero"] = {"channels": [{
            "label": ch["name"], "active": True, "color": (ch["color"] or "#ffffff").lstrip("#").upper(),
            "window": {"start": stored(ch["range"][0], c), "end": stored(ch["range"][1], c),
                       "min": stored(ch["stats"]["min"], c), "max": stored(ch["stats"]["max"], c)}}
            for c, ch in enumerate(channels)]}
    return {"ome": ome}


# ---------------------------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------------------------
def _preview_values(plane, mask, work):
    values = plane.astype(np.float64)
    if mask is not None:
        values[~mask] = np.nan
    f = max(1, math.ceil(max(values.shape) / PREVIEW_SIDE))
    return enc.block_mean(values, f) if f > 1 else values


def _preview_png(planes, channels):
    from matplotlib import colors as mcolors
    from ..images import resolve_lut
    first = next(iter(planes.values()))
    rgb = np.zeros(first.shape + (3,))
    any_valid = np.zeros(first.shape, dtype=bool)
    for c, ch in enumerate(channels):
        v = planes.get(c)
        if v is None:
            continue
        lo, hi = ch["range"]
        valid = np.isfinite(v)
        any_valid |= valid
        t = np.clip((np.where(valid, v, lo) - lo) / (hi - lo), 0.0, 1.0)
        if ch["color"]:
            layer = t[..., None] * np.asarray(mcolors.to_rgb(ch["color"]))
        else:
            layer = resolve_lut(ch["lut"])(t)[..., :3]
        rgb += np.where(valid[..., None], layer, 0.0)
    out = np.empty(first.shape + (4,), dtype=np.uint8)
    out[..., :3] = np.rint(np.clip(rgb, 0.0, 1.0) * 255)
    out[..., 3] = np.where(any_valid, 255, 0)
    return encode_png(out)


# ---------------------------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------------------------
def _write_intensity(root, L, *, level, codec, pool, max_shard_bytes):
    data = L["data"]
    work = _work_dtype(data.dtype, f"layer {L['id']!r}")
    plane = np.ascontiguousarray(data, dtype=work)
    H, W = plane.shape
    acc = enc.ChannelAcc(sample_stride=max(1, math.ceil(H * W / enc.SAMPLE_TARGET)), diff_stride=1)
    if work.kind == "f":
        acc.learn(plane[np.isfinite(plane)])
        acc.learn_diffs(plane)
    else:
        acc.learn(plane.reshape(-1))
    encoding, _long = enc.choose_encoding([acc], work, L["lossless"], what=f"layer {L['id']}")
    note = f"layer {L['id']}: {_short_note(encoding, work, L['lossless'])}"
    if encoding.get("sourceDtype") != data.dtype.name:
        encoding["sourceDtype"] = np.dtype(data.dtype).name
    stored_dtype, fill = _stored_dtype_fill(work, encoding, None, [acc], f"layer {L['id']!r}")
    w = _zarr.ArrayWriter(root, f"layers/{L['id']}", [H, W], stored_dtype, _inner([H, W]), fill, ["y", "x"],
                          level=level, max_shard_bytes=max_shard_bytes, codec=codec, pool=pool)
    for ty, tx, chunk in _tiles(_store(plane, encoding, 0, stored_dtype), w.inner[0], w.inner[1], fill):
        w.add((ty, tx), chunk)
    w.finish()
    st = enc.finish_stats(acc)
    entry = {"id": L["id"], "name": L["name"], "kind": "intensity", "visible": L["visible"], "path": f"layers/{L['id']}",
             "dtype": stored_dtype.name, "encoding": encoding, "color": L["color"], "lut": L["lut"],
             "range": L["range"] or enc.layer_range(st), "opacity": L["opacity"], "stats": st}
    return entry, note


def _short_note(encoding, work, lossless):
    if encoding["kind"] == "linear16":
        return f"16-bit linear16 (step {encoding['guard']['stepOverNeighbour'][0]:.2g} of the neighbour difference)"
    if work.kind != "f":
        return f"exact {work.name}"
    if lossless:
        return f"exact {work.name} (lossless)"
    r = encoding["guard"]["stepOverNeighbour"][0]
    why = "no pixel-to-pixel variation" if r is None else f"step {r:.2g} of the neighbour difference"
    return f"exact {work.name}: a 16-bit copy was refused ({why})"


def _write_text(root, rel, text):
    full = os.path.join(root, rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(text)


def _write_labels(root, L, H, W, *, level, codec, pool, max_shard_bytes):
    lab = L["data"]
    w = _zarr.ArrayWriter(root, f"layers/{L['id']}", [H, W], lab.dtype, _inner([H, W]), 0, ["y", "x"],
                          level=level, max_shard_bytes=max_shard_bytes, codec=codec, pool=pool)
    for ty, tx, chunk in _tiles(lab, w.inner[0], w.inner[1], 0):
        w.add((ty, tx), chunk)
    w.finish()
    outlines = label_outlines(lab)
    cols = L["table"]
    key = L["key"]
    if cols is None:
        cols = {"label": [o.label for o in outlines], "pixels": [o.pixels for o in outlines],
                "cx": [o.cx for o in outlines], "cy": [o.cy for o in outlines]}
    key_of = {int(l): key_text(k) for l, k in zip(cols["label"], cols[key])}
    body = "".join(f'<path id="{_esc(L["id"] + "/" + key_of[o.label])}" data-key="{_esc(key_of[o.label])}" '
                   f'data-label="{o.label}" d="{o.d}"/>' for o in outlines)
    colour = L["color"]
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"><g id="{L["id"]}" fill="{colour}" '
           f'fill-opacity="0.12" fill-rule="evenodd" stroke="{colour}" stroke-width="1">{body}</g></svg>')
    _write_text(root, f"layers/{L['id']}.svg", svg)
    _write_text(root, f"layers/{L['id']}.table.json",
                json.dumps({"key": key, "columns": cols}, ensure_ascii=False, allow_nan=False))
    return {"id": L["id"], "name": L["name"], "kind": "labels", "visible": L["visible"], "path": f"layers/{L['id']}",
            "dtype": lab.dtype.name, "outlines": f"layers/{L['id']}.svg", "table": f"layers/{L['id']}.table.json",
            "key": key, "count": len(outlines), "color": colour, "weights": L["weights"]}


def _write_vector(root, L, H, W):
    lid = L["id"]
    body = "".join(f'<{tag} id="{_esc(lid + "/" + k)}" data-key="{_esc(k)}" '
                   + " ".join(f'{a}="{v}"' for a, v in attrs.items()) + "/>"
                   for k, tag, attrs in L["shapes"])
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"><g id="{lid}" fill="none" '
           f'stroke="{L["color"]}" stroke-width="1">{body}</g></svg>')
    _write_text(root, f"layers/{lid}.svg", svg)
    table_path = None
    if L["table"] is not None:
        table_path = f"layers/{lid}.table.json"
        _write_text(root, table_path, json.dumps({"key": L["key"], "columns": L["table"]}, ensure_ascii=False,
                                                 allow_nan=False))
    return {"id": lid, "name": L["name"], "kind": "vector", "visible": L["visible"], "path": f"layers/{lid}.svg",
            "table": table_path, "key": L["key"] if table_path else None, "count": len(L["shapes"]),
            "color": L["color"]}


# ---------------------------------------------------------------------------------------------
# borrowing layers from another bundle
# ---------------------------------------------------------------------------------------------
def _bundle_path(path) -> str:
    p = os.fspath(path)
    p = p.rstrip("/\\") or p
    if p.endswith(SUFFIX):
        return p
    return p + SUFFIX if not os.path.isdir(p) or os.path.isdir(p + SUFFIX) else p


def _inside(rel, where):
    if not isinstance(rel, str) or not rel or os.path.isabs(rel) or ".." in rel.replace("\\", "/").split("/"):
        raise _zarr.FluximgFormatError(f"path {rel!r} must stay inside the bundle", where)
    return rel


def _layer_files(L, where):
    paths = []
    if L.get("kind") in ("intensity", "labels"):
        paths.append(_inside(L["path"], where))
    if L.get("kind") == "labels":
        paths += [_inside(L["outlines"], where), _inside(L["table"], where)]
    if L.get("kind") == "vector":
        paths.append(_inside(L["path"], where))
        if L.get("table"):
            paths.append(_inside(L["table"], where))
    return paths


def _borrow_plan(sources, img, dest_bundle):
    from .reader import read_manifest
    if sources is None:
        return []
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]
    plan = []
    for src in sources:
        bundle = _bundle_path(src)
        man = read_manifest(bundle)
        p = man["pixels"]
        sh, sw = p["shape"][-2], p["shape"][-1]
        if (sh, sw) != (img.H, img.W):
            _bad(f"layers={os.fspath(src)!r}: that bundle's pixel grid is {sh} x {sw} (y x), this image's is "
                 f"{img.H} x {img.W}; layers can only be shared between images on the same grid")
        for d in ("y", "x"):
            a, b = man["axes"].get(d, {}), img.axes[d]
            if a.get("unit") is not None and b.get("unit") is not None:
                same = (a.get("unit") == b.get("unit")
                        and math.isclose(float(a.get("scale", 1)), float(b.get("scale", 1)), rel_tol=1e-9)
                        and math.isclose(float(a.get("offset", 0)), float(b.get("offset", 0)), rel_tol=1e-9, abs_tol=1e-12))
                if not same:
                    _bad(f"layers={os.fspath(src)!r}: axis {d} differs ({a.get('scale')} {a.get('unit')} vs "
                         f"{b.get('scale')} {b.get('unit')} per pixel); the pixel grids must match")
        rel_from = os.path.relpath(os.path.abspath(bundle), os.path.dirname(os.path.abspath(dest_bundle)))
        for L in man["layers"]:
            entry = dict(L)
            entry["from"] = rel_from.replace(os.sep, "/")
            plan.append((bundle, entry, _layer_files(L, os.path.join(bundle, MANIFEST))))
    return plan


def _copy_tree(src, dst):
    """Copy a file or a directory tree. Never a hard link: a later in-place edit of the source bundle
    must not change this one."""
    if os.path.isdir(src):
        for dirpath, _dirs, files in os.walk(src):
            rel = os.path.relpath(dirpath, src)
            os.makedirs(os.path.join(dst, rel), exist_ok=True)
            for f in files:
                _copy_tree(os.path.join(dirpath, f), os.path.join(dst, rel, f))
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


# ---------------------------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------------------------
def _auto_regions(union, taken):
    from scipy import ndimage
    if union is None or union.all():
        return []
    lab, n = ndimage.label(union, structure=np.ones((3, 3), dtype=bool))
    if n == 0 or n > MAX_AUTO_REGIONS:
        return []
    boxes = [(sl[1].start, sl[0].start, sl[1].stop - sl[1].start, sl[0].stop - sl[0].start)
             for sl in ndimage.find_objects(lab)]
    boxes.sort()
    out, k = [], 0
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        k += 1
        rid = f"roi_{k}"
        while rid in taken:
            k += 1
            rid = f"roi_{k}"
        taken.add(rid)
        out.append({"id": rid, "name": f"ROI {i}", "rect": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}})
    return out


# ---------------------------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------------------------
def _dir_bytes(root):
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total


def _check_destination(bundle):
    if not os.path.lexists(bundle):
        return
    if not os.path.isdir(bundle) or os.path.islink(bundle):
        _bad(f"{bundle} exists and is not a fluximg bundle directory; refusing to replace it")
    if os.listdir(bundle) and not os.path.isfile(os.path.join(bundle, MANIFEST)):
        _bad(f"{bundle} exists and is not a fluximg bundle (no {MANIFEST}); refusing to replace it")


def _swap(tmp, bundle):
    if not os.path.lexists(bundle):
        os.rename(tmp, bundle)
        return
    old = os.path.join(os.path.dirname(bundle), f".{os.path.basename(bundle)}.{uuid.uuid4().hex[:8]}.old")
    os.rename(bundle, old)
    try:
        os.rename(tmp, bundle)
    except BaseException:
        os.rename(old, bundle)
        raise
    shutil.rmtree(old, ignore_errors=True)


def _umask():
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _only_skip(plot_name):
    only = os.environ.get("FLUXPLOT_ONLY", "").strip()
    if not only:
        return None
    pats = [p.strip() for p in only.split(",") if p.strip()]
    if pats and not any(fnmatch.fnmatchcase(plot_name, p) for p in pats):
        return only
    return None


def _save(img, path, *, layers, recipe, regions, zstd_level, max_shard_bytes=None):
    from .. import recipe as _recipe
    from ..version import SPEC_VERSION, __version__
    started = time.perf_counter()
    if regions not in ("auto", True, None, False):
        _bad(f"regions must be 'auto' or None, got {regions!r}")
    level = int(zstd_level)
    if not 1 <= level <= 22:
        _bad(f"zstd_level must be 1..22, got {zstd_level!r}")
    bundle = os.fspath(path)
    if not bundle.endswith(SUFFIX):
        bundle += SUFFIX
    stem = bundle[: -len(SUFFIX)]
    plot_name = os.path.basename(stem)
    name = img.name or plot_name
    raw = int(np.prod(img.shape)) * img.input_dtype.itemsize
    only = _only_skip(plot_name)
    if only is not None:
        print(f"fluxplot: skipped '{plot_name}' (FLUXPLOT_ONLY={only})", file=sys.stderr)
        return SaveReport(path=bundle, name=name, dims=list(img.dims), shape=list(img.shape),
                          dtype=img.input_dtype.name, stored_dtype="", encoding="", reason=f"FLUXPLOT_ONLY={only}",
                          bytes=0, raw_bytes=raw, layers=[], regions=[], skipped=True)
    codec = _zstd.codec()
    _check_destination(bundle)
    borrowed = _borrow_plan(layers, img, bundle)
    own_ids = {L["id"] for L in img._layers}
    seen_ids = set(own_ids)
    for _src, entry, _files in borrowed:
        if entry["id"] in seen_ids:
            _bad(f"layers=: layer {entry['id']!r} exists twice (in this image or another borrowed bundle)")
        seen_ids.add(entry["id"])
    max_shard = int(max_shard_bytes or MAX_SHARD_BYTES)
    parent = os.path.dirname(os.path.abspath(bundle))
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f".{os.path.basename(bundle)}.", suffix=".tmp", dir=parent)
    try:
        os.chmod(tmp, 0o777 & ~_umask())   # mkdtemp makes it private; a bundle is an ordinary directory
        workers = max(1, min(8, os.cpu_count() or 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            px = _write_pixels(tmp, img, level=level, codec=codec, pool=pool, max_shard_bytes=max_shard)
            entries, notes = [], []
            for L in img._layers:
                if L["kind"] == "intensity":
                    entry, note = _write_intensity(tmp, L, level=level, codec=codec, pool=pool,
                                                   max_shard_bytes=max_shard)
                    notes.append(note)
                elif L["kind"] == "labels":
                    entry = _write_labels(tmp, L, img.H, img.W, level=level, codec=codec, pool=pool,
                                          max_shard_bytes=max_shard)
                else:
                    entry = _write_vector(tmp, L, img.H, img.W)
                entries.append(entry)
        by_source = {}
        for src, entry, files in borrowed:
            for rel in files:
                _copy_tree(os.path.join(src, rel), os.path.join(tmp, rel))
            entries.append(entry)
            by_source.setdefault(entry["from"], []).append(entry["id"])
        notes += [f"borrowed from {src}: {', '.join(ids)}" for src, ids in by_source.items()]
        for e in entries:
            if e["kind"] == "labels" and e.get("weights") and not any(x["kind"] == "intensity" and x["id"] == e["weights"]
                                                                      for x in entries):
                _bad(f"layer {e['id']!r} names a weights layer {e['weights']!r} that does not exist")
        region_list = [dict(r) for r in img._regions]
        if regions in ("auto", True):
            region_list += _auto_regions(px["union"], {r["id"] for r in region_list})
        group = _zarr.group_meta_json(_ome_attrs(name, img.dims, img.axes, px["factors"], px["channels"],
                                                 px["pixels"]["encoding"]) if px["pixels"]["ome"] else {})
        _write_text(tmp, "pixels/zarr.json", json.dumps(group, indent=1, ensure_ascii=False, allow_nan=False))
        if px["preview"]:
            with open(os.path.join(tmp, "preview.png"), "wb") as f:
                f.write(_preview_png(px["preview"], px["channels"]))
        recipe_path = stem + ".recipe.json" if recipe not in (None, False) else None
        provenance = {"writer": f"fluxplot {__version__}", "created": _recipe._now_iso()}
        if img.meta is not None:
            provenance["meta"] = img.meta
        if recipe_path:
            provenance["recipe"] = os.path.basename(recipe_path)
        manifest = {"spec": SPEC, "name": name, "pixels": px["pixels"], "axes": img.axes, "channels": px["channels"]}
        if img.photometric:
            manifest["photometric"] = img.photometric
        if px["profiles"]:
            manifest["profiles"] = px["profiles"]
        if img.default_index:
            manifest["defaults"] = {"index": dict(img.default_index)}
        manifest["layers"] = entries
        manifest["regions"] = region_list
        if img.world is not None:
            manifest["world"] = img.world
        manifest["provenance"] = provenance
        # the manifest is written LAST: a bundle without it is an unfinished write
        _write_text(tmp, MANIFEST, json.dumps(manifest, indent=1, ensure_ascii=False, allow_nan=False))
        _swap(tmp, bundle)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if recipe_path:
        rec = _recipe.build_recipe(None if recipe is True else recipe, plot_name=plot_name,
                                   fluximg_filename=os.path.basename(bundle),
                                   manifest_filename=f"{os.path.basename(bundle)}/{MANIFEST}",
                                   spec_version=SPEC_VERSION, base_dir=os.getcwd(), recipe_dir=parent)
        _validate_recipe(rec)
        from ..api import _write_staged
        from ..canonical_json import dumps
        _write_staged([(recipe_path, dumps(rec).encode("utf-8"))])
    report = SaveReport(path=bundle, name=name, dims=list(img.dims), shape=list(img.shape), dtype=img.input_dtype.name,
                        stored_dtype=px["pixels"]["dtype"], encoding=px["pixels"]["encoding"]["kind"],
                        reason=px["note"].split(": ", 1)[1], bytes=_dir_bytes(bundle), raw_bytes=raw,
                        layers=[(e["id"], e["kind"]) for e in entries], regions=[r["name"] for r in region_list],
                        manifest=manifest, recipe=recipe_path, notes=notes,
                        seconds=time.perf_counter() - started)
    print(report)
    return report


def _validate_recipe(rec):
    import jsonschema
    from importlib.resources import files
    schema = json.loads(files("fluxplot").joinpath("schemas/recipe.schema.json").read_text())
    jsonschema.validate(rec, schema)


# ---------------------------------------------------------------------------------------------
# the public functions
# ---------------------------------------------------------------------------------------------
def fluximg(data, dims=None, *, pixel_size=None, unit="µm", channels=None, coords=None, dt=None, t0=None, nodata=None,
            transform=None, meta=None, name=None, lossless=False, photometric=None, default_index=None) -> FluxImage:
    """An N-dimensional scientific image to save as a fluximg bundle (``<stem>.fluximg/``), which
    Flux places, browses and annotates without rerunning Python.

    Add layers with :meth:`FluxImage.add_image`, :meth:`~FluxImage.add_labels` and
    :meth:`~FluxImage.add_shapes`, regions with :meth:`~FluxImage.add_region`, then
    :meth:`~FluxImage.save`. For an image without layers, :func:`imsave` does it in one call.

    Parameters
    ----------
    data
        A numpy array, or an xarray DataArray (its dims, coordinates, channel labels and coordinate
        ``units`` are used; explicit keywords win). A datetime coordinate becomes seconds from its
        first value, which is kept in ``provenance.meta`` as ``<dim>_start``.
    dims
        One name per axis of ``data``: ``y`` and ``x`` (required), optional ``t``, ``c`` (channels),
        ``z`` and any other lower-case names (``trial``). Default ``("y", "x")`` for 2-D data;
        otherwise required. Any order: the pixels are stored as extra dimensions, then t, c, z, y, x.
    pixel_size, unit
        The physical size of a pixel (one number or ``(y, x)``) and its unit (also used by
        ``coords`` of ``z``).
    channels
        Channel names, or dicts ``{"name", "color" | "lut", "range"}``. A ``lut`` that is a colour
        (``"magenta"``, ``"#ff2bd6"``) is the black-to-colour ramp; anything else a colour-map name.
        Defaults: one channel → gray; several → green, magenta, cyan, yellow, orange, azure; the
        display range the 1st to 99.8th percentile.
    coords
        ``{dim: positions}`` (z planes, frame times): evenly spaced values become scale / offset.
    dt, t0
        The ``t`` axis: seconds per frame and the time of frame 0.
    nodata
        Float data: values equal to it become missing (NaN). Integer data: the stored value that
        means "no data".
    transform
        A 4 x 4 matrix (pixel → world), stored verbatim as ``world.matrix``.
    meta
        A dict stored as ``provenance.meta`` (experiment identifiers and the like).
    name
        A human-readable name (default the file stem).
    lossless
        Float data is stored at 16 bits per channel range (``linear16``) when a check on the image
        shows the rounding step is negligible next to its pixel-to-pixel variation; ``True`` always
        stores the floats exactly. Integer data is always stored exactly.
    photometric
        ``"rgb"`` for a true-colour photograph (three channels shown as red, green, blue).
    default_index
        Where the image opens along its other dimensions, e.g. ``{"z": 10}`` (written as
        ``defaults.index``). Without it Flux opens at the middle z plane and the first position
        elsewhere; ``preview.png`` shows the same plane.

    Examples
    --------
    >>> img = fp.fluximg(stack, dims=("c", "y", "x"), pixel_size=0.25, channels=["structure", "signal"])
    >>> img.add_labels("cells", label_map, background=0)
    >>> img.save("plots/cells")
    """
    return FluxImage(data, dims, pixel_size=pixel_size, unit=unit, channels=channels, coords=coords, dt=dt, t0=t0,
                     nodata=nodata, transform=transform, meta=meta, name=name, lossless=lossless,
                     photometric=photometric, default_index=default_index)


def imsave(path, data, dims=None, *, layers=None, recipe=None, regions="auto", zstd_level=DEFAULT_ZSTD_LEVEL,
           **keywords) -> SaveReport:
    """Save an image as a fluximg bundle in one call: ``fp.fluximg(data, dims, **keywords).save(path, …)``.

    Takes every keyword of :func:`fluximg` (``pixel_size``, ``unit``, ``channels``, ``coords``,
    ``dt``, ``t0``, ``nodata``, ``transform``, ``meta``, ``name``, ``lossless``, ``photometric``,
    ``default_index``)
    and of :meth:`FluxImage.save` (``layers``, ``recipe``, ``regions``, ``zstd_level``).

    Examples
    --------
    >>> fp.imsave("plots/stack", stack, dims=("c", "z", "y", "x"), coords={"z": z_um}, pixel_size=0.25)
    >>> fp.imsave("plots/trial1", movie, dims=("c", "t", "x", "y"), dt=1 / 80, layers="plots/stack")
    """
    return FluxImage(data, dims, **keywords).save(path, layers=layers, recipe=recipe, regions=regions,
                                                  zstd_level=zstd_level)
