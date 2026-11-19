"""Read fluximg/1 bundles: :func:`imread`."""
from __future__ import annotations

import json
import os

import numpy as np

from . import _encoding as enc
from . import _zstd
from ._zarr import ArrayReader, FluximgFormatError

__all__ = ["imread", "FluximgFile", "read_manifest", "FluximgFormatError"]

SPEC = "fluximg/1"
MANIFEST = "fluximg.json"
SUFFIX = ".fluximg"


def _resolve(path) -> str:
    p = os.fspath(path).rstrip("/\\") or os.fspath(path)
    if os.path.basename(p) == MANIFEST and os.path.isfile(p):
        return os.path.dirname(p) or "."
    if not os.path.isdir(p) and os.path.isdir(p + SUFFIX):
        return p + SUFFIX
    return p


def read_manifest(path) -> dict:
    """The manifest of the bundle at ``path`` (``.fluximg`` may be omitted); refuses anything that
    is not fluximg/1."""
    bundle = _resolve(path)
    where = os.path.join(bundle, MANIFEST)
    if not os.path.isdir(bundle):
        raise FileNotFoundError(f"fluximg: no bundle at {os.fspath(path)!r}")
    try:
        with open(where, encoding="utf-8") as f:
            m = json.load(f)
    except FileNotFoundError:
        raise FluximgFormatError(f"{bundle} has no {MANIFEST}: not a fluximg bundle, or an unfinished write") from None
    except ValueError:
        raise FluximgFormatError("not JSON", where) from None
    if not isinstance(m, dict) or m.get("spec") != SPEC:
        raise FluximgFormatError(f'spec must be "{SPEC}", got {m.get("spec") if isinstance(m, dict) else m!r}; '
                                 "this fluxplot reads fluximg/1 only", where)
    p = m.get("pixels")
    if (not isinstance(p, dict) or not isinstance(p.get("dims"), list) or not isinstance(p.get("shape"), list)
            or len(p["dims"]) != len(p["shape"]) or not isinstance(p.get("levels"), list) or not p["levels"]):
        raise FluximgFormatError("pixels.dims / shape / levels are malformed", where)
    if (not isinstance(m.get("channels"), list) or not isinstance(m.get("layers"), list)
            or not isinstance(m.get("regions"), list) or not isinstance(m.get("axes"), dict)):
        raise FluximgFormatError("channels, layers, regions and axes are required", where)
    return m


def _inside(bundle, rel):
    if not isinstance(rel, str) or os.path.isabs(rel) or ".." in rel.replace("\\", "/").split("/"):
        raise FluximgFormatError(f"path {rel!r} escapes the bundle", os.path.join(bundle, MANIFEST))
    return os.path.join(bundle, *rel.split("/"))


def _values(reader, encoding, channel_dim=None):
    """Level values of one array: ``exact`` in the stored dtype (NaN for missing floats; integer
    no-data stays the stored ``nodata``), ``linear16`` decoded to float64 (NaN for code 0)."""
    if encoding.get("kind") == "linear16":
        params = encoding.get("channels") or []
        if not params:
            raise FluximgFormatError("linear16 needs scale / offset per channel")

        def decode(samples, coords):
            p = params[coords[channel_dim] if channel_dim is not None else 0]
            return enc.decode_linear16(samples, float(p["scale"]), float(p["offset"]))
        return reader.read(out_dtype=np.float64, missing=np.nan, transform=decode)
    if encoding.get("kind") != "exact":
        raise FluximgFormatError(f"unknown encoding {encoding.get('kind')!r}")
    return reader.read()


class FluximgFile:
    """A fluximg bundle read by :func:`imread`.

    Attributes
    ----------
    values
        The pixel values of the level read, in stored dimension order (``dims``): ``exact`` data in
        its stored dtype (NaN where a float image has no data; integer no-data keeps the stored
        ``nodata`` value), ``linear16`` decoded to float64 (NaN where there is no data).
    dims, shape
        Dimension names (extra dimensions, then t, c, z, y, x) and the shape of ``values``.
    axes
        ``{dim: {"unit", "scale", "offset"} | {"unit", "coords"}}`` for every dimension but ``c``;
        for an overview level, y / x scales are those of its (coarser) pixels.
    channels
        Channel names in stored order (their colours, ranges and statistics are in
        ``manifest["channels"]``).
    manifest
        The bundle's ``fluximg.json``.
    nodata
        The stored value meaning "no data" of integer pixels, or ``None``.
    level, downsample
        Which pixel level ``values`` holds, and its linear downsampling of y and x.
    """

    def __init__(self, path, *, level=0, _read_values=True):
        self.path = _resolve(path)
        self.manifest = read_manifest(self.path)
        m = self.manifest
        p = m["pixels"]
        if not isinstance(level, int) or isinstance(level, bool) or not 0 <= level < len(p["levels"]):
            raise ValueError(f"fluximg: level must be 0..{len(p['levels']) - 1}, got {level!r}")
        self.level = level
        lv = p["levels"][level]
        self.downsample = int(lv.get("downsample", 1))
        self.dims = list(p["dims"])
        self.name = m.get("name")
        self.nodata = p.get("nodata")
        self._codec = _zstd.codec()
        reader = ArrayReader(_inside(self.path, f"{p['path']}/{lv['path']}"), self._codec)
        if reader.meta["dims"] != self.dims:
            raise FluximgFormatError("level dimension_names differ from pixels.dims", reader.directory)
        ci = self.dims.index("c") if "c" in self.dims else None
        self.values = _values(reader, p["encoding"], channel_dim=ci) if _read_values else None
        self.shape = list(reader.shape)
        self.axes = {}
        for d, a in m["axes"].items():
            a = dict(a)
            if d in ("x", "y") and self.downsample != 1 and "scale" in a:
                a["scale"] = float(a["scale"]) * self.downsample
            self.axes[d] = a
        self.channels = [c.get("name") for c in m["channels"]]
        self._layer_cache = {}

    def __repr__(self):
        shape = " × ".join(f"{d}={n}" for d, n in zip(self.dims, self.shape))
        lv = f", level {self.level} (1/{self.downsample})" if self.level else ""
        layers = f", layers [{', '.join(self.layers)}]" if self.layers else ""
        dtype = "" if self.values is None else f", {self.values.dtype}"
        return f"FluximgFile({self.name!r}: {shape}{dtype}{lv}{layers})"

    @property
    def layers(self) -> list:
        """Layer ids in manifest order."""
        return [L["id"] for L in self.manifest["layers"]]

    def _layer_entry(self, id):
        for L in self.manifest["layers"]:
            if L.get("id") == id:
                return L
        have = ", ".join(self.layers) or "none"
        raise KeyError(f"fluximg: no layer {id!r} (layers: {have})")

    def layer(self, id) -> np.ndarray:
        """A raster layer's ``(y, x)`` array: an intensity layer's values (``linear16`` decoded to
        float64, NaN for no data) or a label map (0 = no object). A vector layer has no array:
        use :meth:`table` or the layer's SVG."""
        if id in self._layer_cache:
            return self._layer_cache[id]
        L = self._layer_entry(id)
        kind = L.get("kind")
        if kind not in ("intensity", "labels"):
            raise ValueError(f"fluximg: layer {id!r} is a {kind} layer and has no pixel array; "
                             f"read its rows with .table({id!r})")
        reader = ArrayReader(_inside(self.path, L["path"]), self._codec)
        if kind == "labels":
            out = reader.read()
        else:
            out = _values(reader, L.get("encoding") or {"kind": "exact"})
        self._layer_cache[id] = out
        return out

    def table(self, id) -> dict:
        """A labels / vector layer's table as ``{column: [values…]}`` (``pd.DataFrame(...)`` /
        ``pl.DataFrame(...)`` take it as is); the key column is ``manifest`` layer ``key``."""
        L = self._layer_entry(id)
        rel = L.get("table")
        if not rel:
            raise ValueError(f"fluximg: layer {id!r} has no table")
        where = _inside(self.path, rel)
        with open(where, encoding="utf-8") as f:
            t = json.load(f)
        if not isinstance(t, dict) or not isinstance(t.get("columns"), dict):
            raise FluximgFormatError("table must have key and columns", where)
        return t["columns"]

    def to_xarray(self):
        """The values as an ``xarray.DataArray`` with coordinates from the axes (and channel names
        on ``c``), each coordinate's unit in its ``attrs["units"]``."""
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("to_xarray() needs xarray: pip install xarray") from None
        coords = {}
        for d, n in zip(self.dims, self.shape):
            if d == "c":
                coords["c"] = ("c", list(self.channels))
                continue
            a = self.axes.get(d, {})
            if a.get("labels"):
                values = list(a["labels"])
            elif a.get("coords") is not None:
                values = np.asarray(a["coords"], dtype=float)
            else:
                values = float(a.get("offset", 0.0)) + float(a.get("scale", 1.0)) * np.arange(n)
            attrs = {"units": a["unit"]} if a.get("unit") else {}
            coords[d] = (d, values, attrs)
        attrs = {"fluximg": self.path, "encoding": self.manifest["pixels"]["encoding"].get("kind")}
        if self.nodata is not None:
            attrs["nodata"] = self.nodata
        return xr.DataArray(self.values, dims=self.dims, coords=coords, name=self.name, attrs=attrs)


def imread(path, *, level=0, layer=None):
    """Read a fluximg bundle.

    Parameters
    ----------
    path
        The bundle directory (``.fluximg`` may be omitted).
    level
        Which pixel level to read: 0 is full resolution, 1… the overviews of large planes.
    layer
        Return that layer's array instead (see :meth:`FluximgFile.layer`).

    Returns
    -------
    FluximgFile
        ``.values``, ``.dims``, ``.shape``, ``.axes``, ``.channels``, ``.manifest``,
        ``.layer(id)``, ``.table(id)`` and ``.to_xarray()``; or, with ``layer=``, a numpy array.

    Examples
    --------
    >>> img = fp.imread("plots/dmd1_mean")
    >>> img.values.shape, img.dims
    ((2, 213, 924), ['c', 'y', 'x'])
    >>> footprints = img.layer("footprints")
    """
    if layer is not None:
        return FluximgFile(path, level=0, _read_values=False).layer(layer)
    return FluximgFile(path, level=level)
