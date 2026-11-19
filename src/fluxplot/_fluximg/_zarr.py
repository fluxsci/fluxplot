"""The Zarr v3 profile fluximg/1 uses (SPEC 3.3), and nothing more: one ``sharding_indexed`` codec
whose inner chunks are ``bytes`` (little-endian) + ``zstd``, and whose index (``offset``, ``nbytes``
as little-endian uint64 per inner chunk in C order, then a CRC-32C) sits at the end of each shard.

:class:`ArrayWriter` streams inner chunks, in C order, into shard files split along the first
dimension; :class:`ArrayReader` reads them back.
"""
from __future__ import annotations

import collections
import json
import math
import os
import struct

import numpy as np

__all__ = [
    "FluximgFormatError", "DTYPES", "crc32c", "encode_shard_index", "decode_shard_index", "array_meta_json",
    "group_meta_json", "parse_array_meta", "ArrayWriter", "ArrayReader", "OME_UNITS",
]

#: SPEC 3.2: the stored sample types (little-endian on disk).
DTYPES = ("uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64")
#: an inner chunk that was never written: both index numbers are 2^64 - 1
_MISSING = 0xFFFFFFFFFFFFFFFF

#: OME-Zarr 0.5 unit names for the units fluximg writers commonly see
OME_UNITS = {
    "µm": "micrometer", "um": "micrometer", "μm": "micrometer", "nm": "nanometer", "mm": "millimeter",
    "cm": "centimeter", "m": "meter", "s": "second", "ms": "millisecond", "µs": "microsecond",
    "us": "microsecond", "min": "minute", "h": "hour",
}


class FluximgFormatError(ValueError):
    """A bundle (or one of its files) does not follow fluximg/1."""

    def __init__(self, message, where=""):
        super().__init__(f"{where}: {message}" if where else message)
        self.where = where


# ---------------------------------------------------------------------------------------------
# CRC-32C (Castagnoli), as the Zarr crc32c codec defines it. Only shard indexes are checksummed,
# so a table-driven pure-Python loop is fast enough.
# ---------------------------------------------------------------------------------------------
def _crc_table():
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
        table.append(c)
    return tuple(table)


_CRC_TABLE = _crc_table()


def crc32c(data) -> int:
    """CRC-32C of ``data`` (bytes-like)."""
    table = _CRC_TABLE
    c = 0xFFFFFFFF
    for b in bytes(data):
        c = table[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def encode_shard_index(offsets, lengths) -> bytes:
    """The trailing shard index: per inner chunk ``offset``, ``nbytes`` (uint64 LE; ``None``
    offsets mark a chunk that is not stored) and the CRC-32C of those bytes."""
    n = len(offsets)
    arr = np.empty((n, 2), dtype="<u8")
    for i, (o, ln) in enumerate(zip(offsets, lengths)):
        if o is None:
            arr[i] = (_MISSING, _MISSING)
        else:
            arr[i] = (o, ln)
    body = arr.tobytes()
    return body + struct.pack("<I", crc32c(body))


def decode_shard_index(data, n_chunks, verify=True, where="shard index"):
    """``(offsets, lengths)`` as int arrays, ``-1`` offsets for chunks that are not stored."""
    if len(data) != n_chunks * 16 + 4:
        raise FluximgFormatError(f"expected {n_chunks * 16 + 4} bytes, got {len(data)}", where)
    body = bytes(data[: n_chunks * 16])
    if verify and struct.unpack("<I", bytes(data[n_chunks * 16:]))[0] != crc32c(body):
        raise FluximgFormatError("crc32c mismatch", where)
    arr = np.frombuffer(body, dtype="<u8").reshape(n_chunks, 2)
    missing = (arr[:, 0] == _MISSING) & (arr[:, 1] == _MISSING)
    offsets = np.where(missing, -1, arr[:, 0].astype(np.int64))
    lengths = np.where(missing, 0, arr[:, 1].astype(np.int64))
    return offsets, lengths


# ---------------------------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------------------------
def array_meta_json(shape, dtype, inner, outer, fill, dims, level, attributes=None) -> dict:
    """The exact ``zarr.json`` of SPEC 3.3."""
    fill_value = "NaN" if isinstance(fill, float) and math.isnan(fill) else _plain_number(fill)
    return {
        "zarr_format": 3, "node_type": "array", "shape": [int(s) for s in shape], "data_type": str(dtype),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [int(s) for s in outer]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": fill_value,
        "codecs": [{"name": "sharding_indexed", "configuration": {
            "chunk_shape": [int(s) for s in inner],
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                       {"name": "zstd", "configuration": {"level": int(level), "checksum": False}}],
            "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}],
            "index_location": "end"}}],
        "dimension_names": list(dims),
        "attributes": dict(attributes or {}),
    }


def group_meta_json(attributes=None) -> dict:
    return {"zarr_format": 3, "node_type": "group", "attributes": dict(attributes or {})}


def _plain_number(v):
    v = float(v)
    return int(v) if v.is_integer() else v


def _int_list(v, n=None):
    return (isinstance(v, list) and (n is None or len(v) == n)
            and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in v))


def parse_array_meta(meta, where="zarr.json") -> dict:
    """Validate a ``zarr.json`` against the fluximg/1 profile; return ``{shape, dtype, inner, outer,
    fill, dims, level}``."""
    def bad(message):
        raise FluximgFormatError(message, where)

    if not isinstance(meta, dict):
        bad("not an object")
    if meta.get("zarr_format") != 3 or meta.get("node_type") != "array":
        bad("not a Zarr v3 array")
    shape = meta.get("shape")
    if not _int_list(shape) or len(shape) < 2:
        bad("shape must have at least y and x")
    nd = len(shape)
    if meta.get("data_type") not in DTYPES:
        bad(f"unsupported data_type {meta.get('data_type')!r}")
    grid = meta.get("chunk_grid") or {}
    outer = (grid.get("configuration") or {}).get("chunk_shape")
    if grid.get("name") != "regular" or not _int_list(outer, nd):
        bad("chunk_grid must be regular")
    enc = meta.get("chunk_key_encoding") or {}
    if enc.get("name") != "default" or (enc.get("configuration") or {}).get("separator", "/") != "/":
        bad('chunk_key_encoding must be default with "/"')
    codecs = meta.get("codecs")
    if not isinstance(codecs, list) or len(codecs) != 1 or (codecs[0] or {}).get("name") != "sharding_indexed":
        bad("codecs must be exactly [sharding_indexed]")
    sc = codecs[0].get("configuration") or {}
    inner = sc.get("chunk_shape")
    if not _int_list(inner, nd):
        bad("sharding chunk_shape missing")
    for i in range(nd):
        if inner[i] < 1 or outer[i] < 1 or outer[i] % inner[i]:
            bad("a shard must hold a whole number of inner chunks")
    cc = sc.get("codecs")
    if (not isinstance(cc, list) or len(cc) != 2 or (cc[0] or {}).get("name") != "bytes"
            or (cc[1] or {}).get("name") != "zstd"):
        bad("inner codecs must be [bytes, zstd]")
    if (cc[0].get("configuration") or {}).get("endian", "little") != "little":
        bad("bytes codec must be little-endian")
    ic = sc.get("index_codecs")
    if (not isinstance(ic, list) or len(ic) != 2 or (ic[0] or {}).get("name") != "bytes"
            or (ic[1] or {}).get("name") != "crc32c"):
        bad("index codecs must be [bytes, crc32c]")
    if sc.get("index_location", "end") != "end":
        bad("index_location must be end")
    fv = meta.get("fill_value")
    if fv == "NaN":
        fill = math.nan
    elif isinstance(fv, (int, float)) and not isinstance(fv, bool):
        fill = fv
    else:
        bad('fill_value must be a number or "NaN"')
    dims = meta.get("dimension_names")
    if not isinstance(dims, list) or len(dims) != nd or not all(isinstance(d, str) for d in dims):
        bad("dimension_names missing")
    return {"shape": list(shape), "dtype": meta["data_type"], "inner": list(inner), "outer": list(outer),
            "fill": fill, "dims": list(dims), "level": int((cc[1].get("configuration") or {}).get("level", 0))}


def _ceil_div(a, b):
    return -(-a // b)


def _prod(values):
    out = 1
    for v in values:
        out *= int(v)
    return out


def all_fill(chunk, fill) -> bool:
    """True when every sample of ``chunk`` is the fill value (NaN fill: every sample NaN)."""
    if isinstance(fill, float) and math.isnan(fill):
        return bool(np.isnan(chunk).all())
    if fill == 0:
        return not chunk.any()
    return bool((chunk == fill).all())


# ---------------------------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------------------------
class ArrayWriter:
    """Streams inner chunks, in C order of their chunk coordinates, into shard files.

    A shard spans the whole array along every dimension but the first; along the first it holds
    as many slabs (rows of inner chunks) as keep a file near ``max_shard_bytes``, judged from the
    compressed size of the first slabs (up to eight, the only ones held in memory). After that each
    compressed chunk goes straight into its open shard file, whose index is appended when the
    shard is complete. Chunks that are entirely fill are not stored, and a shard with no stored
    chunk is not written. With a thread ``pool`` compression runs in parallel while at most
    ``max_inflight`` chunks wait for it.
    """

    #: the first slabs whose compressed size fixes the shard height
    DECIDE_SLABS = 8

    def __init__(self, root, path, shape, dtype, inner, fill, dims, *, level, max_shard_bytes, codec,
                 pool=None, max_inflight=None):
        self.root, self.path = root, path
        self.shape, self.inner = [int(s) for s in shape], [int(s) for s in inner]
        self.dtype = np.dtype(dtype).newbyteorder("<")
        self.fill, self.dims, self.level = fill, list(dims), int(level)
        self.max_shard_bytes, self.codec, self.pool = int(max_shard_bytes), codec, pool
        self.max_inflight = max_inflight or (2 * getattr(pool, "_max_workers", 1) + 2)
        self.counts = [_ceil_div(s, i) for s, i in zip(self.shape, self.inner)]
        self.per_slab = _prod(self.counts[1:])
        self.chunk_bytes = _prod(self.inner) * self.dtype.itemsize
        self.queue = collections.deque()  # (flat chunk number, bytes or a Future of them), not yet written
        self.slabs_seen = 0
        self.slabs_per_shard = 0
        self.bytes = 0
        self.files = 0
        self._shard = None                 # the open shard: its number, file, entries and write position
        self._file = None
        self._entries = {}
        self._pos = 0
        os.makedirs(os.path.join(root, path), exist_ok=True)

    def _flat(self, coords):
        k = 0
        for c, n in zip(coords, self.counts):
            k = k * n + int(c)
        return k

    def add(self, coords, chunk):
        """Add one inner chunk (exactly ``inner`` samples, any shape); ``None`` or an all-fill chunk
        stores nothing. Chunks must arrive in C order of ``coords``."""
        slab = int(coords[0])
        self.slabs_seen = max(self.slabs_seen, slab + 1)
        if chunk is not None and not all_fill(chunk, self.fill):
            raw = np.ascontiguousarray(chunk, dtype=self.dtype).reshape(-1).view(np.uint8)
            if raw.nbytes != self.chunk_bytes:
                raise ValueError(f"fluximg: chunk has {raw.nbytes} bytes, expected {self.chunk_bytes}")
            if self.pool is None:
                item = self.codec.compress(raw, self.level)
            else:
                item = self.pool.submit(self.codec.compress, raw, self.level)
            self.queue.append((self._flat(coords), item))
        if not self.slabs_per_shard and slab >= min(self.DECIDE_SLABS, self.counts[0]):
            self._decide(slab)                                        # slabs before this one are complete
        if self.slabs_per_shard:
            self._drain(force=False)
        elif len(self.queue) > self.max_inflight:
            self._resolve(len(self.queue) - self.max_inflight)        # bound the raw chunks still waiting

    def _resolve(self, n):
        """Turn the first ``n`` queued Futures into their bytes."""
        for i in range(n):
            flat, item = self.queue[i]
            if not isinstance(item, (bytes, bytearray)):
                self.queue[i] = (flat, item.result())

    def _decide(self, complete):
        known = max(1, complete)
        self._resolve(len(self.queue))
        total = sum(len(item) for flat, item in self.queue if flat // self.per_slab < known)
        mean = total / known
        n0 = self.counts[0]
        if mean * n0 <= self.max_shard_bytes or mean <= 0:
            self.slabs_per_shard = n0
        else:
            self.slabs_per_shard = max(1, min(n0, int(self.max_shard_bytes * 0.9 // mean)))

    def _drain(self, force):
        while self.queue:
            flat, item = self.queue[0]
            if not isinstance(item, (bytes, bytearray)):
                if not force and not item.done() and len(self.queue) <= self.max_inflight:
                    break
                item = item.result()
            self.queue.popleft()
            self._write_chunk(flat, item)

    def _write_chunk(self, flat, blob):
        k = self.slabs_per_shard
        slab = flat // self.per_slab
        shard = slab // k
        if shard != self._shard:
            self._close_shard()
            key = os.path.join(self.root, self.path, "c", str(shard), *(["0"] * (len(self.shape) - 1)))
            os.makedirs(os.path.dirname(key), exist_ok=True)
            self._file = open(key, "wb")
            self._shard, self._entries, self._pos = shard, {}, 0
        self._file.write(blob)
        self._entries[(slab - shard * k) * self.per_slab + flat % self.per_slab] = (self._pos, len(blob))
        self._pos += len(blob)

    def _close_shard(self):
        if self._file is None:
            return
        n = self.slabs_per_shard * self.per_slab
        offsets, lengths = [None] * n, [0] * n
        for local, (o, ln) in self._entries.items():
            offsets[local], lengths[local] = o, ln
        index = encode_shard_index(offsets, lengths)
        self._file.write(index)
        self._file.close()
        self._file = None
        self.bytes += self._pos + len(index)
        self.files += 1

    def abort(self):
        """Close an open shard file after a failure (the caller removes the directory)."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def finish(self, attributes=None) -> dict:
        """Write the remaining chunks, close the last shard and write ``zarr.json``; return the
        metadata dict."""
        if not self.slabs_per_shard:
            self._decide(self.slabs_seen)
        self._drain(force=True)
        self._close_shard()
        outer = [self.slabs_per_shard * self.inner[0]] + [c * i for c, i in zip(self.counts[1:], self.inner[1:])]
        meta = array_meta_json(self.shape, self.dtype.name, self.inner, outer, self.fill, self.dims, self.level,
                               attributes)
        text = json.dumps(meta, indent=1)
        with open(os.path.join(self.root, self.path, "zarr.json"), "w", encoding="utf-8") as f:
            f.write(text)
        self.bytes += len(text.encode("utf-8"))
        self.files += 1
        return meta


# ---------------------------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------------------------
class ArrayReader:
    """One fluximg Zarr array on disk: ``read()`` returns all of it."""

    def __init__(self, directory, codec, *, verify=True):
        self.directory = directory
        self.codec = codec
        self.verify = verify
        where = os.path.join(directory, "zarr.json")
        try:
            with open(where, encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            raise FluximgFormatError("missing", where) from None
        except ValueError:
            raise FluximgFormatError("not JSON", where) from None
        self.meta = parse_array_meta(raw, where)
        self.shape = self.meta["shape"]
        self.dtype = np.dtype(self.meta["dtype"]).newbyteorder("<")
        self.inner, self.outer = self.meta["inner"], self.meta["outer"]
        self.counts = [_ceil_div(s, i) for s, i in zip(self.shape, self.inner)]
        self.per_shard = [o // i for o, i in zip(self.outer, self.inner)]

    def chunks(self):
        """Yield ``(chunk_coords, samples)`` for every stored inner chunk (samples shaped ``inner``)."""
        n_local = _prod(self.per_shard)
        nbytes = _prod(self.inner) * self.dtype.itemsize
        shard_counts = [_ceil_div(s, o) for s, o in zip(self.shape, self.outer)]
        for sc in np.ndindex(*shard_counts):
            file = os.path.join(self.directory, "c", *map(str, sc))
            try:
                with open(file, "rb") as f:
                    data = f.read()
            except FileNotFoundError:
                continue                                              # never written: all fill
            index_bytes = n_local * 16 + 4
            if len(data) < index_bytes:
                raise FluximgFormatError(f"shorter than its {index_bytes}-byte index", file)
            offsets, lengths = decode_shard_index(data[-index_bytes:], n_local, self.verify, file)
            body = memoryview(data)[: len(data) - index_bytes]
            for local, lc in enumerate(np.ndindex(*self.per_shard)):
                off = int(offsets[local])
                if off < 0:
                    continue
                coords = tuple(s * p + c for s, p, c in zip(sc, self.per_shard, lc))
                if any(c >= n for c, n in zip(coords, self.counts)):
                    raise FluximgFormatError("a chunk outside the array is stored", file)
                end = off + int(lengths[local])
                if end > len(body):
                    raise FluximgFormatError(f"chunk {local} runs past the data", file)
                raw = self.codec.decompress(bytes(body[off:end]), nbytes)
                if len(raw) != nbytes:
                    raise FluximgFormatError(f"chunk decodes to {len(raw)} bytes, expected {nbytes}", f"{file}#{local}")
                yield coords, np.frombuffer(raw, dtype=self.dtype).reshape(self.inner)

    def read(self, *, out_dtype=None, missing=None, transform=None) -> np.ndarray:
        """The whole array. Without arguments: stored samples, missing chunks as the fill value.
        ``transform(samples, chunk_coords)`` maps each stored chunk to ``out_dtype`` values, and
        ``missing`` is what never-written chunks hold then."""
        out_dtype = np.dtype(out_dtype or self.dtype.newbyteorder("="))
        fill = self.meta["fill"] if missing is None else missing
        out = np.full(self.shape, fill, dtype=out_dtype)
        for coords, samples in self.chunks():
            if transform is not None:
                samples = transform(samples, coords)
            dest, src = [], []
            for c, i, s in zip(coords, self.inner, self.shape):
                lo = c * i
                hi = min(lo + i, s)
                dest.append(slice(lo, hi))
                src.append(slice(0, hi - lo))
            out[tuple(dest)] = samples[tuple(src)]
        return out
