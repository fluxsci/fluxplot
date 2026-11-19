"""Stored numbers ⇄ values (SPEC 4) and the statistics a bundle carries (SPEC 5).

``exact`` stores what it was given. ``linear16`` stores a float image at 16 bits across each
channel's own range: code 0 means "no data", and ``value = offset + scale * (code - 1)``. A writer
chooses it only when the guard shows one rounding step is small next to the pixel-to-pixel
variation that is really there.

The sampling below mirrors the reference writer (``writer.ts``) so that both make the same
encoding choice for the same image: a value sample of about a million per channel, and
neighbour differences from about forty evenly spaced planes per channel.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "GUARD_THRESHOLD", "PERCENTILE_KEYS", "ChannelAcc", "finish_stats", "default_range", "layer_range",
    "linear16_params", "encode_linear16", "decode_linear16", "guard_ratio", "guard_passes", "choose_encoding",
    "block_mean", "is_float", "round_half_up",
]

#: compact storage only while one rounding step stays under this fraction of the neighbour difference
GUARD_THRESHOLD = 0.1
PERCENTILE_KEYS = ("0.1", "1", "50", "99", "99.8", "99.9")
SAMPLE_TARGET = 1_000_000
DIFF_TARGET = 2_000_000
DIFF_PLANES = 40


def is_float(dtype) -> bool:
    return np.dtype(dtype).kind == "f"


class ChannelAcc:
    """What one pass over a channel's planes learns: exact min / max, a deterministic value sample
    (every ``sample_stride``-th valid value) and a sample of neighbour differences."""

    def __init__(self, sample_stride=1, diff_stride=1):
        self.min = math.inf
        self.max = -math.inf
        self.sample_stride = max(1, int(sample_stride))
        self.diff_stride = max(1, int(diff_stride))
        self.samples = []
        self.diffs = []
        self.seen = 0

    def learn(self, valid_values):
        """Fold in one plane's valid values (1-D); return ``(sum, count)`` for the profiles."""
        n = valid_values.size
        if n == 0:
            return 0.0, 0
        lo, hi = valid_values.min(), valid_values.max()
        if lo < self.min:
            self.min = float(lo)
        if hi > self.max:
            self.max = float(hi)
        first = (-self.seen) % self.sample_stride
        if first < n:
            # a copy in the input dtype: exact (every stored type converts to float64 exactly), and it
            # does not keep the whole plane alive
            self.samples.append(valid_values[first::self.sample_stride].copy())
        self.seen += n
        return float(valid_values.sum(dtype=np.float64)), int(n)

    def learn_diffs(self, plane):
        """Absolute differences between horizontally adjacent finite samples of one float plane,
        every ``diff_stride``-th pair in row-major order."""
        p = np.asarray(plane, dtype=np.float64)
        d = np.abs(p[:, 1:] - p[:, :-1])
        ok = np.isfinite(p[:, 1:]) & np.isfinite(p[:, :-1])
        vals = d[ok]
        if vals.size:
            self.diffs.append(vals[:: self.diff_stride].copy())       # not a view: it would pin all of vals

    def sample(self) -> np.ndarray:
        """The value sample as float64."""
        if not self.samples:
            return np.empty(0)
        return np.concatenate([np.asarray(s, dtype=np.float64) for s in self.samples])

    def median_diff(self) -> float:
        if not self.diffs:
            return math.nan
        return float(np.median(np.concatenate(self.diffs)))


def _percentile(sorted_values, p):
    r = p / 100 * (len(sorted_values) - 1)
    lo, hi = math.floor(r), math.ceil(r)
    a, b = float(sorted_values[lo]), float(sorted_values[hi])
    return a + (b - a) * (r - lo)


def finish_stats(acc: ChannelAcc) -> dict:
    """SPEC 5 stats from an accumulator: count, exact min / max, percentiles and a 256-bin histogram
    between the 0.1 and 99.9 percentiles."""
    s = np.sort(acc.sample())
    if s.size == 0:
        return {"count": 0, "min": 0.0, "max": 0.0, "percentiles": {k: 0.0 for k in PERCENTILE_KEYS},
                "hist": {"lo": 0.0, "hi": 1.0, "counts": [0] * 256}}
    percentiles = {k: _percentile(s, float(k)) for k in PERCENTILE_KEYS}
    lo, hi = percentiles["0.1"], percentiles["99.9"]
    if not hi > lo:
        lo = acc.min
        hi = acc.max if acc.max > acc.min else acc.min + 1
    k = 256 / (hi - lo)
    b = np.floor((s - lo) * k)
    inside = (b >= 0) & (b < 256)
    counts = np.bincount(b[inside].astype(np.int64), minlength=256)
    counts[255] += int(np.count_nonzero(~inside & (s == hi)))
    return {"count": int(s.size), "min": float(acc.min), "max": float(acc.max), "percentiles": percentiles,
            "hist": {"lo": float(lo), "hi": float(hi), "counts": [int(c) for c in counts]}}


def default_range(stats) -> list:
    """A channel's default display window: the 1st to 99.8th percentile (min..max, then +1, when flat)."""
    lo, hi = stats["percentiles"]["1"], stats["percentiles"]["99.8"]
    if not hi > lo:
        lo, hi = stats["min"], stats["max"]
    if not hi > lo:
        hi = lo + 1
    return [float(lo), float(hi)]


def layer_range(stats) -> list:
    """An intensity layer's default window: the median to the 99.8th percentile (an overlay shows
    what stands out)."""
    lo, hi = stats["percentiles"]["50"], stats["percentiles"]["99.8"]
    return [float(lo), float(hi if hi > lo else lo + 1)]


# ---------------------------------------------------------------------------------------------
# linear16
# ---------------------------------------------------------------------------------------------
def linear16_params(lo, hi) -> dict:
    """``{scale, offset}`` for a channel's finite range; a constant channel gets scale 1."""
    span = hi - lo
    return {"scale": span / 65534 if span > 0 else 1.0, "offset": float(lo)}


def round_half_up(x) -> np.ndarray:
    """Round to nearest, ties toward +infinity: JavaScript's ``Math.round``, so this writer and the
    reference store the same numbers (SPEC allows either tie rule). Exact for ``|x| < 2**52``."""
    x = np.asarray(x, dtype=np.float64)
    fl = np.floor(x)
    with np.errstate(invalid="ignore"):
        return fl + ((x - fl) >= 0.5)


def encode_linear16(values, scale, offset) -> np.ndarray:
    """``k = 1 + round((v - offset) / scale)`` clamped to 1..65535; 0 where ``v`` is not finite.
    Evaluated as the reference writer does: ``(v - offset) * (1 / scale)``, ties rounded up."""
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    with np.errstate(invalid="ignore", over="ignore"):
        k = round_half_up((v - offset) * (1 / scale))
        k += 1
        np.clip(k, 1, 65535, out=k)
    k[~finite] = 0
    return k.astype(np.uint16)


def decode_linear16(codes, scale, offset) -> np.ndarray:
    """``value = offset + scale * (k - 1)`` in float64, in exactly this order; NaN for code 0."""
    k = np.asarray(codes)
    out = k.astype(np.float64)
    out -= 1.0
    out *= scale
    out += offset
    out[k == 0] = np.nan
    return out


# ---------------------------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------------------------
def guard_ratio(scale, median_diff):
    """``scale / m``, or ``None`` when there is no neighbour variation (``m`` is 0 or unknown)."""
    if not median_diff > 0:
        return None
    return scale / median_diff


def guard_passes(ratios, threshold=GUARD_THRESHOLD) -> bool:
    return len(ratios) > 0 and all(r is not None and r < threshold for r in ratios)


def _fmt_ratio(r):
    return "no variation" if r is None else f"{r:.2g}"


def choose_encoding(accs, dtype, lossless, threshold=GUARD_THRESHOLD, what="pixels", channel_names=None):
    """``(encoding, note)``. Integer data is always ``exact``; float data is ``linear16`` when the
    author did not ask for lossless storage and every channel passes the guard."""
    dtype = np.dtype(dtype)
    if not is_float(dtype):
        return {"kind": "exact", "sourceDtype": dtype.name}, f"{what}: stored exactly ({dtype.name}): integer data always is."
    if lossless:
        return {"kind": "exact", "sourceDtype": dtype.name}, f"{what}: stored exactly ({dtype.name}), as asked (lossless=True)."
    params = [linear16_params(a.min if math.isfinite(a.min) else 0.0, a.max if math.isfinite(a.max) else 0.0)
              for a in accs]
    ratios = [guard_ratio(p["scale"], a.median_diff()) for p, a in zip(params, accs)]
    guard = {"stepOverNeighbour": ratios, "threshold": threshold}
    if guard_passes(ratios, threshold):
        worst = max(r for r in ratios)
        across = "each channel's range" if len(accs) > 1 else "its value range"
        note = (f"{what}: stored at 16 bits across {across} (linear16): the rounding step is at most "
                f"{worst:.2g} of the typical difference between neighbouring pixels (limit {threshold:g}), so "
                f"nothing visible or measurable is lost. Pass lossless=True to keep the {dtype.name} values exactly.")
        return {"kind": "linear16", "channels": params, "nodataCode": 0, "sourceDtype": dtype.name, "guard": guard}, note
    names = channel_names or [f"channel {i + 1}" for i in range(len(ratios))]
    failing = [f"{n} ({_fmt_ratio(r)})" for n, r in zip(names, ratios) if r is None or not r < threshold]
    note = (f"{what}: stored exactly ({dtype.name}); a 16-bit copy was refused because its rounding step would not be "
            f"small next to the difference between neighbouring pixels for {', '.join(failing)} (limit {threshold:g}). "
            "That usually means a few extreme values stretch the range (compare the channel's min / max with its "
            "1st-99.9th percentiles in the manifest), or the image is constant or piecewise constant. Nothing is "
            "lost: the values are kept exactly. Pass lossless=True to store floats exactly without this check.")
    return {"kind": "exact", "sourceDtype": dtype.name, "fallback": "guard", "guard": guard}, note


# ---------------------------------------------------------------------------------------------
# overviews
# ---------------------------------------------------------------------------------------------
def block_mean(values, f) -> np.ndarray:
    """Mean of the valid (non-NaN) values in each ``f`` x ``f`` block (fewer at the right and
    bottom edges); NaN where a block has none. ``values`` is float64 ``(H, W)``."""
    H, W = values.shape
    h, w = -(-H // f), -(-W // f)
    padded = np.full((h * f, w * f), np.nan)
    padded[:H, :W] = values
    blocks = padded.reshape(h, f, w, f)
    valid = ~np.isnan(blocks)
    count = valid.sum(axis=(1, 3))
    total = np.where(valid, blocks, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count
    out[count == 0] = np.nan
    return out
