"""A minimal standard-library PNG encoder (8-bit gray, gray + alpha, RGB, RGBA) for ``preview.png``."""
from __future__ import annotations

import struct
import zlib

import numpy as np

__all__ = ["encode_png"]


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


#: PNG colour type by samples per pixel: gray, gray + alpha, RGB, RGBA
_COLOUR_TYPE = {1: 0, 2: 4, 3: 2, 4: 6}


def encode_png(pixels) -> bytes:
    """PNG bytes of a uint8 array ``(H, W)`` (gray), ``(H, W, 2)`` (gray + alpha), ``(H, W, 3)`` (RGB)
    or ``(H, W, 4)`` (RGBA)."""
    a = np.ascontiguousarray(pixels, dtype=np.uint8)
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3 or a.shape[2] not in _COLOUR_TYPE or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError(f"encode_png needs (H, W) or (H, W, 2|3|4) pixels, got {a.shape}")
    h, w, n = a.shape
    rows = np.zeros((h, 1 + w * n), dtype=np.uint8)   # filter byte 0 (None) on every row
    rows[:, 1:] = a.reshape(h, w * n)
    header = struct.pack(">IIBBBBB", w, h, 8, _COLOUR_TYPE[n], 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
            + _chunk(b"IEND", b""))
