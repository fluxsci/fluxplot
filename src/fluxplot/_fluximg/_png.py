"""A minimal standard-library PNG encoder (8-bit RGB / RGBA) for ``preview.png``."""
from __future__ import annotations

import struct
import zlib

import numpy as np

__all__ = ["encode_png"]


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode_png(pixels) -> bytes:
    """PNG bytes of a ``(H, W, 3)`` or ``(H, W, 4)`` uint8 array."""
    a = np.ascontiguousarray(pixels, dtype=np.uint8)
    if a.ndim != 3 or a.shape[2] not in (3, 4) or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError(f"encode_png needs (H, W, 3|4) pixels, got {a.shape}")
    h, w, n = a.shape
    rows = np.zeros((h, 1 + w * n), dtype=np.uint8)   # filter byte 0 (None) on every row
    rows[:, 1:] = a.reshape(h, w * n)
    header = struct.pack(">IIBBBBB", w, h, 8, 6 if n == 4 else 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
            + _chunk(b"IEND", b""))
