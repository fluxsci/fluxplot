"""zstd for fluximg shards: the first available of ``compression.zstd`` (Python 3.14+), ``zstandard``,
``numcodecs`` and ``imagecodecs``. Nothing is imported until a bundle is written or read."""
from __future__ import annotations

import threading

__all__ = ["Codec", "codec", "MissingZstdError"]

INSTALL_HINT = ("fluximg stores pixels zstd-compressed and no zstd library is installed. "
                "Install one with  pip install 'fluxplot[images]'  (or  pip install zstandard), "
                "or use Python 3.14+, which has compression.zstd built in.")


class MissingZstdError(ImportError):
    """No zstd implementation is importable."""


class Codec:
    """``compress(raw, level) -> bytes`` (no checksum, content size in the frame) and
    ``decompress(frame, size) -> bytes``. Safe to call from several threads at once."""

    def __init__(self, name, compress, decompress):
        self.name = name
        self.compress = compress
        self.decompress = decompress

    def __repr__(self):
        return f"Codec({self.name!r})"


def _stdlib():
    from compression import zstd  # Python 3.14+

    def compress(raw, level):
        return zstd.compress(raw, level=level)

    def decompress(frame, size):
        return zstd.decompress(frame)

    return Codec("compression.zstd", compress, decompress)


def _zstandard():
    import zstandard

    local = threading.local()  # a ZstdCompressor is not safe to share between threads

    def compressor(level):
        cache = getattr(local, "c", None)
        if cache is None:
            cache = local.c = {}
        c = cache.get(level)
        if c is None:
            c = cache[level] = zstandard.ZstdCompressor(level=level, write_checksum=False, write_content_size=True)
        return c

    def compress(raw, level):
        out = compressor(level).compress(raw)
        # zstandard returns a bytes object that keeps its compress-bound allocation (about the
        # input size); copy it to its real size so buffered shards cost what they weigh.
        return bytes(memoryview(out))

    def decompress(frame, size):
        d = getattr(local, "d", None)
        if d is None:
            d = local.d = zstandard.ZstdDecompressor()
        return d.decompress(frame, max_output_size=size)

    return Codec("zstandard", compress, decompress)


def _numcodecs():
    import numcodecs

    def compress(raw, level):
        return bytes(numcodecs.Zstd(level=level).encode(raw))

    def decompress(frame, size):
        return bytes(numcodecs.Zstd().decode(frame))

    return Codec("numcodecs", compress, decompress)


def _imagecodecs():
    import imagecodecs

    def compress(raw, level):
        return bytes(imagecodecs.zstd_encode(raw, level=level))

    def decompress(frame, size):
        return bytes(imagecodecs.zstd_decode(frame))

    return Codec("imagecodecs", compress, decompress)


_FACTORIES = (_stdlib, _zstandard, _numcodecs, _imagecodecs)
_cached: Codec | None = None


def codec() -> Codec:
    """The process-wide zstd codec; raises :class:`MissingZstdError` with install advice."""
    global _cached
    if _cached is None:
        for factory in _FACTORIES:
            try:
                _cached = factory()
                break
            except ImportError:
                continue
        else:
            raise MissingZstdError(INSTALL_HINT)
    return _cached
