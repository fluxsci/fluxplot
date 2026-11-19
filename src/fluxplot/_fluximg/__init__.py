"""fluximg/1: first-class N-dimensional scientific images for Flux.

:func:`fluximg` / :func:`imsave` write a bundle (``<stem>.fluximg/``: a sharded Zarr v3 pixel
array, layers, regions and ``fluximg.json``); :func:`imread` reads one back. The on-disk contract
is fluximg/1 (Flux's ``docs/fluximg/SPEC.md``).
"""
from .reader import FluximgFile, FluximgFormatError, imread, read_manifest
from .writer import FluxImage, SaveReport, canonical_dims, fluximg, imsave
from ._zstd import MissingZstdError

__all__ = ["fluximg", "imsave", "imread", "FluxImage", "SaveReport", "FluximgFile", "FluximgFormatError",
           "MissingZstdError", "read_manifest", "canonical_dims"]
