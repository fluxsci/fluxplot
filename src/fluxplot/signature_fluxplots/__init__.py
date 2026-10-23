"""Signature plots with named, editable parts and shared categorical controls."""
from .fluxbox import FluxboxResult, fluxbox
from .meanmark import MeanmarkResult, meanmark
from .hexmatrix import HexMatrixResult, hexmatrix
from ._colour import even_shades, interleaved_order

__all__ = ["meanmark", "MeanmarkResult", "fluxbox", "FluxboxResult", "hexmatrix", "HexMatrixResult", "even_shades", "interleaved_order"]
