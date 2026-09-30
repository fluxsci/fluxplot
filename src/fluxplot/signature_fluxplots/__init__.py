"""Signature fluxplots — preset plot types that are unique to Flux.

Each one is a complete, opinionated plot built from ordinary matplotlib artists and tagged part by
part, so it saves, validates and round-trips through Flux like any hand-built FluxPlot. They take a
DataFrame plus column names, seaborn-style, and are re-exported at the top level (``fp.glowbar``).

* :func:`glowbar` — individual points beside a glowing interval bar with a mean line and a median
  notch; fixed per-unit lanes and colours, optional paired connectors.
* :func:`fluxbox` — the glowbar with a box plot for its summary: a slim translucent box (Q1–Q3)
  with a solid median line, a mean notch and capless whiskers.
* :func:`hexmatrix` — hexagonal binning (counts, densities, reductions of a third variable) or a 2D
  array on a hex lattice; every hexagon a named part, with marginals, colour key and log axes.
"""
from .fluxbox import FluxboxResult, fluxbox
from .hexmatrix import HexMatrixResult, hexmatrix
from .glowbar import GlowbarResult, even_shades, glowbar, interleaved_order

__all__ = ["glowbar", "GlowbarResult", "fluxbox", "FluxboxResult", "hexmatrix", "HexMatrixResult", "even_shades", "interleaved_order"]
