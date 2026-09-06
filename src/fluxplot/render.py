"""Deterministic SVG rendering (P5). Where byte-stability is won or lost.

See ``NOTES_matplotlib_svg.md`` for the verified determinism knobs.
"""
from __future__ import annotations

import io

import matplotlib.pyplot as plt

# rcParams that make matplotlib's SVG output deterministic + editable/addressable.
DETERMINISTIC_RCPARAMS = {
    # keep text as real <text> referencing fonts by name → editable, restyleable, addressable,
    # and font-version-independent (the 'path' default outlines glyphs into nondeterministic d's).
    "svg.fonttype": "none",
    "savefig.bbox": None,
    # path simplification is deterministic given a pinned threshold; pin both explicitly.
    "path.simplify": True,
    "path.simplify_threshold": 0.111111,
}


def render_svg(fig, hashsalt: str, dpi=None) -> bytes:
    """Render ``fig`` to SVG bytes deterministically.

    ``hashsalt`` fixes matplotlib's hashed ids (clip-paths, marker glyphs, ``<defs>``); without it
    they are salted with a fresh uuid4 each run. ``metadata={'Date': None}`` drops the timestamp.
    We never pass ``bbox_inches='tight'`` — it would crop/shift the viewBox *after* coordinate
    anchors were captured, invalidating the data↔pixel mapping.

    ``dpi`` sets the resolution of **rasterized artists only** (see ``raster.py``). SVG user
    space is points, so vector geometry is dpi-invariant — verified in
    ``NOTES_matplotlib_svg.md`` §6 and pinned by ``tests/test_rasterize.py``. Pass it only
    when something is actually rasterized, so ordinary saves keep matplotlib's default.
    """
    rc = dict(DETERMINISTIC_RCPARAMS)
    rc["svg.hashsalt"] = hashsalt
    buf = io.BytesIO()
    extra = {"dpi": dpi} if dpi else {}
    with plt.rc_context(rc):
        fig.savefig(buf, format="svg", metadata={"Date": None}, bbox_inches=None, **extra)
    return buf.getvalue()


from contextlib import contextmanager


@contextmanager
def final_layout(fig):
    """Lay out with the SVG renderer, then freeze it through capture and export.

    Agg and SVG have different font metrics. Public draw_without_rendering avoids
    rasterizing data just to obtain the correct text/layout metrics.
    """
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    canvas, dpi, engine = fig.canvas, fig.dpi, fig.get_layout_engine()
    try:
        FigureCanvasSVG(fig)
        fig.set_dpi(72)
        fig.draw_without_rendering()
        fig.set_layout_engine('none')
        yield
    finally:
        fig.set_layout_engine(engine)
        fig.set_dpi(dpi)
        fig.set_canvas(canvas)
