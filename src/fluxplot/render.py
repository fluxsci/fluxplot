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
    # path simplification is deterministic given a pinned threshold; pin both explicitly.
    "path.simplify": True,
    "path.simplify_threshold": 0.111111,
}


def render_svg(fig, hashsalt: str) -> bytes:
    """Render ``fig`` to SVG bytes deterministically.

    ``hashsalt`` fixes matplotlib's hashed ids (clip-paths, marker glyphs, ``<defs>``); without it
    they are salted with a fresh uuid4 each run. ``metadata={'Date': None}`` drops the timestamp.
    We never pass ``bbox_inches='tight'`` — it would crop/shift the viewBox *after* coordinate
    anchors were captured, invalidating the data↔pixel mapping.
    """
    rc = dict(DETERMINISTIC_RCPARAMS)
    rc["svg.hashsalt"] = hashsalt
    buf = io.BytesIO()
    with plt.rc_context(rc):
        fig.savefig(buf, format="svg", metadata={"Date": None}, bbox_inches=None)
    return buf.getvalue()
