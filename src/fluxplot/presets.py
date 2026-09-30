"""Per-role animation preset hints (spec §8).

The generator has the most semantic knowledge, so it ships sensible *defaults* — the easy path
becomes the beautiful path (Flux Slide can produce an elegant build with zero hand-authoring). These
are hints only; a consumer may override everything.

Tier note (see SciForge_Stack_Decision §3.4 / style_principles): straight-edged marks animate via
compositor-friendly transforms (``grow-from-baseline`` = scaleY); only genuine curves use the
surgical ``draw-on`` (stroke-dashoffset).
"""
from __future__ import annotations

#: The closed vocabulary of build animations a preset may name. Flux maps each to a player
#: preset (``grow-from-baseline`` → ``growBaseline``, ``fade-rise`` → ``fadeRise``, …); a name
#: outside this tuple would silently fall back to a plain fade, so the schema enumerates it.
PRESET_NAMES = ("draw-on", "fade-in", "stagger-in", "grow-from-baseline", "fade-rise", "write-on", "pop-in")

#: What a ``stagger-in`` orders its members by: a data attribute every member carries
#: (``data-x`` / ``data-y`` / ``data-index`` / ``data-value`` / ``data-count``) or its category.
STAGGER_BY = ("x", "y", "index", "value", "count", "category")

ROLE_PRESETS = {
    "line": {"animation": "draw-on", "durationMs": 800},
    "point": {"animation": "stagger-in", "staggerBy": "index", "staggerMs": 40, "durationMs": 240},
    "bar": {"animation": "grow-from-baseline", "durationMs": 500},
    "area": {"animation": "fade-in", "durationMs": 500},
    "errorbar": {"animation": "fade-in", "durationMs": 300},
    "box": {"animation": "grow-from-baseline", "durationMs": 500},
    # composite statistics fade; filled bodies get no speculative draw-on (plan §4)
    "violin": {"animation": "fade-in", "durationMs": 500},
    "whisker": {"animation": "fade-in", "durationMs": 300},
    "cap": {"animation": "fade-in", "durationMs": 300},
    "median": {"animation": "fade-in", "durationMs": 300},
    "flier": {"animation": "fade-in", "durationMs": 240},
    "mean": {"animation": "fade-in", "durationMs": 300},
    "segment": {"animation": "fade-in", "durationMs": 300},
    "axis": {"animation": "draw-on", "durationMs": 400},
    "gridline": {"animation": "fade-in", "durationMs": 300},
    "legend": {"animation": "fade-rise", "durationMs": 300},
    "colorbar": {"animation": "fade-in", "durationMs": 300},
    "title": {"animation": "fade-in", "durationMs": 300},
    "annotation": {"animation": "fade-rise", "delayMs": 150, "durationMs": 300},
    "reference-line": {"animation": "draw-on", "durationMs": 400},
    "significance-bracket": {"animation": "fade-rise", "delayMs": 200, "durationMs": 300},
    "extra": {"animation": "fade-in", "durationMs": 400},
    # colour-mapped fields: a hexmatrix builds up from its emptiest to its fullest hexagon;
    # meshes, images and contour bands fade as one layer
    "x-hexbin": {"animation": "stagger-in", "staggerBy": "value", "staggerMs": 4, "durationMs": 240},
    "x-hex": {"animation": "stagger-in", "staggerBy": "value", "staggerMs": 4, "durationMs": 240},
    "x-heatmap": {"animation": "fade-in", "durationMs": 500},
    "cell": {"animation": "fade-in", "durationMs": 500},
    "x-contour": {"animation": "draw-on", "durationMs": 600},
    "x-contourf": {"animation": "fade-in", "durationMs": 500},
    "contour-level": {"animation": "fade-in", "durationMs": 500},
    # surface (brain) maps: every region is a filled part
    "surface": {"animation": "fade-in", "durationMs": 500},
    "surface-region": {"animation": "fade-in", "durationMs": 500},
    "scalebar": {"animation": "fade-in", "durationMs": 300},
}
assert all(v["animation"] in PRESET_NAMES for v in ROLE_PRESETS.values())
assert all(v.get("staggerBy", "x") in STAGGER_BY for v in ROLE_PRESETS.values())


def presets_for(roles) -> dict:
    """Return the preset map restricted to the roles actually present in a plot."""
    return {r: ROLE_PRESETS[r] for r in roles if r in ROLE_PRESETS}
