"""Per-role animation preset hints (spec §8).

The generator has the most semantic knowledge, so it ships sensible *defaults* — the easy path
becomes the beautiful path (Flux Slide can produce an elegant build with zero hand-authoring). These
are hints only; a consumer may override everything.

Tier note (see SciForge_Stack_Decision §3.4 / style_principles): straight-edged marks animate via
compositor-friendly transforms (``grow-from-baseline`` = scaleY); only genuine curves use the
surgical ``draw-on`` (stroke-dashoffset).
"""
from __future__ import annotations

ROLE_PRESETS = {
    "line": {"animation": "draw-on", "durationMs": 800},
    "point": {"animation": "stagger-in", "staggerMs": 40, "durationMs": 240},
    "bar": {"animation": "grow-from-baseline", "durationMs": 500},
    "area": {"animation": "fade-in", "durationMs": 500},
    "errorbar": {"animation": "fade-in", "durationMs": 300},
    "box": {"animation": "grow-from-baseline", "durationMs": 500},
    "axis": {"animation": "draw-on", "durationMs": 400},
    "gridline": {"animation": "fade-in", "durationMs": 300},
    "legend": {"animation": "fade-rise", "durationMs": 300},
    "annotation": {"animation": "fade-rise", "delayMs": 150, "durationMs": 300},
    "reference-line": {"animation": "draw-on", "durationMs": 400},
    "significance-bracket": {"animation": "fade-rise", "delayMs": 200, "durationMs": 300},
    "extra": {"animation": "fade-in", "durationMs": 400},
}


def presets_for(roles) -> dict:
    """Return the preset map restricted to the roles actually present in a plot."""
    return {r: ROLE_PRESETS[r] for r in roles if r in ROLE_PRESETS}
