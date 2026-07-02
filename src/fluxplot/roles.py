"""Role vocabulary v0 (spec §5, §10). A versioned core + a namespaced ``x-`` extension mechanism.

Unknown roles are NOT rejected — they degrade gracefully (P4): they still get a stable id and a
``data-role`` and are listed in the manifest; consumers that don't recognize them treat them as
opaque addressable groups.
"""
from __future__ import annotations

CORE_ROLES = frozenset(
    {
        # containers
        "figure",
        "panel",
        "plot-area",
        "legend",
        "colorbar",
        "title",
        "subtitle",
        # scaffold / guides
        "axis",
        "spine",
        "tick",
        "tick-label",
        "axis-title",
        "gridline",
        "background",
        # data marks (geoms)
        "series",
        "line",
        "point",
        "bar",
        "area",
        "errorbar",
        "box",
        "violin",
        "contour",
        # composite sub-parts (box / violin / errorbar internals)
        "whisker",
        "cap",
        "flier",
        "median",
        "segment",
        # overlays
        "annotation",
        "reference-line",
        "highlight-region",
        "significance-bracket",
        "label",
        "caption",
        # untagged user-drawn artists swept into addressable "extra" content
        "extra",
        # legend internals
        "legend-entry",
        "legend-swatch",
        "legend-label",
        # a manifest-only container that groups sibling parts (e.g. "all x tick labels")
        "group",
    }
)


def is_core(role: str) -> bool:
    return role in CORE_ROLES


def is_extension(role: str) -> bool:
    return role.startswith("x-")


def validate(role: str) -> str:
    """Return the role unchanged if it is a recognized core role or a well-formed ``x-`` extension.

    Unknown bare roles are allowed (graceful degradation) but normalized into the ``x-`` namespace
    so the core vocabulary stays closed and self-describing.
    """
    if is_core(role) or is_extension(role):
        return role
    return f"x-{role}"
