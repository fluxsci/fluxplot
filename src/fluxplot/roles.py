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
        "mean",
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


# Data-kind hints (text | line | shape | container) per core role — authored truth for
# consumers (Flux's part editors pick the property set by kind: a tick-label edits like a
# text object, a gridline like a line, …). Deliberate omission: "extra" is heterogeneous
# (a swept artist can be a line, a collection or a patch), so its kind is inferred from
# the concrete artist at sweep time (see ``descriptors.artist_kind``). Unknown / ``x-``
# roles are likewise inferred per-artist where possible, else the hint is omitted.
KIND_BY_ROLE = {
    # containers
    "figure": "container",
    "panel": "container",
    "plot-area": "container",
    "legend": "container",
    "colorbar": "container",
    "axis": "container",
    "series": "container",
    "group": "container",
    "legend-entry": "container",
    # text
    "title": "text",
    "subtitle": "text",
    "tick-label": "text",
    "axis-title": "text",
    "legend-label": "text",
    "annotation": "text",
    "label": "text",
    "caption": "text",
    # line
    "line": "line",
    "spine": "line",
    "tick": "line",
    "gridline": "line",
    "reference-line": "line",
    "errorbar": "line",
    "whisker": "line",
    "cap": "line",
    "median": "line",
    "mean": "line",
    "segment": "line",
    "significance-bracket": "line",
    # shape
    "area": "shape",
    "bar": "shape",
    "point": "shape",
    "box": "shape",
    "violin": "shape",
    "contour": "shape",
    "flier": "shape",
    "background": "shape",
    "highlight-region": "shape",
    "legend-swatch": "shape",
}


def kind_for_role(role: str):
    """The data-kind hint for a role, or ``None`` when the role carries no static kind."""
    return KIND_BY_ROLE.get(role)


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
