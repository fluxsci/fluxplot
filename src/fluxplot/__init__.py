"""FluxPlot — matplotlib, but every meaningful thing has a name.

A thin, additive semantic-tagging layer over matplotlib. Plot in real matplotlib (using the
``fp.*`` convenience helpers or by tagging raw artists), then :func:`save` emits a semantic SVG +
``*.fluxplot.json`` manifest + ``*.recipe.json``.

See ``Flux_SemanticSVG_Spec.md`` for the conceptual spec.
"""

from .version import SPEC_VERSION, __version__  # noqa: F401

from .api import (  # noqa: E402,F401
    line,
    scatter,
    bar,
    errorbar,
    area,
    tag,
    tag_points,
    significance_bracket,
    reference_line,
    annotation,
    save,
)

__all__ = [
    "__version__",
    "SPEC_VERSION",
    "line",
    "scatter",
    "bar",
    "errorbar",
    "area",
    "tag",
    "tag_points",
    "significance_bracket",
    "reference_line",
    "annotation",
    "save",
]
