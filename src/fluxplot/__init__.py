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
    tag_seaborn,
    significance_bracket,
    reference_line,
    annotation,
    save,
)

from . import colors  # noqa: E402,F401  (canonical palette/colormaps — colors.green400, colors.maps.emerald, …)
from . import style  # noqa: E402,F401  (house plotting style — fx.use_light(), fx.FLEXOKI, …)
from .style import use_light, use_dark  # noqa: E402,F401  (re-exported for convenience)
from .recipe import params  # noqa: E402,F401  (overridable tunables for rerun-plot/Regenerate)

__all__ = [
    "__version__",
    "SPEC_VERSION",
    "colors",
    "style",
    "use_light",
    "use_dark",
    "params",
    "line",
    "scatter",
    "bar",
    "errorbar",
    "area",
    "tag",
    "tag_points",
    "tag_seaborn",
    "significance_bracket",
    "reference_line",
    "annotation",
    "save",
]
