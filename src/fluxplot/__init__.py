"""FluxPlot — matplotlib, but every meaningful thing has a name.

A thin, additive semantic-tagging layer over matplotlib. Plot in real matplotlib (using the
``fp.*`` convenience helpers or by tagging raw artists), then :func:`save` emits a semantic SVG +
``*.fluxplot.json`` manifest + ``*.recipe.json``.

See ``Flux_SemanticSVG_Spec.md`` for the conceptual spec.
"""

from . import (
    colors,  # noqa: E402,F401  (canonical palette/colormaps — colors.green400, colors.maps.emerald, …)
    style,  # noqa: E402,F401  (house plotting style — fx.use_light(), fx.FLEXOKI, …)
)
from .api import (  # noqa: E402,F401
    annotation,
    area,
    bar,
    box,
    errorbar,
    hist,
    line,
    reference_line,
    save,
    scatter,
    significance_bracket,
    tag,
    tag_points,
    tag_seaborn,
    violin,
)
from .recipe import (
    params,  # noqa: E402,F401  (overridable tunables for rerun-plot/Regenerate)
)
from .style import (  # noqa: E402,F401  (re-exported for convenience)
    use_dark,
    use_light,
    use_paper,
)
from .version import SPEC_VERSION, __version__  # noqa: F401

__all__ = [
    "__version__",
    "SPEC_VERSION",
    "colors",
    "style",
    "use_light",
    "use_dark",
    "use_paper",
    "params",
    "line",
    "scatter",
    "bar",
    "errorbar",
    "area",
    "box",
    "violin",
    "hist",
    "tag",
    "tag_points",
    "tag_seaborn",
    "significance_bracket",
    "reference_line",
    "annotation",
    "save",
]
