"""FluxPlot — matplotlib, but every meaningful thing has a name.

A thin, additive semantic-tagging layer over matplotlib. Plot in real matplotlib (using the
``fp.*`` convenience helpers or by tagging raw artists), then :func:`save` emits a semantic SVG +
``*.fluxplot.json`` manifest + ``*.recipe.json``.

See ``Flux_SemanticSVG_Spec.md`` for the conceptual spec.
"""

from . import (
    colorcheck,  # noqa: E402,F401  (accessibility lint — colorcheck.check_palette, simulate, contrast, …)
    colors,  # noqa: E402,F401  (canonical palette/colormaps — colors.green400, colors.maps.emerald, …)
    colorscale,  # noqa: E402,F401  (the portable colour-scale law — colorscale.apply(record, values))
    stats,  # noqa: E402,F401  (tests behind the plots — stats.welch_hedges(a, b), …)
    style,  # noqa: E402,F401  (house plotting style — fx.use_light(), fx.FLEXOKI, …)
)
from .api import (  # noqa: E402,F401
    annotation,
    area,
    band,
    bar,
    barh,
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
from .surface import (  # noqa: E402,F401
    surface,
)
from .signature_fluxplots import (  # noqa: E402,F401  (preset plot types unique to Flux)
    fluxbox,
    glowbar,
    hexmatrix,
)
from .style import (  # noqa: E402,F401  (re-exported for convenience)
    use_dark,
    use_light,
    use_paper,
)
from .scene3d import Scene3D, scene3d, SCENE3D_SPEC_VERSION
from .mesh3d import mesh3d, can_morph
from .surface3d import surface3d
from .fields import heatmap, contour, contourf, colorbar, color_scale
from .brackets import brackets
from .panels import panel
from .version import SPEC_VERSION, __version__  # noqa: F401

__all__ = [
    "__version__",
    "Scene3D", "scene3d", "mesh3d", "surface3d", "can_morph", "SCENE3D_SPEC_VERSION",
    "surface",
    "glowbar",
    "fluxbox",
    "hexmatrix",
    "panel",
    "heatmap",
    "contour",
    "contourf",
    "colorbar",
    "color_scale",
    "brackets",
    "SPEC_VERSION",
    "colors",
    "colorcheck",
    "colorscale",
    "stats",
    "style",
    "use_light",
    "use_dark",
    "use_paper",
    "params",
    "line",
    "scatter",
    "bar",
    "barh",
    "errorbar",
    "area",
    "band",
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
