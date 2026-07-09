"""fluxplot.style — a consistent, beautiful matplotlib house style (Flexoki + cmasher).

A thin, additive theming layer, in the same spirit as the rest of FluxPlot: you keep plotting in
ordinary matplotlib, and this gives every plot a coherent visual identity. It is **independent of
the semantic API** — use it for quick exploratory plots (``plt.plot``) just as happily as for the
publication figures you ``fp.save``. One import, and everything you make looks like it belongs to
the same set.

    from fluxplot import style as fx
    fx.use_light()                      # apply the theme (call before creating figures)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(x, y)                       # picks up the Flexoki color cycle automatically
    fx.despine(ax); fx.title(ax, "My result", "a short subtitle")

What you get:
- the **Flexoki** palette (Steph Ango, https://stephango.com/flexoki) as :data:`FLEXOKI`, plus
  light/dark categorical cycles installed as matplotlib's ``prop_cycle``;
- perceptually-uniform **continuous colormaps** via `cmasher
  <https://cmasher.readthedocs.io>`_ when installed (:data:`SEQUENTIAL`, :data:`DIVERGING`, …),
  with Flexoki-flavoured fallbacks so a script never breaks without cmasher;
- :func:`use_light` / :func:`use_dark` themes (clean, despined, sensible fonts/DPI);
- small helpers :func:`despine` and :func:`title`.

Notes
-----
- **Backend-agnostic:** this module never calls ``matplotlib.use(...)``, so it won't fight your
  notebook/interactive backend. For headless scripts set ``MPLBACKEND=Agg`` (or call
  ``matplotlib.use("Agg")``) yourself before importing pyplot.
- **It's yours to tune:** the rcParams live here, but every color definition comes from
  :mod:`fluxplot.colors` — the canonical palette/colormap module. Edit hexes *there* and every
  future plot (and this theme) follows.
- Install the optional colormap dependency with ``pip install "fluxplot[style]"``.
"""

from __future__ import annotations

import matplotlib as mpl

from .colors import flex as _flex, maps as _maps

__all__ = [
    "FLEXOKI",
    "CYCLE_LIGHT",
    "CYCLE_DARK",
    "SEQUENTIAL",
    "SEQUENTIAL_WARM",
    "DIVERGING",
    "CYCLIC",
    "FLEXOKI_SEQUENTIAL",
    "FLEXOKI_WARM",
    "FLEXOKI_DIVERGING",
    "TERRAIN",
    "SPECTRUM",
    "HAVE_CMASHER",
    "use_light",
    "use_dark",
    "despine",
    "title",
]

# ---------------------------------------------------------------------------
# Flexoki palette (https://stephango.com/flexoki), built from the canonical
# definitions in fluxplot.colors. 600-weight accents read well on the light
# "paper" background; 400-weight ("*2") on dark. Retune in fluxplot.colors.
# ---------------------------------------------------------------------------
_BASE_LEVELS = (50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 850, 900, 950)
_ACCENTS = ("red", "orange", "yellow", "olive", "green", "cyan", "blue", "purple", "magenta")

FLEXOKI = {"paper": _flex.paper, "black": _flex.black}
for _lvl in _BASE_LEVELS:
    FLEXOKI[f"base{_lvl}"] = _flex.get(f"base-{_lvl}")["hex"]
for _name in _ACCENTS:
    FLEXOKI[_name] = _flex.get(f"{_name}-600")["hex"]  # primary, for light backgrounds
    FLEXOKI[f"{_name}2"] = _flex.get(f"{_name}-400")["hex"]  # lighter, for dark backgrounds

# Categorical cycles — a distinct, harmonious hue order. The active theme installs
# one as matplotlib's prop_cycle, so un-coloured series are assigned from it.
CYCLE_LIGHT = [
    FLEXOKI[c]
    for c in ("blue", "orange", "green", "purple", "cyan", "magenta", "yellow", "red")
]
CYCLE_DARK = [
    FLEXOKI[c]
    for c in (
        "blue2",
        "orange2",
        "green2",
        "purple2",
        "cyan2",
        "magenta2",
        "yellow2",
        "red2",
    )
]


# ---------------------------------------------------------------------------
# Continuous colormaps. Defaults use cmasher's perceptually-uniform maps when it
# is installed (the scientifically honest choice for continuous data), and fall
# back to the Flexoki-flavoured maps otherwise. The Flexoki maps — defined in
# fluxplot.colors, addressable by name (cmap="flexoki_diverging") — stay
# available regardless (FLEXOKI_*); e.g. FLEXOKI_DIVERGING is light-centred
# (blue–paper–red), handy for correlation matrices.
# ---------------------------------------------------------------------------
FLEXOKI_SEQUENTIAL = _maps.flexoki_sequential
FLEXOKI_WARM = _maps.flexoki_warm
FLEXOKI_DIVERGING = _maps.flexoki_diverging
TERRAIN = _maps.flexoki_terrain
SPECTRUM = _maps.flexoki_spectrum

try:  # cmasher: perceptually-uniform continuous maps (the preferred default)
    import cmasher as cmr  # noqa: F401  (importing also registers "cmr.*" names in matplotlib)

    HAVE_CMASHER = True
    # Tweak these picks to taste — any cmasher map works (fx.maps.<name>):
    #   sequential: rainforest, ember, amber, gem, ocean, dusk, eclipse, …
    #   diverging:  fusion, iceburn, redshift, wildfire, pride, …
    #   cyclic:     infinity, emergence
    SEQUENTIAL = cmr.rainforest
    SEQUENTIAL_WARM = cmr.ember
    DIVERGING = cmr.fusion
    CYCLIC = cmr.infinity
except Exception:  # pragma: no cover - cmasher optional
    HAVE_CMASHER = False
    SEQUENTIAL = FLEXOKI_SEQUENTIAL
    SEQUENTIAL_WARM = FLEXOKI_WARM
    DIVERGING = FLEXOKI_DIVERGING
    CYCLIC = SPECTRUM

# Typography. We set the generic family + a fallback chain rather than a single
# face, so a missing font degrades silently to a sane default (no per-figure
# warnings). Arial (sans) / Georgia (serif) are used if present.
SANS_STACK = ["Arial", "Helvetica", "Lato", "Helvetica Neue", "DejaVu Sans"]
SERIF_STACK = [
    "Georgia",
    "Times New Roman",
    "Latin Modern Roman",
    "CMU Serif",
    "DejaVu Serif",
]


def _base_rc(ink, muted, grid, paper, serif):
    return {
        "font.family": "serif" if serif else "sans-serif",
        "font.sans-serif": SANS_STACK,
        "font.serif": SERIF_STACK,
        "font.size": 6,  # 6pt as default
        "axes.titlesize": 8,  # 8pt font for axes titles
        "axes.titleweight": "medium",
        "axes.titlepad": 12,
        "axes.labelsize": 8,  # 8pt font for axes labels
        "axes.labelpad": 6,
        "axes.labelcolor": ink,
        "text.color": ink,
        "axes.edgecolor": muted,
        "axes.linewidth": 1.2,  # 1.2 default linewidth for axes
        "axes.facecolor": paper,
        "figure.facecolor": paper,
        "savefig.facecolor": paper,
        "axes.grid": grid,
        "axes.axisbelow": True,
        "grid.color": FLEXOKI["base150"],
        "grid.linewidth": 0.8,
        "grid.alpha": 0.9,
        "xtick.color": muted,
        "ytick.color": muted,
        "xtick.labelcolor": ink,
        "ytick.labelcolor": ink,
        "xtick.labelsize": 5,
        "ytick.labelsize": 5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "legend.frameon": False,
        "legend.fontsize": 6,
        "legend.handlelength": 1.5,
        "lines.linewidth": 2.0,  # default linewidth
        "lines.markersize": 6.5,
        "lines.solid_capstyle": "round",
        "lines.markeredgewidth": 0.0,
        "lines.dash_capstyle": "round",
        "patch.linewidth": 0.0,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "figure.constrained_layout.use": True,
        "svg.fonttype": "none",  # keep text as text in the SVG (FluxPlot-friendly)
    }


def use_light(
    ink: str = FLEXOKI["black"],
    muted: str = FLEXOKI["base700"],
    grid: bool = True,
    paper: str = FLEXOKI["paper"],
    serif: bool = False,
) -> None:
    """Apply the paper-background theme (the default look). Call before creating figures."""
    mpl.rcParams.update(_base_rc(ink, muted, grid, paper, serif))
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=CYCLE_LIGHT)


def use_dark(serif: bool = False, grid: bool = False, bg: str = "#1C1B1A") -> None:
    """Apply the dark theme — for cosmic / nocturnal plots (star maps, attractors)."""
    ink, muted = FLEXOKI["base200"], FLEXOKI["base500"]
    rc = _base_rc(ink, muted, grid, bg, serif)
    rc.update(
        {
            "grid.color": FLEXOKI["base800"],
            "grid.alpha": 0.6,
            "xtick.labelcolor": ink,
            "ytick.labelcolor": ink,
        }
    )
    mpl.rcParams.update(rc)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=CYCLE_DARK)


# ---------------------------------------------------------------------------
# small Tufte helpers
# ---------------------------------------------------------------------------
def despine(ax, top=True, right=True, left=False, bottom=False) -> None:
    """Hide chart-junk spines (Tufte-style). Defaults: drop top + right."""
    for side, off in (
        ("top", top),
        ("right", right),
        ("left", left),
        ("bottom", bottom),
    ):
        ax.spines[side].set_visible(not off)


def title(ax, main, sub=None) -> None:
    """A left-aligned title with an optional muted subtitle line.

    The main title (a left title) is autotagged ``title`` by :func:`fluxplot.save`;
    the subtitle is tagged ``subtitle`` so it is addressable in the scene graph.
    """
    if sub:
        ax.set_title(f"{main}\n", loc="left")
        ann = ax.annotate(
            sub,
            xy=(0, 1.0),
            xycoords="axes fraction",
            xytext=(0, 10),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=10.5,
            color=FLEXOKI["base500"],
        )
        from . import api as _api  # lazy: style is imported during api's __init__

        _api.tag(ann, role="subtitle", name="figure", text=sub)
    else:
        ax.set_title(main, loc="left")
