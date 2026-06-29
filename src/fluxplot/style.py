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
- **It's yours to tune:** edit the hexes/rcParams here and every future plot follows. (A future
  standalone ``flexoki``-style color package could absorb :data:`FLEXOKI` + the colormaps; for now
  this lives with fluxplot so it's available wherever you already plot.)
- Install the optional colormap dependency with ``pip install "fluxplot[style]"``.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

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
# Flexoki palette (https://stephango.com/flexoki). 600-weight accents read well
# on the light "paper" background; 400-weight ("*2") on dark. Edit here to retune.
# ---------------------------------------------------------------------------
FLEXOKI = {
    "paper": "#FFFCF0", "base50": "#F2F0E5", "base100": "#E6E4D9", "base150": "#DAD8CE",
    "base200": "#CECDC3", "base300": "#B7B5AC", "base400": "#9F9D96", "base500": "#878580",
    "base600": "#6F6E69", "base700": "#575653", "base800": "#403E3C", "base850": "#343331",
    "base900": "#282726", "base950": "#1C1B1A", "black": "#100F0F",
    # accents — 600 (primary, for light backgrounds)
    "red": "#AF3029", "orange": "#BC5215", "yellow": "#AD8301", "green": "#66800B",
    "cyan": "#24837B", "blue": "#205EA6", "purple": "#5E409D", "magenta": "#A02F6F",
    # accents — 400 (lighter, for dark backgrounds)
    "red2": "#D14D41", "orange2": "#DA702C", "yellow2": "#D0A215", "green2": "#879A39",
    "cyan2": "#3AA99F", "blue2": "#4385BE", "purple2": "#8B7EC8", "magenta2": "#CE5D97",
}

# Categorical cycles — a distinct, harmonious hue order. The active theme installs
# one as matplotlib's prop_cycle, so un-coloured series are assigned from it.
CYCLE_LIGHT = [FLEXOKI[c] for c in ("blue", "orange", "green", "purple", "cyan", "magenta", "yellow", "red")]
CYCLE_DARK = [FLEXOKI[c] for c in ("blue2", "orange2", "green2", "purple2", "cyan2", "magenta2", "yellow2", "red2")]


def _cmap(name, colors):
    return LinearSegmentedColormap.from_list(name, colors)


# ---------------------------------------------------------------------------
# Continuous colormaps. Defaults use cmasher's perceptually-uniform maps when it
# is installed (the scientifically honest choice for continuous data), and fall
# back to the Flexoki-flavoured maps otherwise. The Flexoki maps stay available
# by name regardless (FLEXOKI_*) — e.g. FLEXOKI_DIVERGING is light-centred
# (blue–paper–red), handy for correlation matrices.
# ---------------------------------------------------------------------------
FLEXOKI_SEQUENTIAL = _cmap("flexoki_sequential", ["#FFFCF0", "#A8C8E0", FLEXOKI["blue2"], FLEXOKI["blue"], "#163B66"])
FLEXOKI_WARM = _cmap("flexoki_warm", ["#FFFCF0", FLEXOKI["yellow2"], FLEXOKI["orange2"], FLEXOKI["red"], "#55201C"])
FLEXOKI_DIVERGING = _cmap("flexoki_diverging", [FLEXOKI["blue"], FLEXOKI["blue2"], "#FFFCF0", FLEXOKI["red2"], FLEXOKI["red"]])
TERRAIN = _cmap(
    "flexoki_terrain",
    ["#163B66", FLEXOKI["blue"], FLEXOKI["cyan2"], FLEXOKI["green2"], FLEXOKI["yellow2"], FLEXOKI["orange"], "#FFFCF0"],
)
SPECTRUM = _cmap(
    "flexoki_spectrum",
    [FLEXOKI[c] for c in ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta", "red")],
)

try:  # cmasher: perceptually-uniform continuous maps (the preferred default)
    import cmasher as cmr  # noqa: F401  (importing also registers "cmr.*" names in matplotlib)

    HAVE_CMASHER = True
    # Tweak these picks to taste — any cmasher map works (cmr.<name>):
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

# Make the Flexoki maps addressable by name too (so cmap="flexoki_diverging" works).
for _cm in (FLEXOKI_SEQUENTIAL, FLEXOKI_WARM, FLEXOKI_DIVERGING, TERRAIN, SPECTRUM):
    try:
        mpl.colormaps.register(_cm)
    except Exception:  # already registered / older mpl — best-effort
        pass

# Typography. We set the generic family + a fallback chain rather than a single
# face, so a missing font degrades silently to a sane default (no per-figure
# warnings). Lato (sans) / Latin Modern Roman (the LaTeX serif) are used if present.
SANS_STACK = ["Lato", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
SERIF_STACK = ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"]


def _base_rc(ink, muted, grid, paper, serif):
    return {
        "font.family": "serif" if serif else "sans-serif",
        "font.sans-serif": SANS_STACK,
        "font.serif": SERIF_STACK,
        "font.size": 11.5,
        "axes.titlesize": 15, "axes.titleweight": "medium", "axes.titlepad": 12,
        "axes.labelsize": 12.5, "axes.labelpad": 6,
        "axes.labelcolor": ink, "text.color": ink,
        "axes.edgecolor": muted, "axes.linewidth": 1.1,
        "axes.facecolor": paper, "figure.facecolor": paper, "savefig.facecolor": paper,
        "axes.grid": grid, "axes.axisbelow": True,
        "grid.color": FLEXOKI["base150"], "grid.linewidth": 0.8, "grid.alpha": 0.9,
        "xtick.color": muted, "ytick.color": muted,
        "xtick.labelcolor": ink, "ytick.labelcolor": ink,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 4.5, "ytick.major.size": 4.5, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "legend.frameon": False, "legend.fontsize": 10.5, "legend.handlelength": 1.6,
        "lines.linewidth": 2.3, "lines.markersize": 6.5, "lines.solid_capstyle": "round",
        "lines.markeredgewidth": 0.0, "lines.dash_capstyle": "round",
        "patch.linewidth": 0.0, "figure.dpi": 110, "savefig.dpi": 110,
        "figure.constrained_layout.use": True,
        "svg.fonttype": "none",  # keep text as text in the SVG (FluxPlot-friendly)
    }


def use_light(serif: bool = False, grid: bool = True) -> None:
    """Apply the paper-background theme (the default look). Call before creating figures."""
    mpl.rcParams.update(_base_rc(FLEXOKI["black"], FLEXOKI["base600"], grid, FLEXOKI["paper"], serif))
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=CYCLE_LIGHT)


def use_dark(serif: bool = False, grid: bool = False, bg: str = "#1C1B1A") -> None:
    """Apply the dark theme — for cosmic / nocturnal plots (star maps, attractors)."""
    ink, muted = FLEXOKI["base200"], FLEXOKI["base500"]
    rc = _base_rc(ink, muted, grid, bg, serif)
    rc.update({"grid.color": FLEXOKI["base800"], "grid.alpha": 0.6, "xtick.labelcolor": ink, "ytick.labelcolor": ink})
    mpl.rcParams.update(rc)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=CYCLE_DARK)


# ---------------------------------------------------------------------------
# small Tufte helpers
# ---------------------------------------------------------------------------
def despine(ax, top=True, right=True, left=False, bottom=False) -> None:
    """Hide chart-junk spines (Tufte-style). Defaults: drop top + right."""
    for side, off in (("top", top), ("right", right), ("left", left), ("bottom", bottom)):
        ax.spines[side].set_visible(not off)


def title(ax, main, sub=None) -> None:
    """A left-aligned title with an optional muted subtitle line."""
    if sub:
        ax.set_title(f"{main}\n", loc="left")
        ax.annotate(
            sub, xy=(0, 1.0), xycoords="axes fraction", xytext=(0, 10), textcoords="offset points",
            ha="left", va="bottom", fontsize=10.5, color=FLEXOKI["base500"],
        )
    else:
        ax.set_title(main, loc="left")
