"""fluxplot.style applies a coherent house theme without fighting the active backend."""
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.colors import Colormap

from fluxplot import style as fx


def test_palette_and_cycles():
    for k in ("paper", "black", "blue", "red", "green", "blue2"):
        assert fx.FLEXOKI[k].startswith("#")
    assert len(fx.CYCLE_LIGHT) == 8
    assert len(fx.CYCLE_DARK) == 8


def test_colormaps_are_colormaps_and_registered():
    for cm in (fx.SEQUENTIAL, fx.SEQUENTIAL_WARM, fx.DIVERGING, fx.CYCLIC, fx.FLEXOKI_DIVERGING, fx.TERRAIN):
        assert isinstance(cm, Colormap)
    # the Flexoki maps are addressable by name
    assert isinstance(mpl.colormaps["flexoki_diverging"], Colormap)


def test_use_light_sets_paper_and_light_cycle():
    fx.use_light()
    # use_light defaults to PURE WHITE (journal compliance, 2026-07); the warm
    # Flexoki-paper look lives in use_paper() now.
    assert mcolors.to_hex(mpl.rcParams["axes.facecolor"]).lower() == "#ffffff"
    first = next(iter(mpl.rcParams["axes.prop_cycle"]))["color"]
    assert mcolors.to_hex(first).lower() == fx.FLEXOKI["blue"].lower()


def test_use_paper_sets_flexoki_paper():
    fx.use_paper()
    assert mcolors.to_hex(mpl.rcParams["axes.facecolor"]).lower() == fx.FLEXOKI["paper"].lower()
    first = next(iter(mpl.rcParams["axes.prop_cycle"]))["color"]
    assert mcolors.to_hex(first).lower() == fx.FLEXOKI["blue"].lower()
    fx.use_light()  # restore the default for any later tests


def test_use_dark_sets_dark_cycle():
    fx.use_dark()
    first = next(iter(mpl.rcParams["axes.prop_cycle"]))["color"]
    assert mcolors.to_hex(first).lower() == fx.FLEXOKI["blue2"].lower()
    fx.use_light()  # restore the default for any later tests


def test_despine_defaults():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fx.despine(ax)
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert ax.spines["left"].get_visible()
    assert ax.spines["bottom"].get_visible()
    plt.close(fig)
