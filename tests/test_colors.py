"""fluxplot.colors is the canonical palette + colormap source, and style follows it."""
import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import Colormap

from fluxplot import colors as fx
from fluxplot import style


def test_module_level_palette_access():
    assert fx.green400 == "#35AB49"
    assert fx.paper == "#FFFCF0"
    assert fx.black == "#100F0F"
    assert fx.base500 == "#878580"
    assert fx.olive600 == "#66800B"  # the original flexoki green


def test_flex_get_returns_metadata():
    meta = fx.flex.get("green-300")
    assert meta["hex"] == "#44C55A"
    assert meta["l"] == 300
    assert fx.flex.get("green300") == meta  # attribute-style name works too
    assert "green-300" in fx.flex and "green300" in fx.flex


def test_unknown_color_raises():
    with pytest.raises(AttributeError):
        fx.notacolor123
    with pytest.raises(KeyError):
        fx.flex.get("notacolor-123")


def test_custom_maps_available_and_registered():
    import matplotlib as mpl

    for name in fx.maps.names("flexoki"):
        assert isinstance(getattr(fx.maps, name), Colormap)
        assert isinstance(mpl.colormaps[name], Colormap)  # cmap="flexoki_*" works


def test_cmasher_maps_via_attribute():
    pytest.importorskip("cmasher")
    assert isinstance(fx.maps.emerald, Colormap)
    with pytest.raises(AttributeError):
        fx.maps.not_a_real_map


def test_view_map_set():
    pytest.importorskip("cmasher")
    fig = fx.maps.view_map_set("cmasher")
    assert len(fig.axes) >= len(fx.maps.names("cmasher"))
    plt.close(fig)
    fig = fx.maps.view_map_set("flexoki")
    plt.close(fig)
    with pytest.raises(ValueError):
        fx.maps.view_map_set("nope")


def test_style_palette_comes_from_colors():
    # colors.py is canonical: green is the new green, olive is the original flexoki green
    assert style.FLEXOKI["green"] == fx.green600 == "#228833"
    assert style.FLEXOKI["olive"] == fx.olive600
    assert style.FLEXOKI["red"] == fx.red600
    assert style.FLEXOKI["blue2"] == fx.blue400
    assert style.FLEXOKI["base700"] == fx.base700
    assert style.FLEXOKI_DIVERGING is fx.maps.flexoki_diverging
