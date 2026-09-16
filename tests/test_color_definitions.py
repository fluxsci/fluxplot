"""The shipped colour definitions: every collection loads from fluxplot's own JSON
(no upstream package imported), every map is addressable by name in matplotlib,
and the palette collections read back as hex lists."""
import json
import re
from importlib import resources

import matplotlib as mpl
import pytest
from matplotlib.colors import Colormap, ListedColormap

from fluxplot import colors as fx

HEX = re.compile(r"^#[0-9a-f]{6}$")


def _load(name):
    with resources.files("fluxplot").joinpath(f"definitions/{name}.json").open() as f:
        return json.load(f)


def test_colormap_definitions_are_well_formed():
    d = _load("colormaps")
    assert d["schema"] == "fluxplot.colormaps/1"
    ids = [c["id"] for c in d["collections"]]
    assert ids == ["mpl", "crameri", "tol", "cmasher"]
    for c in d["collections"]:
        assert c["maps"], c["id"]
        names = [m["name"] for m in c["maps"]]
        assert len(names) == len(set(names)), f"duplicate names in {c['id']}"
        for m in c["maps"]:
            assert m["type"] in ("sequential", "diverging", "cyclic", "qualitative", "misc")
            assert not m["name"].endswith("_r"), "reversed variants are derived, never stored"
            assert all(HEX.match(h) for h in m["colors"]), m["name"]
            assert len(m["colors"]) == (len(m["colors"]) if m.get("discrete") else d["samples"])


def test_every_shipped_map_is_registered_with_matplotlib():
    d = _load("colormaps")
    for c in d["collections"]:
        for m in c["maps"]:
            full = f"{c['id']}.{m['name']}"
            assert isinstance(mpl.colormaps[full], Colormap), full
            assert isinstance(mpl.colormaps[full + "_r"], Colormap), full + "_r"


def test_registry_resolves_qualified_bare_and_reversed_names():
    assert fx.maps.collections() == ["flexoki", "mpl", "crameri", "tol", "cmasher"]
    assert "batlow" in fx.maps.names("crameri")
    assert "viridis" in fx.maps.names("mpl") and "amber" in fx.maps.names("cmasher")
    batlow = fx.maps.get("crameri.batlow")
    assert batlow is fx.maps.get("batlow") is fx.maps.batlow
    rev = fx.maps.get("batlow_r")
    assert rev.name == "crameri.batlow_r"
    assert tuple(rev(0.0)) == tuple(batlow(1.0))
    # a bare name shared between collections resolves to matplotlib's first; the qualified name picks Tol's
    assert fx.maps.get("PRGn").name == "mpl.PRGn" and fx.maps.get("tol.PRGn").name == "tol.PRGn"
    assert fx.maps.get("cmr.amber").name == "cmasher.amber"  # the legacy cmasher prefix still works
    with pytest.raises(KeyError):
        fx.maps.get("nosuchmap")


def test_registry_info_and_discrete_maps():
    assert fx.maps.info("tol.sunset")["type"] == "diverging"
    assert fx.maps.info("crameri.batlowS")["discrete"] and isinstance(fx.maps.get("crameri.batlowS"), ListedColormap)
    assert fx.maps.info("viridis")["family"] == "perceptually uniform"
    assert fx.maps.info("flexoki_diverging")["collection"] == "flexoki"


def test_flexoki_custom_maps_still_come_first():
    assert fx.maps.names("flexoki") == sorted(fx.maps._custom)
    assert fx.maps.get("flexoki_diverging") is fx.maps.flexoki_diverging


def test_palette_collections():
    d = _load("palettes")
    assert d["schema"] == "fluxplot.palettes/1"
    assert fx.palettes.collections() == ["flexoki", "brewer", "tol"]
    blues = fx.palettes.get("brewer", "Blues")
    assert len(blues) == 9 and blues[0] == "#f7fbff" and blues[-1] == "#08306b"
    assert len(fx.palettes.get("brewer", "Spectral")) == 11 and len(fx.palettes.get("brewer", "Set1")) == 9
    bright = fx.palettes.get("tol", "bright")
    assert bright[0] == "#4477aa" and len(bright) == 7
    assert fx.palettes.flexoki["green"] and fx.palettes.brewer["YlGn"]
    assert "base" in fx.palettes.names("flexoki")
    for c in d["collections"]:
        for g in c["groups"]:
            assert all(HEX.match(s["hex"]) for s in g["swatches"]), (c["id"], g["name"])
    with pytest.raises(KeyError):
        fx.palettes.get("brewer", "nosuch")
