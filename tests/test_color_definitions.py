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
    assert ids == ["flexoki", "mpl", "crameri", "tol", "cmasher"]
    for c in d["collections"]:
        assert c["maps"], c["id"]
        names = [m["name"] for m in c["maps"]]
        assert len(names) == len(set(names)), f"duplicate names in {c['id']}"
        for m in c["maps"]:
            assert m["type"] in ("sequential", "diverging", "cyclic", "qualitative", "misc")
            assert not m["name"].endswith("_r"), "reversed variants are derived, never stored"
            assert all(HEX.match(h) for h in m["colors"]), m["name"]
            # colors IS the exact lookup table: N of a listed map's own colours, else 256 samples
            assert len(m["colors"]) == m["N"], m["name"]
            assert m["N"] == d["samples"] or m["N"] <= 128
            # one "discrete" threshold: classes are listed maps of at most DISCRETE_MAX colours
            assert bool(m.get("discrete")) == (m["N"] != d["samples"] and m["N"] <= fx.DISCRETE_MAX), m["name"]
            assert m.get("uniform") in (None, True, False)
    flexoki = next(c for c in d["collections"] if c["id"] == "flexoki")
    assert [m["name"] for m in flexoki["maps"]] == list(fx.FLEXOKI_MAP_ANCHORS)
    assert all(m["uniform"] is False for m in flexoki["maps"]), "linear ramps through palette anchors are not uniform"


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
    # a 100-colour category set is an exact listed map, but not a set of "classes" (DISCRETE_MAX = 32)
    cats = fx.maps.info("crameri.batlowS")
    assert cats["discrete"] is False and cats["N"] == 100 and isinstance(fx.maps.get("crameri.batlowS"), ListedColormap)
    assert fx.maps.info("viridis")["family"] == "perceptually uniform"
    assert fx.maps.info("flexoki_diverging")["collection"] == "flexoki"


def test_flexoki_house_maps_come_first_and_alias_the_shipped_collection():
    import numpy as np
    # the "flexoki" set is the shipped collection (qualified names) plus any user-registered map;
    # the historical bare names are aliases of the very same tables
    assert fx.maps.names("flexoki") == [f"flexoki.{n}" for n in fx.FLEXOKI_MAP_ANCHORS]
    assert fx.maps.get("flexoki_diverging") is fx.maps.flexoki_diverging
    shipped = fx.maps.get("flexoki.diverging")
    assert np.array_equal(shipped(np.arange(256)), fx.maps.flexoki_diverging(np.arange(256)))
    assert fx.maps.info("flexoki_diverging") == fx.maps.info("flexoki.diverging")
    assert fx.maps.info("flexoki_diverging")["uniform"] is False
    # register() keeps the reversed twin too
    assert fx.maps.get("flexoki_diverging_r") is fx.maps._custom["flexoki_diverging_r"]
    assert mpl.colormaps["flexoki_diverging_r"].name == "flexoki_diverging_r"
    # a map the user registers joins the set after the shipped ones
    from matplotlib.colors import LinearSegmentedColormap
    fx.maps.register(LinearSegmentedColormap.from_list("mine_test_map", ["#000000", "#ffffff"]))
    try:
        assert fx.maps.names("flexoki")[-1] == "mine_test_map"
    finally:
        fx.maps._custom.pop("mine_test_map", None); fx.maps._custom.pop("mine_test_map_r", None)


def test_info_reports_size_and_uniformity():
    assert fx.maps.info("viridis")["uniform"] is True and fx.maps.info("viridis")["N"] == 256
    assert fx.maps.info("crameri.batlow")["uniform"] is True
    cats = fx.maps.info("crameri.batlowS")
    assert cats["N"] == 100 and cats["discrete"] is False and cats["type"] == "qualitative"
    assert isinstance(fx.maps.get("crameri.batlowS"), ListedColormap) and fx.maps.get("crameri.batlowS").N == 100
    assert fx.maps.info("tab10")["discrete"] is True and fx.maps.info("tab10")["N"] == 10
    assert fx.maps.info("jet")["uniform"] is None  # not assessed


def test_truncate_and_bracket_names():
    import numpy as np
    t = fx.maps.truncate("batlow", 0.2, 0.8)
    assert t.name == "crameri.batlow[0.2:0.8]" and t.N == 256
    base = fx.maps.get("batlow")
    assert np.allclose(t(0.0), base(0.2)) and np.allclose(t(1.0), base(0.8))
    again = fx.maps.get("batlow[0.2:0.8]")
    assert again.name == t.name and np.array_equal(again(np.arange(256)), t(np.arange(256)))
    assert fx.maps.info("batlow[0.2:0.8]")["collection"] == "crameri"
    assert fx.maps.truncate(mpl.colormaps["viridis"], 0.5, 1.0, n=16).N == 16
    with pytest.raises(ValueError):
        fx.maps.truncate("viridis", 0.8, 0.2)
    with pytest.raises(KeyError):
        fx.maps.get("viridis[a:b]")


def test_discretize_builds_a_binned_scale():
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import BoundaryNorm, to_hex
    cmap, norm = fx.maps.discretize("viridis", [0, 2, 5, 10])
    assert isinstance(norm, BoundaryNorm) and cmap.N == 3 and list(norm.boundaries) == [0, 2, 5, 10]
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    assert to_hex(m.to_rgba(1)) != to_hex(m.to_rgba(3)) and to_hex(m.to_rgba(3)) == to_hex(m.to_rgba(4.9))
    both, bnorm = fx.maps.discretize("viridis", [0, 2, 5, 10], extend="both")
    assert both.N == 5 and bnorm.extend == "both"
    mb = cm.ScalarMappable(norm=bnorm, cmap=both)
    colours = [to_hex(mb.to_rgba(v)) for v in (-1, 1, 3, 7, 11)]
    assert len(set(colours)) == 5, "under, three bins and over are five distinct colours"
    assert colours[0] == to_hex(both(0)) and colours[-1] == to_hex(both(4))
    n_cmap, n_norm = fx.maps.discretize("viridis", n=4, vmin=0, vmax=8)
    assert list(n_norm.boundaries) == [0, 2, 4, 6, 8] and n_cmap.N == 4
    with pytest.raises(ValueError):
        fx.maps.discretize("viridis", n=4)


def test_single_source_of_truth_for_the_flexoki_palette():
    """The design-token export is canonical: colors.py's table and palettes.json derive from it."""
    tokens = _load("flexoki.tokens")
    flat = {}
    def walk(node):
        if isinstance(node, dict):
            if "$value" in node:
                return
            for k, v in node.items():
                if isinstance(v, dict) and "$value" in v and not k.endswith("-opacity-10"):
                    flat[k] = v["$value"]["hex"].upper()
                else:
                    walk(v)
    walk(tokens)
    table = {k: v["hex"] for k, v in fx._flex_data.items() if k not in ("base-0", "base-1000")}
    assert table == flat
    assert fx._flex_data["base-0"]["hex"] == fx.paper and fx._flex_data["base-1000"]["hex"] == fx.black
    assert fx.green600 == "#228833" and fx.olive600 == "#66800B"  # fluxplot's green is the tokens' GRN; Flexoki's green is olive
    flexoki = next(c for c in _load("palettes")["collections"] if c["id"] == "flexoki")
    swatches = {s["name"]: s["hex"].upper() for g in flexoki["groups"] for s in g["swatches"]}
    assert swatches == table


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
