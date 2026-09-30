"""Colour scales: the recipe controls (A0, A5) and the portable colorScales contract (A1, A2, A3, A6).

What this file pins down: replaying the controls fluxplot itself recorded never breaks a rerun
(custom colormaps included), fluxplot's own map names work everywhere a colormap is accepted,
the house default map is really the default, a colour control arriving as a JSON string is
understood, and the control key names the series rather than the axes' position. Then the
contract: every colour-mapped mark records the exact lookup table and norm matplotlib painted
with, the reference law reproduces matplotlib hex for hex, every coloured element carries its
value, colour keys are exact vector gradients with data anchors, raw mappables get anonymous
scales, and every v2 control round-trips.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

import fluxplot as fp  # noqa: E402
from fluxplot import style as fx  # noqa: E402
from fluxplot._fieldmap import resolve_colormap  # noqa: E402


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _save(fig, tmp_path, name="p"):
    res = fp.save(fig, str(tmp_path / f"{name}.svg"), recipe=False)
    man = json.loads((tmp_path / f"{name}.fluxplot.json").read_text())
    rec = json.loads((tmp_path / f"{name}.recipe.json").read_text())
    return res, man, rec


M = np.arange(12, dtype=float).reshape(3, 4)


# ---- A0.1: custom colormaps survive a regeneration --------------------------------------------
def test_recorded_controls_replay_a_custom_colormap(tmp_path, monkeypatch):
    custom = ListedColormap(["#112233", "#445566", "#778899"])
    fig, ax = plt.subplots()
    fp.heatmap(ax, M, series="m", cmap=custom)
    _res, man, rec = _save(fig, tmp_path)
    controls = rec["params"]["__fluxplot__"]
    assert man["series"][0]["field"]["cmap"] == custom.name == "unnamed"  # not a resolvable name
    # Flux replays exactly what was recorded → the script's own object is kept, no error
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": controls}))
    fig, ax = plt.subplots()
    image = fp.heatmap(ax, M, series="m", cmap=custom)
    assert image.get_cmap() is custom
    assert np.array_equal(image.get_cmap()(np.arange(3)), custom(np.arange(3)))
    # an edited colormap still wins
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": {"m": {"cmap": "magma"}}}))
    fig, ax = plt.subplots()
    assert fp.heatmap(ax, M, series="m", cmap=custom).get_cmap().name == "magma"


def test_unknown_colormap_override_names_the_key(monkeypatch):
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": {"m": {"cmap": "no-such-map"}}}))
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="colour control 'm': unknown colormap 'no-such-map'"):
        fp.heatmap(ax, M, series="m")


# ---- A0.2: fluxplot map names work in heatmap / contour -----------------------------------------
@pytest.mark.parametrize("name", ["emerald", "crameri.batlow", "batlow", "viridis_r",
                                  "flexoki_diverging", "flexoki_diverging_r"])
@pytest.mark.parametrize("helper", ["heatmap", "contourf", "contour"])
def test_fluxplot_map_names_resolve_everywhere(tmp_path, name, helper):
    fig, ax = plt.subplots()
    if helper == "heatmap":
        fp.heatmap(ax, M, series="f", cmap=name)
    else:
        getattr(fp, helper)(ax, M, series="f", cmap=name)
    _res, man, _rec = _save(fig, tmp_path)
    assert man["series"][0]["field"]["cmap"] == resolve_colormap(name).name


# ---- A0.3: the house default colormap is applied -----------------------------------------------
@pytest.mark.parametrize("theme", ["use_light", "use_lighttable", "use_paper", "use_dark"])
def test_house_default_colormap(tmp_path, theme):
    try:
        getattr(fx, theme)()
        assert mpl.rcParams["image.cmap"] == fx.SEQUENTIAL.name == "cmasher.rainforest"
        assert fx.DEFAULT_DIVERGING == fx.DIVERGING.name
        fig, ax = plt.subplots()
        fp.heatmap(ax, M, series="m")
        _res, man, _rec = _save(fig, tmp_path)
        assert man["series"][0]["field"]["cmap"] == fx.SEQUENTIAL.name
    finally:
        mpl.rcdefaults()
        fx.use_light()


# ---- A0.5: a __fluxplot__ block arriving as a string ---------------------------------------------
def test_string_controls_are_parsed(monkeypatch):
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": json.dumps({"m": {"cmap": "magma"}})}))
    assert fp.params()["__fluxplot__"] == {"m": {"cmap": "magma"}}
    fig, ax = plt.subplots()
    assert fp.heatmap(ax, M, series="m").get_cmap().name == "magma"
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": "{not json"}))
    with pytest.raises(ValueError, match="__fluxplot__ is not valid JSON"):
        fp.params()


# ---- A0.6: control keys name the series, not the axes' position -------------------------------
def test_control_key_does_not_depend_on_axes_order(tmp_path):
    fig, ax = plt.subplots()
    fp.heatmap(ax, M, series="m")
    _res, man_a, rec_a = _save(fig, tmp_path, "a")
    fig, (_extra, ax) = plt.subplots(1, 2)  # a subplot added before the heatmap
    fp.heatmap(ax, M, series="m")
    _res, man_b, rec_b = _save(fig, tmp_path, "b")
    assert man_a["series"][0]["field"]["controlKey"] == "m"
    assert next(s for s in man_b["series"] if s["name"] == "m")["field"]["controlKey"] == "m"
    scales = lambda rec: set(rec["params"]["__fluxplot__"]) - {"theme"}  # noqa: E731  (the theme rides beside)
    assert scales(rec_a) == scales(rec_b) == {"m"}


def test_same_series_name_twice_gets_distinct_keys(tmp_path):
    fig, (a, b) = plt.subplots(1, 2)
    fp.heatmap(a, M, series="m")
    fp.heatmap(b, M, series="m")
    _res, man, rec = _save(fig, tmp_path)
    keys = [s["field"]["controlKey"] for s in man["series"]]
    assert keys == ["m", "axes.2.m"]
    assert set(rec["params"]["__fluxplot__"]) - {"theme"} == {"m", "axes.2.m"}
    fig, (a, b) = plt.subplots(1, 2)
    fp.panel(b, "right")
    fp.heatmap(a, M, series="m")
    fp.heatmap(b, M, series="m")
    _res, man, _rec = _save(fig, tmp_path, "named")
    assert sorted(s["field"]["controlKey"] for s in man["series"]) == ["m", "panel.right.m"]


def test_legacy_positional_override_still_applies(monkeypatch):
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": {"axes.1.m": {"cmap": "magma", "vmax": 40}}}))
    fig, ax = plt.subplots()
    image = fp.heatmap(ax, M, series="m")
    assert image.get_cmap().name == "magma" and image.norm.vmax == 40
    # …but the series key wins when both are present
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": {"axes.1.m": {"cmap": "magma"},
                                                                   "m": {"cmap": "plasma"}}}))
    fig, ax = plt.subplots()
    assert fp.heatmap(ax, M, series="m").get_cmap().name == "plasma"


# =================================================================================================
# A1 — the colorScales contract and the reference law
# =================================================================================================
import subprocess  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

from matplotlib import cm  # noqa: E402
from matplotlib import colors as mc  # noqa: E402
from lxml import etree  # noqa: E402

from fluxplot import colorscale as cs  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "colorscale_vectors.json"
SVG = "{http://www.w3.org/2000/svg}"


def _save_all(fig, tmp_path, name="p"):
    res = fp.save(fig, str(tmp_path / f"{name}.svg"), recipe=False)
    man = json.loads((tmp_path / f"{name}.fluxplot.json").read_text())
    rec = json.loads((tmp_path / f"{name}.recipe.json").read_text())
    root = etree.parse(str(tmp_path / f"{name}.svg")).getroot()
    return res, man, rec, root


def _el(root, gid):
    return next(el for el in root.iter() if el.get("id") == gid)


def _fill(el):
    """The painted fill: matplotlib's SVG writer omits a black fill (the SVG default)."""
    for decl in (el.get("style") or "").split(";"):
        k, _, v = decl.partition(":")
        if k.strip() == "fill":
            return v.strip()
    return "#000000"


def test_reference_law_matches_matplotlib_for_every_norm_kind():
    rng = np.random.default_rng(5)
    listed = ListedColormap(["#ff0000", "#00ff00", "#0000ff", "#000000"])
    norms = [mc.Normalize(0, 10), mc.Normalize(0, 10, clip=True), mc.LogNorm(1, 100),
             mc.SymLogNorm(2, vmin=-50, vmax=50), mc.PowerNorm(0.5, 0, 10), mc.TwoSlopeNorm(3, 0, 10),
             mc.CenteredNorm(2, 5), mc.BoundaryNorm([0, 2, 5, 10], 256, extend="both"),
             mc.BoundaryNorm([0, 2, 5, 10], 4), mc.BoundaryNorm([0, 2, 5, 10], 256, extend="max")]
    for norm in norms:
        for cmap in ("viridis", listed):
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            v = np.concatenate([rng.uniform(-60, 120, 1000), [0, 1, 10, 100, -50, 50, 3, 2, 5, np.nan, -1e-12, 10 + 1e-12]])
            ref = [mc.to_hex(c, keep_alpha=True) for c in mappable.to_rgba(np.ma.masked_invalid(v))]
            assert cs.apply(cs.scale_record("k", mappable), v) == ref, (type(norm).__name__, getattr(cmap, "name", cmap))


def test_parity_vectors_fixture_is_current():
    doc = json.loads(FIXTURE.read_text())
    assert doc["spec"] == "fluxplot/colorscale-vectors" and doc["cases"]
    for case in doc["cases"]:
        values = [np.nan if v is None else v for v in case["values"]]
        assert cs.apply(case["record"], values) == case["expected"], case["name"]
    kinds = {c["record"]["norm"]["kind"] for c in doc["cases"]}
    assert {"linear", "log", "symlog", "power", "twoslope", "centered", "boundary"} <= kinds


def test_colormap_record_is_the_exact_lut():
    rec = cs.colormap_record(mpl.colormaps["viridis"])
    assert rec["N"] == 256 and len(rec["lut"]) == 256 and rec["source"] == "matplotlib" and not rec["discrete"]
    assert rec["lut"][0] == "#440154ff" and rec["lut"][-1] == "#fde725ff" and rec["bad"] == "#00000000"
    custom = ListedColormap(["#112233", "#445566"]); custom.set_under("#000000"); custom.set_over("#ffffff")
    rec = cs.colormap_record(custom)
    assert rec == {"name": "unnamed", "source": "custom", "N": 2, "lut": ["#112233ff", "#445566ff"],
                   "under": "#000000ff", "over": "#ffffffff", "bad": "#00000000", "discrete": True}
    assert cs.colormap_record(fp.colors.maps.get("crameri.batlow"))["source"] == "crameri"
    assert cs.colormap_record(fp.colors.maps.flexoki_diverging)["source"] == "fluxplot"
    big = cs.colormap_record(mpl.colormaps["viridis"].resampled(4096))
    assert big["N"] == 1024 and big["approximate"] is True
    from fluxplot.signature_fluxplots.hexmatrix import _mono_cmap
    mono = _mono_cmap("#4cb391")
    assert cs.colormap_record(mono)["lut"] == [mc.to_hex(c, keep_alpha=True) for c in mono(np.arange(mono.N))]


def test_heatmap_scale_record_and_aliases(tmp_path):
    fig, ax = plt.subplots(figsize=(3, 2.4))
    im = fp.heatmap(ax, M, series="m", cells=True, norm=mc.LogNorm(1, 50))
    fp.colorbar(im, label="Rate")
    res, man, rec, root = _save_all(fig, tmp_path)
    (scale,) = man["colorScales"]
    assert scale["id"] == "m" and scale["kind"] == "continuous" and scale["recolor"] == "live"
    assert scale["norm"]["kind"] == "log" and scale["norm"]["base"] == 10 and scale["norm"]["extend"] == "neither"
    assert len(scale["colormap"]["lut"]) == 256 and scale["mappables"] == ["m.x-heatmap"]
    assert scale["colorbars"] == ["colorbar.color"] and scale["label"] == "Rate"
    assert scale["editable"] == {"cmap": True, "limits": True, "normKinds": ["linear", "log", "power", "symlog"], "center": False}
    field = man["series"][0]["field"]
    assert field["colorScale"] == "m" and field["cmap"] == scale["colormap"]["name"] and field["normalization"]["kind"] == "LogNorm"
    assert man["series"][0]["color"] == {"scale": "m"}
    key = next(g for g in man["guides"] if g["role"] == "colorbar")
    assert key["colorScale"] == "m"
    # imshow is a raster: the scale needs a regeneration
    fig, ax = plt.subplots()
    fp.heatmap(ax, M, series="m")
    _res, man, _rec, _root = _save_all(fig, tmp_path, "img")
    assert man["colorScales"][0]["recolor"] == "regenerate"


# =================================================================================================
# A2 — every coloured element carries its value
# =================================================================================================
def test_cells_carry_values_and_the_group_its_scale(tmp_path):
    data = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    fig, ax = plt.subplots()
    fp.heatmap(ax, data, series="m", cells=True)
    _res, man, _rec, root = _save_all(fig, tmp_path)
    group = _el(root, "m.x-heatmap")
    assert group.get("data-color-scale") == "m" and group.get("data-paint") == "fill"
    assert float(_el(root, "m.x-heatmap.cell.1.2").get("data-value")) == 6.0
    missing = _el(root, "m.x-heatmap.cell.0.1")
    assert missing.get("data-missing") == "1" and missing.get("data-value") is None
    # the recorded law reproduces every painted cell
    scale = man["colorScales"][0]
    cells = [el for el in root.iter() if el.get("data-role") == "cell" and el.get("data-value")]
    painted = [_fill(el) for el in cells]
    assert [c[:7] for c in cs.apply(scale, [float(el.get("data-value")) for el in cells])] == painted


def test_hexmatrix_bins_name_their_hexagons(tmp_path):
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(figsize=(3, 3))
    hm = fp.hexmatrix(x=rng.normal(size=200), y=rng.normal(size=200), ax=ax, gridsize=6, series="h", edgecolor="face")
    _res, man, _rec, root = _save_all(fig, tmp_path)
    s = man["series"][0]
    bins = s["hexmatrix"]["bins"]
    assert all(b["svgId"] == hm.hex_id(b["row"], b["col"]) for b in bins)
    group = _el(root, "h.hexes")
    assert group.get("data-color-scale") == "h" and group.get("data-paint") == "fill stroke"
    scale = man["colorScales"][0]
    assert scale["mappables"] == ["h.hexes"] and scale["label"] == "Count per hexbin"
    for b in bins[:5]:
        hex_el = _el(root, b["svgId"])
        assert float(hex_el.get("data-value")) == b["value"]
        assert _fill(hex_el) == cs.apply(scale, [b["value"]])[0][:7]


def test_contour_bands_and_lines_carry_their_levels(tmp_path):
    fig, ax = plt.subplots()
    x, y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
    cf = fp.contourf(ax, x, y, x * x + y * y, series="e", levels=[1, 3, 5], extend="both")
    fp.contour(ax, x, y, x * x + y * y, series="l", levels=[1, 3, 5])
    _res, man, _rec, root = _save_all(fig, tmp_path)
    bands = [el for el in root.iter() if el.get("data-series") == "e" and el.get("data-role") == "contour-level"]
    assert [(b.get("data-level-low"), b.get("data-level-high")) for b in bands] == [("-inf", "1.0"), ("1.0", "3.0"), ("3.0", "5.0"), ("5.0", "inf")]
    scale = next(s for s in man["colorScales"] if s["id"] == "e")
    # matplotlib colours each band by its layer value, an extend band by a far stand-in that
    # resolves to under / over: the recorded data-value reproduces exactly that
    painted = [_fill(b) for b in bands]
    assert [c[:7] for c in cs.apply(scale, [float(b.get("data-value")) for b in bands])] == painted
    assert [mc.to_hex(c) for c in cf.to_rgba(cf.cvalues)] == painted
    assert scale["norm"]["extend"] == "both" and _el(root, "e.x-contourf").get("data-paint") == "fill"
    lines = [el for el in root.iter() if el.get("data-series") == "l" and el.get("data-role") == "contour-level"]
    assert [float(l.get("data-value")) for l in lines] == [1.0, 3.0, 5.0]
    assert _el(root, "l.x-contour").get("data-paint") == "stroke"


def test_value_raster_travels_beside_the_svg(tmp_path):
    fig, ax = plt.subplots()
    fp.heatmap(ax, [[1.0, np.nan], [3.0, 4.0]], series="m", value_raster=True)
    res, man, _rec, _root = _save_all(fig, tmp_path)
    scale = man["colorScales"][0]
    assert scale["recolor"] == "raster" and scale["valueRaster"] == "p.m.values.json"
    payload = json.loads((tmp_path / "p.m.values.json").read_text())
    assert payload == {"spec": "fluxplot/values", "scale": "m", "shape": [2, 2], "values": [1.0, None, 3.0, 4.0]}


# =================================================================================================
# A3 — coverage: scatter c=, raw mappables, seaborn fields
# =================================================================================================
def test_scatter_c_is_a_colour_scale(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    x, y, c = rng.normal(size=30), rng.normal(size=30), rng.uniform(size=30)
    fig, ax = plt.subplots()
    coll = fp.scatter(ax, x, y, c=c, s=rng.uniform(10, 60, 30), series="pts", cmap="magma")
    fp.colorbar(coll)
    res, man, rec, root = _save_all(fig, tmp_path)
    assert res.warnings == []
    (scale,) = man["colorScales"]
    assert scale["id"] == "pts" and scale["mappables"] == ["pts.points"] and scale["colorbars"] == ["colorbar.color"]
    s = man["series"][0]
    assert s["color"] == {"scale": "pts"} and s["field"]["kind"] == "scatter" and s["field"]["controlKey"] == "pts"
    for k in (0, 7, 29):
        assert float(_el(root, f"pts.point.{k}").get("data-value")) == c[k]
    assert _el(root, "pts.points").get("data-paint") == "fill stroke"
    assert rec["params"]["__fluxplot__"]["pts"]["cmap"] == "magma"
    # an edited vmax applies on rerun
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": {"pts": {"vmax": 0.5}}}))
    fig, ax = plt.subplots()
    coll = fp.scatter(ax, x, y, c=c, series="pts", cmap="magma")
    assert coll.norm.vmax == 0.5
    # colour names are not values
    fig, ax = plt.subplots()
    plain = fp.scatter(ax, [1, 2, 3], [1, 2, 3], c=["r", "g", "b"], series="rgb")
    assert plain.get_array() is None
    _res, man, _rec, _root = _save_all(fig, tmp_path, "rgb")
    assert "colorScales" not in man


def test_raw_mappables_get_anonymous_scales(tmp_path):
    fig, (a, b) = plt.subplots(1, 2)
    fp.panel(a, "img"); fp.panel(b, "mesh")  # imshow shrinks its axes, which would reorder the letters
    raw = a.imshow(np.arange(16.0).reshape(4, 4))
    fig.colorbar(raw, ax=a)
    b.pcolormesh(np.arange(6.0).reshape(2, 3))
    b.scatter([0, 1, 2], [0, 1, 2], c=[0.1, 0.5, 0.9])
    _res, man, _rec, root = _save_all(fig, tmp_path)
    scales = {s["id"]: s for s in man["colorScales"]}
    assert set(scales) == {"panel.img.extra.image.0", "panel.mesh.extra.collection.0", "panel.mesh.extra.collection.1"}
    assert scales["panel.img.extra.image.0"]["recolor"] == "regenerate" and scales["panel.img.extra.image.0"]["colorbars"] == ["panel.img.colorbar.color"]
    assert scales["panel.mesh.extra.collection.0"]["recolor"] == "live"
    assert _el(root, "panel.mesh.extra.collection.1").get("data-color-scale") == "panel.mesh.extra.collection.1"
    assert man["series"] == []  # scales, not series


def test_seaborn_heatmap_is_a_field(tmp_path):
    import pandas as pd
    import seaborn as sns
    fig, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(np.arange(12.0).reshape(3, 4)), ax=ax)
    tagged = fp.tag_seaborn(ax, plot="heatmap")
    assert tagged == {"heatmap": ["x-heatmap"]}
    res, man, _rec, root = _save_all(fig, tmp_path)
    (s,) = man["series"]
    assert s["field"]["kind"] == "heatmap" and s["field"]["shape"] == [3, 4] and s["kind"] == "heatmap"
    assert "heatmap.x-heatmap.cell.2.3" in set(root.xpath("//@id"))
    (scale,) = man["colorScales"]
    assert scale["mappables"] == ["heatmap.x-heatmap"] and scale["colorbars"] == ["colorbar.color"]
    fig, ax = plt.subplots()
    rng = np.random.default_rng(1)
    sns.kdeplot(x=rng.normal(size=200), y=rng.normal(size=200), fill=True, ax=ax)
    assert fp.tag_seaborn(ax, plot="kdeplot") == {"heatmap": ["contourf"]}


# =================================================================================================
# A6 — colour keys are exact vector gradients with data anchors
# =================================================================================================
def test_colorbar_is_a_vector_gradient_with_anchors(tmp_path):
    fig, ax = plt.subplots(figsize=(3, 2.4))
    im = fp.heatmap(ax, M, series="m", norm=mc.LogNorm(1, 50), cells=True)
    fp.colorbar(im, extend="both")
    res, man, _rec, root = _save_all(fig, tmp_path)
    assert res.warnings == [] and res.rasterized == []
    assert not root.xpath("//*[local-name()='image']")  # neither the cells nor the key are rasters
    (grad,) = root.findall(f".//{SVG}linearGradient")
    assert grad.get("id") == "colorbar.color.solids.gradient" and len(grad) == 2 * 256
    solids = _el(root, "colorbar.color.solids")
    (rect,) = list(solids)
    assert rect.tag == f"{SVG}rect" and rect.get("style") == "fill: url(#colorbar.color.solids.gradient); stroke: none"
    stops = [(float(st.get("offset")), st.get("stop-color")) for st in grad]
    lut = man["colorScales"][0]["colormap"]["lut"]
    assert [c for _, c in stops[::2]] == [h[:7] for h in lut] and stops[0][0] == 0.0 and stops[-1][0] == 1.0
    assert stops[1][0] == stops[2][0]  # hard steps: an entry ends where the next begins
    key = next(g for g in man["guides"] if g["role"] == "colorbar")
    assert key["orientation"] == "vertical" and key["tickLocator"] == "log" and key["tickFormatter"] == "log"
    assert key["extend"] == "both" and set(key["extendParts"]) == {"min", "max"}
    assert _el(root, key["extendParts"]["min"]).get("data-role") == "colorbar-extend"
    lo, hi = key["anchors"]
    assert (lo["value"], hi["value"]) == (1.0, 50.0)
    y0, h = float(rect.get("y")), float(rect.get("height"))
    assert abs(lo["svg"] - (y0 + h)) < 0.01 and abs(hi["svg"] - y0) < 0.01
    assert abs(key["axisLength"] - h) < 0.01
    # the gradient runs along the axis between the anchors, in user space
    assert grad.get("gradientUnits") == "userSpaceOnUse"
    assert abs(float(grad.get("y1")) - lo["svg"]) < 1e-9 and abs(float(grad.get("y2")) - hi["svg"]) < 1e-9


def test_twoslope_key_records_its_centre_anchor(tmp_path):
    fig, ax = plt.subplots()
    im = fp.heatmap(ax, M - 5, series="d", norm=mc.TwoSlopeNorm(0, -5, 6), cmap="RdBu_r")
    fp.colorbar(im)
    _res, man, _rec, _root = _save_all(fig, tmp_path)
    key = next(g for g in man["guides"] if g["role"] == "colorbar")
    assert [a["value"] for a in key["anchors"]] == [-5.0, 0.0, 6.0]
    scale = man["colorScales"][0]
    assert scale["norm"]["kind"] == "twoslope" and scale["norm"]["vcenter"] == 0.0 and scale["editable"]["center"] is True


@pytest.mark.skipif(shutil.which("rsvg-convert") is None, reason="rsvg-convert not installed")
@pytest.mark.parametrize("norm,extend", [(None, "neither"), (mc.LogNorm(1, 50), "both")])
def test_vector_colorbar_renders_like_matplotlib(tmp_path, norm, extend):
    import io
    from PIL import Image
    fig, ax = plt.subplots(figsize=(3, 2.4))
    im = fp.heatmap(ax, np.random.default_rng(0).uniform(1, 50, (4, 5)), series="m", norm=norm)
    fp.colorbar(im, extend=extend)
    _res, _man, _rec, root = _save_all(fig, tmp_path)
    scale = 4
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72 * scale)
    ref = np.asarray(Image.open(buf).convert("RGB"), dtype=float)
    png = subprocess.run(["rsvg-convert", "-w", str(ref.shape[1]), "-h", str(ref.shape[0]), str(tmp_path / "p.svg")],
                         capture_output=True, check=True).stdout
    mine = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=float)
    assert mine.shape == ref.shape
    rect = next(el for el in root.iter() if el.tag == f"{SVG}rect" and "gradient" in (el.get("style") or ""))
    x0, y0, w, h = (float(rect.get(k)) for k in ("x", "y", "width", "height"))
    region = (slice(int((y0 + 1) * scale), int((y0 + h - 1) * scale)), slice(int((x0 + 1) * scale), int((x0 + w - 1) * scale)))
    diff = np.abs(mine[region] - ref[region])
    assert diff.mean() < 6 and np.percentile(diff, 99) < 40, (diff.mean(), np.percentile(diff, 99))


# =================================================================================================
# A5 — colour controls v2 round-trip
# =================================================================================================
def _rerun(monkeypatch, controls, **kw):
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": controls}))
    fig, ax = plt.subplots()
    image = fp.heatmap(ax, M, series="m", **kw)
    return image


@pytest.mark.parametrize("override", [
    {"norm": {"kind": "log"}, "vmin": 1, "vmax": 20},
    {"norm": {"kind": "symlog", "linthresh": 2, "linscale": 1.5}, "vmin": -5, "vmax": 11},
    {"norm": {"kind": "power", "gamma": 0.5}},
    {"norm": {"kind": "twoslope", "vcenter": 4}, "vmin": 0, "vmax": 11},
    {"norm": {"kind": "centered", "vcenter": 5}, "vmin": 0, "vmax": 11},
    {"cmap": "plasma", "reversed": True},
    {"cmap": {"lut": ["#112233", "#445566", "#778899"], "under": "#000000", "over": "#ffffff", "bad": "#ff00ff"}},
    {"extend": "max"},
])
def test_v2_controls_round_trip(tmp_path, monkeypatch, override):
    image = _rerun(monkeypatch, {"m": override})
    fp.colorbar(image)
    res, man, rec, _root = _save_all(image.axes.figure, tmp_path)
    state = rec["params"]["__fluxplot__"]["m"]
    scale = man["colorScales"][0]
    if "norm" in override:
        assert state["norm"]["kind"] == override["norm"]["kind"] == scale["norm"]["kind"]
        for k, v in override["norm"].items():
            if k != "kind":
                assert state["norm"][k] == v
    for k in ("vmin", "vmax"):
        if k in override and override.get("norm", {}).get("kind") != "centered":
            assert state[k] == override[k] == scale["norm"][k]
    if override.get("norm", {}).get("kind") == "centered":  # symmetric about the centre by construction
        assert state["vmax"] - 5 == 5 - state["vmin"] == 6
    if override.get("cmap") == "plasma":
        assert state["cmap"] == "plasma_r" and scale["colormap"]["name"] == "plasma_r"
    if isinstance(override.get("cmap"), dict):
        assert state["cmap"]["lut"] == ["#112233ff", "#445566ff", "#778899ff"] and state["cmap"]["over"] == "#ffffffff"
        assert scale["colormap"]["source"] == "custom" and scale["colormap"]["N"] == 3
    if "extend" in override:
        assert state["extend"] == "max" == scale["norm"]["extend"]
        key = next(g for g in man["guides"] if g["role"] == "colorbar")
        assert key["extend"] == "max" and set(key["extendParts"]) == {"max"}
    # replaying the written state is a fixed point: the same state comes back
    again = _rerun(monkeypatch, rec["params"]["__fluxplot__"])
    fp.colorbar(again)
    _res, _man, rec2, _root = _save_all(again.axes.figure, tmp_path, "again")
    assert rec2["params"]["__fluxplot__"]["m"] == state


def test_v2_controls_keep_the_scripts_objects_when_unchanged(monkeypatch):
    norm = mc.LogNorm(1, 50)
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": {"m": {"norm": {"kind": "log"}, "vmin": 1, "vmax": 50, "cmap": "viridis"}}}))
    fig, ax = plt.subplots()
    image = fp.heatmap(ax, M, series="m", norm=norm, cmap="viridis")
    assert type(image.norm) is mc.LogNorm and image.norm is not norm  # copied, never mutated
    assert (image.norm.vmin, image.norm.vmax) == (1, 50) and image.get_cmap().name == "viridis"


@pytest.mark.parametrize("override,message", [
    ({"norm": {"kind": "log"}, "vmin": 0, "vmax": 5}, "log norm needs vmin > 0"),
    ({"norm": {"kind": "twoslope", "vcenter": 8}, "vmin": 0, "vmax": 5}, "twoslope needs vcenter < vmax"),
    ({"norm": {"kind": "twoslope", "vcenter": -1}, "vmin": 0, "vmax": 5}, "twoslope needs vmin < vcenter"),
    ({"norm": {"kind": "spiral"}}, "cannot build a 'spiral' norm"),
    ({"extend": "sideways"}, "extend must be"),
    ({"cmap": {"lut": []}}, "non-empty 'lut'"),
])
def test_invalid_controls_name_the_key(monkeypatch, override, message):
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": {"m": override}}))
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match=f"colour control 'm'.*{message}"):
        fp.heatmap(ax, M, series="m")
