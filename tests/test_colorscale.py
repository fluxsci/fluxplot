"""Colour scales as recipe controls: the A0 fixes of the colour-system plan.

What this file pins down: replaying the controls fluxplot itself recorded never breaks a rerun
(custom colormaps included), fluxplot's own map names work everywhere a colormap is accepted,
the house default map is really the default, a colour control arriving as a JSON string is
understood, and the control key names the series rather than the axes' position.
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
    assert set(rec_a["params"]["__fluxplot__"]) == set(rec_b["params"]["__fluxplot__"]) == {"m"}


def test_same_series_name_twice_gets_distinct_keys(tmp_path):
    fig, (a, b) = plt.subplots(1, 2)
    fp.heatmap(a, M, series="m")
    fp.heatmap(b, M, series="m")
    _res, man, rec = _save(fig, tmp_path)
    keys = [s["field"]["controlKey"] for s in man["series"]]
    assert keys == ["m", "axes.2.m"]
    assert set(rec["params"]["__fluxplot__"]) == {"m", "axes.2.m"}
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
