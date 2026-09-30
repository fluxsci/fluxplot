"""Semantic series colours and cross-figure consistency (B2): every series records its primary
paint (and the palette token / cycle slot that names it), categories keep one colour across
figures through ``fp.colors.categories`` and its project file, and the palette and per-series
colours are recipe controls."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex, to_rgba  # noqa: E402

import fluxplot as fp  # noqa: E402
from fluxplot import colors as fx  # noqa: E402
from fluxplot import style  # noqa: E402
from fluxplot._fieldmap import categorical_colors  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no project colours file anywhere above
    monkeypatch.delenv("FLUXPLOT_COLORS", raising=False)
    fx.categories.reset()
    style.use_light()
    yield
    plt.close("all")
    fx.categories.reset()
    mpl.rcdefaults()
    style.use_light()


def _table():
    return {"type": ["SD", "S", "SD", "S", "SD", "S"], "APP": [1.0, 2.0, 1.5, 2.5, 1.2, 2.2],
            "subject": ["a", "b", "c", "d", "e", "f"]}


def _save(fig, tmp_path, name="p"):
    fp.save(fig, str(tmp_path / f"{name}.svg"), recipe=False)
    return json.loads((tmp_path / f"{name}.fluxplot.json").read_text())


def test_series_record_their_colour_token_and_cycle_slot(tmp_path):
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [0, 1], series="green", color=fx.green400)
    fp.line(ax, [0, 1], [1, 0], series="cycled")  # the theme's first cycle colour
    fp.bar(ax, [0, 1], [1, 2], series="bars", color=fx.palettes.get("tol", "bright")[2], alpha=0.5)
    fp.scatter(ax, [0, 1], [2, 2], series="dots", color="#123456")
    man = _save(fig, tmp_path)
    colour = {s["id"]: s["color"] for s in man["series"]}
    assert colour["green"] == {"hex": "#35ab49", "alpha": 1.0, "token": "flexoki.green-400"}
    assert colour["cycled"]["token"] == "flexoki.blue-600" and colour["cycled"]["palette"] == {"name": "flexoki.light", "index": 0}
    # Tol's bright green IS Flexoki's green-600 (fluxplot's custom green came from it): Flexoki names first
    assert colour["bars"]["token"] == "flexoki.green-600" and colour["bars"]["alpha"] == 0.5
    assert fx.token_of(fx.palettes.get("tol", "bright")[0]) == "tol.bright.blue"
    assert colour["dots"] == {"hex": "#123456", "alpha": 1.0}


def test_category_colours_do_not_depend_on_which_categories_are_present():
    assert categorical_colors(["b", "c"])["b"] == categorical_colors(["a", "b", "c"])["b"]
    first = categorical_colors(["b", "c"])
    assert first["b"] == to_hex(style.CYCLE_LIGHT[0]).lower() and first["c"] == to_hex(style.CYCLE_LIGHT[1]).lower()
    assert categorical_colors(["a", "b", "c"])["a"] == to_hex(style.CYCLE_LIGHT[2]).lower()  # the next free slot
    assert categorical_colors(["b"], palette={"b": "#000000"})["b"] == "#000000"  # an explicit palette still wins


def test_project_colours_file_pins_glowbar_groups(tmp_path, monkeypatch):
    path = tmp_path / "fluxplot.colors.json"
    path.write_text(json.dumps({"spec": "fluxplot/colors", "version": 1,
                                "categories": {"SD": "#bc5215"}, "palette": "flexoki"}))
    fx.categories.reset()  # the next use auto-discovers the file in the working directory
    fig, ax = plt.subplots()
    gb = fp.glowbar(_table(), x="type", y="APP", units="subject", ax=ax)
    assert gb.group_colors["SD"] == pytest.approx(to_rgba("#bc5215"))
    assert gb.group_colors["S"] != gb.group_colors["SD"]
    man = _save(fig, tmp_path)
    sd = next(s for s in man["series"] if s["id"] == "sd")
    assert sd["glowbar"]["groupColor"] == "#bc5215" and sd["glowbar"]["palette"] == "#bc5215"
    s_ = next(s for s in man["series"] if s["id"] == "s")
    assert s_["glowbar"]["palette"] == "YlOrRd"  # the default map for the second category, recorded
    assert fx.categories.path == str(path)
    # nothing is ever written implicitly; save() is explicit
    fx.categories.assign({"Sleep": fx.blue600})
    assert json.loads(path.read_text())["categories"] == {"SD": "#bc5215"}
    fx.categories.save()
    assert json.loads(path.read_text())["categories"] == {"SD": "#bc5215", "Sleep": "#205ea6"}
    monkeypatch.setenv("FLUXPLOT_COLORS", str(path))
    fx.categories.reset()
    assert fx.categories.get("Sleep") == "#205ea6"


def test_pinned_categories_recolour_seaborn_hue_series(tmp_path):
    import pandas as pd
    import seaborn as sns
    fx.categories.assign({"Male": "#00ff00"})
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"day": rng.choice(list("ab"), 40), "sex": rng.choice(["Male", "Female"], 40), "v": rng.normal(size=40)})
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="day", y="v", hue="sex", errorbar=None, ax=ax)
    fp.tag_seaborn(ax, plot="barplot")
    man = _save(fig, tmp_path)
    colour = {s["id"]: s["color"]["hex"] for s in man["series"]}
    assert colour["male"] == "#00ff00" and colour["female"] != "#00ff00"


def test_recipe_series_colours_and_palette(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": {
        "series": {"ctl": {"color": "#00ff00"}, "sd": {"color": "#112233"}}, "palette": "tol.bright"}}))
    style.use_light()
    cycle = [c["color"] for c in mpl.rcParams["axes.prop_cycle"]]
    assert [to_hex(c) for c in cycle] == fx.palettes.get("tol", "bright")
    fig, (a, b) = plt.subplots(1, 2)
    ln = fp.line(a, [0, 1], [0, 1], series="ctl", color="red")
    assert to_hex(ln.get_color()) == "#00ff00"  # the recipe's colour wins over the script's
    other = fp.line(a, [0, 1], [1, 0], series="other")
    assert to_hex(other.get_color()) == fx.palettes.get("tol", "bright")[0]  # …and the cycle is the palette
    gb = fp.glowbar(_table(), x="type", y="APP", units="subject", ax=b)
    assert gb.group_colors["SD"] == pytest.approx(to_rgba("#112233"))
    man = _save(fig, tmp_path)
    assert next(s for s in man["series"] if s["id"] == "panel.a.ctl")["color"]["hex"] == "#00ff00"
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": {"series": "nope"}}))
    with pytest.raises(ValueError, match="__fluxplot__.series"):
        fp.line(a, [0, 1], [0, 1], series="ctl")


def test_auto_series_colours_from_the_registry():
    fx.categories.auto_series = True
    fx.categories.assign({"treatment": "#bc5215"})
    fig, ax = plt.subplots()
    ln = fp.line(ax, [0, 1], [0, 1], series="treatment")
    pts = fp.scatter(ax, [0, 1], [0, 1], series="control")
    assert to_hex(ln.get_color()) == "#bc5215"
    assert to_hex(pts.get_facecolor()[0]) == to_hex(style.CYCLE_LIGHT[0]).lower()  # the next free slot
    explicit = fp.line(ax, [0, 1], [2, 2], series="treatment", color="k")
    assert explicit.get_color() == "k"  # a colour the call gives stands


def test_palette_colours_resolve_house_and_shipped_palettes():
    assert fx.palette_colors("flexoki") == list(style.CYCLE_LIGHT)
    assert fx.palette_colors("flexoki.dark") == list(style.CYCLE_DARK)
    assert fx.palette_colors("brewer.Set2") == fx.palettes.get("brewer", "Set2")
    assert fx.palette_colors("bright") == fx.palettes.get("tol", "bright")
    with pytest.raises(KeyError):
        fx.palette_colors("no.such")
    assert fx.token_of("#35AB49") == "flexoki.green-400" and fx.token_of("#123456") is None
