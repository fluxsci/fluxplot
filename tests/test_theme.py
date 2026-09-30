"""Theme-able plots (B1): the manifest records the theme and its tokens, every scaffold element
says which token painted it, the grounds are parts, ``theme_vars`` rewrites paints to CSS
variables, and a recipe can ask for another theme."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from lxml import etree  # noqa: E402

import fluxplot as fp  # noqa: E402
from fluxplot import style as fx  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_theme():
    yield
    plt.close("all")
    mpl.rcdefaults()
    fx.use_light()


def _figure():
    fig, ax = plt.subplots(figsize=(3, 2.4))
    fp.line(ax, [0, 1, 2], [1, 2, 3], series="s", label="S")
    fp.scatter(ax, [0, 1, 2], [3, 2, 1], series="p", c=[0.1, 0.5, 0.9])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("T"); ax.grid(True); ax.legend()
    fp.significance_bracket(ax, x0=0, x1=1, y=3.2, label="*")
    fp.annotation(ax, name="a", text="note", xy=(1, 2))
    ax.text(1.5, 1.5, "free")
    return fig


def _save(fig, tmp_path, name="p", **kw):
    fp.save(fig, str(tmp_path / f"{name}.svg"), recipe=False, **kw)
    man = json.loads((tmp_path / f"{name}.fluxplot.json").read_text())
    root = etree.parse(str(tmp_path / f"{name}.svg")).getroot()
    return man, root


def _ink(root, gid):
    el = next(e for e in root.iter() if e.get("id") == gid)
    return el.get("data-ink-fill"), el.get("data-ink-stroke")


def test_every_scaffold_element_names_its_token(tmp_path):
    fx.use_light()
    man, root = _save(_figure(), tmp_path)
    assert man["style"]["theme"] == "light"
    tokens = man["style"]["tokens"]
    assert tokens == {"ink": "#100f0f", "label": "#100f0f", "tick": "#575653", "axis": "#575653",
                      "grid": "#dad8ce", "plot": "#ffffff", "paper": "#ffffff"}
    tagged = {el.get("id"): (el.get("data-ink-fill"), el.get("data-ink-stroke")) for el in root.iter()
              if el.get("data-ink-fill") or el.get("data-ink-stroke")}
    assert tagged["axis.x.title"] == ("label", None)
    assert tagged["axis.x.ticklabel.1"] == ("ink", None)
    assert tagged["axis.x.tick.1"] == ("tick", "tick") and tagged["axis.x.tick.2"] == ("tick", "tick")
    assert tagged["axis.x.spine.bottom"] == (None, "axis")
    assert tagged["axis.x.gridline.1"] == (None, "grid")
    assert tagged["figure.title"] == ("ink", None)
    assert tagged["legend.entry.0.label"] == ("ink", None)
    assert tagged["annotation.a"] == ("ink", None) and tagged["annotation.0"] == ("ink", None)
    assert tagged["significance-bracket.0"] == (None, "ink") and tagged["significance-bracket.0.label"] == ("ink", None)
    assert tagged["axes.background"] == ("plot", None) and tagged["figure.background"] == ("paper", None)
    # every scaffold role present carries a tag; no data mark does
    roles = {el.get("data-role") for el in root.iter() if el.get("data-ink-fill") or el.get("data-ink-stroke")}
    assert {"axis-title", "tick-label", "tick", "spine", "gridline", "title", "legend-label", "annotation",
            "significance-bracket", "background"} <= roles
    for el in root.iter():
        if el.get("data-role") in ("line", "point", "legend-swatch"):
            assert not el.get("data-ink-fill") and not el.get("data-ink-stroke"), el.get("id")
    # the grounds are parts of the tree
    def leaves(node, out):
        if "ref" in node: out.append((node["ref"], node.get("role")))
        for c in node.get("children", []): leaves(c, out)
        return out
    refs = dict(leaves(man["parts"], []))
    assert refs["axes.background"] == "background" and refs["figure.background"] == "background"
    assert man["parts"]["children"][0]["ref"] == "figure.background"


def test_dark_theme_is_recorded_and_tagged(tmp_path):
    fx.use_dark()
    man, root = _save(_figure(), tmp_path)
    assert man["style"]["theme"] == "dark" and man["style"]["tokens"]["paper"] == "#1c1b1a"
    assert _ink(root, "axis.x.ticklabel.1") == ("ink", None) and _ink(root, "axes.background") == ("plot", None)
    # a hand-changed rcParam after the theme: the tokens are still recorded, the name is not claimed
    mpl.rcParams["text.color"] = "#ff0000"
    man, root = _save(_figure(), tmp_path, "edited")
    assert man["style"]["theme"] is None and man["style"]["tokens"]["ink"] == "#ff0000"
    # the tick labels still wear the theme's grey — which is now what the `label` token names
    assert _ink(root, "axis.x.ticklabel.1") == ("label", None)


def test_theme_vars_rewrite_the_paints(tmp_path):
    fx.use_light()
    _man, root = _save(_figure(), tmp_path, "vars", theme_vars=True)
    title = next(e for e in root.iter() if e.get("id") == "axis.x.title")
    text = next(d for d in title.iter() if d.tag.endswith("text"))
    assert "fill: var(--fx-label, #100f0f)" in text.get("style")
    spine = next(e for e in root.iter() if e.get("id") == "axis.x.spine.bottom")
    assert "stroke: var(--fx-axis, #575653)" in next(d for d in spine.iter() if d.tag.endswith("path")).get("style")
    _man, plain = _save(_figure(), tmp_path, "plain")
    assert "var(" not in etree.tostring(plain).decode()


def test_recipe_theme_override_applies_to_every_use_call(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": {"theme": "dark"}}))
    fx.use_light()
    assert fx.ACTIVE["name"] == "dark" and mpl.rcParams["figure.facecolor"].lower() == "#1c1b1a"
    fx.use_paper()
    assert fx.ACTIVE["name"] == "dark"
    man, _root = _save(_figure(), tmp_path)
    assert man["style"]["theme"] == "dark"
    rec = json.loads((tmp_path / "p.recipe.json").read_text())
    assert rec["params"]["__fluxplot__"]["theme"] == "dark"
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": {"theme": "sepia"}}))
    with pytest.raises(ValueError, match="theme must be one of"):
        fx.use_light()


def test_theme_and_scales_share_the_control_block(tmp_path):
    fx.use_light()
    man, _root = _save(_figure(), tmp_path)
    rec = json.loads((tmp_path / "p.recipe.json").read_text())
    assert set(rec["params"]["__fluxplot__"]) == {"p", "theme"}
    (scale,) = man["colorScales"]
    assert scale["id"] == "p"  # a series literally named "theme" could never claim that key
    fig, ax = plt.subplots()
    fp.heatmap(ax, np.eye(3), series="theme")
    man, _root = _save(fig, tmp_path, "reserved")
    assert man["colorScales"][0]["id"] == "axes.1.theme"


def test_theme_output_is_deterministic(tmp_path):
    fx.use_paper()
    outs = []
    for k in range(2):
        fig = _figure()
        res = fp.save(fig, str(tmp_path / "d.svg"), recipe=False, theme_vars=True)
        outs.append((open(res.svg, "rb").read(), open(res.manifest, "rb").read()))
        plt.close(fig)
    assert outs[0] == outs[1]
