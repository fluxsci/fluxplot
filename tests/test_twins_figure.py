"""C4 — twin axes are one panel; C5 — figure-scope artists are named at figure scope."""
import json
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fluxplot as fp  # noqa: E402


def _save(fig, tmp_path, name="t.svg"):
    res = fp.save(fig, str(tmp_path / name), recipe=False)
    return res, json.loads(open(res.manifest).read()), open(res.svg).read()


def test_twinx_is_one_panel_with_a_y2_axis(tmp_path):
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1, 2], [1, 2, 3], series="rate", label="Rate")
    ax.set_ylabel("rate (Hz)")
    ax2 = ax.twinx()
    fp.line(ax2, [0, 1, 2], [100, 50, 25], series="temp", label="Temp")
    ax2.set_ylabel("temperature (°C)")
    ax2.spines["right"].set_visible(True)  # the house style hides right spines; a twin wants its own
    res, man, svg = _save(fig, tmp_path)
    assert "panels" not in man  # one panel, not two
    ids = set(re.findall(r'id="([^"]+)"', svg))
    assert "axis.y2" in ids and "axis.y2.title" in ids and "axis.y2.spine.right" in ids
    assert not any(i.startswith("panel.") for i in ids)
    assert len([i for i in ids if i.startswith("axis.x.spine")]) == 1  # one x axis: one bottom spine
    assert "axis.y2.tick.1" in ids and "axis.y2.ticklabel.1" in ids  # the twin's own ticks are its primary side
    assert not any(i.endswith("-2") for i in ids if i.startswith("axis."))
    (axes,) = man["axes"]
    assert axes["y2"]["label"] == "temperature (°C)" and axes["y2"]["domain"] == [float(v) for v in ax2.get_ylim()]
    assert axes["y"]["label"] == "rate (Hz)"
    by_id = {s["id"]: s for s in man["series"]}
    assert by_id["temp"]["axis"] == "y2" and "axis" not in by_id["rate"]
    tree = json.dumps(man["parts"])
    assert '"axis": "y2"' in tree
    assert "plot-area.y2" in ids  # the twin's own group, named after the panel
    assert any(g["id"] == "axis.y2" for g in man["guides"])


def test_twiny_and_promoted_labels_on_the_twin(tmp_path):
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [0, 1], series="a")
    ax2 = ax.twiny()
    ax2.spines["top"].set_visible(True)
    ax2.plot([10, 20], [0, 1], label="Raw on twin")  # promoted at save
    res, man, svg = _save(fig, tmp_path, "w.svg")
    ids = set(re.findall(r'id="([^"]+)"', svg))
    assert "axis.x2" in ids and "axis.x2.spine.top" in ids
    by_id = {s["id"]: s for s in man["series"]}
    assert by_id["raw-on-twin"]["axis"] == "x2" and "x2" in man["axes"][0]
    # two panels each with a twin: prefixed ids, one panel each
    fig, (a, b) = plt.subplots(1, 2)
    fp.line(a, [0, 1], [0, 1], series="a")
    a.twinx().plot([0, 1], [5, 6], label="A2")
    fp.line(b, [0, 1], [0, 1], series="b")
    res, man, svg = _save(fig, tmp_path, "p.svg")
    assert [p["id"] for p in man["panels"]] == ["panel.a", "panel.b"]
    ids = set(re.findall(r'id="([^"]+)"', svg))
    assert "panel.a.axis.y2" in ids and "panel.a.plot-area.y2" in ids


def test_figure_scope_artists(tmp_path):
    fig, (a, b) = plt.subplots(1, 2)
    la = fp.line(a, [0, 1], [0, 1], series="ctl", label="Control")
    lb = fp.line(b, [0, 1], [1, 0], series="drug", label="Drug")
    fig.suptitle("Two panels")
    fig.supxlabel("time (s)")
    fig.legend(loc="lower center", ncol=2)
    fig.text(0.01, 0.99, "A", fontweight="bold")
    res, man, svg = _save(fig, tmp_path, "f.svg")
    ids = set(re.findall(r'id="([^"]+)"', svg))
    assert "figure.title" in ids and "figure.xlabel" in ids and "figure.legend" in ids and "figure.annotation.0" in ids
    assert "figure.legend.entry.0.label" in ids and "figure.legend.entry.1.swatch" in ids
    assert not any(i.startswith("panel.") and "figure.title" in i for i in ids)  # the suptitle is not a panel's
    block = man["figure"]
    assert block["title"] == "figure.title" and block["xlabel"] == "figure.xlabel" and block["background"] == "figure.background"
    assert "figure.background" in ids and not any(i.endswith(".figure.background") for i in ids)
    assert man["idAliases"]["panel.a.figure.background"] == "figure.background"  # the pre-0.3.2 id
    assert block["legends"] == ["figure.legend"] and block["annotations"] == [{"id": "figure.annotation.0", "text": "A"}]
    (legend,) = [g for g in man["guides"] if g["id"] == "figure.legend"]
    assert {e["text"]: e["series"] for e in legend["entries"]} == {"Control": "panel.a.ctl", "Drug": "panel.b.drug"}
    roots = {c.get("id") or c.get("ref") for c in man["parts"]["children"]}
    assert {"panel.a", "panel.b", "figure.title", "figure.xlabel", "figure.legend", "figure.annotation.0"} <= roots
    assert man["build"]["order"].index("figure.title") < man["build"]["order"].index("panel.a.ctl.line")
    overlays = {o["id"]: o for o in man["overlays"]}
    assert overlays["figure.annotation.0"]["text"] == "A"
    # a figure legend entry standing for a bar container (members only, no group id) joins too
    fig, (a, b) = plt.subplots(1, 2)
    fp.bar(a, [0, 1], [1, 2], series="bars", label="Bars")
    fp.scatter(b, [0, 1], [1, 2], series="dots", label="Dots")
    fig.legend(loc="lower center", ncol=2)
    res, man, svg = _save(fig, tmp_path, "fb.svg")
    (legend,) = [g for g in man["guides"] if g["id"] == "figure.legend"]
    assert {e["text"]: e["series"] for e in legend["entries"]} == {"Bars": "panel.a.bars", "Dots": "panel.b.dots"}
    # a single-panel figure: the axes title keeps figure.title unless a suptitle claims it first
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [0, 1], series="s")
    ax.set_title("Axes title")
    res, man, svg = _save(fig, tmp_path, "s.svg")
    ids = set(re.findall(r'id="([^"]+)"', svg))
    assert "figure.title" in ids and man["figure"] == {"background": "figure.background"}  # the ground only
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [0, 1], series="s")
    fig.suptitle("Sup")
    ax.set_title("Axes title")
    res, man, svg = _save(fig, tmp_path, "s2.svg")
    assert man["figure"]["title"] == "figure.title"
    ids = set(re.findall(r'id="([^"]+)"', svg))
    assert {"figure.title", "figure.title-2"} <= ids  # the suptitle claimed figure.title; the axes title follows
    assert any(g["role"] == "title" and g["id"] == "figure.title-2" for g in man["guides"]) or "figure.title-2" in json.dumps(man["parts"])
