"""The Flux ↔ fluxplot parts contract: build presets use the closed vocabulary and carry stagger
hints (F1); every parts-tree leaf states its role, every group its members' role, and series /
legend-entry nodes their label, so a consumer never guesses a role from an id (F2).
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import fluxplot as fp  # noqa: E402
from fluxplot import presets  # noqa: E402


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _save(fig, tmp_path, name="p", **kw):
    fp.save(fig, str(tmp_path / f"{name}.svg"), recipe=False, **kw)
    return json.loads((tmp_path / f"{name}.fluxplot.json").read_text())


# ---- F1 -----------------------------------------------------------------------------------------
def test_presets_use_the_closed_vocabulary(tmp_path):
    assert set(presets.PRESET_NAMES) >= {v["animation"] for v in presets.ROLE_PRESETS.values()}
    x, y = np.random.default_rng(4).normal(size=(2, 300))
    fig, ax = plt.subplots()
    fp.hexmatrix(x=x, y=y, ax=ax, gridsize=8, series="h")
    ax.legend(handles=[plt.Line2D([], [], label="k")])
    man = _save(fig, tmp_path)
    pre = man["build"]["presets"]
    assert pre["x-hexbin"] == {"animation": "stagger-in", "staggerBy": "value", "staggerMs": 4, "durationMs": 240}
    assert pre["colorbar"]["animation"] == "fade-in" and pre["legend"]["animation"] == "fade-rise"
    assert pre["x-hex"]["staggerBy"] == "value"


# ---- F2 -----------------------------------------------------------------------------------------
def _leaves(node, out):
    if "ref" in node:
        out.append(node)
    for c in node.get("children", []):
        _leaves(c, out)
    return out


def _nodes(node, out):
    out.append(node)
    for c in node.get("children", []):
        _nodes(c, out)
    return out


def test_every_parts_leaf_carries_its_role(tmp_path):
    fig, (a, b) = plt.subplots(1, 2)
    fp.line(a, [0, 1, 2], [1, 2, 3], series="ctl", marker="o", label="Control")
    fp.bar(b, [0, 1], [1, 2], series="n", label="Counts")
    fp.annotation(a, name="peak", text="peak", xy=(1, 2))
    a.text(0.5, 0.5, "free text")
    fp.significance_bracket(a, x0=0, x1=1, y=3, label="*")
    fp.heatmap(b, np.eye(3), series="m", cells=True)
    a.legend()
    b.set_title("B")
    b.set_xlabel("x")
    man = _save(fig, tmp_path)
    leaves = _leaves(man["parts"], [])
    assert leaves and all(leaf.get("role") for leaf in leaves), [l for l in leaves if not l.get("role")]
    by_ref = {leaf["ref"]: leaf["role"] for leaf in leaves}
    assert by_ref["panel.a.annotation.peak"] == "annotation"
    assert by_ref["panel.a.annotation.0"] == "annotation"
    assert by_ref["panel.a.significance-bracket.0"] == "significance-bracket"
    assert by_ref["panel.a.ctl.line"] == "line"
    assert by_ref["panel.b.axis.x.title"] == "axis-title"
    assert by_ref["panel.b.figure.title"] == "title"
    groups = [n for n in _nodes(man["parts"], []) if n.get("role") == "group"]
    assert all(g.get("memberRole") for g in groups)
    # a scaffold group's members share its role; a field layer's members have their own
    assert all(g["memberRole"] == g["groupRole"] for g in groups if g["id"].startswith("panel.a.axis"))
    cells = next(g for g in groups if g["id"] == "panel.b.m.x-heatmap")
    assert cells["memberRole"] == "cell" and cells["kind"] == "shape"
    series_nodes = [n for n in _nodes(man["parts"], []) if n.get("role") == "series"]
    assert {n["id"]: n["label"] for n in series_nodes} == {"panel.a.ctl": "Control", "panel.b.n": "Counts",
                                                           "panel.b.m": "m"}
    entry = next(n for n in _nodes(man["parts"], []) if n.get("role") == "legend-entry")
    assert entry["label"] == "Control"
    comp = next(c for c in next(s for s in man["series"] if s["id"] == "panel.b.m")["components"])
    assert comp["memberRole"] == "cell"
