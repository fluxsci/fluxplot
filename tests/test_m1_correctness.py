"""Milestone-1 correctness fixes: bubble scatters keep per-point ids (C1), seaborn KDE curves are
lines not error bars (C2), notebooks never name a helper module as the script (C3) and brackets
take the theme's ink (B5).
"""
import json
import sys
import types

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from lxml import etree  # noqa: E402

import fluxplot as fp  # noqa: E402
from fluxplot import provenance  # noqa: E402


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _save(fig, tmp_path, name="p", **kw):
    res = fp.save(fig, str(tmp_path / f"{name}.svg"), recipe=False, **kw)
    man = json.loads((tmp_path / f"{name}.fluxplot.json").read_text())
    root = etree.parse(str(tmp_path / f"{name}.svg")).getroot()
    return res, man, root


# ---- C1 -----------------------------------------------------------------------------------------
def test_bubble_scatter_keeps_per_point_ids(tmp_path):
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=40), rng.normal(size=40)
    fig, ax = plt.subplots()
    fp.scatter(ax, x, y, s=np.linspace(5, 50, 40), series="b")
    res, man, root = _save(fig, tmp_path)
    assert res.warnings == []
    ids = set(root.xpath("//@id"))
    assert {f"b.point.{k}" for k in range(40)} <= ids
    assert len(man["series"][0]["points"]) == 40
    node = root.xpath("//*[@id='b.point.7']")[0]
    assert node.tag.endswith("path") and node.get("data-index") == "7"
    assert float(node.get("data-x")) == x[7]


def test_bubble_scatter_with_colour_and_colorbar(tmp_path):
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots()
    coll = fp.scatter(ax, rng.normal(size=30), rng.normal(size=30), s=rng.uniform(5, 60, 30),
                      c=rng.uniform(size=30), series="c")
    fp.colorbar(coll)
    res, man, root = _save(fig, tmp_path)
    assert not any("skipping per-point ids" in w for w in res.warnings)
    assert len(man["series"][0]["points"]) == 30


# ---- C2 -----------------------------------------------------------------------------------------
def test_kde_over_histogram_is_a_line_not_an_errorbar(tmp_path):
    import seaborn as sns
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots()
    sns.histplot(rng.normal(size=200), kde=True, ax=ax)
    tagged = fp.tag_seaborn(ax, plot="histplot")
    roles = sorted(r for rs in tagged.values() for r in rs)
    assert roles == ["bar", "line"]
    res, man, _root = _save(fig, tmp_path)
    (s,) = man["series"]
    assert "errorbar" not in s["roles"] and "line" in s["roles"]


def test_barplot_error_bars_still_join(tmp_path):
    import pandas as pd
    import seaborn as sns
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"g": rng.choice(list("abc"), 90), "v": rng.normal(size=90)})
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="g", y="v", errorbar="sd", capsize=0.3, ax=ax)
    tagged = fp.tag_seaborn(ax, plot="barplot")
    assert list(tagged.values())[0].count("errorbar") == 3


# ---- C3 -----------------------------------------------------------------------------------------
def test_notebook_without_main_file_records_no_script(tmp_path, monkeypatch):
    fake_main = types.ModuleType("__main__")  # ipykernel: __main__ has no __file__
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    assert provenance.discover_script() is None
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [0, 1], series="s")
    res = fp.save(fig, str(tmp_path / "nb.svg"))
    rec = json.loads(open(res.recipe).read())
    assert rec["script"] is None and rec["provenance"]["scriptDiscovery"] == "unavailable"
    assert "command" not in rec


def test_runner_entry_point_still_finds_the_caller():
    # under pytest, __main__ is the runner's entry point; the caller (this file) is the script
    found = provenance.discover_script()
    assert found is not None and found.endswith("test_m1_correctness.py")


# ---- B5 (bracket) -------------------------------------------------------------------------------
def test_bracket_takes_the_theme_ink_and_paints_its_label():
    fig, ax = plt.subplots()
    with mpl.rc_context({"text.color": "#abcdef"}):
        br = fp.significance_bracket(ax, x0=0, x1=1, y=1, label="*")
    assert mpl.colors.to_hex(br.get_color()) == "#abcdef"
    label = [t for t in ax.texts if t.get_text() == "*"][0]
    assert mpl.colors.to_hex(label.get_color()) == "#abcdef"
    br2 = fp.significance_bracket(ax, x0=0, x1=1, y=2, label="**", color="red",
                                  text_kw={"fontsize": 3})
    label2 = [t for t in ax.texts if t.get_text() == "**"][0]
    assert br2.get_color() == "red" and label2.get_color() == "red" and label2.get_fontsize() == 3
    ax.set_yscale("log")
    with pytest.raises(ValueError, match="positive on a log axis"):
        fp.significance_bracket(ax, x0=0, x1=1, y=0, label="ns")
