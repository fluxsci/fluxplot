"""C11 — seaborn's categorical plots and hue splits get exact identity."""
import json
import re
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fluxplot as fp  # noqa: E402

sns = pytest.importorskip("seaborn")
pd = pytest.importorskip("pandas")

RNG = np.random.default_rng(0)
DF = pd.DataFrame({"g": np.repeat(["a", "b", "c"], 20), "y": RNG.normal(size=60),
                   "h": np.tile(["p", "q"], 30), "x": RNG.normal(size=60)})


def _save(fig, tmp_path, name="s.svg"):
    res = fp.save(fig, str(tmp_path / name), recipe=False)
    return res, json.loads(open(res.manifest).read()), set(re.findall(r'id="([^"]+)"', open(res.svg).read()))


def test_boxplot_per_category_and_per_hue(tmp_path):
    fig, ax = plt.subplots()
    sns.boxplot(data=DF, x="g", y="y", ax=ax)
    out = fp.tag_seaborn(ax, plot="boxplot", data=DF, x="g", y="y")
    assert list(out) == ["a", "b", "c"] and out["a"][:1] == ["box"] and set(out["a"]) >= {"box", "whisker", "cap", "median"}
    res, man, ids = _save(fig, tmp_path)
    assert {"a.box", "a.whisker", "a.whisker.1", "a.cap", "a.median", "b.box", "c.box"} <= ids
    assert [s["id"] for s in man["series"]] == ["a", "b", "c"] and man["series"][0]["kind"] == "box"
    # hue: series = hue level, category = named part
    fig, ax = plt.subplots()
    sns.boxplot(data=DF, x="g", y="y", hue="h", ax=ax)
    out = fp.tag_seaborn(ax, plot="boxplot", data=DF, x="g", y="y", hue="h")
    assert list(out) == ["p", "q"]
    res, man, ids = _save(fig, tmp_path, "h.svg")
    assert {"p.a", "p.b", "p.c", "q.a", "p.a-whisker", "p.a-median", "q.c-cap"} <= ids
    assert [s["id"] for s in man["series"]] == ["p", "q"]
    (legend,) = [g for g in man["guides"] if g["role"] == "legend"]
    assert {e["text"]: e.get("series") for e in legend["entries"]} == {"p": "p", "q": "q"}
    # without a frame: categories from the tick labels, hues from the legend
    fig, ax = plt.subplots()
    sns.boxplot(data=DF, x="g", y="y", hue="h", ax=ax)
    out = fp.tag_seaborn(ax, plot="boxplot", hue="h")
    assert list(out) == ["p", "q"] and len(out["p"]) >= 12


def test_violinplot_bodies_and_inner_lines(tmp_path):
    fig, ax = plt.subplots()
    sns.violinplot(data=DF, x="g", y="y", ax=ax)
    out = fp.tag_seaborn(ax, plot="violinplot", data=DF, x="g", y="y")
    assert list(out) == ["a", "b", "c"] and out["a"] == ["violin", "segment"]
    res, man, ids = _save(fig, tmp_path)
    assert {"a.violin", "a.inner", "b.violin", "c.violin"} <= ids
    fig, ax = plt.subplots()
    sns.violinplot(data=DF, x="g", y="y", hue="h", split=True, ax=ax)
    out = fp.tag_seaborn(ax, plot="violinplot", data=DF, x="g", y="y", hue="h")
    assert list(out) == ["p", "q"] and out["p"].count("violin") == 3
    res, man, ids = _save(fig, tmp_path, "v.svg")
    assert {"p.a", "q.a", "p.c", "q.c"} <= ids
    # the bodies really are the hue's: their face colour equals the legend swatch's
    leg = ax.get_legend()
    handle_rgb = {t.get_text(): tuple(np.round(h.get_facecolor()[:3], 4)) for t, h in zip(leg.get_texts(), leg.legend_handles)}
    marks = [m for m in fp.tagger.registry_for(fig).marks if m.role == "violin"]
    for m in marks:
        assert tuple(np.round(m.artists[0].get_facecolor()[0][:3], 4)) == handle_rgb[m.series]


def test_strip_and_swarm_split_hue_levels_by_row(tmp_path):
    fig, ax = plt.subplots()
    sns.stripplot(data=DF, x="g", y="y", ax=ax, jitter=False)
    out = fp.tag_seaborn(ax, plot="stripplot", data=DF, x="g", y="y")
    assert list(out) == ["a", "b", "c"]
    res, man, ids = _save(fig, tmp_path)
    assert {"a.points", "a.point.0", "a.point.19", "c.point.5"} <= ids
    fig, ax = plt.subplots()
    sns.swarmplot(data=DF, x="g", y="y", hue="h", ax=ax)
    out = fp.tag_seaborn(ax, plot="swarmplot", data=DF, x="g", y="y", hue="h")
    assert list(out) == ["p", "q"] and out["p"].count("point") == 3
    res, man, ids = _save(fig, tmp_path, "w.svg")
    assert {"p.a.points", "p.a.point.0", "q.a.point.1", "p.b.point.0", "q.c.point.19"} <= ids
    assert "q.a.points" not in ids  # the collection is one <g>, owned by the first hue's mark
    by_id = {s["id"]: s for s in man["series"]}
    p_pts = {pt["svgId"] for pt in by_id["p"]["points"]}
    q_pts = {pt["svgId"] for pt in by_id["q"]["points"]}
    assert len(p_pts) == 30 and len(q_pts) == 30 and not (p_pts & q_pts)
    # every tagged point's colour is its hue's colour (the split follows the frame rows exactly)
    leg = ax.get_legend()
    handle_rgb = {t.get_text(): tuple(np.round(matplotlib.colors.to_rgba(h.get_markerfacecolor())[:3], 4))
                  for t, h in zip(leg.get_texts(), leg.legend_handles)}  # swarm legend handles are marker lines
    for m in fp.tagger.registry_for(fig).marks:
        if m.role == "point":
            faces = m.artists[0].get_facecolor()
            for k in m.data["point_subset"]:
                assert tuple(np.round(faces[k][:3], 4)) == handle_rgb[m.series]
    comps = [c for c in by_id["q"]["components"] if c["role"] == "point"]
    assert all(c["groupId"].startswith("q.") and c["svgId"].startswith("p.") for c in comps)
    tree = json.dumps(man["parts"])
    assert '"id": "q.a.points"' in tree and '"id": "p.a.points"' in tree


def test_scatterplot_hue_split_and_barplot_hue_equals_x(tmp_path):
    fig, ax = plt.subplots()
    sns.scatterplot(data=DF, x="x", y="y", hue="h", ax=ax)
    out = fp.tag_seaborn(ax, plot="scatterplot", data=DF, x="x", y="y", hue="h")
    assert list(out) == ["p", "q"]
    res, man, ids = _save(fig, tmp_path)
    by_id = {s["id"]: s for s in man["series"]}
    assert len(by_id["p"]["points"]) == 30 and len(by_id["q"]["points"]) == 30
    assert "p.points" in ids and "p.point.0" in ids and "q.point.1" in ids
    assert by_id["q"]["svg"]["points"] == "p.points"  # shared collection
    # rows alternate p, q: the first point is p's index 0, the second q's index 1
    assert by_id["p"]["points"][0]["index"] == 0 and by_id["q"]["points"][0]["index"] == 1
    assert by_id["p"]["points"][0]["x"] == pytest.approx(DF.x[0]) and by_id["q"]["points"][0]["x"] == pytest.approx(DF.x[1])
    # barplot with hue equal to x: seaborn draws no legend; the frame names the bars
    fig, ax = plt.subplots()
    sns.barplot(data=DF, x="g", y="y", hue="g", ax=ax, errorbar=None)
    out = fp.tag_seaborn(ax, plot="barplot", data=DF, x="g", y="y", hue="g")
    assert list(out) == ["a", "b", "c"] and all(v == ["bar"] for v in out.values())
    res, man, ids = _save(fig, tmp_path, "b.svg")
    assert {"a.bar.0", "b.bar.0", "c.bar.0"} <= ids


def test_pointplot_means_points_and_errors(tmp_path):
    fig, ax = plt.subplots()
    sns.pointplot(data=DF, x="g", y="y", hue="h", ax=ax)
    out = fp.tag_seaborn(ax, plot="pointplot", data=DF, x="g", y="y", hue="h")
    assert out == {"p": ["point", "line", "errorbar"], "q": ["point", "line", "errorbar"]}
    res, man, ids = _save(fig, tmp_path)
    assert {"p.line", "p.points", "p.point.0", "p.point.2", "p.errorbar", "p.errorbar.2", "q.line"} <= ids
    by_id = {s["id"]: s for s in man["series"]}
    assert by_id["p"]["data"]["x"] == [0.0, 1.0, 2.0] or len(by_id["p"]["data"]["x"]) == 3
    fig, ax = plt.subplots()
    sns.pointplot(data=DF, x="g", y="y", ax=ax, errorbar=None)
    out = fp.tag_seaborn(ax, plot="pointplot", data=DF, x="g", y="y")
    assert out == {"y": ["point", "line"]}


def test_nothing_tagged_warns():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with pytest.warns(UserWarning, match="nothing tagged"):
        fp.tag_seaborn(ax, plot="boxplot")
    fig, ax = plt.subplots()
    sns.boxplot(data=DF, x="g", y="y", ax=ax)
    with pytest.warns(UserWarning, match="nothing tagged"):
        out = fp.tag_seaborn(ax, plot="violinplot")  # the wrong kind: no bodies to name
    assert out == {}
