"""C6, C7, C12 — swept anchored artists and tables, side-named spines with aliases, seaborn small
items, recorded tagger warnings, and the gridlines build token only when gridlines exist."""
import json
import re

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fluxplot as fp  # noqa: E402


def _save(fig, tmp_path, name="c.svg", **kw):
    res = fp.save(fig, str(tmp_path / name), recipe=False, **kw)
    return res, json.loads(open(res.manifest).read()), open(res.svg).read()


def test_spines_are_named_by_side_with_aliases_for_the_old_ids(tmp_path):
    fig, ax = plt.subplots()
    ax.imshow(np.arange(12).reshape(3, 4))
    for sp in ax.spines.values():  # all four spines visible (the house style hides two)
        sp.set_visible(True)
    fp.line(ax, [0, 1], [0, 1], series="s")
    res, man, svg = _save(fig, tmp_path)
    ids = set(re.findall(r'id="([^"]+)"', svg))
    spines = sorted(i for i in ids if ".spine" in i)
    assert spines == ["axis.x.spine.bottom", "axis.x.spine.top", "axis.y.spine.left", "axis.y.spine.right"]
    assert not any(i.endswith("-2") for i in ids)
    assert man["idAliases"] == {"axis.x.spine": "axis.x.spine.bottom", "axis.x.spine-2": "axis.x.spine.top",
                                "axis.y.spine": "axis.y.spine.left", "axis.y.spine-2": "axis.y.spine.right"}
    # a house-style plot (bottom/left only) aliases just the two
    fig, ax = plt.subplots()
    ax.spines[["top", "right"]].set_visible(False)
    fp.line(ax, [0, 1], [0, 1], series="s")
    res, man, svg = _save(fig, tmp_path, "h.svg")
    assert man["idAliases"] == {"axis.x.spine": "axis.x.spine.bottom", "axis.y.spine": "axis.y.spine.left"}
    # panels prefix both sides of an alias
    fig, (a, b) = plt.subplots(1, 2)
    for sp in b.spines.values():
        sp.set_visible(True)
    fp.line(a, [0, 1], [0, 1], series="s")
    fp.line(b, [0, 1], [0, 1], series="t")
    res, man, svg = _save(fig, tmp_path, "p.svg")
    assert man["idAliases"]["panel.a.axis.x.spine"] == "panel.a.axis.x.spine.bottom"
    assert man["idAliases"]["panel.b.axis.y.spine-2"] == "panel.b.axis.y.spine.right"


def test_series_whose_slug_changed_get_a_root_alias(tmp_path):
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [0, 1], series="IL-6 (pg/mL)")
    fp.line(ax, [0, 1], [1, 0], series="IL-6 [pg/mL]")
    fp.scatter(ax, [0, 1], [0, 1], series="α")
    res, man, svg = _save(fig, tmp_path)
    ids = {s["id"] for s in man["series"]}
    assert len(ids) == 3 and "alpha" in ids
    aliases = man["idAliases"]
    assert aliases["series"] == "alpha"  # the old rule turned α into the fallback
    assert "il-6-pg-ml" not in aliases  # two series shared the old slug: no alias can be honest
    assert all(v in ids or ".spine" in v for v in aliases.values())


def test_anchored_artists_and_tables_are_swept(tmp_path):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((10, 10)))
    bar = AnchoredSizeBar(ax.transData, 3, "3 px", "lower right", frameon=False)
    ax.add_artist(bar)
    ax.table(cellText=[["a", "b"]], loc="top")
    res, man, svg = _save(fig, tmp_path)
    ids = set(re.findall(r'id="([^"]+)"', svg))
    assert "extra.artist.0" in ids and "extra.artist.0.label" in ids
    assert "extra.table.0" in ids
    overlays = {o["id"]: o for o in man["overlays"]}
    assert overlays["extra.artist.0"]["kind"] == "shape"
    assert overlays["extra.table.0"]["kind"] == "container"
    assert overlays["extra.artist.0.label"]["role"] == "label" or overlays["extra.artist.0.label"]["kind"] == "text"


def test_build_order_names_gridlines_only_when_there_are_any(tmp_path):
    fig, ax = plt.subplots()
    ax.grid(False)
    fp.line(ax, [0, 1], [0, 1], series="s")
    res, man, svg = _save(fig, tmp_path)
    assert "gridlines" not in man["build"]["order"]
    fig, ax = plt.subplots()
    ax.grid(True)
    fp.line(ax, [0, 1], [0, 1], series="s")
    res, man, svg = _save(fig, tmp_path, "g.svg")
    assert "gridlines" in man["build"]["order"]


def test_tag_seaborn_removes_only_seaborn_proxies_and_names_facets():
    sns = pytest.importorskip("seaborn")
    import pandas as pd
    df = pd.DataFrame({"x": np.tile(np.arange(5), 2), "y": np.r_[np.arange(5), np.arange(5) * 2],
                       "g": ["a"] * 5 + ["b"] * 5})
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="x", y="y", hue="g", ax=ax, errorbar=None)
    (mine,) = ax.plot([], [], label="Placeholder")  # an empty line the USER labelled
    fp.tag_seaborn(ax, plot="lineplot")
    assert mine in ax.lines
    assert not any(len(ln.get_xdata()) == 0 and str(ln.get_label()).startswith("_") for ln in ax.lines)
    # a facet without a legend is named after its title, not "data"
    fig, ax = plt.subplots()
    ax.set_title("cortex")
    sns.lineplot(data=df[df.g == "a"], x="x", y="y", ax=ax, errorbar=None)
    ax.set_ylabel("")
    out = fp.tag_seaborn(ax)
    assert list(out) == ["cortex"]


def test_legend_swatch_problems_are_reported_not_swallowed(tmp_path):
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [0, 1], series="s", label="S")
    leg = ax.legend()
    leg.legend_handles.append(object())  # not an Artist: the swatch cannot be tagged
    res = fp.save(fig, str(tmp_path / "w.svg"), recipe=False)
    assert any("legend entry 1" in w and "could not be tagged" in w for w in res.warnings)
    assert fp.tagger.registry_for(fig).warnings == []  # consumed by the save
