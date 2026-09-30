"""F4 / F7 (fluxplot half): tick scheme kinds per axis, complete bar geometry, cell and hex
bounds, stable member keys and capabilities.valueMorph."""
import json
import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from lxml import etree  # noqa: E402

import fluxplot as fp  # noqa: E402


def _save(fig, tmp_path, name="g.svg"):
    res = fp.save(fig, str(tmp_path / name), recipe=False)
    root = etree.parse(res.svg).getroot()
    return json.loads(open(res.manifest).read()), {el.get("id"): el for el in root.iter() if el.get("id")}


def test_axes_record_their_tick_scheme(tmp_path):
    fig, (a, b) = plt.subplots(1, 2)
    fp.line(a, [1, 10, 100], [1, 2, 3], series="s")
    a.set_xscale("log")
    fp.bar(b, ["ctl", "drug"], [1, 2], series="t")
    man, ids = _save(fig, tmp_path)
    axes = {ax["panelId"]: ax for ax in man["axes"]}
    assert axes["panel.a"]["x"]["tickLocator"] == "log" and axes["panel.a"]["x"]["tickFormatter"] == "log"
    assert axes["panel.a"]["y"]["tickLocator"] == "auto" and axes["panel.a"]["y"]["tickFormatter"] == "plain"
    assert axes["panel.b"]["x"]["tickLocator"] == "category" and axes["panel.b"]["x"]["tickFormatter"] == "category"


def test_bars_carry_geometry_and_category_keys(tmp_path):
    fig, ax = plt.subplots()
    fp.bar(ax, ["SD", "Sleep", "Wake"], [3.0, 1.5, 2.0], series="rates", width=0.6)
    man, ids = _save(fig, tmp_path)
    (s,) = man["series"]
    bar = s["bar"]
    assert bar["keys"] == ["SD", "Sleep", "Wake"] and bar["center"] == [0.0, 1.0, 2.0]
    assert bar["width"] == [0.6, 0.6, 0.6] and bar["length"] == [3.0, 1.5, 2.0] and bar["baseline"] == [0.0, 0.0, 0.0]
    assert [ids[f"rates.bar.{k}"].get("data-key") for k in range(3)] == ["SD", "Sleep", "Wake"]
    assert s["capabilities"]["valueMorph"] is True
    # numeric positions: the key is the position as text (or the tick label sitting there)
    fig, ax = plt.subplots()
    fp.bar(ax, [0.5, 1.5], [1, 2], series="n", width=0.4)
    ax.set_xticks([0.5, 1.5], ["low", "high"])
    man, ids = _save(fig, tmp_path, "n.svg")
    assert man["series"][0]["bar"]["keys"] == ["low", "high"]
    fig, ax = plt.subplots()
    fp.barh(ax, [0, 1], [1, 2], series="h")
    man, ids = _save(fig, tmp_path, "h.svg")
    assert man["series"][0]["bar"]["keys"] == ["0", "1"] and man["series"][0]["bar"]["orientation"] == "horizontal"


def test_cells_and_hexes_carry_bounds_and_keys(tmp_path):
    fig, ax = plt.subplots()
    fp.heatmap(ax, [[1.0, 2.0], [3.0, 4.0]], series="m", cells=True, x=[0, 1, 2], y=[10, 20, 30])
    man, ids = _save(fig, tmp_path)
    cell = ids["m.x-heatmap.cell.1.0"]
    assert cell.get("data-key") == "1.0"
    assert [float(cell.get(k)) for k in ("data-x0", "data-x1", "data-y0", "data-y1")] == [0.0, 1.0, 20.0, 30.0]
    assert float(ids["m.x-heatmap.cell.0.1"].get("data-x0")) == 1.0
    (s,) = man["series"]
    assert s["capabilities"]["valueMorph"] is True
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots()
    fp.hexmatrix(x=rng.normal(size=100), y=rng.normal(size=100), ax=ax, gridsize=5, series="h", colorbar=False)
    man, ids = _save(fig, tmp_path, "x.svg")
    hexes = [el for gid, el in ids.items() if gid.startswith("h.hex.")]
    assert hexes
    for el in hexes:
        assert el.get("data-key") == f'{el.get("data-row")}.{el.get("data-column")}'
        x0, x1, y0, y1 = (float(el.get(k)) for k in ("data-x0", "data-x1", "data-y0", "data-y1"))
        assert x0 < float(el.get("data-x")) < x1 and y0 < float(el.get("data-y")) < y1
    (s,) = man["series"]
    assert s["capabilities"]["valueMorph"] is True
    # a plain line has no keyed members
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [0, 1], series="l")
    man, ids = _save(fig, tmp_path, "l.svg")
    assert man["series"][0]["capabilities"]["valueMorph"] is False
