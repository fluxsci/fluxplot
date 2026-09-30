"""D2 — fp.band beside its series' line; fp.area records its inputs."""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fluxplot as fp  # noqa: E402


def _save(fig, tmp_path, name="b.svg"):
    res = fp.save(fig, str(tmp_path / name), recipe=False)
    return json.loads(open(res.manifest).read()), open(res.svg).read()


def test_band_sits_beside_its_line_with_the_inputs_recorded(tmp_path):
    x = np.arange(5.0)
    y = x**2
    lo, hi = y - 1, y + 1
    fig, ax = plt.subplots()
    fp.line(ax, x, y, series="ctl", color="#c03030")
    poly = fp.band(ax, x, lo, hi, series="ctl", what="SEM")
    man, svg = _save(fig, tmp_path)
    (ctl,) = [s for s in man["series"] if s["id"] == "ctl"]
    assert ctl["svg"]["line"] == "ctl.line" and ctl["svg"]["area"] == "ctl.band"
    assert sorted(ctl["roles"]) == ["area", "line"]
    assert ctl["band"] == {"x": list(x), "lo": list(lo), "hi": list(hi), "what": "SEM"}
    assert 'id="ctl.band"' in svg and 'id="ctl.line"' in svg
    # the band took the line's colour, translucent, without an edge
    assert matplotlib.colors.to_hex(poly.get_facecolor()[0][:3]) == "#c03030"
    assert poly.get_facecolor()[0][3] == 0.25 and poly.get_linewidth()[0] == 0
    assert "band" in fp.__all__


def test_band_defaults_and_overrides():
    fig, ax = plt.subplots()
    poly = fp.band(ax, [0, 1], [0, 0], [1, 1], series="drug", color="#3030c0", alpha=0.5)
    assert matplotlib.colors.to_hex(poly.get_facecolor()[0][:3]) == "#3030c0" and poly.get_facecolor()[0][3] == 0.5
    mark = next(m for m in fp.tagger.registry_for(fig).marks if m.series == "drug")
    assert mark.data["band"]["what"] == "95% CI" and mark.data["band"]["lo"] == [0.0, 0.0]
    # scalar bounds broadcast
    fp.band(ax, [0, 1, 2], 0.5, [1, 2, 3], series="s2")
    mark = next(m for m in fp.tagger.registry_for(fig).marks if m.series == "s2")
    assert mark.data["band"]["lo"] == [0.5, 0.5, 0.5]


def test_area_records_x_y1_y2_with_a_broadcast_baseline(tmp_path):
    fig, ax = plt.subplots()
    fp.area(ax, [0, 1, 2], [1, 3, 2], series="cum", alpha=0.3)
    fp.area(ax, [0, 1, 2], [4, 5, 6], [3, 3, 4], series="stack")
    man, _ = _save(fig, tmp_path, "a.svg")
    by = {s["id"]: s for s in man["series"]}
    assert by["cum"]["band"] == {"x": [0.0, 1.0, 2.0], "y1": [1.0, 3.0, 2.0], "y2": [0.0, 0.0, 0.0]}
    assert by["stack"]["band"] == {"x": [0.0, 1.0, 2.0], "y1": [4.0, 5.0, 6.0], "y2": [3.0, 3.0, 4.0]}
    assert "paths" not in by["cum"]["band"]
    # a raw labelled fill_between still records only its polygon
    fig, ax = plt.subplots()
    ax.fill_between([0, 1, 2], [1, 2, 1], [2, 3, 2], label="Band", alpha=0.4)
    man, _ = _save(fig, tmp_path, "r.svg")
    (raw,) = man["series"]
    assert set(raw["band"]) == {"paths"}
