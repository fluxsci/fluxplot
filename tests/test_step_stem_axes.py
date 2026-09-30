"""D4 step/stem, D5 categorical and date data labels, D6 insets and secondary axes."""
import json
import re
from datetime import datetime, timedelta

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fluxplot as fp  # noqa: E402


def _save(fig, tmp_path, name="d.svg"):
    res = fp.save(fig, str(tmp_path / name), recipe=False)
    return json.loads(open(res.manifest).read()), set(re.findall(r'id="([^"]+)"', open(res.svg).read()))


def test_step_records_where_and_drawstyle(tmp_path):
    fig, ax = plt.subplots()
    fp.step(ax, [0, 1, 2, 3], [1, 3, 2, 4], series="s", where="post")
    man, ids = _save(fig, tmp_path)
    (s,) = man["series"]
    assert s["kind"] == "step" and s["step"] == {"where": "post", "drawstyle": "steps-post"}
    assert s["svg"]["line"] == "s.line" and s["data"]["y"] == [1.0, 3.0, 2.0, 4.0]
    with pytest.raises(ValueError, match="where must be"):
        fp.step(ax, [0, 1], [0, 1], series="bad", where="center")


def test_stem_parts_under_one_series(tmp_path):
    fig, ax = plt.subplots()
    fp.stem(ax, [1, 2, 3], [2.0, 1.0, 3.0], series="spikes")
    man, ids = _save(fig, tmp_path)
    assert {"spikes.points", "spikes.point.0", "spikes.point.2", "spikes.segment", "spikes.baseline"} <= ids
    (s,) = man["series"]
    assert s["kind"] == "stem" and sorted(s["roles"]) == ["point", "reference-line", "segment"]
    assert s["stem"]["baseline"] == 0.0 and s["stem"]["orientation"] == "vertical"
    assert [p["y"] for p in s["points"]] == [2.0, 1.0, 3.0]
    tree = json.dumps(man["parts"])
    assert "spikes.baseline" in tree and "spikes.segment" in tree


def test_categorical_and_date_axes_label_their_values(tmp_path):
    fig, (a, b) = plt.subplots(1, 2)
    fp.bar(a, ["ctl", "drug"], [1, 2], series="bars")
    fp.line(a, ["ctl", "drug"], [1.5, 2.5], series="ln")
    days = [datetime(2024, 3, 1) + timedelta(days=i) for i in range(3)]
    fp.line(b, days, [1, 2, 3], series="ts")
    fp.scatter(b, days, [3, 2, 1], series="pts")
    man, ids = _save(fig, tmp_path)
    by = {s["id"]: s for s in man["series"]}
    assert by["panel.a.bars"]["data"]["xLabels"] == ["ctl", "drug"] and by["panel.a.ln"]["data"]["xLabels"] == ["ctl", "drug"]
    assert "yLabels" not in by["panel.a.bars"]["data"]
    assert by["panel.b.ts"]["data"]["xIso"][0].startswith("2024-03-01T00:00:00") and by["panel.b.pts"]["data"]["xIso"][2].startswith("2024-03-03")
    assert "xIso" not in by["panel.a.ln"]["data"] and "xLabels" not in by["panel.b.ts"]["data"]
    axes = {ax["panelId"]: ax for ax in man["axes"]}
    assert axes["panel.a"]["x"]["units"]["kind"] == "category" and axes["panel.b"]["x"]["units"]["kind"] == "date"


def test_inset_panels_know_their_host(tmp_path):
    fig, ax = plt.subplots()
    fp.panel(ax, "main")
    fp.line(ax, [0, 1], [0, 1], series="s")
    inset = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
    fp.panel(inset, "detail")
    fp.line(inset, [0, 1], [1, 0], series="d")
    man, ids = _save(fig, tmp_path)
    panels = {p["id"]: p for p in man["panels"]}
    assert panels["panel.detail"]["insetOf"] == "panel.main" and "insetOf" not in panels["panel.main"]
    # unnamed inset: lettered, still an inset of its host
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [0, 1], series="s")
    inset = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
    fp.line(inset, [0, 1], [1, 0], series="d")
    man, ids = _save(fig, tmp_path, "u.svg")
    insets = [p for p in man["panels"] if p.get("insetOf")]
    assert len(insets) == 1 and insets[0]["insetOf"] in {p["id"] for p in man["panels"]}


def test_secondary_axis_is_the_panels_x2_with_sampled_transform(tmp_path):
    fig, ax = plt.subplots()
    fp.line(ax, [1, 2, 3, 4], [1, 4, 9, 16], series="s")
    ax.set_xlabel("frequency (Hz)")
    sec = fp.secondary_axis(ax, "top", functions=(lambda f: 1000 / f, lambda w: 1000 / w), label="wavelength (mm)")
    man, ids = _save(fig, tmp_path)
    assert "panels" not in man  # the secondary axis is no panel
    assert "axis.x2" in ids and "axis.x2.title" in ids and "axis.x2.spine.top" in ids and "axis.x2.tick.1" in ids
    (axes,) = man["axes"]
    x2 = axes["x2"]
    assert x2["label"] == "wavelength (mm)" and x2["secondary"]["of"] == "x"
    lo, hi = ax.get_xlim()
    samples = x2["secondary"]["samples"]
    assert samples[0][0] == pytest.approx(lo) and samples[-1][0] == pytest.approx(hi)
    assert all(b == pytest.approx(1000 / a) for a, b in samples)
    assert not any(i.startswith("extra.") for i in ids)
    # a secondary y axis on the right
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [0, 100], series="c")
    fp.secondary_axis(ax, "right", functions=(lambda c: c * 9 / 5 + 32, lambda f: (f - 32) * 5 / 9), label="°F")
    man, ids = _save(fig, tmp_path, "y.svg")
    assert "axis.y2.title" in ids and man["axes"][0]["y2"]["secondary"]["of"] == "y"
    with pytest.raises(ValueError, match="location must be"):
        fp.secondary_axis(ax, "middle", functions=(lambda v: v, lambda v: v))
