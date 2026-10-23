"""Statistical, geometric and Flux export contracts for the mean circle signature."""
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from lxml import etree
from matplotlib.colors import to_rgba
from scipy.stats import t

import fluxplot as fp
from test_fluxbox import _pixel, _render


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def draw(values=(1, 2, 3, 5, 9), **kw):
    ax = kw.pop("ax", None)
    if ax is None:
        _, ax = plt.subplots()
    return fp.meanmark(x=["a"] * len(values), y=values, ax=ax, **kw)


@pytest.mark.parametrize("interval", ["sem", "sd", "ci", "range", lambda a: (a.min() - 1, a.max() + 2)])
def test_exact_statistics_and_interval(interval):
    v = np.array([1, 2, 3, 5, 9], dtype=float)
    st = draw(list(v) + [None, np.inf, np.nan], interval=interval, confidence=.9).stats["a"]
    mean, sd = v.mean(), v.std(ddof=1)
    sem = sd / np.sqrt(len(v))
    assert st["n"] == len(v)
    assert (st["mean"], st["sd"], st["sem"]) == pytest.approx((mean, sd, sem))
    if callable(interval):
        expected = interval(v)
    elif interval == "range":
        expected = v.min(), v.max()
    else:
        half = {"sem": sem, "sd": sd, "ci": t.ppf(.95, len(v) - 1) * sem}[interval]
        expected = mean - half, mean + half
    assert (st["low"], st["high"]) == pytest.approx(expected)
    assert st["meanKind"] == "arithmetic"
    assert st["confidence"] == (.9 if interval == "ci" else None)


@pytest.mark.parametrize("interval", ["sem", "sd", "ci"])
def test_singleton_missing_and_constant_categories(interval, tmp_path):
    fig, ax = plt.subplots()
    mm = fp.meanmark(x=["one", "empty", "same", "same"], y=[3, None, 4, 4], ax=ax,
                    order=["one", "empty", "same", "absent"], interval=interval)
    assert list(mm.stats) == ["one", "same"]
    assert mm.stats["one"]["low"] is None and mm.stats["one"]["sem"] is None
    assert mm.stats["same"]["low"] == mm.stats["same"]["high"] == 4
    assert len(mm.artists["mean"]) == 2 and len(mm.artists["error_low"]) == 1
    result = fp.save(fig, str(tmp_path / "special.svg"))
    assert result.warnings == []
    assert "NaN" not in Path(result.manifest).read_text()
    assert [t.get_text() for t in ax.get_xticklabels()] == ["one", "empty", "same", "absent"]


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
@pytest.mark.parametrize("inverted", [False, True])
def test_circle_and_two_arms_track_data_dpi_scale_and_limits(orientation, inverted):
    mm = draw([10, 12, 18, 24], show_individual_points=False, mark_size=12,
              show_caps=True, cap_size=4, orientation=orientation, value_scale="log")
    ax, st = mm.ax, mm.stats["a"]
    if inverted:
        (ax.invert_xaxis if orientation == "horizontal" else ax.invert_yaxis)()
    for dpi in (90, 180):
        ax.figure.set_dpi(dpi)
        ax.figure.canvas.draw()
        circle = mm.artists["mean"][0].get_path().get_extents()
        assert (circle.width, circle.height) == pytest.approx((12*dpi/72,)*2)
        center = ax.transData.transform((st["x"], st["y"]))
        assert (circle.x0+circle.width/2, circle.y0+circle.height/2) == pytest.approx(center)
        for side in ("low", "high"):
            bound = st[side]
            expected = ax.transData.transform((bound, 0) if orientation == "horizontal" else (0, bound))
            line = mm.artists["error_"+side][0].get_path().vertices
            assert line[1] == pytest.approx(expected)
            assert np.linalg.norm(line[0]-center) == pytest.approx(6*dpi/72)
            cap = mm.artists["cap_"+side][0].get_path().vertices
            assert cap.mean(axis=0) == pytest.approx(expected)
            assert np.linalg.norm(cap[1]-cap[0]) == pytest.approx(4*dpi/72)
    # Changing limits must move the parts without changing their point-sized circle.
    (ax.set_xlim if orientation == "horizontal" else ax.set_ylim)(40, 2) if inverted else \
        (ax.set_xlim if orientation == "horizontal" else ax.set_ylim)(2, 40)
    ax.figure.canvas.draw()
    assert mm.artists["mean"][0].get_path().get_extents().width == pytest.approx(30)
    assert st["mean"] == 16  # arithmetic even on log axes


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_true_notch_reveals_background_and_is_exactly_at_mean(orientation):
    fig, ax = plt.subplots(dpi=200)
    mm = draw([1, 1, 2, 2], ax=ax, show_individual_points=False, mark_size=30,
              mark_edge_width=1, show_notch=True, notch_depth=10, notch_height=16,
              orientation=orientation)
    st = mm.stats["a"]
    (ax.set_xlim if orientation == "horizontal" else ax.set_ylim)(0, 3)
    behind = (ax.axvspan if orientation == "horizontal" else ax.axhspan)(1.2, 1.8, color="#00ff00", zorder=-10)
    ax.grid(False)
    # 12 points into the circle's perpendicular radius, at the mean value.
    def pixel(image, distance):
        center = ax.transData.transform((st["x"], st["y"]))
        xy = center + np.array([0, distance] if orientation == "horizontal" else [distance, 0])*fig.dpi/72
        return image[int(fig.canvas.get_width_height()[1]-xy[1]), int(xy[0])]
    image = _render(fig)
    assert pixel(image, 12) == pytest.approx([0, 1, 0, 1], abs=.02)
    behind.remove()
    image = _render(fig)
    assert pixel(image, 12)[3] == 0
    assert pixel(image, 0)[3] == 1
    path = mm.artists["mean"][0].get_path().vertices
    center = ax.transData.transform((st["x"], st["y"]))
    relative = (path-center)/(fig.dpi/72)
    if orientation == "horizontal":
        relative = relative[:, ::-1]
    assert any(np.allclose(p, [5, 0]) for p in relative)
    assert any(np.allclose(p, [-5, 0]) for p in relative)


def test_hollow_and_solid_fill_and_independent_styling():
    mm = draw(mark_color="#205ea6", mark_fill_color="stroke", mark_fill_alpha=.4,
              mark_edge_width=2, error_width=3, error_color="red", show_caps=True,
              cap_width=4, cap_color="green")
    mark = mm.artists["mean"][0]
    assert mark.get_facecolor() == pytest.approx((*to_rgba("#205ea6")[:3], .4))
    assert mark.get_edgecolor() == to_rgba("#205ea6") and mark.get_linewidth() == 2
    assert mm.artists["error_low"][0].get_edgecolor() == to_rgba("red")
    assert mm.artists["error_high"][0].get_linewidth() == 3
    assert mm.artists["cap_low"][0].get_edgecolor() == to_rgba("green")
    assert mm.artists["cap_high"][0].get_linewidth() == 4
    mm = draw(show_individual_points=False, mark_fill_color="none", mark_size=30)
    image = _render(mm.ax.figure)
    y, x = _pixel(mm.ax.figure, mm.ax, 0, 0, mm.stats["a"]["mean"])
    assert image[y, x, 3] == 0  # no error arm through the hollow centre


def test_default_tint_is_opaque_and_derived_from_stroke():
    mm = draw(mark_color="#205ea6")
    ink = np.array(to_rgba("#205ea6")[:3])
    assert mm.artists["mean"][0].get_facecolor() == pytest.approx((* (.75+.25*ink), 1))
    assert len(mm.artists["cap_low"]) == len(mm.artists["cap_high"]) == 1
    assert mm.stats["a"]["interval"] == "ci"
    assert mm.stats["a"]["confidence"] == .95


def test_short_intervals_do_not_reverse_arms_or_paint_through_circle():
    mm = draw([1, 1, 1], show_individual_points=False, mark_fill_color="none")
    mm.ax.figure.canvas.draw()
    for key in ("error_low", "error_high"):
        a, b = mm.artists[key][0].get_path().vertices
        assert a == pytest.approx(b)


@pytest.mark.parametrize("kw,match", [
    ({"interval":"bad"}, "interval"), ({"confidence":1}, "confidence"),
    ({"mark_size":0}, "mark_size"), ({"mark_edge_width":-1}, "mark_edge_width"),
    ({"error_width":np.nan}, "error_width"), ({"mark_offset":np.inf}, "mark_offset"),
    ({"mark_side":"bad"}, "mark_side"), ({"orientation":"bad"}, "orientation"),
    ({"mark_fill_alpha":2}, "mark_fill_alpha"), ({"mark_fill_mix":-1}, "mark_fill_mix"),
    ({"show_notch":True,"notch_depth":4}, "notch_depth"),
    ({"show_notch":True,"notch_height":7}, "notch_height"),
    ({"interval":lambda a:(5, 1)}, "bounds"), ({"interval":lambda a:(1, np.inf)}, "bounds"),
    ({"interval":lambda a:(1,)}, "callable"), ({"value_scale":"symlog"}, "value_scale"),
])
def test_invalid_controls_fail_clearly(kw, match):
    with pytest.raises(ValueError, match="meanmark:.*"+match):
        draw(**kw)


def test_log_scale_refuses_nonpositive_data_or_bounds_before_drawing():
    fig, ax = plt.subplots()
    for values, kw in (([-1, 2], {}), ([1, 2, 100], {"interval":"sd"})):
        with pytest.raises(ValueError, match="positive"):
            draw(values, ax=ax, value_scale="log", **kw)
        assert not ax.patches and not ax.collections


@pytest.mark.parametrize("units", [False, True])
def test_flux_export_each_part_has_stable_identity_and_correct_editing_kind(tmp_path, units):
    fig, ax = plt.subplots()
    kw = {"units":["u1", "u2", "u3"]} if units else {}
    mm = fp.meanmark(x=["a"]*3, y=[1, 2, 4], ax=ax, show_caps=True,
                     show_notch=True, **kw)
    saved = fp.save(fig, str(tmp_path / "parts.svg"), recipe={"params":{"show_caps":True}})
    assert saved.warnings == []
    man = json.loads(Path(saved.manifest).read_text())
    root = etree.parse(saved.svg)
    ids = root.xpath("//@id")
    assert len(ids) == len(set(ids))
    assert man["plotType"] == "meanmark"
    series = next(s for s in man["series"] if s["id"] == "a")
    assert series["kind"] == "meanmark"
    def nodes(node):
        yield node
        for child in node.get("children", []):
            yield from nodes(child)
    mean_group = next(n for n in nodes(man["parts"]) if n.get("id") == "a.means")
    assert mean_group["kind"] == "shape" and mean_group["members"] == ["a.mean"]
    assert set(series["svg"]["errorbars"]) == {"a.error-low", "a.error-high"}
    assert set(series["svg"]["caps"]) == {"a.cap-low", "a.cap-high"}
    for part, role, kind in (("mean","mean","shape"), ("error-low","errorbar","line"),
                             ("error-high","errorbar","line"), ("cap-low","cap","line"),
                             ("cap-high","cap","line")):
        node, = root.xpath(f'//*[@id="a.{part}"]')
        assert node.attrib["data-kind"] == kind
        assert node.attrib["data-role"] == role
        assert {"role":role, "svgId":"a."+part} in series["components"]
    assert series["meanmark"]["low"] == pytest.approx(mm.stats["a"]["low"])
    recipe = json.loads(Path(saved.recipe).read_text())
    assert recipe["params"]["show_caps"] is True
    for artist in mm.artists["cap_low"] + mm.artists["cap_high"]:
        artist.set_visible(False)
    second = fp.save(fig, str(tmp_path / "parts.svg"))
    new_ids = etree.parse(second.svg).xpath("//@id")
    assert all("a."+s in new_ids for s in ("mean", "error-low", "error-high"))
    assert "a.cap-low" not in new_ids


def test_points_and_connectors_rotate_together():
    kw = dict(x=["a", "b"]*3, y=[1, 2, 3, 4, 5, 6], units=["u1"]*2+["u2"]*2+["u3"]*2,
              connect_identical_points_across_x_values=True)
    fig, (a, b) = plt.subplots(1, 2)
    v, h = fp.meanmark(ax=a, **kw), fp.meanmark(ax=b, orientation="horizontal", **kw)
    for first, second in zip(v.artists["points"], h.artists["points"]):
        np.testing.assert_allclose(first.get_offsets(), second.get_offsets()[:, ::-1])
    for first, second in zip(v.artists["lines"], h.artists["lines"]):
        assert first.get_xydata() == pytest.approx(second.get_xydata()[:, ::-1])
    with pytest.raises(ValueError, match="vertical"):
        h.brackets([])


def test_summary_only_centres_on_categories_and_error_toggle():
    mm = draw(show_individual_points=False, show_error=False, show_caps=True)
    assert mm.stats["a"]["categoryPosition"] == 0
    assert all(mm.artists[key] == [] for key in ("points", "error_low", "error_high", "cap_low", "cap_high"))
    mm = draw(mark_side="left", mark_offset=1.2)
    assert mm.stats["a"]["categoryPosition"] == -1.2
    assert mm.ax.get_xlim()[0] < -1.2


def test_recipe_reruns_with_changed_visual_controls(tmp_path):
    import shutil
    script = tmp_path / "meanmark_example.py"
    shutil.copy(Path(__file__).parents[1] / "examples" / script.name, script)
    subprocess.run([sys.executable, str(script)], cwd=tmp_path, check=True, capture_output=True)
    output = tmp_path / "out" / "meanmark.svg"
    recipe_path = output.with_suffix(".recipe.json")
    recipe = json.loads(recipe_path.read_text())
    before = output.read_text()
    overrides = dict(recipe["params"], show_caps=False, mark_size=12, show_notch=True)
    subprocess.run([recipe["command"], *recipe["args"]], cwd=recipe_path.parent / recipe["cwd"],
                   env=dict(os.environ, FLUX_PARAMS=json.dumps(overrides)),
                   check=True, capture_output=True)
    after = output.read_text()
    assert 'id="sws.cap-low"' in before and 'id="sws.cap-low"' not in after
    assert 'id="sws.mean"' in before and 'id="sws.mean"' in after
    assert json.loads(recipe_path.read_text())["params"]["mark_size"] == 12


def test_summary_only_manifest_colour_uses_the_visible_stroke(tmp_path):
    mm = draw(show_individual_points=False, mark_fill_color="none", group_color="#205ea6", error_color="red")
    saved = fp.save(mm.ax.figure, str(tmp_path / "colour.svg"))
    series = json.loads(Path(saved.manifest).read_text())["series"][0]
    assert series["color"]["hex"] == "#205ea6" and series["color"]["alpha"] == 1


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_caps_never_cross_a_hollow_zero_error_circle(orientation):
    mm = draw([1, 1, 1], show_individual_points=False, mark_fill_color="none",
              mark_size=30, show_caps=True, orientation=orientation)
    mm.ax.grid(False)
    img = _render(mm.ax.figure)
    center = mm.ax.transData.transform((mm.stats["a"]["x"], mm.stats["a"]["y"]))
    pixel = img[int(img.shape[0]-center[1]), int(center[0])]
    assert pixel[3] == 0


def test_brackets_clear_summary_only_uncertainty():
    from fluxplot.brackets import _data_top
    fig, ax = plt.subplots()
    mm = fp.meanmark(x=["a"]*3+["b"]*3, y=[1, 2, 4, 2, 4, 8], ax=ax,
                     show_individual_points=False, interval="ci", show_caps=True)
    fig.canvas.draw()
    high = max(s["high"] for s in mm.stats.values())
    assert _data_top(ax, 0, 1) >= high - 1e-10
    rows = [{"groups":["a", "b"], "p-value":.01}]
    bracket, = mm.brackets(rows, p_column="p-value")
    assert min(bracket.get_ydata()) > high


def test_default_pairing_is_automatic_and_can_be_disabled():
    kw = dict(x=["a", "a", "b", "b"], y=[1, 2, 3, 4])
    unpaired = fp.meanmark(**kw)
    assert unpaired.artists["lines"] == []
    paired = fp.meanmark(**kw, units=["u1", "u2"] * 2)
    assert len(paired.artists["lines"]) == 2
    assert paired.stats["a"]["categoryPosition"] == .25
    assert paired.stats["b"]["categoryPosition"] == 1.25
    disabled = fp.meanmark(**kw, units=["u1", "u2"] * 2,
                          connect_identical_points_across_x_values=False)
    assert disabled.artists["lines"] == []
    summary = fp.meanmark(**kw, show_individual_points=False)
    assert summary.stats["a"]["categoryPosition"] == 0
    with pytest.raises(ValueError, match="units"):
        fp.meanmark(**kw, connect_identical_points_across_x_values=True)
    with pytest.raises(ValueError, match="connect_identical_points_across_x_values"):
        fp.meanmark(**kw, connect_identical_points_across_x_values="invalid")
