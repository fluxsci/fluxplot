"""fp.fluxbox — the glowbar with a box plot for its summary.

What this file pins down: the box, whiskers and outliers are exactly matplotlib's box-plot statistics,
every piece is a named part with a manifest payload, the mean notch never silently disappears, and
everything outside the box — lanes, colours, names, connectors — is the glowbar's, dot for dot.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import cbook  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

import fluxplot as fp  # noqa: E402
from fluxplot.signature_fluxplots._colour import perceptual as _perceptual  # noqa: E402

S = [1.471, 2.378, 1.898, 2.297, 1.648, 1.823]
SD = [2.765, 2.876, 2.085, 3.468, 2.802, 3.329]  # 2.085 lies below Q1 - 1.5 IQR: an outlier


def _table():
    subj = [f"B6_{i}" for i in range(1, 13)]
    group = ["S" if i % 2 else "SD" for i in range(1, 13)]
    s_it, sd_it = iter(S), iter(SD)
    app = [next(s_it) if g == "S" else next(sd_it) for g in group]
    return {"subject": subj, "type": group, "APP": app}


def _save(fig, tmp_path, name="fb"):
    out = tmp_path / f"{name}.svg"
    res = fp.save(fig, str(out))
    plt.close(fig)
    return res, json.loads((tmp_path / f"{name}.fluxplot.json").read_text()), out.read_text()


def _series(manifest):
    return {s["id"]: s for s in manifest["series"]}


def test_every_piece_is_a_named_part_with_the_statistics_in_the_manifest(tmp_path):
    fig, ax = plt.subplots()
    fb = fp.fluxbox(_table(), x="type", y="APP", units="subject", ax=ax)
    res, man, svg = _save(fig, tmp_path)
    assert res.warnings == []
    assert man["plotType"] == "fluxbox"
    series = _series(man)
    for cat in ("s", "sd"):
        for part in ("box", "whiskers", "median", "mean"):
            assert f'id="{cat}.{part}"' in svg
        assert f'id="{cat}.caps"' not in svg  # whiskers are capless by default
        payload = series[cat]["fluxbox"]
        assert payload["part"] == "summary" and payload["whis"] == 1.5
    assert series["sd"]["fluxbox"]["outliers"] == [2.085]
    assert 'id="sd.fliers"' not in svg  # the points already show the outlier
    assert 'id="b6-8.points"' in svg and 'id="b6-8.point.0"' in svg
    assert series["b6-8"]["fluxbox"] == {
        "part": "unit", "units": "subject", "unit": "B6_8", "categories": ["SD"],
        "colors": series["b6-8"]["fluxbox"]["colors"]}
    assert fb.categories == ["S", "SD"] and fb.series == {"S": "S", "SD": "SD"}


@pytest.mark.parametrize("whis", [1.5, 0.5, 3.0, "range", (5, 95)])
def test_box_and_whiskers_are_exactly_matplotlibs(whis):
    rng = np.random.default_rng(4)
    v = np.r_[rng.lognormal(0, 0.6, 30), 9.0, 0.01]
    fig, ax = plt.subplots()
    st = fp.fluxbox({"g": ["a"] * v.size, "v": v}, x="g", y="v", ax=ax, whis=whis).stats["a"]
    plt.close(fig)
    ref = cbook.boxplot_stats(v, whis=(0, 100) if whis == "range" else whis)[0]
    assert (st["q1"], st["median"], st["q3"]) == pytest.approx((ref["q1"], ref["med"], ref["q3"]))
    assert (st["whiskerLow"], st["whiskerHigh"]) == pytest.approx((ref["whislo"], ref["whishi"]))
    assert st["outliers"] == pytest.approx(sorted(ref["fliers"]))
    assert st["mean"] == pytest.approx(v.mean()) and st["n"] == v.size
    assert st["sd"] == pytest.approx(v.std(ddof=1)) and st["sem"] == pytest.approx(v.std(ddof=1) / np.sqrt(v.size))


def test_the_box_spans_the_quartiles_at_its_stated_width():
    fig, ax = plt.subplots()
    fb = fp.fluxbox(_table(), x="type", y="APP", ax=ax, box_width=6.0)
    st, body = fb.stats["SD"], fb.artists["box"][1]
    assert list(body.get_ydata()) == pytest.approx([st["q1"], st["q3"]])
    assert body.get_linewidth() == 6.0 and body.get_solid_capstyle() == "butt"
    lo_seg, hi_seg = np.split(fb.artists["whiskers"][1].get_ydata(), [2])
    assert list(lo_seg) == pytest.approx([st["q1"], st["whiskerLow"]])
    assert list(hi_seg[1:]) == pytest.approx([st["q3"], st["whiskerHigh"]])  # hi_seg[0] is the break
    plt.close(fig)


def test_everything_outside_the_box_is_the_glowbars():
    """Swapping fp.glowbar for fp.fluxbox must keep every dot, colour, lane and name."""
    kw = {"x": "type", "y": "APP", "units": "subject", "palette": ["cmasher.emerald", "YlOrRd"],
          "connect_identical_points_across_x_values": True}
    t = _table()
    t["subject"] = [f"m{(i % 6) + 1}" for i in range(12)]  # paired: six mice in both groups
    fig, (a, b) = plt.subplots(1, 2)
    gb, fb = fp.glowbar(t, ax=a, **kw), fp.fluxbox(t, ax=b, **kw)
    assert gb.categories == fb.categories and gb.series == fb.series and gb.unit_series == fb.unit_series
    assert gb.group_colors == fb.group_colors and gb.point_colors == fb.point_colors
    for g, f in zip(gb.artists["points"], fb.artists["points"]):
        assert np.array_equal(g.get_offsets(), f.get_offsets())
        assert np.array_equal(g.get_facecolors(), f.get_facecolors())
    assert [ln.get_xydata().tolist() for ln in gb.artists["lines"]] == \
        [ln.get_xydata().tolist() for ln in fb.artists["lines"]]
    assert {c: s["x"] for c, s in gb.stats.items()} == {c: s["x"] for c, s in fb.stats.items()}
    plt.close(fig)


def test_the_mean_notch_is_cut_in_the_box_and_drawn_solid_outside_it():
    t = {"g": ["in"] * 6 + ["out"] * 6 + ["one"],
         "v": [1, 2, 3, 4, 5, 6] + [0, 0.1, 0.2, 0.3, 0.2, 20] + [7]}  # "out": mean ≈ 3.5 ≫ Q3
    fig, ax = plt.subplots()
    fb = fp.fluxbox(t, x="g", y="v", ax=ax)
    white = to_rgba("white")
    face = {c: to_rgba(m.get_markerfacecolor()) for c, m in zip(fb.categories, fb.artists["mean"])}
    assert face["in"] == white                                    # cut out of the box
    ink = {c: to_rgba(m.get_color()) for c, m in zip(fb.categories, fb.artists["median"])}
    assert face["out"] == pytest.approx(ink["out"])   # nothing to cut: drawn solid, in the ink
    assert face["one"] == pytest.approx(ink["one"])   # a box of no height: drawn solid
    assert fb.stats["out"]["mean"] > fb.stats["out"]["q3"]
    plt.close(fig)


def _lightness(c):
    return _perceptual(np.array(to_rgba(c)[:3]))[0]


@pytest.mark.parametrize("ground", ["white", "#1C1B1A"])
@pytest.mark.parametrize("colour", ["#205EA6", "#E31A1C", "#F2C12E", "#AAAAAA"])
def test_the_median_is_a_solid_line_that_stands_off_its_box(ground, colour):
    """Whatever the group colour and the background, the median keeps its hue and reads clearly."""
    fig, ax = plt.subplots()
    ax.set_facecolor(ground)
    fb = fp.fluxbox(_table(), x="type", y="APP", ax=ax, group_color=colour)
    body, med = fb.artists["box"][0], fb.artists["median"][0]
    assert body.get_alpha() == 0.5 and med.get_alpha() is None     # a wash of a box; an opaque line
    assert med.get_markersize() == body.get_linewidth() == 7.0       # exactly as wide as the box
    col, bg = np.array(to_rgba(colour)[:3]), np.array(to_rgba(ground)[:3])
    box = 0.5 * col + 0.5 * bg                                       # the box as it renders
    ink = np.array(to_rgba(med.get_color())[:3])
    assert abs(_lightness(ink) - _lightness(box)) >= 30 - 0.5
    if abs(_lightness(col) - _lightness(box)) >= 30:
        assert ink == pytest.approx(col)                             # already clear: the group colour itself
    toward = np.zeros(3) if _lightness(col) <= _lightness(box) else np.ones(3)
    k = np.linalg.norm(ink - col) / np.linalg.norm(toward - col)
    assert ink == pytest.approx(col + k * (toward - col), abs=1e-9)  # same hue: only deepened / lifted
    plt.close(fig)


def test_the_whiskers_caps_and_fliers_share_the_medians_ink():
    fig, ax = plt.subplots()
    fb = fp.fluxbox(_table(), x="type", y="APP", ax=ax, group_color="#6CB4EE",  # pale: ink is deepened
                    show_caps=True, show_individual_points=False)
    ink = to_rgba(fb.artists["median"][1].get_color())
    assert ink != pytest.approx(to_rgba("#6CB4EE"))
    assert to_rgba(fb.artists["whiskers"][1].get_color()) == ink
    assert to_rgba(fb.artists["caps"][1].get_color()) == ink
    assert tuple(fb.artists["fliers"][0].get_facecolors()[0]) == pytest.approx(ink)  # only SD has one
    plt.close(fig)


def test_the_median_colour_and_box_alpha_can_be_set():
    fig, ax = plt.subplots()
    fb = fp.fluxbox(_table(), x="type", y="APP", ax=ax, median_color="#100F0F", box_alpha=1.0)
    assert to_rgba(fb.artists["median"][0].get_color()) == to_rgba("#100F0F")
    assert to_rgba(fb.artists["whiskers"][0].get_color()) == to_rgba("#100F0F")  # the whiskers follow
    assert fb.artists["box"][0].get_alpha() == 1.0
    fb = fp.fluxbox(_table(), x="type", y="APP", ax=ax, whisker_color="#AF3029", show_caps=True)
    assert to_rgba(fb.artists["whiskers"][0].get_color()) == to_rgba("#AF3029")
    assert to_rgba(fb.artists["caps"][0].get_color()) == to_rgba("#AF3029")         # caps follow whiskers
    assert to_rgba(fb.artists["median"][0].get_color()) != to_rgba("#AF3029")
    plt.close(fig)
    with pytest.raises(ValueError, match="box_alpha"):
        fp.fluxbox(_table(), x="type", y="APP", box_alpha=1.5)


def test_fliers_appear_only_when_the_points_do_not_already_show_them(tmp_path):
    fig, ax = plt.subplots()
    fb = fp.fluxbox(_table(), x="type", y="APP", units="subject", ax=ax, show_individual_points=False)
    assert fb.artists["points"] == []
    (fl,) = fb.artists["fliers"]                      # only SD has an outlier
    assert fl.get_offsets().tolist() == [[fb.stats["SD"]["x"], 2.085]]
    _, _, svg = _save(fig, tmp_path)
    assert 'id="sd.fliers"' in svg and 'id="s.fliers"' not in svg
    fig, ax = plt.subplots()
    assert len(fp.fluxbox(_table(), x="type", y="APP", ax=ax, show_fliers=True).artists["fliers"]) == 1
    assert fp.fluxbox(_table(), x="type", y="APP", ax=ax, show_individual_points=False,
                      show_fliers=False).artists["fliers"] == []
    plt.close(fig)
    with pytest.raises(ValueError, match="show_fliers"):
        fp.fluxbox(_table(), x="type", y="APP", show_fliers="yes")


def test_a_whisker_of_no_length_draws_no_whisker_or_cap():
    t = {"g": ["a"] * 5, "v": [5, 5, 5, 6, 7]}  # Q1 = min = 5: nothing below the box
    fig, ax = plt.subplots()
    fb = fp.fluxbox(t, x="g", y="v", ax=ax, show_caps=True)
    st = fb.stats["a"]
    assert st["whiskerLow"] == st["q1"] == 5
    assert list(fb.artists["whiskers"][0].get_ydata()) == [st["q3"], st["whiskerHigh"]]
    assert list(fb.artists["caps"][0].get_ydata()) == [st["whiskerHigh"]]
    plt.close(fig)


def test_every_part_can_be_switched_off(tmp_path):
    fig, ax = plt.subplots()
    fb = fp.fluxbox(_table(), x="type", y="APP", units="subject", ax=ax, show_mean=False,
                    show_median=False, show_whiskers=False, show_caps=False, show_fliers=False,
                    show_individual_points=False)
    assert all(fb.artists[k] == [] for k in ("mean", "median", "whiskers", "caps", "fliers", "points"))
    _, man, svg = _save(fig, tmp_path)
    assert 'id="sd.box"' in svg and 'id="sd.mean"' not in svg and "b6-8" not in svg
    assert _series(man)["sd"]["fluxbox"]["median"] == pytest.approx(np.median(SD))  # still recorded


@pytest.mark.parametrize("side, expected", [
    ("outer", {"S": -0.42, "SD": 1.42}),
    ("left", {"S": -0.42, "SD": 0.58}),
    ("right", {"S": 0.42, "SD": 1.42}),
])
def test_box_side(side, expected):
    fig, ax = plt.subplots()
    kw = {} if side == "outer" else {"box_side": side}
    fb = fp.fluxbox(_table(), x="type", y="APP", units="subject", ax=ax, **kw)
    assert {c: s["x"] for c, s in fb.stats.items()} == pytest.approx(expected)
    plt.close(fig)
    with pytest.raises(ValueError, match="box_side"):
        fp.fluxbox(_table(), x="type", y="APP", box_side="inside")


@pytest.mark.parametrize("whis", [-1, "tukey", (95, 5), (0, 101), True])
def test_bad_whisker_rules_are_refused(whis):
    with pytest.raises(ValueError, match="whis"):
        fp.fluxbox(_table(), x="type", y="APP", whis=whis)


def test_errors_name_the_fluxbox():
    t = {"t": ["a", "b"], "v": [1.0, 2.0]}
    with pytest.raises(ValueError, match="^fluxbox: connect_identical_points_across_x_values needs units"):
        fp.fluxbox(t, x="t", y="v", connect_identical_points_across_x_values=True)
    with pytest.raises(KeyError, match="fluxbox: y='nope' is not a column"):
        fp.fluxbox(t, x="t", y="nope")


def test_whisker_range_is_json_safe(tmp_path):
    fig, ax = plt.subplots()
    fb = fp.fluxbox(_table(), x="type", y="APP", ax=ax, whis=float("inf"))
    assert fb.stats["SD"]["whis"] == "range" and fb.stats["SD"]["whiskerLow"] == min(SD)
    _, man, _ = _save(fig, tmp_path)
    assert _series(man)["sd"]["fluxbox"]["whis"] == "range"


def test_fluxbox_is_deterministic(tmp_path):
    outs = []
    for _ in range(2):
        fig, ax = plt.subplots(figsize=(2, 2))
        fp.fluxbox(_table(), x="type", y="APP", units="subject", ax=ax)
        fp.save(fig, str(tmp_path / "d.svg"))
        plt.close(fig)
        outs.append((tmp_path / "d.svg").read_bytes())
    assert outs[0] == outs[1]
