"""fp.glowbar — the first signature fluxplot.

What this file pins down: the statistics drawn are exactly the documented ones, every piece is a
named part with a manifest payload, and a unit's lane/colour comes from the TABLE, never from the
plotted values — so separate plots of different measures of the same animals agree.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import fluxplot as fp  # noqa: E402
from fluxplot.signature_fluxplots.glowbar import _perceptual  # noqa: E402

S = [1.471, 2.378, 1.898, 2.297, 1.648, 1.823]
SD = [2.765, 2.876, 2.085, 3.468, 2.802, 3.329]


def _table(apv=None):
    subj = [f"B6_{i}" for i in range(1, 13)]
    group = ["S" if i % 2 else "SD" for i in range(1, 13)]
    app = [None] * 12
    s_it, sd_it = iter(S), iter(SD)
    for k, g in enumerate(group):
        app[k] = next(s_it) if g == "S" else next(sd_it)
    return {"subject": subj, "type": group, "APP": app if apv is None else apv}


def _save(fig, tmp_path, name="gb"):
    out = tmp_path / f"{name}.svg"
    res = fp.save(fig, str(out))
    plt.close(fig)
    return res, json.loads((tmp_path / f"{name}.fluxplot.json").read_text()), out.read_text()


def _series(manifest):
    return {s["id"]: s for s in manifest["series"]}


def test_every_piece_is_a_named_part_with_the_statistics_in_the_manifest(tmp_path):
    fig, ax = plt.subplots()
    gb = fp.glowbar(_table(), x="type", y="APP", units="subject", ax=ax, interval="iqr")
    res, man, svg = _save(fig, tmp_path)
    assert res.warnings == []
    assert man["plotType"] == "glowbar"
    series = _series(man)
    for cat in ("s", "sd"):
        for part in ("glow", "caps", "mean", "median"):
            assert f'id="{cat}.{part}"' in svg
        payload = series[cat]["glowbar"]
        assert payload["part"] == "summary" and payload["interval"] == "iqr"
    # one series per unit, named by the unit, carrying its identity
    assert 'id="b6-8.points"' in svg and 'id="b6-8.point.0"' in svg
    assert series["b6-8"]["glowbar"] == {
        "part": "unit", "units": "subject", "unit": "B6_8", "categories": ["SD"],
        "colors": series["b6-8"]["glowbar"]["colors"]}
    assert gb.categories == ["S", "SD"] and gb.series == {"S": "S", "SD": "SD"}


def test_statistics_are_exactly_the_documented_ones(tmp_path):
    fig, ax = plt.subplots()
    gb = fp.glowbar(_table(), x="type", y="APP", ax=ax, interval="iqr")
    v = np.array(SD)
    st = gb.stats["SD"]
    assert st["n"] == 6
    assert st["mean"] == pytest.approx(v.mean())
    assert st["median"] == pytest.approx(np.median(v))
    assert st["sd"] == pytest.approx(v.std(ddof=1))
    assert st["sem"] == pytest.approx(v.std(ddof=1) / np.sqrt(6))
    assert (st["low"], st["high"]) == pytest.approx(tuple(np.percentile(v, [25, 75])))
    assert st["center"] == "median"  # IQR glows around the median
    _, man, _ = _save(fig, tmp_path)
    assert _series(man)["sd"]["glowbar"]["q3"] == pytest.approx(np.percentile(v, 75))

    for interval, lo, hi, centre in (("sem", v.mean() - st["sem"], v.mean() + st["sem"], "mean"),
                                     ("sd", v.mean() - st["sd"], v.mean() + st["sd"], "mean"),
                                     (lambda a: (a.min(), a.max()), v.min(), v.max(), "mean")):
        fig, ax = plt.subplots()
        s2 = fp.glowbar(_table(), x="type", y="APP", ax=ax, interval=interval).stats["SD"]
        assert (s2["low"], s2["high"]) == pytest.approx((lo, hi))
        assert s2["center"] == centre
        plt.close(fig)


def test_a_units_lane_and_colour_come_from_the_table_not_the_values():
    """Two measures of the same animals — one with a missing value — must agree unit by unit."""
    a = _table()
    other = list(np.array([v * 10 for v in a["APP"]])[::-1])  # a different measure, reordered values
    other[2] = None                                           # B6_3 not measured
    b = dict(a, APP=other)
    got = []
    for t in (a, b):
        fig, ax = plt.subplots()
        gb = fp.glowbar(t, x="type", y="APP", units="subject", ax=ax)
        coll = {name: art for name, art in zip(
            [u for u in gb.unit_series if not (t is b and u == "B6_3")], gb.artists["points"])}
        got.append((gb.point_colors, {u: c.get_offsets()[0][0] for u, c in coll.items()}))
        plt.close(fig)
    (cols_a, x_a), (cols_b, x_b) = got
    assert cols_a == cols_b                      # colour per unit: identical
    assert {u: x_a[u] for u in x_b} == x_b       # lane per unit: identical, B6_3's lane stays reserved


def test_shades_are_equal_perceptual_steps_and_neighbours_contrast():
    import matplotlib as mpl
    shades = fp.signature_fluxplots.even_shades(mpl.colormaps["YlGnBu"], 6, 88, 22)
    u = _perceptual(np.array([s[:3] for s in shades]))
    steps = np.linalg.norm(np.diff(u, axis=0), axis=1)
    assert steps.max() / steps.min() < 1.1          # equal visible differences
    assert u[0, 0] == pytest.approx(88, abs=1) and u[-1, 0] == pytest.approx(22, abs=1)
    order = fp.signature_fluxplots.interleaved_order(6)
    assert order == [0, 3, 1, 4, 2, 5]
    assert min(abs(a - b) for a, b in zip(order, order[1:])) >= 2
    for n in range(1, 10):
        assert sorted(fp.signature_fluxplots.interleaved_order(n)) == list(range(n))
    # a reversed source map is re-oriented light -> dark: same shades
    fig, ax = plt.subplots()
    fwd = fp.glowbar(_table(), x="type", y="APP", units="subject", ax=ax, palette="YlGnBu").point_colors
    rev = fp.glowbar(_table(), x="type", y="APP", units="subject", ax=ax, palette="YlGnBu_r").point_colors
    assert fwd.keys() == rev.keys()
    assert all(np.allclose(fwd[u][c], rev[u][c]) for u in fwd for c in fwd[u])
    plt.close(fig)


def test_connectors_join_a_unit_across_categories_and_break_at_gaps(tmp_path):
    t = {"mouse": ["m1", "m2", "m3"] * 3, "t": ["pre"] * 3 + ["mid"] * 3 + ["post"] * 3,
         "v": [1, 2, 3, 2, 3, np.nan, 3, 4, 5]}          # m3 has no "mid" value
    fig, ax = plt.subplots()
    gb = fp.glowbar(t, x="t", y="v", units="mouse", ax=ax, connect_identical_points_across_x_values=True)
    assert gb.categories == ["pre", "mid", "post"]
    assert len(gb.artists["lines"]) == 2                  # m3: pre and post are not adjacent → no line
    _, man, svg = _save(fig, tmp_path)
    assert 'id="m1.line"' in svg and 'id="m3.line"' not in svg
    assert sorted(_series(man)["m1"]["svg"]) == ["line", "points"]
    with pytest.raises(ValueError, match="needs units"):
        fp.glowbar(t, x="t", y="v", connect_identical_points_across_x_values=True)


def test_without_units_points_belong_to_their_category(tmp_path):
    fig, ax = plt.subplots()
    gb = fp.glowbar(_table(), x="type", y="APP", ax=ax, palette={"S": "#205EA6", "SD": "Reds"})
    assert all(len(set(map(tuple, c.get_facecolors()))) == 1 for c in gb.artists["points"])  # group colour
    assert gb.group_colors["S"] == pytest.approx(matplotlib.colors.to_rgba("#205EA6"))
    _, man, svg = _save(fig, tmp_path)
    assert 'id="s.points"' in svg and 'id="s.point.5"' in svg and 'id="s.glow"' in svg


def test_names_can_be_overridden_and_clashes_are_refused(tmp_path):
    fig, ax = plt.subplots()
    fp.glowbar(_table(), x="type", y="APP", units="subject", ax=ax,
               series={"S": "sleep", "SD": "sleep-dep"}, unit_series=lambda u: f"animal {u[3:]}")
    _, _, svg = _save(fig, tmp_path)
    assert 'id="sleep.glow"' in svg and 'id="sleep-dep.median"' in svg and 'id="animal-8.points"' in svg
    clash = {"subject": ["S", "b"], "type": ["S", "T"], "APP": [1.0, 2.0]}
    with pytest.raises(ValueError, match="share the series id"):
        fp.glowbar(clash, x="type", y="APP", units="subject")


def test_every_part_can_be_switched_off(tmp_path):
    fig, ax = plt.subplots()
    gb = fp.glowbar(_table(), x="type", y="APP", units="subject", ax=ax, show_mean=False,
                    show_median=False, show_caps=False, show_individual_points=False)
    assert gb.artists["points"] == [] and gb.artists["mean"] == [] and gb.artists["median"] == []
    _, man, svg = _save(fig, tmp_path)
    assert 'id="sd.glow"' in svg and 'id="sd.mean"' not in svg and "b6-8" not in svg
    assert _series(man)["sd"]["glowbar"]["median"] == pytest.approx(np.median(SD))  # stats still recorded


def test_a_mean_outside_the_interval_keeps_the_glow_inside_it():
    skew = {"g": ["a"] * 6, "v": [0, 0, 0, 0, 0.1, 20]}  # mean ≈ 3.35, far above Q3
    fig, ax = plt.subplots()
    gb = fp.glowbar(skew, x="g", y="v", ax=ax, center="mean")
    st, segs = gb.stats["a"], gb.artists["glow"][0].get_segments()
    ys = np.concatenate([s[:, 1] for s in segs])
    assert ys.min() >= st["low"] - 1e-12 and ys.max() <= st["high"] + 1e-12
    plt.close(fig)


def test_dataframes_and_arrays(tmp_path):
    t = _table()
    fig, ax = plt.subplots()
    ref = fp.glowbar(t, x="type", y="APP", units="subject", ax=ax).stats
    plt.close(fig)
    fig, ax = plt.subplots()
    arrays = fp.glowbar(x=np.array(t["type"]), y=t["APP"], units=t["subject"], ax=ax).stats
    plt.close(fig)
    assert arrays["SD"]["mean"] == pytest.approx(ref["SD"]["mean"])
    for mod in ("pandas", "polars"):
        lib = pytest.importorskip(mod)
        fig, ax = plt.subplots()
        got = fp.glowbar(lib.DataFrame(t), x="type", y="APP", units="subject", ax=ax)
        assert got.stats["S"]["median"] == pytest.approx(ref["S"]["median"])
        assert ax.get_ylabel() == "APP"
        plt.close(fig)
    with pytest.raises(KeyError, match="not a column"):
        fp.glowbar(t, x="type", y="nope")


def test_glowbar_is_deterministic(tmp_path):
    outs = []
    for k in range(2):
        fig, ax = plt.subplots(figsize=(2, 2))
        fp.glowbar(_table(), x="type", y="APP", units="subject", ax=ax)
        fp.save(fig, str(tmp_path / "d.svg"))  # same name → same hashsalt
        plt.close(fig)
        outs.append((tmp_path / "d.svg").read_bytes())
    assert outs[0] == outs[1]


@pytest.mark.parametrize("side, expected", [
    ("outer", {"S": -0.42, "SD": 1.42}),   # default: the bars frame the comparison from outside
    ("left", {"S": -0.42, "SD": 0.58}),
    ("right", {"S": 0.42, "SD": 1.42}),
])
def test_bar_side(side, expected):
    fig, ax = plt.subplots()
    kw = {} if side == "outer" else {"bar_side": side}
    gb = fp.glowbar(_table(), x="type", y="APP", units="subject", ax=ax, **kw)
    assert {c: s["x"] for c, s in gb.stats.items()} == pytest.approx(expected)
    plt.close(fig)
    with pytest.raises(ValueError, match="bar_side"):
        fp.glowbar(_table(), x="type", y="APP", bar_side="inside")


def test_point_fill_alpha_leaves_the_rim_opaque():
    fig, ax = plt.subplots()
    gb = fp.glowbar(_table(), x="type", y="APP", units="subject", ax=ax, point_fill_alpha=0.4)
    for coll in gb.artists["points"]:
        assert np.allclose(coll.get_facecolors()[:, 3], 0.4)
        assert np.allclose(coll.get_edgecolors()[:, 3], 1.0)
    plt.close(fig)
    with pytest.raises(ValueError, match="point_fill_alpha"):
        fp.glowbar(_table(), x="type", y="APP", point_fill_alpha=1.5)


@pytest.mark.parametrize("spec", ["cmasher.emerald", "emerald", "crameri.batlow", "tol.sunset",
                                  "brewer.Set2", "tol.bright", "flexoki.blue", "RdBu", "viridis",
                                  ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]])
def test_any_fluxplot_map_or_palette_gives_distinct_points_and_a_solid_group_colour(spec):
    fig, ax = plt.subplots()
    gb = fp.glowbar(_table(), x="type", y="APP", units="subject", ax=ax, palette={"S": spec, "SD": spec})
    cols = [gb.point_colors[u]["SD"] for u in gb.point_colors if "SD" in gb.point_colors[u]]
    u = _perceptual(np.array([c[:3] for c in cols]))
    assert len({tuple(np.round(c, 4)) for c in cols}) == 6            # six distinct colours
    assert u[:, 0].max() <= 88.5                                        # none too pale for white
    rep = _perceptual(np.array(gb.group_colors["SD"][:3]))
    assert 20 <= rep[0] <= 80 and gb.group_colors["SD"][3] == 1.0       # a solid, mid-tone group colour
    plt.close(fig)


def test_group_color_overrides_the_representative():
    fig, ax = plt.subplots()
    gb = fp.glowbar(_table(), x="type", y="APP", units="subject", ax=ax, palette="cmasher.emerald",
                    group_color={"SD": "#AF3029"})
    assert gb.group_colors["SD"] == pytest.approx(matplotlib.colors.to_rgba("#AF3029"))
    assert gb.group_colors["S"] != gb.group_colors["SD"]
    plt.close(fig)


def test_defaults_are_the_house_glowbar():
    fig, ax = plt.subplots()
    gb = fp.glowbar(_table(), x="type", y="APP", units="subject", ax=ax)
    st = gb.stats["SD"]
    assert st["interval"] == "sem" and st["center"] == "mean"
    assert gb.artists["mean"][0].get_markeredgewidth() == 1.2
    assert gb.artists["points"][0].get_sizes()[0] == 18
    import matplotlib as mpl
    assert gb.group_colors["SD"] == pytest.approx(mpl.colormaps["YlOrRd"](0.75))
    plt.close(fig)
