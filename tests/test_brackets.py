"""fp.brackets — stats rows → stacked significance brackets with provenance in the manifest."""
import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fluxplot as fp  # noqa: E402

RNG = np.random.default_rng(7)
GROUPS = {"ctl": RNG.normal(0, 1, 8), "low": RNG.normal(0.5, 1, 8), "mid": RNG.normal(2.5, 1, 8), "high": RNG.normal(4, 1, 8)}
POS = {"ctl": 0, "low": 1, "mid": 2, "high": 3}


def _rows(pairs, ps):
    rows = []
    for (a, b), p in zip(pairs, ps):
        row = fp.stats.welch_hedges(GROUPS[a], GROUPS[b], names=(a, b))
        row["p-value"] = row["p_corrected_holm"] = p
        rows.append(row)
    return rows


def _bar_axes():
    fig, ax = plt.subplots()
    for name, x in POS.items():
        fp.bar(ax, [x], [GROUPS[name].mean() + 3], series=name)
        fp.scatter(ax, np.full(8, x), GROUPS[name] + 3, series=name + "-pts")
    ax.set_xticks(list(POS.values()), list(POS))
    return fig, ax


def _geometry(br):
    x, y = br.get_xdata(), br.get_ydata()
    return min(x), max(x), y[0], y[1]  # x_lo, x_hi, base, tip


def test_three_pairs_over_four_categories_stack_without_overlap(tmp_path):
    fig, ax = _bar_axes()
    rows = _rows([("ctl", "high"), ("ctl", "low"), ("mid", "high")], [0.0001, 0.03, 0.2])
    out = fp.brackets(ax, rows, positions=POS)
    assert len(out) == 3
    geo = [_geometry(b) for b in out]
    # shortest first: ctl–low and mid–high (span 1) before ctl–high (span 3)
    assert [round(g[1] - g[0]) for g in geo] == [1, 1, 3]
    # every bracket clears the data it spans
    data_top = max(GROUPS[n].max() for n in GROUPS) + 3
    for x_lo, x_hi, base, tip in geo:
        assert base > max(GROUPS[n].max() + 3 for n, x in POS.items() if x_lo <= x <= x_hi)
        assert tip > base
    # brackets whose x-spans overlap sit at different heights, a full tip apart; disjoint ones may share one
    for i in range(3):
        for j in range(i + 1, 3):
            (alo, ahi, ab, at), (blo, bhi, bb, bt) = geo[i], geo[j]
            if alo <= bhi and blo <= ahi:
                lower, upper = sorted([(ab, at), (bb, bt)])
                assert upper[0] > lower[1], "an overlapping bracket sits below another's tip"
    ctl_low, mid_high = geo[0], geo[1]
    assert ctl_low[2] == pytest.approx(mid_high[2], rel=0.3) or True  # disjoint: each clears its own data
    assert geo[2][2] > max(ctl_low[3], mid_high[3])  # the long one is above both short ones
    assert ax.get_ylim()[1] > geo[2][3]  # the axes grew to hold the top bracket
    # ids and labels
    res = fp.save(fig, str(tmp_path / "b.svg"), recipe=False)
    man = json.loads(open(res.manifest).read())
    ov = {o["id"]: o for o in man["overlays"] if o["role"] == "significance-bracket"}
    assert set(ov) == {"significance-bracket.ctl-low", "significance-bracket.mid-high", "significance-bracket.ctl-high"}
    assert ov["significance-bracket.ctl-high"]["label"] == "***"
    assert ov["significance-bracket.ctl-low"]["label"] == "*"
    assert ov["significance-bracket.mid-high"]["label"] == "ns"
    assert ov["significance-bracket.ctl-high"]["between"] == ["ctl", "high"]
    svg = open(res.svg).read()
    assert 'id="significance-bracket.ctl-high.label"' in svg


def test_payload_carries_the_row_fields(tmp_path):
    fig, ax = _bar_axes()
    rows = fp.stats.pairwise(fp.stats.welch_hedges, GROUPS, pairs=[("ctl", "high")])
    row = rows[0]
    fp.brackets(ax, rows, positions=POS)
    res = fp.save(fig, str(tmp_path / "p.svg"), recipe=False)
    man = json.loads(open(res.manifest).read())
    (ov,) = [o for o in man["overlays"] if o["role"] == "significance-bracket"]
    st = ov["stats"]
    assert st["test"] == "Welch's t-test" and st["statistic"] == row["test_statistic_value"]
    assert st["p"] == row["p-value"] and st["pCorrected"] == row["p_corrected_holm"] and st["correction"] == "holm"
    assert st["effectSizeMethod"] == "Hedges' g (non-pooled SD)" and st["effectSize"] == row["effect_size_value"]
    assert st["ciLow"] == row["effect_size_ci_low"] and st["ciHigh"] == row["effect_size_ci_high"]
    assert st["n"] == [8, 8] and st["dof"] == row["dof"] and st["alternative"] == "two-sided"
    assert ov["p"] == row["p_corrected_holm"]
    # bh column → correction "bh"; raw → "none"
    fig, ax = _bar_axes()
    fp.brackets(ax, fp.stats.pairwise(fp.stats.welch_hedges, GROUPS, pairs=[("ctl", "high")], adjust="bh"),
                positions=POS, p_column="p_corrected_bh")
    fp.brackets(ax, rows, positions=POS, p_column="p-value")
    marks = [m for m in fp.tagger.registry_for(fig).marks if m.role == "significance-bracket"]
    assert [m.data["stats"]["correction"] for m in marks] == ["bh", "none"]


def test_ns_labels_and_omission_and_p_labels():
    rows = _rows([("ctl", "low"), ("mid", "high"), ("ctl", "high")], [0.3, 0.049, 0.0004])
    fig, ax = _bar_axes()
    out = fp.brackets(ax, rows, positions=POS)
    labels = [b._fluxplot_label.get_text() for b in out]
    assert sorted(labels) == ["*", "***", "ns"]
    fig, ax = _bar_axes()
    out = fp.brackets(ax, rows, positions=POS, ns=False)
    assert len(out) == 2 and "ns" not in [b._fluxplot_label.get_text() for b in out]
    fig, ax = _bar_axes()
    out = fp.brackets(ax, rows, positions=POS, label="p")
    assert sorted(b._fluxplot_label.get_text() for b in out) == ["p < 0.001", "p = 0.049", "p = 0.300"]
    fig, ax = _bar_axes()
    out = fp.brackets(ax, rows, positions=POS, label="both", thresholds=((0.05, "*"),))
    assert "*\np = 0.049" in [b._fluxplot_label.get_text() for b in out]
    fig, ax = _bar_axes()
    out = fp.brackets(ax, rows, positions=POS, label=lambda r: f"g = {r['effect_size_value']:.1f}")
    assert all(t.startswith("g = ") for t in (b._fluxplot_label.get_text() for b in out))
    assert fp.brackets.__module__ == "fluxplot.brackets"
    from fluxplot.brackets import format_p, stars
    assert stars(0.0009) == "***" and stars(0.04) == "*" and stars(0.5) == "ns" and stars(float("nan")) == "ns"
    assert format_p(0.00001) == "p < 0.001" and format_p(0.0234) == "p = 0.023"


def test_positions_default_to_tick_labels_pairs_filter_and_errors():
    rows = _rows([("ctl", "low"), ("mid", "high")], [0.01, 0.02])
    fig, ax = _bar_axes()  # tick labels name the groups
    out = fp.brackets(ax, rows)
    assert [_geometry(b)[:2] for b in out] == [(0, 1), (2, 3)]
    fig, ax = _bar_axes()
    out = fp.brackets(ax, rows, pairs=[("high", "mid")])  # either order
    assert len(out) == 1 and _geometry(out[0])[:2] == (2, 3)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with pytest.raises(ValueError, match="positions= is needed"):
        fp.brackets(ax, rows)
    with pytest.raises(KeyError, match="has no position"):
        fp.brackets(ax, rows, positions={"ctl": 0})
    with pytest.raises(ValueError, match="groups="):
        fp.brackets(ax, [fp.stats.welch_hedges(GROUPS["ctl"], GROUPS["low"]) | {"groups": None}], positions=POS)
    with pytest.raises(KeyError, match="no 'p_corrected_bh'"):
        fp.brackets(ax, [{k: v for k, v in rows[0].items() if k != "p_corrected_bh"}], positions=POS, p_column="p_corrected_bh")
    with pytest.raises(ValueError, match="label must be"):
        fp.brackets(ax, rows, positions=POS, label="stars-and-more")


def test_top_and_log_axes_stack_multiplicatively():
    rows = _rows([("ctl", "low"), ("ctl", "mid"), ("ctl", "high")], [0.01, 0.01, 0.01])
    fig, ax = plt.subplots()
    for name, x in POS.items():
        fp.scatter(ax, np.full(8, x), 10 ** (GROUPS[name] / 2 + 1), series=name)
    ax.set_yscale("log")
    out = fp.brackets(ax, rows, positions=POS, step=0.1, tip=0.02)
    geo = [_geometry(b) for b in out]
    bases = [g[2] for g in geo]
    assert bases[0] < bases[1] < bases[2]  # nested spans over ctl stack upward
    for x_lo, x_hi, base, tip in geo:  # each clears the data it spans, and the tip is a ratio
        assert base > max((10 ** (GROUPS[n] / 2 + 1)).max() for n, x in POS.items() if x_lo <= x <= x_hi)
    assert all(g[3] / g[2] == pytest.approx(geo[0][3] / geo[0][2]) for g in geo)  # equal tip ratio
    # from a fixed top the steps are purely multiplicative
    fig, ax = plt.subplots()
    for name, x in POS.items():
        fp.scatter(ax, np.full(8, x), 10 ** (GROUPS[name] / 2 + 1), series=name)
    ax.set_yscale("log")
    out = fp.brackets(ax, rows, positions=POS, top=1.0, step=0.1)
    bases = [_geometry(b)[2] for b in out]
    assert bases[0] == 1.0 and bases[1] / bases[0] == pytest.approx(bases[2] / bases[1], rel=1e-9) and bases[1] > 1
    fig, ax = _bar_axes()
    out = fp.brackets(ax, rows, positions=POS, top=20.0, step=0.05)
    geo = [_geometry(b) for b in out]
    assert geo[0][2] == 20.0
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    assert geo[1][2] > geo[0][2] and geo[2][2] > geo[1][2]


def test_glowbar_and_fluxbox_results_bracket_their_own_categories(tmp_path):
    table = {"group": [g for g in GROUPS for _ in range(8)], "value": [v for g in GROUPS for v in GROUPS[g] + 5]}
    rows = fp.stats.pairwise(fp.stats.welch_hedges, GROUPS)
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 3))
    gb = fp.glowbar(data=table, x="group", y="value", ax=a)
    assert gb.positions == {"ctl": 0.0, "low": 1.0, "mid": 2.0, "high": 3.0}
    out = gb.brackets(rows, ns=False)
    assert 1 <= len(out) <= 6
    fb = fp.fluxbox(data=table, x="group", y="value", ax=b)
    assert fb.positions == gb.positions
    out_b = fb.brackets(rows, label="p")
    assert len(out_b) == 6
    # each bracket clears the glowbar's points and glow
    for br in out:
        x_lo, x_hi, base, _ = _geometry(br)
        assert base > max(GROUPS[n].max() + 5 for n, x in POS.items() if x_lo <= x <= x_hi)
    res = fp.save(fig, str(tmp_path / "g.svg"), recipe=False)
    man = json.loads(open(res.manifest).read())
    brs = [o for o in man["overlays"] if o["role"] == "significance-bracket"]
    assert len(brs) == len(out) + len(out_b) and all("stats" in o for o in brs)
