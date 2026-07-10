"""fp.box / fp.violin / fp.hist (plan §4): exact semantic composites over matplotlib returns.

Each wrapper registers only the pieces matplotlib's documented return structure exposes; every
declared member must resolve in the SVG and appear in build order exactly once.
"""
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import fluxplot as fp  # noqa: E402

VALUES_A = [1.0, 1.4, 1.9, 2.3, 2.8, 3.1, 3.6, 4.2, 4.9, 9.5]  # 9.5 = flier bait
VALUES_B = [2.0, 2.6, 3.3, 3.9, 4.4, 5.0, 5.8, 6.1, 6.9]


def _save(fig, tmp_path, name="w.svg"):
    res = fp.save(fig, str(tmp_path / name))
    plt.close(fig)
    return res, json.load(open(res.manifest)), open(res.svg).read()


def _series(man, sid):
    return next(s for s in man["series"] if s["id"] == sid)


def _svg_ids(svg):
    return set(re.findall(r'id="([^"]+)"', svg))


def _find_group(node, gid):
    if node.get("id") == gid and node.get("role") == "group":
        return node
    for c in node.get("children", []):
        if isinstance(c, dict):
            hit = _find_group(c, gid)
            if hit:
                return hit
    return None


def _assert_members_resolve_once(man, svg, sid):
    """Every declared composite member exists in the SVG and appears in build order once."""
    ids = _svg_ids(svg)
    order = man["build"]["order"]
    s = _series(man, sid)
    for key, val in s["svg"].items():
        for ref in val if isinstance(val, list) else [val]:
            assert ref in ids, f"{sid}: svg.{key} member {ref} missing from SVG"
    for key in ("whiskers", "caps", "medians", "fliers", "means", "segments", "errorbars", "bars"):
        for ref in s["svg"].get(key, []):
            assert order.count(ref) == 1, f"{ref} must appear in build order exactly once"


def test_box_components_and_groups(tmp_path):
    fig, ax = plt.subplots()
    bp = fp.box(ax, VALUES_A, series="control", label="Control")
    assert set(bp) >= {"boxes", "whiskers", "caps", "medians", "fliers"}  # mpl contract
    res, man, svg = _save(fig, tmp_path)

    s = _series(man, "control")
    assert s["kind"] == "box" and s["label"] == "Control"
    assert s["svg"]["box"] == "control.box"
    assert len(s["svg"]["whiskers"]) == 2 and len(s["svg"]["caps"]) == 2
    assert len(s["svg"]["medians"]) == 1 and len(s["svg"]["fliers"]) == 1
    assert "distribution" not in s  # raw samples are opt-in
    _assert_members_resolve_once(man, svg, "control")

    # X-ray grouping: one group node per statistic family
    for gid, n in (("control.whiskers", 2), ("control.caps", 2), ("control.medians", 1)):
        grp = _find_group(man["parts"], gid)
        assert grp is not None and len(grp["members"]) == n
    # presets cover the composite roles present
    assert "whisker" in man["build"]["presets"] and "median" in man["build"]["presets"]


def test_box_optional_pieces_create_no_dead_parts(tmp_path):
    fig, ax = plt.subplots()
    fp.box(ax, VALUES_A, series="bare", showfliers=False, showcaps=False, showmeans=True)
    _, man, svg = _save(fig, tmp_path)
    s = _series(man, "bare")
    assert "fliers" not in s["svg"] and "caps" not in s["svg"]
    assert len(s["svg"]["means"]) == 1  # showmeans=True registered
    _assert_members_resolve_once(man, svg, "bare")


def test_box_rejects_multiple_groups(tmp_path):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="one box per call"):
        fp.box(ax, [VALUES_A, VALUES_B], series="both")
    plt.close(fig)


def test_two_boxes_two_series_with_legend(tmp_path):
    fig, ax = plt.subplots()
    fp.box(ax, VALUES_A, series="control", label="Control", positions=[1])
    fp.box(ax, VALUES_B, series="treatment", label="Treatment", positions=[2])
    ax.legend()
    _, man, svg = _save(fig, tmp_path)
    assert {s["id"] for s in man["series"]} == {"control", "treatment"}
    _assert_members_resolve_once(man, svg, "control")
    _assert_members_resolve_once(man, svg, "treatment")
    ids = _svg_ids(svg)
    assert len([i for i in ids if i.startswith("control.")] ) != 0
    assert not any(i.endswith("-2") for i in ids if i.startswith(("control.", "treatment."))), \
        "no collision-suffixed ids across repeated wrapper calls"


def test_violin_components(tmp_path):
    fig, ax = plt.subplots()
    vp = fp.violin(ax, VALUES_A, series="dist", label="Distribution", showmedians=True)
    assert set(vp) >= {"bodies", "cbars", "cmins", "cmaxes", "cmedians"}  # mpl contract
    _, man, svg = _save(fig, tmp_path)
    s = _series(man, "dist")
    assert s["kind"] == "violin" and s["svg"]["violin"] == "dist.violin"
    assert len(s["svg"]["caps"]) == 2  # cmins + cmaxes
    assert len(s["svg"]["whiskers"]) == 1 and len(s["svg"]["medians"]) == 1
    _assert_members_resolve_once(man, svg, "dist")


def test_violin_without_extrema_has_no_dead_parts(tmp_path):
    fig, ax = plt.subplots()
    fp.violin(ax, VALUES_A, series="plain", showextrema=False)
    _, man, svg = _save(fig, tmp_path)
    s = _series(man, "plain")
    assert "caps" not in s["svg"] and "whiskers" not in s["svg"]
    assert s["svg"]["violin"] == "plain.violin"
    _assert_members_resolve_once(man, svg, "plain")


def test_hist_distribution_matches_numpy(tmp_path):
    rng = np.random.default_rng(7)
    values = rng.normal(5.0, 1.5, 200)
    fig, ax = plt.subplots()
    counts, edges, _ = fp.hist(ax, values, series="observations", bins=20, label="Observations")
    _, man, svg = _save(fig, tmp_path)

    np_counts, np_edges = np.histogram(values, bins=20)
    assert [float(c) for c in np_counts] == list(counts)
    s = _series(man, "observations")
    dist = s["distribution"]
    assert len(dist["binEdges"]) == 21 and len(dist["counts"]) == 20
    assert [round(float(e), 4) for e in np_edges] == dist["binEdges"]  # canonical 4-decimal JSON
    assert "values" not in dist  # raw observations absent by default
    assert len(s["svg"]["bars"]) == 20
    _assert_members_resolve_once(man, svg, "observations")
    assert 'id="observations.bar.19"' in svg


def test_hist_include_values_opt_in(tmp_path):
    fig, ax = plt.subplots()
    fp.hist(ax, VALUES_A, series="obs", bins=4, include_values=True)
    _, man, _ = _save(fig, tmp_path)
    assert _series(man, "obs")["distribution"]["values"] == VALUES_A


def test_hist_rejects_unsupported_structures(tmp_path):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="per-bar contract"):
        fp.hist(ax, [VALUES_A, VALUES_B], series="multi")
    with pytest.raises(ValueError, match="per-bar contract"):
        fp.hist(ax, VALUES_A, series="step", histtype="step")
    plt.close(fig)


def test_composites_are_deterministic(tmp_path):
    def build():
        fig, ax = plt.subplots(figsize=(6, 4))
        fp.box(ax, VALUES_A, series="control", label="Control", positions=[1])
        fp.violin(ax, VALUES_B, series="treatment", positions=[2], showmedians=True)
        fp.hist(ax, VALUES_A, series="obs", bins=4)
        res = fp.save(fig, str(tmp_path / "d.svg"))
        plt.close(fig)
        return open(res.svg, "rb").read(), open(res.manifest, "rb").read()

    s1, m1 = build()
    s2, m2 = build()
    assert s1 == s2 and m1 == m2
