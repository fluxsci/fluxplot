"""Save-time promotion of labeled raw artists (plan §3).

Identity only from public artist labels; data only from exact artist state; ambiguity declines
to extra.* with one actionable warning; explicit tags always win.
"""
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import fluxplot as fp  # noqa: E402


def _save(fig, tmp_path, name="p.svg"):
    res = fp.save(fig, str(tmp_path / name))
    plt.close(fig)
    return res, json.load(open(res.manifest)), open(res.svg).read()


def _series(man, sid):
    return next((s for s in man["series"] if s["id"] == sid), None)


def test_raw_labeled_plot_scatter_bar_promote(tmp_path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    t = [0, 1, 2, 3]
    ax.plot(t, [1, 2, 1.5, 3], label="Control", marker="o")
    ax.scatter(t, [2, 3, 2.5, 4], label="Treatment")
    ax.bar([5, 6], [1.0, 2.0], label="Counts")
    ax.legend()
    res, man, svg = _save(fig, tmp_path)

    ctrl = _series(man, "control")
    assert ctrl is not None and ctrl["kind"] == "line"
    assert ctrl["data"] == {"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 1.5, 3.0]}
    assert ctrl["label"] == "Control"
    assert ctrl["capture"] == {"identity": "artist-label", "data": "artist"}
    assert ctrl["svg"]["line"] == "control.line"
    assert 'id="control.line"' in svg

    treat = _series(man, "treatment")
    assert treat is not None and treat["kind"] == "scatter"
    assert [p["index"] for p in treat["points"]] == [0, 1, 2, 3]  # one per exact offset
    assert 'id="treatment.point.2"' in svg

    counts = _series(man, "counts")
    assert counts is not None and counts["kind"] == "bar"
    assert counts["data"]["x"] == [5.0, 6.0] and counts["data"]["y"] == [1.0, 2.0]
    assert counts["svg"]["bars"] == ["counts.bar.0", "counts.bar.1"]

    # all promoted parts participate in the build order
    order = man["build"]["order"]
    assert "control.line" in order and "treatment.points" in order and "counts.bar.0" in order
    # nothing left over as extras
    assert not [o for o in man["overlays"] if o["role"] == "extra"]
    assert res.warnings == []


def test_labeled_fill_between_promotes_as_area_without_invented_data(tmp_path):
    fig, ax = plt.subplots()
    x = [0, 1, 2]
    ax.fill_between(x, [1, 2, 1], [2, 3, 2], label="Band", alpha=0.4)
    _, man, svg = _save(fig, tmp_path)
    band = _series(man, "band")
    assert band is not None and band["kind"] == "area"
    assert band["data"] == {}  # original y1/y2 vectors are not recoverable — never invented
    assert 'id="band.area"' in svg


def test_private_and_default_labels_stay_extras(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])  # default _childN label
    ln, = ax.plot([0, 1], [1, 0])
    ln.set_label("_nolegend_")
    _, man, _ = _save(fig, tmp_path)
    assert man["series"] == []
    extras = [o for o in man["overlays"] if o["role"] == "extra"]
    assert len(extras) == 2  # still visible/addressable, just not named


def test_duplicate_public_labels_decline_with_one_warning(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="Same")
    ax.plot([0, 1], [1, 0], label="Same")
    with pytest.warns(UserWarning, match="ambiguous"):
        res = fp.save(fig, str(tmp_path / "d.svg"))
    plt.close(fig)
    man = json.load(open(res.manifest))
    assert man["series"] == []
    assert len([o for o in man["overlays"] if o["role"] == "extra"]) == 2
    assert len(res.warnings) == 1


def test_label_colliding_with_explicit_series_defers_to_helper(tmp_path):
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [0, 1], series="control", label="Control")
    ax.plot([0, 1], [1, 0], label="Control")  # raw artist reusing the explicit identity
    with pytest.warns(UserWarning, match="ambiguous"):
        res = fp.save(fig, str(tmp_path / "c.svg"))
    plt.close(fig)
    man = json.load(open(res.manifest))
    ctrl = _series(man, "control")
    assert ctrl["data"]["y"] == [0.0, 1.0]  # the explicit helper's mark, untouched
    assert len([o for o in man["overlays"] if o["role"] == "extra"]) == 1


def test_nonfinite_scatter_offsets_decline(tmp_path):
    fig, ax = plt.subplots()
    ax.scatter([0, 1, 2], [1, np.nan, 2], label="Holes")
    with pytest.warns(UserWarning, match="non-finite"):
        res = fp.save(fig, str(tmp_path / "n.svg"))
    plt.close(fig)
    man = json.load(open(res.manifest))
    assert _series(man, "holes") is None
    assert [o for o in man["overlays"] if o["role"] == "extra"]


def test_labeled_threshold_is_generic_line_not_reference_line(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 24], [1.0, 1.0], label="Threshold")  # horizontal rule, but labeled
    _, man, svg = _save(fig, tmp_path)
    thr = _series(man, "threshold")
    assert thr is not None and thr["roles"] == ["line"]  # never guessed as reference-line
    assert 'data-role="reference-line"' not in svg


def test_unlabeled_raw_artists_still_swept_to_extras(tmp_path):
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [0, 1], series="alpha", label="Alpha")
    ax.plot([0.5, 0.5], [0, 1], color="red")  # unlabeled rule → stays extra
    _, man, _ = _save(fig, tmp_path)
    assert [o["id"] for o in man["overlays"] if o["role"] == "extra"] == ["extra.line.0"]


def test_promotion_is_idempotent_across_resaves(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [1, 2, 3], label="Control")
    r1 = fp.save(fig, str(tmp_path / "r.svg"))
    m1 = open(r1.manifest, "rb").read()
    r2 = fp.save(fig, str(tmp_path / "r.svg"))
    m2 = open(r2.manifest, "rb").read()
    plt.close(fig)
    assert m1 == m2  # no duplicated marks or shifted ids on re-save


def test_tag_extracts_exact_xy_when_omitted(tmp_path):
    fig, ax = plt.subplots()
    (ln,) = ax.plot([0, 1, 2], [3, 4, 5])
    fp.tag(ln, role="line", series="fit")
    coll = ax.scatter([0, 1], [1, 2])
    fp.tag_points(coll, series="obs")
    _, man, svg = _save(fig, tmp_path)
    fit = _series(man, "fit")
    assert fit["data"] == {"x": [0.0, 1.0, 2.0], "y": [3.0, 4.0, 5.0]}
    obs = _series(man, "obs")
    assert len(obs["points"]) == 2 and 'id="obs.point.1"' in svg


def test_tag_without_supported_artist_keeps_xy_absent(tmp_path):
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots()
    patch = ax.add_patch(mpatches.Rectangle((0, 0), 1, 1))
    fp.tag(patch, role="highlight-region", name="roi")
    _, man, _ = _save(fig, tmp_path)
    roi = [o for o in man["overlays"] if o.get("name") == "roi"]
    assert roi  # addressable, with no invented spatial claim
