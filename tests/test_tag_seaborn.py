"""tag_seaborn: one-call semantic tagging of seaborn axes-level plots.

Seaborn's artist layout is an implicit contract this helper rides on (hue-ordered lines/
bands/containers, empty legend-proxy lines, loose 2-point error segments) — so like
test_marker_dom.py, these tests double as a tripwire: a seaborn upgrade that changes the
layout should fail here loudly instead of silently mis-tagging.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import seaborn as sns  # noqa: E402

import fluxplot as fp  # noqa: E402


@pytest.fixture
def ax():
    fig, ax = plt.subplots(figsize=(5, 3))
    yield ax
    plt.close(fig)


def _fmri_stim():
    rng = np.random.default_rng(7)
    rows = []
    for region in ("parietal", "frontal"):
        for subject in range(6):
            for t in range(10):
                rows.append({"region": region, "timepoint": t,
                             "signal": np.sin(t / 3) + rng.normal(0, 0.1)})
    return pd.DataFrame(rows)


def _tips():
    rng = np.random.default_rng(11)
    n = 80
    return pd.DataFrame({
        "day": rng.choice(["Thur", "Fri", "Sat", "Sun"], n),
        "sex": rng.choice(["Male", "Female"], n),
        "total_bill": rng.normal(20, 6, n).clip(3),
        "tip": rng.normal(3, 1, n).clip(0.5),
    })


def _save(ax, tmp_path, name):
    res = fp.save(ax.figure, str(tmp_path / f"{name}.svg"))
    assert res.warnings == []
    return json.loads((tmp_path / f"{name}.fluxplot.json").read_text())


def test_lineplot_hue(ax, tmp_path):
    sns.lineplot(data=_fmri_stim(), x="timepoint", y="signal", hue="region", ax=ax)
    tagged = fp.tag_seaborn(ax, plot="lineplot")
    assert tagged == {"parietal": ["line", "area"], "frontal": ["line", "area"]}
    # the empty legend-proxy lines are gone
    assert all(len(ln.get_xdata()) for ln in ax.lines)
    man = _save(ax, tmp_path, "lineplot")
    assert [(s["id"], s["roles"]) for s in man["series"]] == [
        ("parietal", ["area", "line"]),
        ("frontal", ["area", "line"]),
    ]
    # the mean line's data was captured from the artist
    assert len(man["series"][0]["data"]["x"]) == 10


def test_lineplot_no_hue_falls_back_to_ylabel(ax, tmp_path):
    sns.lineplot(data=_fmri_stim(), x="timepoint", y="signal", ax=ax)
    tagged = fp.tag_seaborn(ax)
    assert tagged == {"signal": ["line", "area"]}


def test_scatterplot_per_point(ax, tmp_path):
    sns.scatterplot(data=_tips(), x="total_bill", y="tip", ax=ax)
    tagged = fp.tag_seaborn(ax)
    assert tagged == {"tip": ["point"]}
    man = _save(ax, tmp_path, "scatter")
    (s,) = man["series"]
    assert len(s["points"]) == 80  # every point individually addressable
    assert s["points"][0]["svgId"] == "tip.point.0"


def test_scatterplot_hue_fuses_to_one_group(ax, tmp_path):
    # seaborn draws ALL hue groups as one collection; identity below hue is unrecoverable
    sns.scatterplot(data=_tips(), x="total_bill", y="tip", hue="sex", ax=ax)
    tagged = fp.tag_seaborn(ax, plot="scatterplot")
    assert tagged == {"tip": ["point"]}


def test_barplot_hue_bars_and_errorbars(ax, tmp_path):
    tips = _tips()
    sns.barplot(data=tips, x="day", y="total_bill", hue="sex", errorbar="sd", ax=ax)
    tagged = fp.tag_seaborn(ax, plot="barplot")
    assert set(tagged) == {"Male", "Female"}
    assert tagged["Male"][0] == "bar" and tagged["Male"].count("errorbar") == 4
    man = _save(ax, tmp_path, "bars")
    male = next(s for s in man["series"] if s["id"] == "male")
    # container ↔ legend pairing is by draw order — verify against the actual data
    expected = (
        tips[tips.sex == "Male"].groupby("day", observed=True)["total_bill"].mean()
    )
    order = [t.get_text() for t in ax.get_xticklabels()]
    # Scientific values round-trip without destructive decimal quantization.
    assert male["data"]["y"] == pytest.approx([float(expected[d]) for d in order], rel=1e-14)


def test_series_override_and_compose_with_helpers(ax, tmp_path):
    sns.lineplot(data=_fmri_stim(), x="timepoint", y="signal", hue="region", ax=ax)
    # a raw line the user tagged themselves must be skipped by the helper
    (extra,) = ax.plot([0, 9], [0, 0], ":")
    fp.tag(extra, role="reference-line", name="zero")
    tagged = fp.tag_seaborn(ax, series=["par", "fro"])
    assert tagged == {"par": ["line", "area"], "fro": ["line", "area"]}
    man = _save(ax, tmp_path, "compose")
    assert {s["id"] for s in man["series"]} == {"par", "fro"}
    assert any(o["id"] == "reference-line.zero" for o in man["overlays"])


def test_regplot_one_series(ax, tmp_path):
    sns.regplot(data=_tips(), x="total_bill", y="tip", ax=ax)
    tagged = fp.tag_seaborn(ax)
    assert tagged == {"tip": ["line", "area", "point"]}


def test_untaggable_left_for_orphan_sweep(ax, tmp_path):
    # boxplot: per-category composite the helper deliberately does NOT guess at
    sns.boxplot(data=_tips(), x="day", y="total_bill", ax=ax)
    tagged = fp.tag_seaborn(ax)
    assert tagged == {}
    man = _save(ax, tmp_path, "box")
    assert man["series"] == []
    # ...but everything is still addressable via the extra.* sweep
    assert any(o["role"] == "extra" for o in man["overlays"])
