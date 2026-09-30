"""C9 — the errorbar composite, part by part; C10 — legend entries joined to series by artist."""
import json
import re

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fluxplot as fp  # noqa: E402


def _save(fig, tmp_path, name="e.svg"):
    res = fp.save(fig, str(tmp_path / name), recipe=False)
    return res, json.loads(open(res.manifest).read()), open(res.svg).read()


def test_errorbar_parts_are_addressable_and_errors_broadcast(tmp_path):
    x, y = [0, 1, 2, 3], [1.0, 2.0, 1.5, 3.0]
    fig, ax = plt.subplots()
    fp.errorbar(ax, x, y, yerr=0.3, series="s", fmt="o-", capsize=3)
    res, man, svg = _save(fig, tmp_path)
    ids = set(re.findall(r'id="([^"]+)"', svg))
    assert {"s.line", "s.points", "s.point.0", "s.point.3", "s.cap", "s.cap.1", "s.errorbar"} <= ids
    (s,) = man["series"]
    assert s["kind"] == "errorbar" and sorted(s["roles"]) == ["cap", "errorbar", "line", "point"]
    assert s["svg"]["line"] == "s.line" and s["svg"]["points"] == "s.points" and s["svg"]["errorbar"] == "s.errorbar"
    assert s["svg"]["caps"] == ["s.cap", "s.cap.1"]
    assert s["uncertainty"] == {"xerr": None, "yerr": [0.3, 0.3, 0.3, 0.3], "errShape": "scalar"}
    assert s["data"] == {"x": [0.0, 1.0, 2.0, 3.0], "y": y}
    assert [p["index"] for p in s["points"]] == [0, 1, 2, 3]
    # symmetric and asymmetric arrays; xerr
    fig, ax = plt.subplots()
    fp.errorbar(ax, x, y, yerr=[[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5]], xerr=[0.1, 0.2, 0.3, 0.4], series="a", fmt="s")
    res, man, svg = _save(fig, tmp_path, "a.svg")
    (a,) = man["series"]
    assert a["uncertainty"]["errShape"] == "asymmetric" and a["uncertainty"]["yerr"] == [[0.1] * 4, [0.5] * 4]
    assert a["uncertainty"]["xerr"] == [0.1, 0.2, 0.3, 0.4]
    ids = set(re.findall(r'id="([^"]+)"', svg))
    assert "a.points" in ids and "a.line" not in ids  # fmt="s": markers only, no data line
    assert "a.point.2" in ids
    with pytest.raises(ValueError, match="scalar or a 1D or"):  # matplotlib rejects it first
        fp.errorbar(ax, x, y, yerr=np.zeros((3, 4)), series="bad")


def test_errorbar_line_only_and_legend_still_joins(tmp_path):
    fig, ax = plt.subplots()
    fp.errorbar(ax, [0, 1, 2], [1, 2, 3], yerr=0.2, series="ctl", fmt="-", label="Control")
    ax.legend()
    res, man, svg = _save(fig, tmp_path)
    (s,) = man["series"]
    assert "point" not in s["roles"] and s["svg"]["line"] == "ctl.line"
    (legend,) = [g for g in man["guides"] if g["role"] == "legend"]
    assert legend["entries"][0]["series"] == "ctl" and legend["entries"][0]["text"] == "Control"


def test_legend_entries_join_series_by_artist_not_text(tmp_path):
    fig, ax = plt.subplots()
    ln = fp.line(ax, [0, 1], [0, 1], series="ctl")
    fp.line(ax, [0, 1], [1, 0], series="drug", label="Drug")
    fp.legend(ax, [ln], ["Control"])  # explicit handle; the series has no label of its own
    res, man, svg = _save(fig, tmp_path)
    (legend,) = [g for g in man["guides"] if g["role"] == "legend"]
    assert legend["entries"] == [{"series": "ctl", "text": "Control", "swatch": "legend.entry.0.swatch", "label": "legend.entry.0.label"}]
    # duplicated label texts: the text join is ambiguous, the artist join is not
    fig, ax = plt.subplots()
    a = fp.line(ax, [0, 1], [0, 1], series="a", label="Same")
    b = fp.line(ax, [0, 1], [1, 0], series="b", label="Same")
    ax.legend()
    res, man, svg = _save(fig, tmp_path, "d.svg")
    (legend,) = [g for g in man["guides"] if g["role"] == "legend"]
    assert [e["series"] for e in legend["entries"]] == ["a", "b"]
    # bar containers and scatter collections join too
    fig, ax = plt.subplots()
    fp.bar(ax, [0, 1], [1, 2], series="bars", label="Bars")
    fp.scatter(ax, [0, 1], [2, 3], series="dots", label="Dots")
    ax.legend()
    res, man, svg = _save(fig, tmp_path, "b.svg")
    (legend,) = [g for g in man["guides"] if g["role"] == "legend"]
    assert {e["text"]: e["series"] for e in legend["entries"]} == {"Bars": "bars", "Dots": "dots"}
    # a hand-made legend through raw ax.legend with reordered texts yields no false claim
    fig, ax = plt.subplots()
    a = fp.line(ax, [0, 1], [0, 1], series="a", label="A")
    b = fp.line(ax, [0, 1], [1, 0], series="b", label="B")
    ax.legend([b, a], ["B first", "A second"])
    res, man, svg = _save(fig, tmp_path, "r.svg")
    (legend,) = [g for g in man["guides"] if g["role"] == "legend"]
    assert all("series" not in e for e in legend["entries"])
    fig, ax = plt.subplots()
    a = fp.line(ax, [0, 1], [0, 1], series="a", label="A")
    b = fp.line(ax, [0, 1], [1, 0], series="b", label="B")
    fp.legend(ax, [b, a], ["B first", "A second"])
    res, man, svg = _save(fig, tmp_path, "r2.svg")
    (legend,) = [g for g in man["guides"] if g["role"] == "legend"]
    assert [e["series"] for e in legend["entries"]] == ["b", "a"]
