"""B6 — value × confidence: alpha_by on hexmatrix, heatmap and scatter; D8 — hexmatrix follow-ups."""
import json
import re
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from lxml import etree  # noqa: E402

import fluxplot as fp  # noqa: E402
from fluxplot.fields import alpha_channel  # noqa: E402


def _save(fig, tmp_path, name="a.svg", **kw):
    res = fp.save(fig, str(tmp_path / name), recipe=False, **kw)
    root = etree.parse(res.svg).getroot()
    return res, json.loads(open(res.manifest).read()), {el.get("id"): el for el in root.iter() if el.get("id")}


def _opacity(el):
    style = el.get("style") or ""
    m = re.search(r"fill-opacity:\s*([0-9.]+)", style)
    return float(m.group(1)) if m else (float(el.get("fill-opacity")) if el.get("fill-opacity") else 1.0)


def test_alpha_channel_is_monotone_and_recorded():
    alphas, rec = alpha_channel([1, 2, 4, 8], alpha_range=(0.2, 1.0))
    assert alphas[0] == 0.2 and alphas[-1] == 1.0 and np.all(np.diff(alphas) > 0)
    assert rec == {"source": "values", "range": [0.2, 1.0], "norm": {"kind": "linear", "vmin": 1.0, "vmax": 8.0}}
    log_alphas, rec = alpha_channel([1, 10, 100], alpha_norm="log", alpha_range=(0.0, 1.0))
    assert log_alphas == pytest.approx([0.0, 0.5, 1.0]) and rec["norm"]["kind"] == "log"
    a, _ = alpha_channel([np.nan, 3.0, 5.0], alpha_range=(0.25, 1.0))
    assert a[0] == 0.25 and a[2] == 1.0  # a missing value takes the low alpha
    a, rec = alpha_channel([2.0, 2.0])
    assert list(a) == [1.0, 1.0] and rec["norm"]["vmin"] == rec["norm"]["vmax"] == 2.0
    with pytest.raises(ValueError, match="alpha_norm"):
        alpha_channel([1, 2], alpha_norm="sqrt")
    with pytest.raises(ValueError, match="alpha_range"):
        alpha_channel([1, 2], alpha_range=(0, 2))


def test_hexmatrix_alpha_by_count_recolours_and_round_trips(tmp_path):
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=400), rng.normal(size=400)
    fig, ax = plt.subplots()
    hm = fp.hexmatrix(x=x, y=y, C=rng.normal(size=400), ax=ax, gridsize=6, series="h", alpha_by="count",
                      alpha_range=(0.3, 1.0), colorbar=False)
    res, man, ids = _save(fig, tmp_path)
    (scale,) = man["colorScales"]
    counts = [b["count"] for b in hm.bins if b["count"] is not None] if isinstance(hm.bins, list) else None
    assert scale["alpha"]["source"] == "count" and scale["alpha"]["range"] == [0.3, 1.0]
    assert scale["alpha"]["norm"]["kind"] == "linear" and scale["alpha"]["norm"]["vmax"] >= scale["alpha"]["norm"]["vmin"] >= 1
    hexes = [el for gid, el in ids.items() if gid.startswith("h.hex.")]
    assert hexes and all(el.get("data-alpha-value") is not None for el in hexes)
    # opacity is monotone in the recorded alpha value: recompute from the channel and compare
    lo, hi = scale["alpha"]["norm"]["vmin"], scale["alpha"]["norm"]["vmax"]
    a0, a1 = scale["alpha"]["range"]
    for el in hexes:
        v = float(el.get("data-alpha-value"))
        expect = a0 + (v - lo) / (hi - lo) * (a1 - a0) if hi > lo else a1
        assert _opacity(el) == pytest.approx(expect, abs=1e-3)
    # the manifest's hexmatrix payload agrees with the elements
    (s,) = man["series"]
    by_key = {f'{b["row"]}.{b["col"]}': b for b in s["hexmatrix"]["bins"]}
    for el in hexes:
        assert by_key[el.get("data-key")]["count"] == float(el.get("data-alpha-value"))


def test_hexmatrix_alpha_by_column_and_matrix_and_errors(tmp_path):
    rng = np.random.default_rng(5)
    table = {"x": rng.normal(size=300), "y": rng.normal(size=300), "p": rng.uniform(1e-4, 1, 300)}
    fig, ax = plt.subplots()
    hm = fp.hexmatrix(table, x="x", y="y", ax=ax, gridsize=5, series="h", alpha_by="p", alpha_norm="log", colorbar=False)
    res, man, ids = _save(fig, tmp_path)
    assert man["colorScales"][0]["alpha"] == {"source": "p", "range": [0.25, 1.0],
                                             "norm": {"kind": "log", "vmin": pytest.approx(man["colorScales"][0]["alpha"]["norm"]["vmin"]), "vmax": pytest.approx(man["colorScales"][0]["alpha"]["norm"]["vmax"])}}
    hexes = [el for gid, el in ids.items() if gid.startswith("h.hex.")]
    assert all(0 < float(el.get("data-alpha-value")) <= 1 for el in hexes)
    # per-hexagon array (one per drawn hexagon)
    fig, ax = plt.subplots()
    n_hex = len(hm.bins["row"])
    fp.hexmatrix(table, x="x", y="y", ax=ax, gridsize=5, series="h", alpha_by=np.linspace(0, 1, n_hex), colorbar=False)
    # matrix mode
    fig, ax = plt.subplots()
    M = rng.normal(size=(4, 5))
    P = rng.uniform(size=(4, 5))
    fp.hexmatrix(matrix=M, ax=ax, series="m", alpha_by=P, alpha_norm="log", colorbar=False)
    res, man, ids = _save(fig, tmp_path, "m.svg")
    hexes = {gid: el for gid, el in ids.items() if gid.startswith("m.hex.")}
    el = hexes["m.hex.1.2"]
    assert float(el.get("data-alpha-value")) == pytest.approx(P[1, 2])
    with pytest.raises(ValueError, match="matrix shape"):
        fp.hexmatrix(matrix=M, ax=ax, series="bad", alpha_by=P[:2], colorbar=False)
    with pytest.raises(ValueError, match="alpha_by must be 'count'"):
        fp.hexmatrix(table, x="x", y="y", ax=ax, gridsize=5, series="bad2", alpha_by=[1, 2, 3], colorbar=False)


def test_heatmap_and_scatter_alpha_by(tmp_path):
    vals = np.arange(6.0).reshape(2, 3)
    pv = np.array([[0.001, 0.01, 0.5], [0.04, 0.9, 0.0001]])
    fig, ax = plt.subplots()
    fp.heatmap(ax, vals, series="corr", cells=True, alpha_by=pv, alpha_norm="log", alpha_range=(0.2, 1.0))
    res, man, ids = _save(fig, tmp_path)
    (scale,) = man["colorScales"]
    assert scale["alpha"]["norm"]["kind"] == "log" and scale["alpha"]["norm"]["vmin"] == 0.0001 and scale["alpha"]["norm"]["vmax"] == 0.9
    cell = ids["corr.x-heatmap.cell.1.2"]
    assert float(cell.get("data-alpha-value")) == 0.0001 and _opacity(cell) == pytest.approx(0.2, abs=1e-3)
    cell = ids["corr.x-heatmap.cell.1.1"]
    assert _opacity(cell) == pytest.approx(1.0, abs=1e-3)
    with pytest.raises(ValueError, match="matrix shape"):
        fp.heatmap(ax, vals, series="bad", alpha_by=pv[0])
    # imshow heatmap: the alpha array rides on the image (no per-cell nodes)
    fig, ax = plt.subplots()
    im = fp.heatmap(ax, vals, series="img", alpha_by=pv)
    assert np.asarray(im.get_alpha()).shape == (2, 3)
    # scatter
    fig, ax = plt.subplots()
    conf = np.array([1.0, 0.5, 0.25, 0.1])
    fp.scatter(ax, [0, 1, 2, 3], [0, 1, 2, 3], c=[1, 2, 3, 4], series="pts", alpha_by=conf, alpha_range=(0.1, 1.0))
    res, man, ids = _save(fig, tmp_path, "s.svg")
    (scale,) = man["colorScales"]
    assert scale["alpha"]["source"] == "alpha_by" and scale["alpha"]["norm"] == {"kind": "linear", "vmin": 0.1, "vmax": 1.0}
    pts = [ids[f"pts.point.{k}"] for k in range(4)]
    assert [float(p.get("data-alpha-value")) for p in pts] == list(conf)
    assert _opacity(pts[0]) == pytest.approx(1.0, abs=1e-3) and _opacity(pts[3]) == pytest.approx(0.1, abs=1e-3)
    with pytest.raises(ValueError, match="one value per point"):
        fp.scatter(ax, [0, 1], [0, 1], series="bad", alpha_by=[1.0])


def test_hexmatrix_vector_limit_and_shared_axes_aspect(tmp_path):
    rng = np.random.default_rng(6)
    fig, ax = plt.subplots()
    hm = fp.hexmatrix(x=rng.normal(size=20000), y=rng.normal(size=20000), ax=ax, gridsize=45, series="big", colorbar=False)
    n_hex = len(hm.bins["row"])
    assert 800 < n_hex <= 5000  # above the generic threshold, below the hexmatrix default
    res = fp.save(fig, str(tmp_path / "v.svg"), recipe=False)
    assert res.rasterized == []
    svg = open(res.svg).read()
    assert svg.count('id="big.hex.') == n_hex
    fig, ax = plt.subplots()
    fp.hexmatrix(x=rng.normal(size=20000), y=rng.normal(size=20000), ax=ax, gridsize=45, series="big", colorbar=False,
                 vector_limit=None)  # the save's generic threshold applies
    res = fp.save(fig, str(tmp_path / "r.svg"), recipe=False)
    assert res.rasterized == ["big.hexes"]
    # aspect="auto" on shared axes: no box-aspect lock, one warning
    fig, (a, b) = plt.subplots(1, 2, sharey=True)
    with pytest.warns(UserWarning, match="box aspect unlocked"):
        fp.hexmatrix(x=rng.normal(size=300), y=rng.normal(size=300), ax=a, gridsize=6, series="l", colorbar=False)
    assert a.get_box_aspect() is None
