"""fp.hexmatrix — hexagonal binning and hex-lattice maps.

What this file pins down: every point lands in the hexagon that geometrically contains it (both
orientations, linear and log axes), the statistics are exact, every hexagon is a named part carrying
its lattice address and value, the colour scale is a Flux recipe control that round-trips (including
the single-colour ramps), and hexagon names depend only on the lattice — not on the data.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.path import Path  # noqa: E402

import fluxplot as fp  # noqa: E402


def _cloud(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.gamma(2.0, 0.7, n)
    return x, -0.5 * x + rng.normal(0, 1, n)


def _save(fig, tmp_path, name="hx", **kw):
    out = tmp_path / f"{name}.svg"
    res = fp.save(fig, str(out), **kw)
    plt.close(fig)
    man = json.loads((tmp_path / f"{name}.fluxplot.json").read_text())
    rec = json.loads((tmp_path / f"{name}.recipe.json").read_text())
    return res, man, rec, out.read_text()


def _series(man, sid):
    return next(s for s in man["series"] if s["id"] == sid)


@pytest.mark.parametrize("orientation", ["pointy", "flat"])
@pytest.mark.parametrize("scale", ["linear", "log"])
def test_every_point_lands_in_the_hexagon_that_contains_it(orientation, scale):
    rng = np.random.default_rng(1)
    x = 10 ** rng.uniform(-2, 1, 400) if scale == "log" else rng.normal(0, 3, 400)
    y = 10 ** rng.uniform(-1, 2, 400) if scale == "log" else rng.normal(0, 1, 400)
    fig, ax = plt.subplots()
    hm = fp.hexmatrix(x=x, y=y, ax=ax, gridsize=12, orientation=orientation, xscale=scale,
                      yscale=scale, colorbar=False)
    lat = hm.artists["lattice"]
    u, v = lat.to_unit(x, y)
    rows, cols = lat.index(u, v)
    polys = lat.polygons(rows, cols)
    for k in range(len(x)):  # the polygon drawn for the point's hexagon contains the point
        poly = np.stack(lat.to_unit(polys[k][:, 0], polys[k][:, 1]), axis=1)
        assert Path(poly).contains_point((u[k], v[k]), radius=1e-9) or \
            Path(poly).contains_point((u[k], v[k]), radius=-1e-9)
    # …and its centre is the nearest lattice centre (brute force over every drawn hexagon)
    cu, cv = lat.centre(hm.bins["row"], hm.bins["col"])
    d = (u[:, None] - cu[None, :]) ** 2 + (v[:, None] - cv[None, :]) ** 2
    nearest = np.argmin(d, axis=1)
    assert np.array_equal(hm.bins["row"][nearest], rows)
    assert np.array_equal(hm.bins["col"][nearest], cols)
    assert hm.bins["count"].sum() == len(x)
    plt.close(fig)


def test_statistics_are_exact():
    x, y = _cloud()
    w = np.linspace(0.5, 2.0, len(x))
    fig, axs = plt.subplots(1, 4)
    count = fp.hexmatrix(x=x, y=y, ax=axs[0], gridsize=10, colorbar=False, series="count")
    dens = fp.hexmatrix(x=x, y=y, ax=axs[1], gridsize=10, stat="density", colorbar=False, series="dens")
    prob = fp.hexmatrix(x=x, y=y, ax=axs[2], gridsize=10, stat="probability", weights=w,
                        colorbar=False, series="prob")
    mean = fp.hexmatrix(x=x, y=y, C=x * y, ax=axs[3], gridsize=10, colorbar=False, series="mean")
    assert count.bins["value"].sum() == len(x)
    area = dens.artists["lattice"].area()
    assert np.isclose((dens.bins["value"] * area).sum(), 1.0)
    assert np.isclose(prob.bins["value"].sum(), 1.0)
    lat = mean.artists["lattice"]
    rows, cols = lat.index(*lat.to_unit(x, y))
    for k in range(len(mean.bins["row"])):
        sel = (rows == mean.bins["row"][k]) & (cols == mean.bins["col"][k])
        assert np.isclose(mean.bins["value"][k], (x * y)[sel].mean())
        assert mean.bins["count"][k] == sel.sum()
    med = fp.hexmatrix(x=x, y=y, C=x * y, reduce="median", ax=axs[3], gridsize=10,
                       colorbar=False, series="med")
    sel = (rows == med.bins["row"][0]) & (cols == med.bins["col"][0])
    assert np.isclose(med.bins["value"][0], np.median((x * y)[sel]))
    plt.close(fig)


def test_every_hexagon_is_a_named_part_with_its_address_and_value(tmp_path):
    x, y = _cloud()
    fig, ax = plt.subplots()
    hm = fp.hexmatrix({"wake": x, "nrem": y}, x="wake", y="nrem", ax=ax, gridsize=14, series="rates")
    res, man, rec, svg = _save(fig, tmp_path)
    assert res.warnings == [] or all("colorbar" in w for w in res.warnings)
    assert man["plotType"] == "hexmatrix"
    s = _series(man, "rates")
    assert s["svg"] == {"x-hexbin": "rates.hexes"}
    members = s["components"][0]["members"]
    assert len(members) == len(hm.bins["row"]) == s["hexmatrix"]["nBins"]
    r, c = int(hm.bins["row"][0]), int(hm.bins["col"][0])
    assert members[0] == hm.hex_id(r, c) == f"rates.hex.{r}.{c}"
    assert f'id="rates.hex.{r}.{c}"' in svg and 'data-role="x-hex"' in svg
    assert f'data-row="{r}" data-column="{c}"' in svg
    payload = s["hexmatrix"]
    assert payload["mode"] == "points" and payload["n"] == len(x) and payload["dropped"] == 0
    assert payload["bins"][0]["count"] == hm.bins["count"][0]
    assert ax.get_xlabel() == "wake" and ax.get_ylabel() == "nrem"
    assert s["field"]["kind"] == "hexbin" and s["field"]["controlKey"] == hm.control_key
    assert any(g["role"] == "colorbar" for g in man["guides"])


def test_colour_scale_is_a_recipe_control_that_flux_can_override(tmp_path, monkeypatch):
    x, y = _cloud()
    fig, ax = plt.subplots()
    fp.hexmatrix(x=x, y=y, ax=ax, gridsize=10, color="#4CB391", series="h")
    _res, _man, rec, _svg = _save(fig, tmp_path, "a")
    ctl = rec["params"]["__fluxplot__"]["h"]  # keyed by the series, not the axes' position
    assert ctl["cmap"] == "hexmatrix.mono:#4cb391" and ctl["vmin"] == 1.0
    # replaying the recorded controls rebuilds the same single-colour ramp
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": {"h": ctl}}))
    fig, ax = plt.subplots()
    again = fp.hexmatrix(x=x, y=y, ax=ax, gridsize=10, color="#4CB391", series="h")
    assert again.cmap.name == ctl["cmap"]
    plt.close(fig)
    # a different single-colour ramp is a legitimate edit through the hexmatrix's own names
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": {"h": {"cmap": "hexmatrix.mono:#aa3311"}}}))
    fig, ax = plt.subplots()
    recoloured = fp.hexmatrix(x=x, y=y, ax=ax, gridsize=10, color="#4CB391", series="h")
    assert recoloured.cmap.name == "hexmatrix.mono:#aa3311"
    plt.close(fig)
    # an edited colormap / range from Flux's Color scales editor wins over the script's, also
    # when it was saved under the pre-0.3.1 positional key
    monkeypatch.setenv("FLUX_PARAMS", json.dumps(
        {"__fluxplot__": {"axes.1.h": {"cmap": "magma", "vmin": 2.0, "vmax": 5.0}}}))
    fig, ax = plt.subplots()
    edited = fp.hexmatrix(x=x, y=y, ax=ax, gridsize=10, color="#4CB391", norm="log", series="h")
    assert edited.cmap.name == "magma"
    assert type(edited.norm).__name__ == "LogNorm"  # the scale survives the limit override
    assert (edited.norm.vmin, edited.norm.vmax) == (2.0, 5.0)
    plt.close(fig)


def test_hexagon_names_depend_only_on_the_lattice():
    rng = np.random.default_rng(3)
    ext = (0, 10, 0, 10)
    fig, (a, b) = plt.subplots(1, 2)
    one = fp.hexmatrix(x=rng.uniform(2, 8, 300), y=rng.uniform(2, 8, 300), ax=a, extent=ext,
                       gridsize=10, colorbar=False, series="a")
    two = fp.hexmatrix(x=rng.uniform(4, 6, 50), y=rng.uniform(4, 6, 50), ax=b, extent=ext,
                       gridsize=10, colorbar=False, series="b")
    assert one.lookup(5.1, 5.1) == two.lookup(5.1, 5.1)
    first = dict(zip(zip(one.bins["row"], one.bins["col"]), zip(one.bins["x"], one.bins["y"])))
    for key, centre in zip(zip(two.bins["row"], two.bins["col"]), zip(two.bins["x"], two.bins["y"])):
        if key in first:
            assert np.allclose(first[key], centre)
    plt.close(fig)


def test_hexagons_stay_regular_through_layout():
    x, y = _cloud()
    fig, ax = plt.subplots(figsize=(4, 2.5), layout="constrained")
    hm = fp.hexmatrix(x=x, y=y, ax=ax, gridsize=12)
    rho = ax.get_box_aspect()
    assert rho is not None  # locked, so constrained layout cannot squash the hexagons
    fig.canvas.draw()
    verts = hm.hexes.get_paths()[0].vertices[:6]
    px = ax.transData.transform(verts)
    sides = np.hypot(*np.diff(np.vstack([px, px[:1]]), axis=0).T)
    assert np.allclose(sides, sides.mean(), rtol=1e-6)
    plt.close(fig)
    fig, ax = plt.subplots()
    fp.hexmatrix(x=x, y=y, ax=ax, aspect="equal", binwidth=0.4, colorbar=False)
    assert ax.get_aspect() == 1.0
    plt.close(fig)


def test_matrix_mode(tmp_path):
    M = np.arange(12, dtype=float).reshape(3, 4)
    M[1, 2] = np.nan
    fig, ax = plt.subplots()
    hm = fp.hexmatrix(matrix=M, ax=ax, series="som", gap=0.1)
    assert len(hm.bins["row"]) == 11 and (1, 2) not in set(zip(hm.bins["row"], hm.bins["col"]))
    assert hm.bins["count"] is None
    top = hm.bins["y"][hm.bins["row"] == 0].mean()
    assert top > hm.bins["y"][hm.bins["row"] == 2].mean()  # origin="upper": row 0 on top
    assert np.array_equal(hm.bins["value"], np.delete(M.ravel(), 6))
    _res, man, _rec, svg = _save(fig, tmp_path)
    assert 'id="som.hex.2.3"' in svg and 'id="som.hex.1.2"' not in svg
    assert _series(man, "som")["hexmatrix"]["shape"] == [3, 4]


def test_empty_and_sparse_hexagons(tmp_path):
    x, y = _cloud()
    fig, (a, b) = plt.subplots(1, 2)
    full = fp.hexmatrix(x=x, y=y, ax=a, gridsize=8, mincnt=0, colorbar=False, series="full")
    some = fp.hexmatrix(x=x, y=y, ax=b, gridsize=8, colorbar=False, series="some")
    assert (full.bins["count"] == 0).any() and len(full.bins["row"]) > len(some.bins["row"])
    assert full.bins["count"].sum() == len(x)
    plt.close(fig)
    fig, ax = plt.subplots()
    sp = fp.hexmatrix(x=x, y=y, ax=ax, gridsize=12, sparse=3, series="sp")
    assert sp.bins["count"].min() >= 3
    n_points = len(sp.points.get_offsets())
    assert n_points + sp.bins["count"].sum() == len(x)
    _res, man, _rec, svg = _save(fig, tmp_path)
    assert 'id="sp-points.points"' in svg


def test_marginals_and_identity_line(tmp_path):
    rng = np.random.default_rng(5)
    x = 10 ** rng.normal(-0.7, 0.5, 500)
    fig, ax = plt.subplots(figsize=(3, 3))
    hm = fp.hexmatrix(x=x, y=x * 10 ** rng.normal(0, 0.2, 500), ax=ax, xscale="log", yscale="log",
                      norm="log", marginals=True, identity_line=True, series="r")
    assert set(hm.marginal_axes) == {"x", "y"}
    _res, man, _rec, svg = _save(fig, tmp_path)
    ids = {p["id"] for p in man["panels"]}
    assert {"panel.r-x-marginal", "panel.r-y-marginal"} <= ids
    assert _series(man, "panel.r-x-marginal.r-x")["distribution"]["counts"]
    assert 'id="panel.a.reference-line.identity"' in svg
    assert _series(man, "panel.a.r")["hexmatrix"]["scale"] == {"x": "log", "y": "log"}


def test_refuses_what_it_cannot_draw():
    x, y = _cloud(50)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="give x and y"):
        fp.hexmatrix(x=x, ax=ax)
    with pytest.raises(ValueError, match="excludes"):
        fp.hexmatrix(matrix=np.eye(3), x=x, ax=ax)
    with pytest.raises(ValueError, match="orientation"):
        fp.hexmatrix(x=x, y=y, ax=ax, orientation="round")
    with pytest.raises(ValueError, match="cmap or color"):
        fp.hexmatrix(x=x, y=y, ax=ax, cmap="viridis", color="red")
    with pytest.raises(ValueError, match="no finite points"):
        fp.hexmatrix(x=[np.nan], y=[1.0], ax=ax)
    with pytest.raises(ValueError, match="center= needs a linear norm"):
        fp.hexmatrix(x=x, y=y, C=y, ax=ax, center=0, norm="log")
    with pytest.raises(ValueError, match="center must lie between vmin and vmax"):
        fp.hexmatrix(x=x, y=y, C=y, ax=ax, center=0, vmin=0.5, vmax=3)
    hm = fp.hexmatrix(x=x, y=y, C=y, ax=ax, center=0, colorbar=False)  # symmetric auto limits
    assert hm.norm.vmin == -hm.norm.vmax and type(hm.norm).__name__ == "TwoSlopeNorm"
    with pytest.raises(KeyError):
        fp.hexmatrix({"a": x}, x="a", y="b", ax=ax)
    plt.close(fig)
