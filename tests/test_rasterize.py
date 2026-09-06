"""Auto-rasterization of pathologically heavy layers (the safety default).

A ``LineCollection`` of per-edge segments emits one ``<path>`` per segment and a scatter
emits one ``<use>`` per point, so a single realistic panel can carry 10^4–10^5 SVG nodes.
Downstream editors inline that markup as live DOM, where it is ruinous. ``fp.save`` now
rasterizes such layers to a single embedded ``<image>`` by default.

The contract these tests pin:

1. heavy layers become ONE ``<image>``; light artists (axes, ticks, labels, legend) stay vector;
2. the layer KEEPS its gid, ``data-role``/``data-series`` and manifest entry — matplotlib drops
   the gid when it rasterizes, and ``raster.reattach`` puts it back (the load-bearing bit);
3. with several heavy layers, each image gets the gid of the artist it actually came from;
4. ``force_vectors`` (argument or env) restores full vector output and reports the cost;
5. an ordinary plot is byte-for-byte unaffected;
6. the figure the caller handed in is not mutated, and output stays deterministic.
"""
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

import fluxplot as fp  # noqa: E402
from fluxplot import raster as _raster  # noqa: E402

HEAVY = 5000  # comfortably over the 800 default threshold
# fp.tag with a non-core role namespaces it as an "x-" extension (roles.py), so the
# gid for role="morphology" on series "axon" is "axon.x-morphology".
AXON = "axon.x-morphology"
DENDRITE = "dendrite.x-morphology"


def _edge_segments(n, *, x0=0.0, x1=1.0, seed=0):
    """``n`` disconnected 2-point segments — the shape that explodes into ``n`` <path>s."""
    rng = np.random.default_rng(seed)
    pts = np.column_stack([rng.uniform(x0, x1, n), rng.uniform(0, 1, n)])
    ends = pts + rng.uniform(-0.01, 0.01, (n, 2))
    return np.stack([pts, ends], axis=1)


def _heavy_fig(n=HEAVY, seed=0):
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    lc = LineCollection(_edge_segments(n, seed=seed), colors="#205EA6", linewidths=0.2)
    ax.add_collection(lc)
    fp.tag(lc, role="morphology", series="axon")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x label")
    ax.set_ylabel("y label")
    return fig, ax, lc


def _read(res):
    return open(res.svg).read(), json.load(open(res.manifest))


def _svg_ids(svg):
    return set(re.findall(r'\bid="([^"]+)"', svg))


# --------------------------------------------------------------------------------------
# 1. heavy → one <image>; the rest stays vector
# --------------------------------------------------------------------------------------
def test_heavy_layer_becomes_one_image(tmp_path):
    fig, ax, lc = _heavy_fig()
    res = fp.save(fig, str(tmp_path / "heavy.svg"))
    plt.close(fig)
    svg, _man = _read(res)

    assert svg.count("<image") == 1, "the heavy collection should collapse to a single <image>"
    assert svg.count("<path") < 100, f"expected the {HEAVY} segments gone, got {svg.count('<path')}"
    # the scaffold is untouched: real text, real tick paths
    assert "x label" in svg and "y label" in svg
    assert svg.count("<text") >= 2


def test_light_artists_are_never_rasterized(tmp_path):
    """Axes/ticks/labels and any modest series stay vector alongside a rasterized layer."""
    fig, ax, lc = _heavy_fig()
    fp.line(ax, [0, 0.5, 1], [0.1, 0.5, 0.9], series="trend", label="Trend")
    ax.legend()
    res = fp.save(fig, str(tmp_path / "mixed.svg"))
    plt.close(fig)
    svg, man = _read(res)

    assert svg.count("<image") == 1
    trend = next(s for s in man["series"] if s["id"] == "trend")
    assert trend["svg"]["line"] in _svg_ids(svg)
    assert not trend.get("rasterized"), "a 3-point line must not be rasterized"


# --------------------------------------------------------------------------------------
# 2. the gid survives — matplotlib drops it, we put it back
# --------------------------------------------------------------------------------------
def test_rasterized_layer_keeps_its_gid_and_semantics(tmp_path):
    fig, ax, lc = _heavy_fig()
    res = fp.save(fig, str(tmp_path / "gid.svg"))
    plt.close(fig)
    svg, man = _read(res)

    assert f'id="{AXON}"' in svg, "the gid must be re-attached to the raster <image>"
    assert not re.search(r'id="image[0-9a-f]{6,}"', svg), "matplotlib's generated id must be gone"
    img = re.search(r"<image[^>]*>", svg).group(0)
    assert 'data-role="x-morphology"' in img
    assert 'data-series="axon"' in img
    assert 'data-rasterized="1"' in img

    series = next(s for s in man["series"] if s["id"] == "axon")
    assert series["svg"]["x-morphology"] == AXON
    assert series["rasterized"] is True
    assert series["svg"]["x-morphology"] in _svg_ids(svg), "manifest must resolve into the SVG"
    assert res.rasterized == [AXON]


def test_rasterized_points_report_no_spurious_warning(tmp_path):
    """A rasterized cloud has no per-point <use>; that is the intent, not a shortfall."""
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    rng = np.random.default_rng(1)
    fp.scatter(ax, rng.random(HEAVY), rng.random(HEAVY), series="cells")
    res = fp.save(fig, str(tmp_path / "pts.svg"))
    plt.close(fig)
    svg, man = _read(res)

    assert svg.count("<image") == 1
    assert not [w for w in res.warnings if "<use> vs" in w], (
        f"the per-point split must degrade silently when rasterized: {res.warnings}"
    )
    series = next(s for s in man["series"] if s["id"] == "cells")
    assert series["svg"]["points"] in _svg_ids(svg)
    assert not series.get("points"), "no per-point entries survive rasterization"
    assert series["rasterized"] is True


def test_manifest_svg_integrity_under_rasterization(tmp_path):
    """Every id the manifest names still exists in the SVG (the X-ray contract)."""
    fig, ax, lc = _heavy_fig()
    fp.line(ax, [0, 1], [0, 1], series="trend", label="Trend")
    ax.legend()
    res = fp.save(fig, str(tmp_path / "integrity.svg"))
    plt.close(fig)
    svg, man = _read(res)
    ids = _svg_ids(svg)

    refs = set()
    for s in man["series"]:
        for v in s["svg"].values():
            refs.update(v if isinstance(v, list) else [v])
    for o in man["overlays"]:
        refs.add(o["svgId"])
    missing = {r for r in refs if r not in ids}
    assert not missing, f"manifest references ids absent from the SVG: {missing}"


# --------------------------------------------------------------------------------------
# 3. several heavy layers → each image gets the gid of ITS OWN artist
# --------------------------------------------------------------------------------------
def test_two_heavy_layers_are_not_swapped(tmp_path):
    """Matching is by draw order; a swap would silently mislabel one layer as the other.

    The layers are placed in disjoint halves of the axes, so the emitted images' x extents
    identify which artist produced which — independent of the matching code under test.
    """
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    left = LineCollection(_edge_segments(HEAVY, x0=0.0, x1=0.3, seed=2), colors="#205EA6")
    right = LineCollection(_edge_segments(HEAVY, x0=0.7, x1=1.0, seed=3), colors="#BC5215")
    ax.add_collection(left)
    ax.add_collection(right)
    fp.tag(left, role="morphology", series="axon")
    fp.tag(right, role="morphology", series="dendrite")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    res = fp.save(fig, str(tmp_path / "two.svg"))
    plt.close(fig)
    svg, _man = _read(res)

    imgs = re.findall(r"<image[^>]*>", svg)
    assert len(imgs) == 2
    by_id = {
        re.search(r'id="([^"]+)"', i).group(1): float(re.search(r'\bx="(-?[\d.eE+]+)"', i).group(1))
        for i in imgs
    }
    assert set(by_id) == {AXON, DENDRITE}, by_id
    assert by_id[AXON] < by_id[DENDRITE], (
        f"gids landed on the wrong images: {by_id}"
    )


def test_raster_identity_does_not_depend_on_backend_generated_ids(tmp_path, monkeypatch):
    """Explicit draw scopes retain identity even when backend image naming changes."""
    fig, ax, lc = _heavy_fig()
    monkeypatch.setattr(_raster, "AUTO_IMAGE_ID", re.compile(r"^never-matches$"))
    res = fp.save(fig, str(tmp_path / "ambig.svg"))
    plt.close(fig)
    svg, _man = _read(res)
    assert not any("generated <image>" in w for w in res.warnings)
    assert f'id="{AXON}"' in svg
    assert res.rasterized == [AXON]


def test_force_vectors_keeps_vectors_and_reports_the_cost(tmp_path):
    fig, ax, lc = _heavy_fig()
    res = fp.save(fig, str(tmp_path / "vec.svg"), force_vectors=True)
    plt.close(fig)
    svg, man = _read(res)

    assert svg.count("<image") == 0
    assert svg.count("<path") > HEAVY
    assert res.rasterized == []
    assert any("force_vectors=True" in w and AXON in w for w in res.warnings), (
        f"the cost must still be reported: {res.warnings}"
    )
    assert not next(s for s in man["series"] if s["id"] == "axon").get("rasterized")


def test_env_force_vectors(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXPLOT_FORCE_VECTORS", "1")
    fig, ax, lc = _heavy_fig()
    res = fp.save(fig, str(tmp_path / "env.svg"))
    plt.close(fig)
    svg, _man = _read(res)
    assert svg.count("<image") == 0 and svg.count("<path") > HEAVY


def test_per_artist_rasterized_false_does_not_override_the_default(tmp_path):
    """matplotlib's factory ``rasterized=False`` is not a considered choice — force_vectors is."""
    fig, ax, lc = _heavy_fig()
    lc.set_rasterized(False)
    res = fp.save(fig, str(tmp_path / "explicit.svg"))
    plt.close(fig)
    svg, _man = _read(res)
    assert svg.count("<image") == 1


def test_threshold_is_configurable(tmp_path):
    fig, ax, lc = _heavy_fig()
    res = fp.save(fig, str(tmp_path / "thresh.svg"), raster_threshold=HEAVY * 10)
    plt.close(fig)
    svg, _man = _read(res)
    assert svg.count("<image") == 0, "nothing is over a threshold set above the data"


def test_raster_dpi_controls_only_the_raster_weight(tmp_path):
    """Higher dpi ⇒ more pixels in the embedded PNG, same vector scaffold."""
    small = fp.save(*_dpi_case(tmp_path, "lo"), raster_dpi=100)
    large = fp.save(*_dpi_case(tmp_path, "hi"), raster_dpi=600)
    lo, hi = open(small.svg).read(), open(large.svg).read()
    assert len(hi) > len(lo) * 2
    assert lo.count("<text") == hi.count("<text")


def _dpi_case(tmp_path, name):
    fig, _ax, _lc = _heavy_fig()
    return fig, str(tmp_path / f"{name}.svg")


# --------------------------------------------------------------------------------------
# 5-6. no collateral damage: light plots, the caller's figure, determinism
# --------------------------------------------------------------------------------------
def test_ordinary_plot_is_byte_identical_either_way(tmp_path):
    """Nothing is heavy ⇒ the rasterization pass must be a complete no-op."""
    def build():
        fig, ax = plt.subplots(figsize=(4, 3))
        fp.line(ax, [0, 1, 2, 3], [1, 2, 1.5, 3], series="alpha", marker="o", label="Alpha")
        ax.set_xlabel("t")
        return fig

    # same basename in different dirs: the hashsalt is derived from the plot NAME, so
    # renaming the file would change matplotlib's clip/marker ids and fake a difference.
    da, db = tmp_path / "auto", tmp_path / "forced"
    da.mkdir()
    db.mkdir()
    f1 = build()
    a = fp.save(f1, str(da / "plot.svg"))
    plt.close(f1)
    f2 = build()
    b = fp.save(f2, str(db / "plot.svg"), force_vectors=True)
    plt.close(f2)

    assert open(a.svg, "rb").read() == open(b.svg, "rb").read()
    assert a.rasterized == [] and a.warnings == b.warnings == []


def test_save_does_not_mutate_the_callers_figure(tmp_path):
    fig, ax, lc = _heavy_fig()
    assert lc.get_rasterized() is False
    before_composite = fig.suppressComposite
    fp.save(fig, str(tmp_path / "mut.svg"))
    assert lc.get_rasterized() is False, "save must hand the figure back exactly as given"
    assert fig.suppressComposite == before_composite
    plt.close(fig)


def test_rasterized_output_is_deterministic(tmp_path):
    outs = []
    for name in ("d1", "d2"):
        fig, _ax, _lc = _heavy_fig(seed=7)
        res = fp.save(fig, str(tmp_path / f"{name}.svg"))
        plt.close(fig)
        outs.append(open(res.svg, "rb").read())
    assert outs[0] == outs[1], "embedded rasters must be byte-stable across runs"


def test_vector_geometry_is_dpi_invariant(tmp_path):
    """Pins NOTES_matplotlib_svg.md §6 — raising dpi for rasters must not move vector content."""
    from fluxplot import render as _render

    def vector_svg(dpi):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([0, 1, 2], [1, 3, 2])
        ax.set_xlabel("t")
        out = _render.render_svg(fig, hashsalt="dpi-probe", dpi=dpi)
        plt.close(fig)
        return out

    assert vector_svg(100) == vector_svg(600)


# --------------------------------------------------------------------------------------
# the sweep addition that makes the image match unambiguous
# --------------------------------------------------------------------------------------
def test_untagged_image_is_swept_to_extra(tmp_path):
    """Raw ax.imshow now gets a gid — both for addressability and so that a generated
    image id can only ever mean "rasterization produced this"."""
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.imshow(np.random.default_rng(0).random((8, 8)))
    res = fp.save(fig, str(tmp_path / "img.svg"))
    plt.close(fig)
    svg, man = _read(res)

    extras = [o for o in man["overlays"] if o["role"] == "extra"]
    assert any(o["svgId"].startswith("extra.image") for o in extras), extras
    assert 'id="extra.image.0"' in svg


def test_primitive_count_shapes():
    """The estimator reads the artists we actually care about, and never raises."""
    fig, ax = plt.subplots()
    lc = LineCollection(_edge_segments(120))
    ax.add_collection(lc)
    assert _raster.primitive_count(lc) == 120
    sc = ax.scatter(np.zeros(37), np.zeros(37))
    assert _raster.primitive_count(sc) == 37
    (line,) = ax.plot([0, 1, 2], [0, 1, 2], marker="o")
    assert _raster.primitive_count(line) == 4  # 3 markers + the line itself
    (bare,) = ax.plot([0, 1, 2], [0, 1, 2])
    assert _raster.primitive_count(bare) == 1
    assert _raster.primitive_count(object()) == 0  # not an Artist → "unknown", never heavy
    plt.close(fig)
