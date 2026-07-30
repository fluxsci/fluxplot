"""fp.surface: a per-vertex field on a mesh as named, addressable parts.

The contract this file pins down is the reason the primitive exists: the value→colour mapping must
stay DATA (a named part per category, the palette/range recorded in the manifest), never pixels.
Everything here runs on a synthetic mesh, so the tests carry no data dependency.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import fluxplot as fp  # noqa: E402


def _sphere(n_theta=24, n_phi=24, x_offset=0.0):
    """A closed UV-sphere ``(vertices, faces)`` — a stand-in for one hemisphere surface."""
    th = np.linspace(0, np.pi, n_theta)
    ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    v = np.column_stack([
        (np.sin(T) * np.cos(P)).ravel() + x_offset,
        (np.sin(T) * np.sin(P)).ravel(),
        np.cos(T).ravel(),
    ])
    faces = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            a = i * n_phi + j
            b = i * n_phi + (j + 1) % n_phi
            c = (i + 1) * n_phi + j
            d = (i + 1) * n_phi + (j + 1) % n_phi
            faces += [[a, b, c], [b, d, c]]
    return v, np.asarray(faces, dtype=int)


@pytest.fixture
def mesh():
    return {"left": _sphere(x_offset=-1.2), "right": _sphere(x_offset=+1.2)}


def _manifest(fig, tmp_path, name="surf"):
    out = tmp_path / f"{name}.svg"
    fp.save(fig, str(out))
    return json.loads((tmp_path / f"{name}.fluxplot.json").read_text()), out.read_text()


def _labels(mesh, codes=(0, 1, 2)):
    n = mesh["left"][0].shape[0]
    per = np.zeros(n, dtype=float)
    for k, c in enumerate(codes):
        per[k * n // len(codes):(k + 1) * n // len(codes)] = c
    return {"left": per, "right": per.copy()}


def test_label_map_gives_one_named_part_per_category(mesh, tmp_path):
    """Each category is its own addressable part — this is what makes a region recolourable."""
    fig, ax = plt.subplots()
    fp.surface(ax, _labels(mesh), series="atlas", surfaces=mesh, kind="label",
               categories={0: "frontal", 1: "parietal", 2: "temporal"},
               palette={"frontal": "#4C78A8", "parietal": "#F58518", "temporal": "#54A24B"})
    man, svg = _manifest(fig, tmp_path)
    entry = man["series"][0]
    assert entry["kind"] == "surface"
    assert entry["svg"]["regions"] == ["atlas.frontal", "atlas.parietal", "atlas.temporal"]
    for gid in entry["svg"]["regions"]:
        assert f'id="{gid}"' in svg, f"{gid} must exist in the SVG to be addressable"
    plt.close(fig)


def test_palette_is_recorded_and_not_remapped(mesh, tmp_path):
    """The id→colour mapping round-trips verbatim; no silent reassignment."""
    fig, ax = plt.subplots()
    palette = {"frontal": "#4C78A8", "parietal": "#F58518", "temporal": "#54A24B"}
    fp.surface(ax, _labels(mesh), series="atlas", surfaces=mesh, kind="label",
               categories={0: "frontal", 1: "parietal", 2: "temporal"}, palette=palette)
    man, _ = _manifest(fig, tmp_path)
    surf = man["series"][0]["surface"]
    assert {k: v.lower() for k, v in surf["palette"].items()} == {
        k: v.lower() for k, v in palette.items()}
    by_part = {p["part"]: p for p in surf["parts"]}
    assert by_part["parietal"]["color"].lower() == "#f58518"
    assert by_part["parietal"]["ref"] == "atlas.parietal"
    plt.close(fig)


def test_missing_is_its_own_part_and_zero_is_data(mesh, tmp_path):
    """NaN → the 'missing' part; a real 0 stays a category and must NOT be swallowed by it."""
    vals = _labels(mesh)
    vals["left"][:50] = np.nan
    fig, ax = plt.subplots()
    fp.surface(ax, vals, series="atlas", surfaces=mesh, kind="label",
               categories={0: "zero-category", 1: "b", 2: "c"})
    man, _ = _manifest(fig, tmp_path)
    entry = man["series"][0]
    assert entry["svg"]["surface-missing"] == "atlas.missing"
    assert "atlas.zero-category" in entry["svg"]["regions"], "0 is a real value, not missing"
    assert entry["surface"]["nMissing"] == 50
    plt.close(fig)


def test_sentinel_below_threshold_becomes_missing(mesh, tmp_path):
    """A -1 sentinel greys out instead of becoming a spurious extra category."""
    vals = _labels(mesh)
    vals["right"][:] = -1.0
    fig, ax = plt.subplots()
    fp.surface(ax, vals, series="atlas", surfaces=mesh, kind="label",
               categories={0: "a", 1: "b", 2: "c"}, missing_below=0)
    man, _ = _manifest(fig, tmp_path)
    surf = man["series"][0]["surface"]
    assert "category--1" not in {p["part"] for p in surf["parts"]}
    assert surf["nMissing"] == mesh["right"][0].shape[0]
    plt.close(fig)


def test_continuous_records_range_and_colormap(mesh, tmp_path):
    """A continuous field carries its full mapping so a range edit is declarative."""
    n = mesh["left"][0].shape[0]
    vals = {"left": np.linspace(0, 100, n), "right": np.linspace(0, 100, n)}
    fig, ax = plt.subplots()
    fp.surface(ax, vals, series="field", surfaces=mesh, kind="continuous",
               cmap="viridis", color_range=(10, 90), colorbar=True, cbar_label="units")
    man, _ = _manifest(fig, tmp_path, "cont")
    surf = man["series"][0]["surface"]
    assert surf["kind"] == "continuous"
    assert surf["cmap"] == "viridis"
    assert (surf["vmin"], surf["vmax"]) == (10.0, 90.0)
    cbar = [p for p in surf["parts"] if p["part"] == "colorbar"]
    assert cbar and set(cbar[0]["editable"]) == {"vmin", "vmax", "cmap"}
    plt.close(fig)


def test_percentile_clip_sets_the_range(mesh, tmp_path):
    n = mesh["left"][0].shape[0]
    vals = {"left": np.linspace(0, 100, n), "right": np.linspace(0, 100, n)}
    fig, ax = plt.subplots()
    fp.surface(ax, vals, series="field", surfaces=mesh, kind="continuous", percentile=(5, 95))
    man, _ = _manifest(fig, tmp_path, "pct")
    surf = man["series"][0]["surface"]
    assert surf["vmin"] == pytest.approx(5.0, abs=0.5)
    assert surf["vmax"] == pytest.approx(95.0, abs=0.5)
    assert surf["percentile"] == [5, 95]
    plt.close(fig)


def test_auto_kind_detects_labels_vs_continuous(mesh, tmp_path):
    n = mesh["left"][0].shape[0]
    fig, ax = plt.subplots()
    fp.surface(ax, _labels(mesh), series="a", surfaces=mesh)          # few integers -> label
    man, _ = _manifest(fig, tmp_path, "auto1")
    assert man["series"][0]["surface"]["kind"] == "label"
    plt.close(fig)

    fig, ax = plt.subplots()
    vals = {"left": np.random.default_rng(0).normal(size=n),
            "right": np.random.default_rng(1).normal(size=n)}
    fp.surface(ax, vals, series="a", surfaces=mesh)                    # continuous floats
    man, _ = _manifest(fig, tmp_path, "auto2")
    assert man["series"][0]["surface"]["kind"] == "continuous"
    plt.close(fig)


def test_wrong_value_count_is_rejected(mesh):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="one value per vertex"):
        fp.surface(ax, {"left": np.zeros(3), "right": np.zeros(3)}, series="a", surfaces=mesh)
    plt.close(fig)


def test_backface_culling_halves_the_drawn_geometry(mesh):
    """Only the camera-facing half is drawn — what keeps per-region collections from occluding."""
    fig, ax = plt.subplots()
    colls = fp.surface(ax, _labels(mesh), series="a", surfaces=mesh, kind="label",
                       views=("lateral",), hemispheres=("left",))
    drawn = sum(len(c.get_paths()) for c in colls)
    total = mesh["left"][1].shape[0]
    assert 0 < drawn < total, "back faces must be culled, front faces kept"
    plt.close(fig)


def test_flat_array_is_split_across_hemispheres(mesh, tmp_path):
    """A single concatenated array (left then right) is accepted, like a CIFTI-style vector."""
    n = mesh["left"][0].shape[0]
    flat = np.concatenate([np.zeros(n), np.ones(n)])
    fig, ax = plt.subplots()
    fp.surface(ax, flat, series="a", surfaces=mesh, kind="label",
               categories={0: "left-only", 1: "right-only"})
    man, _ = _manifest(fig, tmp_path, "flat")
    assert set(man["series"][0]["svg"]["regions"]) == {"a.left-only", "a.right-only"}
    plt.close(fig)


def _guides(man, role):
    return [g for g in man["guides"] if g["role"] == role]


def test_label_map_gets_a_legend_whose_entries_resolve_to_parts(mesh, tmp_path):
    """The legend is a real one — same manifest shape as any other plot — and each entry points at
    the region part it keys, so 'recolour the block this swatch names' is resolvable."""
    fig, ax = plt.subplots()
    fp.surface(ax, _labels(mesh), series="atlas", surfaces=mesh, kind="label",
               categories={0: "frontal", 1: "parietal", 2: "temporal"},
               palette={"frontal": "#4C78A8", "parietal": "#F58518", "temporal": "#54A24B"})
    man, svg = _manifest(fig, tmp_path, "leg")
    legend = _guides(man, "legend")
    assert legend, "a categorical map must ship a key by default"
    entries = legend[0]["entries"]
    assert [e["text"] for e in entries] == ["frontal", "parietal", "temporal"]
    for e in entries:
        assert e["part"] == f"atlas.{e['text']}"       # entry -> the addressable region
        assert e["series"] == "atlas"
        assert f'id="{e["swatch"]}"' in svg and f'id="{e["label"]}"' in svg
    plt.close(fig)


def test_region_name_does_not_become_the_series_label(mesh, tmp_path):
    """A region names a PART; letting it become the series label would make the series masquerade
    as its own first category and hijack the legend join."""
    fig, ax = plt.subplots()
    fp.surface(ax, _labels(mesh), series="atlas", surfaces=mesh, kind="label",
               categories={0: "frontal", 1: "parietal", 2: "temporal"})
    man, _ = _manifest(fig, tmp_path, "leglabel")
    assert man["series"][0].get("label") is None
    plt.close(fig)


def test_continuous_has_no_legend_by_default_and_legend_can_be_disabled(mesh, tmp_path):
    """The colorbar is a continuous map's key, so it gets no category legend; label maps can opt out."""
    n = mesh["left"][0].shape[0]
    vals = {"left": np.linspace(0, 1, n), "right": np.linspace(0, 1, n)}
    fig, ax = plt.subplots()
    fp.surface(ax, vals, series="f", surfaces=mesh, kind="continuous", colorbar=True)
    man, _ = _manifest(fig, tmp_path, "nolegend")
    assert not _guides(man, "legend")
    plt.close(fig)

    fig, ax = plt.subplots()
    fp.surface(ax, _labels(mesh), series="a", surfaces=mesh, kind="label", legend=False)
    man, _ = _manifest(fig, tmp_path, "legoff")
    assert not _guides(man, "legend")
    plt.close(fig)


def test_boundary_faces_take_the_majority_label_not_missing(mesh, tmp_path):
    """A face straddling two categories must be drawn as one of them — assigning it to 'missing'
    would both mean 'no data' and etch a pale crack along every boundary."""
    from fluxplot.surface import _face_labels
    values = np.array([0.0, 0.0, 1.0, np.nan])
    faces = np.array([[0, 1, 2],      # 2x label 0, 1x label 1 -> majority 0
                      [0, 2, 2],      # 1x label 0, 2x label 1 -> majority 1
                      [0, 1, 3]])     # touches a missing vertex -> missing
    out = _face_labels(values, faces)
    assert out[0] == 0.0 and out[1] == 1.0 and np.isnan(out[2])
