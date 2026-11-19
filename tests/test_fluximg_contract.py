"""fluximg/1 conformance, judged by Flux's pinned contract tools (Node + tsx).

- READER: ``fp.imread`` of every reference bundle returns ``expected/<name>.npy`` exactly.
- WRITER: ``fp.imsave`` of every ``inputs/<name>.npy`` passes ``fluximg-validate``, decodes within
  ``maxAbsError`` of the input with missing data in the same places, and picks the fixture's encoding.
- The label-outline port gives the reference ``outlines.ts`` output character for character.
- The owner-approved workflow and a bundle with every layer kind on a tiled grid validate.

The tools live in a Flux worktree (``$FLUXIMG_CONTRACT``, default below). Without them, or without
Node, these tests skip and say why; on the development machine they must pass.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import fluxplot as fp
from fluxplot._fluximg import writer as W
from fluxplot._fluximg._outlines import label_outlines

CONTRACT = Path(os.environ.get("FLUXIMG_CONTRACT", "/home/driessen2/flux/flux-wt/fluximg-contract"))
NODE = shutil.which("node")


def _skip_reason():
    if not (CONTRACT / "scripts" / "fluximg-validate.ts").is_file():
        return f"the pinned fluximg contract tools are not at {CONTRACT} (set FLUXIMG_CONTRACT)"
    if NODE is None:
        return "node is not on PATH"
    if not (CONTRACT / "node_modules" / "tsx").is_dir():
        return f"tsx is not installed in {CONTRACT} (run npm install there)"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=str(_skip_reason()))


def node(*args, timeout=600):
    return subprocess.run([NODE, "--import", "tsx", *map(str, args)], cwd=CONTRACT, capture_output=True, text=True,
                          timeout=timeout)


def validate(bundle, values=None):
    """``(ok, report)`` from fluximg-validate --json (and the values a correct reader returns)."""
    args = ["scripts/fluximg-validate.ts", bundle, "--json"]
    if values is not None:
        args += ["--values", values]
    r = node(*args)
    assert r.returncode in (0, 1), f"fluximg-validate failed to run:\n{r.stdout}\n{r.stderr}"
    return r.returncode == 0, json.loads(r.stdout)


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    out = tmp_path_factory.mktemp("fluximg-fixtures")
    r = node("scripts/fluximg-fixtures.ts", out)
    assert r.returncode == 0, f"fluximg-fixtures failed:\n{r.stdout}\n{r.stderr}"
    return out, json.loads((out / "fixtures.json").read_text())


def _values(img):
    v = img.values.astype(np.float64)
    if img.nodata is not None:
        v[img.values == img.nodata] = np.nan
    return v


def test_reader_returns_exactly_the_expected_values(fixtures):
    out, index = fixtures
    assert {"mid_tiled_u16", "big_tiled_u16", "mean_f64_layers", "movie_f32", "sparse_f32"} <= set(index)
    wrong = []
    for name, f in index.items():
        img = fp.imread(out / f"{name}.fluximg")
        want = np.load(out / "expected" / f"{name}.npy")
        assert img.dims == f["dims"] and img.shape == f["shape"], name
        if not np.array_equal(_values(img), want, equal_nan=True):
            wrong.append(name)
        for lv, factor in enumerate(f["levels"]):
            if lv:
                over = fp.imread(out / f"{name}.fluximg", level=lv)
                assert over.downsample == factor and over.values.shape[-1] == -(-f["shape"][-1] // factor), name
        for layer in img.manifest["layers"]:
            if layer["kind"] != "vector":
                assert img.layer(layer["id"]).shape == tuple(f["shape"][-2:]), (name, layer["id"])
            if layer.get("table"):
                cols = img.table(layer["id"])
                assert layer["key"] in cols, (name, layer["id"])
    assert not wrong, f"imread differs from expected/*.npy for {wrong}"


def test_writer_conforms_on_every_fixture_input(fixtures, tmp_path, monkeypatch):
    out, index = fixtures
    problems = []
    for name, f in index.items():
        data = np.load(out / "inputs" / f"{name}.npy")
        monkeypatch.setattr(W, "MAX_SHARD_BYTES", f["options"].get("maxShardBytes", 64 * 1024 * 1024))
        rep = fp.imsave(tmp_path / name, data, dims=f["dims"], nodata=f["nodata"], lossless=f["lossless"])
        ok, report = validate(rep.path, tmp_path / f"{name}.values.npy")
        if not ok:
            problems.append(f"{name}: {report['issues']}")
            continue
        if rep.encoding != f["encoding"]:
            problems.append(f"{name}: encoding {rep.encoding}, the fixture's is {f['encoding']}")
        levels = [lv["downsample"] for lv in rep.manifest["pixels"]["levels"]]
        if levels != f["levels"]:
            problems.append(f"{name}: levels {levels}, the fixture's are {f['levels']}")
        got = np.load(tmp_path / f"{name}.values.npy")
        x = data.astype(np.float64)
        if f["nodata"] is not None:
            x[data == f["nodata"]] = np.nan
        if not np.array_equal(np.isnan(got), np.isnan(x)):
            problems.append(f"{name}: missing data in different places")
            continue
        dims = f["dims"]
        per_channel = np.moveaxis(np.abs(got - x), dims.index("c"), 0) if "c" in dims else np.abs(got - x)[None]
        for c, bound in enumerate(f["maxAbsError"] if "c" in dims else f["maxAbsError"][:1]):
            err = np.nanmax(per_channel[c]) if np.isfinite(per_channel[c]).any() else 0.0
            if not err <= bound * (1 + 1e-9):
                problems.append(f"{name}: channel {c} error {err} > maxAbsError {bound}")
    assert not problems, "\n".join(problems)


def test_writer_tiles_the_mid_sized_fixture_like_the_contract(fixtures, tmp_path):
    out, index = fixtures
    f = index["mid_tiled_u16"]
    rep = fp.imsave(tmp_path / "mid", np.load(out / "inputs" / "mid_tiled_u16.npy"), dims=f["dims"])
    levels = [(lv["downsample"], lv["shape"]) for lv in rep.manifest["pixels"]["levels"]]
    assert levels == [(1, [2, 1100, 1300]), (4, [2, 275, 325])]
    z = json.loads((Path(rep.path) / "pixels" / "0" / "zarr.json").read_text())
    assert z["codecs"][0]["configuration"]["chunk_shape"] == [1, 512, 512]


OUTLINES_SCRIPT = """
import { readFileSync } from "node:fs";
import { labelOutlines } from "%s";
const cases = JSON.parse(readFileSync(process.argv[2], "utf8"));
console.log(JSON.stringify(cases.map((c) => labelOutlines(Uint32Array.from(c.labels), c.width, c.height))));
"""


def _label_cases():
    rng = np.random.default_rng(11)
    cases = []
    checker = (np.indices((9, 11)).sum(axis=0) % 2).astype(np.uint32) * 5       # every pixel touches diagonally
    cases.append(checker)
    ring = np.zeros((12, 12), np.uint32)
    ring[1:11, 1:11] = 3
    ring[3:9, 3:9] = 0
    ring[5:7, 5:7] = 3                                                        # an island in a hole
    cases.append(ring)
    for density, labels, shape in [(0.3, 1, (20, 25)), (0.55, 3, (30, 17)), (0.7, 8, (40, 40)), (0.5, 200, (16, 64))]:
        lab = (rng.random(shape) < density) * rng.integers(1, labels + 1, shape)
        cases.append(lab.astype(np.uint32))
    blobs = np.zeros((50, 60), np.uint32)
    for k in range(1, 13):
        y, x, r = rng.integers(5, 45), rng.integers(5, 55), rng.integers(2, 8)
        yy, xx = np.ogrid[:50, :60]
        blobs[(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = k
    cases.append(blobs)
    return cases


def test_outline_port_matches_the_reference_exactly(tmp_path):
    outlines_ts = (CONTRACT / "src" / "lib" / "fluximg" / "outlines.ts").resolve()
    script = tmp_path / "outlines_check.ts"
    script.write_text(OUTLINES_SCRIPT % outlines_ts.as_uri())
    cases = _label_cases()
    (tmp_path / "cases.json").write_text(json.dumps(
        [{"labels": c.ravel().tolist(), "width": c.shape[1], "height": c.shape[0]} for c in cases]))
    r = node(script, tmp_path / "cases.json")
    assert r.returncode == 0, r.stderr
    reference = json.loads(r.stdout)
    for lab, ref in zip(cases, reference):
        mine = label_outlines(lab)
        assert [o.label for o in mine] == [o["label"] for o in ref]
        for o, want in zip(mine, ref):
            assert o.d == want["d"], f"label {o.label}"
            assert o.pixels == want["pixels"]
            assert list(o.box) == [want["box"][k] for k in ("x", "y", "width", "height")]
            assert (o.cx, o.cy) == (want["cx"], want["cy"])


def test_the_approved_workflow_validates(tmp_path):
    from test_fluximg import approved_workflow
    out = approved_workflow(str(tmp_path))
    for key in ("mean", "stack", "trial"):
        ok, report = validate(out[key].path)
        assert ok, f"{key}: {report['issues']}"


def test_every_layer_kind_on_a_tiled_grid_validates(tmp_path):
    rng = np.random.default_rng(12)
    H, W_ = 1100, 1030
    data = (100 + 10 * rng.standard_normal((2, 3, H, W_))).astype(np.float32)
    data[:, :, :50] = np.nan
    lab = np.zeros((H, W_), np.int32)
    for k in range(1, 40):
        y, x = rng.integers(60, H - 20), rng.integers(10, W_ - 20)
        lab[y:y + 9, x:x + 7] = k * 1000                          # big label numbers: uint32 not needed, sparse ids
    img = fp.fluximg(data, dims=("c", "trial", "y", "x"), pixel_size=0.2, channels=["a", {"name": "b", "lut": "viridis"}],
                     coords={"trial": ["x1", "x2", "x3"]}, default_index={"trial": 1})
    img.add_image("speed", rng.random((H, W_)), lut="cmr.ember", opacity="value")
    img.add_image("count", rng.integers(0, 50, (H, W_)).astype(np.uint8), lossless=True)
    img.add_labels("cells", lab, weights=np.where(lab > 0, rng.random((H, W_)), np.nan))
    img.add_shapes("marks", circles={"row": [100, 200], "col": [100, 300], "id": ["p", "q"]}, diameter=2.0,
                   polygons={"tri": [[0, 60], [50, 60], [25, 100]]})
    img.add_region("Corner", (0, 50, 100, 100))
    rep = img.save(tmp_path / "every")
    m = rep.manifest
    assert m["pixels"]["dims"] == ["trial", "c", "y", "x"] and m["pixels"]["ome"] is False
    assert [lv["downsample"] for lv in m["pixels"]["levels"]] == [1, 4]
    z = json.loads((Path(rep.path) / "layers" / "cells" / "zarr.json").read_text())
    assert z["codecs"][0]["configuration"]["chunk_shape"] == [512, 512]
    ok, report = validate(rep.path)
    assert ok, report["issues"]
    photo = fp.imsave(tmp_path / "photo", rng.integers(0, 256, (30, 40, 3)).astype(np.uint8), photometric="rgb")
    ok, report = validate(photo.path)
    assert ok, report["issues"]
