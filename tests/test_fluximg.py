"""fp.fluximg / fp.imsave / fp.imread: write → read round trips and every documented behaviour.

Self-contained (no Node); the contract checks against Flux's pinned tools live in
test_fluximg_contract.py. All data is synthetic.
"""
import itertools
import json
import os
import stat

import numpy as np
import pytest

import fluxplot as fp
from fluxplot._fluximg import _encoding as E
from fluxplot._fluximg import _zarr as Z
from fluxplot._fluximg import _zstd
from fluxplot._fluximg import writer as W
from fluxplot._fluximg.reader import FluximgFormatError


def noisy(shape, seed=0, dtype=np.float32, base=100.0, amp=10.0):
    """Smooth structure plus noise: an image the guard lets through."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 3, shape[-2]), np.linspace(0, 3, shape[-1]), indexing="ij")
    return (base + amp * np.sin(yy) * np.cos(xx) + amp * 0.3 * rng.standard_normal(shape)).astype(dtype)


def man(rep_or_path):
    path = rep_or_path.path if hasattr(rep_or_path, "path") else os.fspath(rep_or_path)
    with open(os.path.join(path, "fluximg.json"), encoding="utf-8") as f:
        return json.load(f)


def save(tmp_path, data, dims=None, name="img", **kw):
    return fp.imsave(tmp_path / name, data, dims=dims, **kw)


def as_values(img):
    """Values as float64 with NaN for no data (integer no-data included)."""
    v = img.values.astype(np.float64)
    if img.nodata is not None:
        v[img.values == img.nodata] = np.nan
    return v


# ---------------------------------------------------------------------------------------------
# round trips
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"])
def test_exact_round_trip_for_every_stored_dtype(tmp_path, dtype):
    rng = np.random.default_rng(1)
    info = np.iinfo(dtype) if np.dtype(dtype).kind in "iu" else None
    if info is not None:
        data = rng.integers(info.min, info.max, (2, 9, 13), endpoint=True).astype(dtype)
    else:
        data = rng.standard_normal((2, 9, 13)).astype(dtype) * 1e3
        data[0, :2] = np.nan
    rep = save(tmp_path, data, ("c", "y", "x"), lossless=True)
    m = man(rep)
    assert m["pixels"]["dtype"] == dtype and m["pixels"]["encoding"]["kind"] == "exact"
    img = fp.imread(rep.path)
    assert img.values.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(img.values, data)
    assert img.dims == ["c", "y", "x"] and img.shape == [2, 9, 13]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_linear16_round_trip_is_within_half_a_step(tmp_path, dtype):
    data = noisy((3, 2, 40, 50), dtype=dtype)
    data[:, :, :10, :] = np.nan
    data[1, 1] = np.nan                                      # a whole plane without data
    rep = save(tmp_path, data, ("t", "c", "y", "x"))
    m = man(rep)
    enc = m["pixels"]["encoding"]
    assert enc["kind"] == "linear16" and m["pixels"]["dtype"] == "uint16" and enc["nodataCode"] == 0
    assert enc["sourceDtype"] == dtype and len(enc["channels"]) == 2
    img = fp.imread(rep.path)
    assert img.values.dtype == np.float64
    np.testing.assert_array_equal(np.isnan(img.values), np.isnan(data))
    for c in range(2):
        p = enc["channels"][c]
        finite = data[:, c][np.isfinite(data[:, c])].astype(np.float64)
        assert p["offset"] == finite.min() and p["scale"] == (finite.max() - finite.min()) / 65534
        err = np.nanmax(np.abs(img.values[:, c] - data[:, c].astype(np.float64)))
        assert err <= p["scale"] / 2 * (1 + 1e-9)


def test_linear16_values_follow_the_decode_formula(tmp_path):
    data = noisy((40, 50), dtype=np.float64)
    rep = save(tmp_path, data)
    p = man(rep)["pixels"]["encoding"]["channels"][0]
    codes = Z.ArrayReader(os.path.join(rep.path, "pixels", "0"), _zstd.codec()).read()
    expect = p["offset"] + p["scale"] * (codes.astype(np.float64) - 1)
    np.testing.assert_array_equal(fp.imread(rep.path).values, expect)


def test_integer_data_is_always_exact_and_nodata_is_recorded(tmp_path):
    data = np.random.default_rng(2).integers(0, 4000, (30, 40)).astype(np.int16)
    data[:5] = -1
    rep = save(tmp_path, data, nodata=-1)
    m = man(rep)
    assert m["pixels"]["encoding"] == {"kind": "exact", "sourceDtype": "int16"} and m["pixels"]["nodata"] == -1
    z = json.loads(open(os.path.join(rep.path, "pixels", "0", "zarr.json")).read())
    assert z["fill_value"] == -1
    img = fp.imread(rep.path)
    assert img.nodata == -1
    np.testing.assert_array_equal(img.values, data)
    assert m["channels"][0]["stats"]["min"] == data[data != -1].min()


def test_float_nodata_becomes_missing(tmp_path):
    data = noisy((2, 3, 20, 30))
    data[:, :, :4] = -1
    rep = save(tmp_path, data, ("c", "z", "y", "x"), nodata=-1, lossless=True)
    m = man(rep)
    assert "nodata" not in m["pixels"]
    z = json.loads(open(os.path.join(rep.path, "pixels", "0", "zarr.json")).read())
    assert z["fill_value"] == "NaN"
    v = fp.imread(rep.path).values
    assert np.isnan(v[:, :, :4]).all() and np.isfinite(v[:, :, 4:]).all()


def test_masked_arrays(tmp_path):
    data = np.ma.masked_less(noisy((20, 30)), 100)
    v = fp.imread(save(tmp_path, data, lossless=True).path).values
    np.testing.assert_array_equal(np.isnan(v), data.mask)
    ints = np.ma.array(np.arange(12, dtype=np.int32).reshape(3, 4), mask=[[1, 0, 0, 0]] + [[0] * 4] * 2)
    with pytest.raises(ValueError, match="nodata"):
        fp.fluximg(ints)
    v = fp.imread(save(tmp_path, ints, name="ints", nodata=-7).path).values
    assert v[0, 0] == -7 and v[2, 3] == 11


def test_dtypes_outside_the_stored_set(tmp_path):
    small = np.arange(20, dtype=np.int64).reshape(4, 5)
    m = man(save(tmp_path, small, name="a"))
    assert m["pixels"]["dtype"] == "uint8" and m["pixels"]["encoding"]["sourceDtype"] == "int64"
    big = (np.arange(20, dtype=np.int64) - 40_000).reshape(4, 5)
    assert man(save(tmp_path, big, name="b"))["pixels"]["dtype"] == "int32"
    np.testing.assert_array_equal(fp.imread(tmp_path / "b").values, big)
    with pytest.raises(ValueError, match=r"0\.\.5000000000"):
        save(tmp_path, np.array([[0, 5_000_000_000]], dtype=np.uint64), name="c")
    m = man(save(tmp_path, np.eye(4, dtype=bool), name="d"))
    assert m["pixels"]["dtype"] == "uint8" and m["pixels"]["encoding"]["sourceDtype"] == "bool"
    m = man(save(tmp_path, noisy((10, 12), dtype=np.float16), name="e", lossless=True))
    assert m["pixels"]["dtype"] == "float32" and m["pixels"]["encoding"]["sourceDtype"] == "float16"
    with pytest.raises(ValueError, match="integers, booleans or floats"):
        fp.fluximg(np.ones((3, 3), complex))
    with pytest.raises(ValueError, match="numeric"):
        fp.fluximg(np.array([["a", "b"]], dtype=object))


# ---------------------------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------------------------
def test_guard_passes_on_noisy_data(tmp_path):
    m = man(save(tmp_path, noisy((2, 30, 40)), ("c", "y", "x")))
    g = m["pixels"]["encoding"]["guard"]
    assert m["pixels"]["encoding"]["kind"] == "linear16"
    assert g["threshold"] == 0.1 and len(g["stepOverNeighbour"]) == 2 and max(g["stepOverNeighbour"]) < 0.1


def test_guard_refuses_a_constant_image(tmp_path, capsys):
    rep = save(tmp_path, np.full((20, 30), 3.25, np.float32))
    enc = man(rep)["pixels"]["encoding"]
    assert enc["kind"] == "exact" and enc["fallback"] == "guard" and enc["guard"]["stepOverNeighbour"] == [None]
    assert "refused" in rep.reason and "image (no variation)" in rep.reason and "constant" in rep.reason
    assert "why      stored exactly (float32); a 16-bit copy was refused" in capsys.readouterr().out


def test_guard_refuses_one_extreme_value(tmp_path, capsys):
    data = noisy((40, 40), amp=0.05, base=2.0)
    data[20, 20] = 2e5
    rep = save(tmp_path, data)
    enc = man(rep)["pixels"]["encoding"]
    assert enc["kind"] == "exact" and enc["fallback"] == "guard" and enc["guard"]["stepOverNeighbour"][0] > 0.1
    assert "extreme values" in rep.reason and "refused" in capsys.readouterr().out
    np.testing.assert_array_equal(fp.imread(rep.path).values, data)


def test_guard_refuses_piecewise_constant_and_any_failing_channel(tmp_path):
    steps = np.repeat(np.repeat(np.arange(16, dtype=np.float64).reshape(4, 4), 8, 0), 8, 1)
    assert man(save(tmp_path, steps, name="a"))["pixels"]["encoding"]["kind"] == "exact"
    two = np.stack([noisy((30, 30), dtype=np.float64), np.full((30, 30), 1.0)])
    enc = man(save(tmp_path, two, ("c", "y", "x"), name="b"))["pixels"]["encoding"]
    assert enc["kind"] == "exact" and enc["guard"]["stepOverNeighbour"][0] < 0.1
    assert enc["guard"]["stepOverNeighbour"][1] is None


def test_lossless_skips_the_guard(tmp_path):
    enc = man(save(tmp_path, noisy((20, 20)), lossless=True))["pixels"]["encoding"]
    assert enc == {"kind": "exact", "sourceDtype": "float32"}


# ---------------------------------------------------------------------------------------------
# tiling, overviews, shards, missing planes
# ---------------------------------------------------------------------------------------------
def _zarr_json(bundle, rel):
    with open(os.path.join(bundle, *rel.split("/"), "zarr.json"), encoding="utf-8") as f:
        return json.load(f)


def test_a_plane_up_to_1024_px_is_one_chunk(tmp_path):
    rep = save(tmp_path, np.zeros((2, 1024, 300), np.uint8) + 7, ("c", "y", "x"))
    m = man(rep)
    assert [lv["downsample"] for lv in m["pixels"]["levels"]] == [1]
    assert _zarr_json(rep.path, "pixels/0")["codecs"][0]["configuration"]["chunk_shape"] == [1, 1024, 300]


def test_large_planes_are_tiled_with_overviews(tmp_path):
    rng = np.random.default_rng(4)
    data = rng.integers(1, 3000, (2, 1100, 1300)).astype(np.uint16)
    data[1, :, :40] = 0
    rep = save(tmp_path, data, ("c", "y", "x"), nodata=0)
    m = man(rep)
    assert [(lv["downsample"], lv["shape"]) for lv in m["pixels"]["levels"]] == [(1, [2, 1100, 1300]), (4, [2, 275, 325])]
    z0 = _zarr_json(rep.path, "pixels/0")
    assert z0["codecs"][0]["configuration"]["chunk_shape"] == [1, 512, 512]
    outer = z0["chunk_grid"]["configuration"]["chunk_shape"]
    assert outer[1:] == [1536, 1536] and outer[0] in (1, 2)       # whole tiles, padded edge tiles
    img = fp.imread(rep.path)
    np.testing.assert_array_equal(img.values, data)
    over = fp.imread(rep.path, level=1)
    assert over.shape == [2, 275, 325] and over.downsample == 4 and over.level == 1
    vals = data.astype(float)
    vals[vals == 0] = np.nan
    for c in range(2):
        want = E.round_half_up(E.block_mean(vals[c], 4))
        want = np.where(np.isnan(E.block_mean(vals[c], 4)), 0, want)
        np.testing.assert_array_equal(over.values[c], want)


def test_overview_levels_stop_at_512_and_keep_physical_scale(tmp_path):
    data = noisy((2100, 600), dtype=np.float32)
    rep = save(tmp_path, data, pixel_size=0.5)
    m = man(rep)
    assert [lv["downsample"] for lv in m["pixels"]["levels"]] == [1, 4, 8]
    assert m["pixels"]["levels"][2]["shape"] == [263, 75]
    ov = fp.imread(rep.path, level=2)
    assert ov.axes["x"]["scale"] == 4.0 and fp.imread(rep.path).axes["x"]["scale"] == 0.5
    p = m["pixels"]["encoding"]["channels"][0]
    want = E.block_mean(data.astype(np.float64), 8)
    assert np.nanmax(np.abs(ov.values - want)) <= p["scale"] / 2 * (1 + 1e-9)
    with pytest.raises(ValueError, match="level must be 0..2"):
        fp.imread(rep.path, level=3)


def test_float_overview_is_missing_where_a_block_has_no_data(tmp_path):
    data = noisy((1200, 40), dtype=np.float64)
    data[:8] = np.nan
    ov = fp.imread(save(tmp_path, data, lossless=True).path, level=1).values
    assert np.isnan(ov[:2]).all() and np.isfinite(ov[2:]).all()


def test_shards_split_along_the_first_dimension(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "MAX_SHARD_BYTES", 8 * 1024)
    data = np.random.default_rng(5).integers(1, 256, (40, 32, 32)).astype(np.uint8)
    rep = save(tmp_path, data, ("t", "y", "x"))
    shards = sorted(os.listdir(os.path.join(rep.path, "pixels", "0", "c")), key=int)
    assert len(shards) > 3
    k = _zarr_json(rep.path, "pixels/0")["chunk_grid"]["configuration"]["chunk_shape"][0]
    assert len(shards) == -(-40 // k)
    np.testing.assert_array_equal(fp.imread(rep.path).values, data)


def test_planes_without_data_are_not_stored(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "MAX_SHARD_BYTES", 4 * 1024)
    data = noisy((8, 32, 32))
    data[2:6] = np.nan
    rep = save(tmp_path, data, ("t", "y", "x"), lossless=True)
    reader = Z.ArrayReader(os.path.join(rep.path, "pixels", "0"), _zstd.codec())
    assert sorted(c[0] for c, _ in reader.chunks()) == [0, 1, 6, 7]
    k = reader.outer[0]
    written = {int(s) for s in os.listdir(os.path.join(rep.path, "pixels", "0", "c"))}
    assert written == {i // k for i in (0, 1, 6, 7)}
    np.testing.assert_array_equal(fp.imread(rep.path).values, data)


@pytest.mark.parametrize("order", list(itertools.permutations(("t", "c", "y", "x"))))
def test_every_dimension_order_is_stored_canonically(tmp_path, order):
    shape = {"t": 3, "c": 2, "y": 5, "x": 7}
    data = np.arange(np.prod(list(shape.values())), dtype=np.int16).reshape([shape[d] for d in order])
    rep = save(tmp_path, data, order)
    img = fp.imread(rep.path)
    assert img.dims == ["t", "c", "y", "x"]
    np.testing.assert_array_equal(img.values, np.transpose(data, [order.index(d) for d in img.dims]))


def test_extra_dimensions_come_first(tmp_path):
    data = noisy((4, 3, 2, 10, 12)).astype(np.float32)
    rep = save(tmp_path, data, ("c", "x", "trial", "t", "y"))
    m = man(rep)
    assert m["pixels"]["dims"] == ["trial", "t", "c", "y", "x"] and m["pixels"]["ome"] is False
    group = _zarr_json(rep.path, "pixels")
    assert group == {"zarr_format": 3, "node_type": "group", "attributes": {}}
    assert set(m["profiles"]) == {"trial", "t"}


# ---------------------------------------------------------------------------------------------
# dims, axes, channels, world, meta, defaults
# ---------------------------------------------------------------------------------------------
def test_dims_are_required_and_checked():
    with pytest.raises(ValueError, match=r"dims="):
        fp.fluximg(np.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match="names 2 dimensions"):
        fp.fluximg(np.zeros((2, 3, 4)), dims=("y", "x"))
    with pytest.raises(ValueError, match="repeats"):
        fp.fluximg(np.zeros((2, 3, 4)), dims=("y", "y", "x"))
    with pytest.raises(ValueError, match="lower-case"):
        fp.fluximg(np.zeros((2, 3, 4)), dims=("Z", "y", "x"))
    with pytest.raises(ValueError, match="must include"):
        fp.fluximg(np.zeros((2, 3, 4)), dims=("c", "z", "x"))
    with pytest.raises(ValueError, match="empty"):
        fp.fluximg(np.zeros((0, 3)))
    assert fp.fluximg(np.zeros((3, 4))).dims == ["y", "x"]


def test_axes_from_pixel_size_coords_dt_and_t0(tmp_path):
    data = noisy((3, 4, 10, 12))
    z = [-61.5, -60.5, -59.5, -58.5]
    m = man(save(tmp_path, data, ("t", "z", "y", "x"), pixel_size=(0.5, 0.25), unit="µm", coords={"z": z},
                 dt=1 / 80, t0=2.5))
    assert m["axes"]["y"] == {"unit": "µm", "scale": 0.5, "offset": 0.0}
    assert m["axes"]["x"] == {"unit": "µm", "scale": 0.25, "offset": 0.0}
    assert m["axes"]["z"] == {"unit": "µm", "scale": 1.0, "offset": -61.5}
    assert m["axes"]["t"] == {"unit": "s", "scale": 0.0125, "offset": 2.5}
    m = man(save(tmp_path, data, ("t", "z", "y", "x"), name="b", coords={"z": [0, 1, 3, 7], "t": [0, 1, 2]}))
    assert m["axes"]["z"] == {"unit": "µm", "coords": [0.0, 1.0, 3.0, 7.0]}
    assert m["axes"]["t"] == {"unit": "s", "scale": 1.0, "offset": 0.0}
    assert m["axes"]["x"] == {"unit": None, "scale": 1.0, "offset": 0.0}     # no pixel size: pixel units


def test_categorical_and_datetime_coordinates(tmp_path):
    data = noisy((3, 2, 8, 9))
    times = np.array(["2026-11-19T10:00:00", "2026-11-19T10:00:00.5"], dtype="datetime64[ms]")
    m = man(save(tmp_path, data, ("trial", "t", "y", "x"), coords={"trial": ["a", "b", "c"], "t": times}))
    assert m["axes"]["trial"] == {"unit": None, "scale": 1.0, "offset": 0.0, "labels": ["a", "b", "c"]}
    assert m["axes"]["t"] == {"unit": "s", "scale": 0.5, "offset": 0.0}
    assert m["provenance"]["meta"] == {"t_start": "2026-11-19T10:00:00.000"}
    deltas = np.array([10, 20], dtype="timedelta64[ms]")
    m = man(save(tmp_path, data, ("trial", "t", "y", "x"), name="b", coords={"t": deltas}, meta={"t_start": "mine"}))
    assert m["axes"]["t"] == {"unit": "s", "scale": 0.01, "offset": 0.01}


def test_axis_errors():
    with pytest.raises(ValueError, match="evenly spaced"):
        fp.fluximg(np.zeros((3, 4)), coords={"x": [0, 1, 3, 4]})
    with pytest.raises(ValueError, match="not a dimension"):
        fp.fluximg(np.zeros((2, 3, 4)), dims=("c", "y", "x"), coords={"c": [0, 1]})
    with pytest.raises(ValueError, match="3 values for 4"):
        fp.fluximg(np.zeros((4, 3, 4)), dims=("z", "y", "x"), coords={"z": [0, 1, 2]})
    with pytest.raises(ValueError, match="'t' dimension"):
        fp.fluximg(np.zeros((3, 4)), dt=0.1)
    with pytest.raises(ValueError, match="positive"):
        fp.fluximg(np.zeros((3, 4)), pixel_size=-1)
    with pytest.raises(ValueError, match="one number or"):
        fp.fluximg(np.zeros((3, 4)), pixel_size=(1, 2, 3))
    with pytest.raises(ValueError, match="finite"):
        fp.fluximg(np.zeros((2, 3, 4)), dims=("z", "y", "x"), coords={"z": [0, np.nan]})


def test_channels_names_colours_luts_and_ranges(tmp_path):
    data = noisy((4, 10, 12))
    m = man(save(tmp_path, data, ("c", "y", "x"), channels=[
        "dapi", {"name": "gfp", "lut": "#2bff00"}, {"name": "act", "lut": "cmr.ember", "range": (90, 110)},
        {"color": "magenta"}]))
    ch = m["channels"]
    assert [c["name"] for c in ch] == ["dapi", "gfp", "act", "channel 4"]
    assert (ch[0]["color"], ch[0]["lut"]) == ("#00ff00", None)
    assert (ch[1]["color"], ch[1]["lut"]) == ("#2bff00", None)
    assert (ch[2]["color"], ch[2]["lut"]) == (None, "cmasher.ember") and ch[2]["range"] == [90.0, 110.0]
    assert ch[3]["color"] == "#ff00ff"
    for c in ch:
        assert c["range"][1] > c["range"][0] and len(c["stats"]["hist"]["counts"]) == 256
    single = man(save(tmp_path, data[0], name="one"))["channels"]
    assert single == [{**single[0], "name": "image", "color": None, "lut": "gray"}]
    assert single[0]["range"] == E.default_range(single[0]["stats"])
    many = man(save(tmp_path, noisy((7, 4, 4)), ("c", "y", "x"), name="many"))["channels"]
    assert [c["color"] for c in many] == list(W.DEFAULT_COLOURS) + [W.DEFAULT_COLOURS[0]]


def test_channel_errors():
    with pytest.raises(ValueError, match="2 entries for 3"):
        fp.fluximg(np.zeros((3, 4, 4)), dims=("c", "y", "x"), channels=["a", "b"])
    with pytest.raises(ValueError, match="not both"):
        fp.fluximg(np.zeros((4, 4)), channels=[{"color": "red", "lut": "gray"}])
    with pytest.raises(ValueError, match="unknown keys"):
        fp.fluximg(np.zeros((4, 4)), channels=[{"colour_map": "gray"}])
    with pytest.raises(ValueError, match="neither a colour nor a colour-map"):
        fp.fluximg(np.zeros((4, 4)), channels=[{"lut": "not-a-thing"}])
    with pytest.raises(ValueError, match="white > black"):
        fp.fluximg(np.zeros((4, 4)), channels=[{"range": (5, 5)}])


def test_photometric_rgb(tmp_path):
    photo = np.random.default_rng(6).integers(0, 256, (20, 30, 3)).astype(np.uint8)
    rep = save(tmp_path, photo, photometric="rgb")
    m = man(rep)
    assert m["photometric"] == "rgb" and m["pixels"]["dims"] == ["c", "y", "x"]
    assert [(c["name"], c["color"], c["range"]) for c in m["channels"]] == [
        ("red", "#ff0000", [0.0, 255.0]), ("green", "#00ff00", [0.0, 255.0]), ("blue", "#0000ff", [0.0, 255.0])]
    np.testing.assert_array_equal(fp.imread(rep.path).values, np.moveaxis(photo, -1, 0))
    with pytest.raises(ValueError, match="three channels"):
        fp.fluximg(np.zeros((2, 4, 4)), dims=("c", "y", "x"), photometric="rgb")
    with pytest.raises(ValueError, match="photometric"):
        fp.fluximg(np.zeros((4, 4)), photometric="cmyk")


def test_world_meta_name_and_default_index(tmp_path):
    mat = np.array([[0.25, 0, 0, -160], [0, 0.25, 0, -100], [0, 0, 1, -61.5], [0, 0, 0, 1]])
    m = man(save(tmp_path, noisy((5, 6, 8)), ("z", "y", "x"), transform=mat, unit="µm", pixel_size=0.25,
                 meta={"subject": "S1", "dmd": np.int64(1), "gain": np.float32(0.5)}, name="mean image",
                 default_index={"z": 3}))
    assert m["world"] == {"matrix": mat.tolist(), "unit": "µm"}
    assert m["provenance"]["meta"] == {"subject": "S1", "dmd": 1, "gain": 0.5}
    assert m["provenance"]["writer"].startswith("fluxplot ") and m["provenance"]["created"].endswith("Z")
    assert m["name"] == "mean image" and m["defaults"] == {"index": {"z": 3}}
    m = man(save(tmp_path, noisy((6, 8)), name="w", transform={"matrix": np.eye(4), "note": "affine"}))
    assert m["world"] == {"matrix": np.eye(4).tolist(), "note": "affine"} and m["name"] == "w"
    with pytest.raises(ValueError, match="4 x 4"):
        fp.fluximg(np.zeros((4, 4)), transform=np.eye(3))
    with pytest.raises(ValueError, match="meta must be a dict"):
        fp.fluximg(np.zeros((4, 4)), meta=[1])
    with pytest.raises(ValueError, match="it can name z"):
        fp.fluximg(np.zeros((2, 4, 4)), dims=("z", "y", "x"), default_index={"t": 0})
    with pytest.raises(ValueError, match=r"integer 0\.\.1"):
        fp.fluximg(np.zeros((2, 4, 4)), dims=("z", "y", "x"), default_index={"z": 2})
    with pytest.raises(ValueError, match="dict"):
        fp.fluximg(np.zeros((2, 4, 4)), dims=("z", "y", "x"), default_index=1)


def test_profiles_are_per_channel_means_along_each_extra_dimension(tmp_path):
    data = np.ones((3, 2, 4, 5), np.float32)
    data *= np.arange(1, 4, dtype=np.float32)[:, None, None, None]
    data[:, 1] *= 10
    data[2] = np.nan
    m = man(save(tmp_path, data, ("t", "c", "y", "x"), lossless=True))
    assert m["profiles"] == {"t": {"mean": [[1.0, 2.0, None], [10.0, 20.0, None]]}}


def test_ome_metadata(tmp_path):
    data = noisy((2, 3, 10, 12))
    rep = save(tmp_path, data, ("c", "z", "y", "x"), pixel_size=0.25, coords={"z": [0, 2, 4]},
               channels=[{"name": "a", "color": "#ff2bd6"}, {"name": "b", "lut": "gray"}])
    m = man(rep)
    assert m["pixels"]["ome"] is True
    ome = _zarr_json(rep.path, "pixels")["attributes"]["ome"]
    assert ome["version"] == "0.5"
    ms = ome["multiscales"][0]
    assert ms["axes"] == [{"name": "c", "type": "channel"}, {"name": "z", "type": "space", "unit": "micrometer"},
                          {"name": "y", "type": "space", "unit": "micrometer"},
                          {"name": "x", "type": "space", "unit": "micrometer"}]
    assert ms["datasets"] == [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 2.0, 0.25, 0.25]}]}]
    om = ome["omero"]["channels"]
    assert [c["color"] for c in om] == ["FF2BD6", "FFFFFF"] and om[0]["label"] == "a" and om[0]["active"] is True
    p = m["pixels"]["encoding"]["channels"][0]
    assert om[0]["window"]["start"] == pytest.approx(1 + (m["channels"][0]["range"][0] - p["offset"]) / p["scale"])
    assert om[0]["window"]["min"] == pytest.approx(1.0)
    no_c = _zarr_json(save(tmp_path, noisy((4, 4)), name="g").path, "pixels")["attributes"]["ome"]
    assert "omero" not in no_c


# ---------------------------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------------------------
def test_add_image(tmp_path):
    base = noisy((2, 20, 30))
    act = noisy((20, 30), seed=3, base=0.0, amp=0.02).astype(np.float64)
    img = fp.fluximg(base, dims=("c", "y", "x"))
    assert img.add_image("activity", act, lut="magenta", opacity="value") is img
    img.add_image("speed", act, lut="viridis", range=(0, 1), opacity=0.5, visible=True, lossless=True, name="Speed")
    img.add_image("count", np.arange(600, dtype=np.uint16).reshape(20, 30))
    rep = img.save(tmp_path / "a")
    layers = {L["id"]: L for L in man(rep)["layers"]}
    a = layers["activity"]
    assert (a["kind"], a["color"], a["lut"], a["opacity"], a["visible"], a["name"]) == (
        "intensity", "#ff00ff", None, "value", False, "activity")
    assert a["encoding"]["kind"] == "linear16" and a["dtype"] == "uint16"
    assert a["range"] == [a["stats"]["percentiles"]["50"], a["stats"]["percentiles"]["99.8"]]
    s = layers["speed"]
    assert (s["color"], s["lut"], s["range"], s["opacity"], s["visible"], s["name"]) == (
        None, "viridis", [0.0, 1.0], 0.5, True, "Speed")
    assert s["encoding"] == {"kind": "exact", "sourceDtype": "float64"} and s["dtype"] == "float64"
    assert layers["count"]["encoding"]["kind"] == "exact" and layers["count"]["color"] == "#ff00ff"
    back = fp.imread(rep.path)
    np.testing.assert_array_equal(back.layer("speed"), act)
    assert np.nanmax(np.abs(back.layer("activity") - act)) <= a["encoding"]["channels"][0]["scale"] / 2 * (1 + 1e-9)
    np.testing.assert_array_equal(fp.imread(rep.path, layer="count"), np.arange(600).reshape(20, 30))
    assert back.layers == ["activity", "speed", "count"]


def test_add_image_errors():
    img = fp.fluximg(np.zeros((20, 30)))
    with pytest.raises(ValueError, match="looks transposed"):
        img.add_image("a", np.zeros((30, 20)))
    with pytest.raises(ValueError, match=r"\(20, 30\)"):
        img.add_image("a", np.zeros((10, 10)))
    with pytest.raises(ValueError, match="2-D"):
        img.add_image("a", np.zeros((2, 20, 30)))
    with pytest.raises(ValueError, match="opacity"):
        img.add_image("a", np.zeros((20, 30)), opacity=2)
    with pytest.raises(ValueError, match="lower-case"):
        img.add_image("Activity", np.zeros((20, 30)))
    img.add_image("a", np.zeros((20, 30)))
    with pytest.raises(ValueError, match="already a layer 'a'"):
        img.add_image("a", np.zeros((20, 30)))


def _label_map():
    lab = np.full((30, 40), -1, dtype=np.int64)
    lab[2:6, 3:8] = 4
    lab[10:16, 10:16] = 7
    lab[12:14, 12:14] = -1                                     # a hole
    lab[20:23, 30:33] = 9
    lab[23:26, 33:36] = 9                                      # touching at a corner
    return lab


def test_add_labels_with_a_table_keyed_by_an_integer_column(tmp_path):
    pd = pytest.importorskip("pandas")
    lab = _label_map()
    weights = np.where(lab > 0, 0.5, np.nan)
    sources = pd.DataFrame({"source_id": [4, 7, 9, 11], "snr": [1.5, 2.5, np.nan, 4.0]})
    img = fp.fluximg(noisy((2, 30, 40)), dims=("c", "y", "x"), pixel_size=0.25)
    img.add_labels("footprints", lab, background=-1, weights=weights, table=sources, key="source_id", name="Footprints")
    rep = img.save(tmp_path / "a")
    m = man(rep)
    assert [L["id"] for L in m["layers"]] == ["footprints_weights", "footprints"]
    w, L = m["layers"]
    assert w["kind"] == "intensity" and w["encoding"]["kind"] == "exact" and w["dtype"] == "float64"
    assert w["visible"] is False and w["color"] == L["color"] == "#ffd400"
    assert (L["kind"], L["dtype"], L["key"], L["count"], L["weights"], L["visible"], L["name"]) == (
        "labels", "uint16", "source_id", 3, "footprints_weights", True, "Footprints")
    assert L["outlines"] == "layers/footprints.svg" and L["table"] == "layers/footprints.table.json"
    table = json.load(open(os.path.join(rep.path, L["table"])))
    assert table["key"] == "source_id"
    assert table["columns"] == {"source_id": [4, 7, 9, 11], "snr": [1.5, 2.5, None, 4.0], "label": [4, 7, 9, 11]}
    svg = open(os.path.join(rep.path, L["outlines"])).read()
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 30"><g id="footprints"')
    assert '<path id="footprints/7" data-key="7" data-label="7" d="M10,10 L16,10 L16,16 L10,16 Z M12,12 L12,14 L14,14 L14,12 Z"/>' in svg
    assert 'data-label="9" d="M30,20 L33,20 L33,23 L30,23 Z M33,23 L36,23 L36,26 L33,26 Z"' in svg
    back = fp.imread(rep.path)
    np.testing.assert_array_equal(back.layer("footprints"), np.where(lab < 0, 0, lab))
    np.testing.assert_array_equal(back.layer("footprints_weights"), weights)
    assert back.table("footprints") == table["columns"]


def test_add_labels_without_a_table_and_with_big_labels(tmp_path):
    lab = np.zeros((10, 12), np.int32)
    lab[1:3, 1:4] = 70_000
    lab[5, 5] = 2
    rep = fp.fluximg(noisy((10, 12))).add_labels("cells", lab).save(tmp_path / "a")
    L = man(rep)["layers"][0]
    assert L["dtype"] == "uint32" and L["key"] == "label" and L["count"] == 2
    t = json.load(open(os.path.join(rep.path, L["table"])))
    assert t == {"key": "label", "columns": {"label": [2, 70000], "pixels": [1, 6], "cx": [5.5, 2.5], "cy": [5.5, 2.0]}}
    svg = open(os.path.join(rep.path, L["outlines"])).read()
    assert 'id="cells/70000" data-key="70000" data-label="70000" d="M1,1 L4,1 L4,3 L1,3 Z"' in svg


def test_add_labels_with_a_label_column_and_string_keys(tmp_path):
    lab = np.zeros((6, 6), np.uint8)
    lab[0:2, 0:2] = 1
    lab[3:5, 3:5] = 2
    rep = fp.fluximg(noisy((6, 6))).add_labels(
        "rois", lab, table={"label": [1, 2], "name": ['a"<b>&', "c"]}, key="name").save(tmp_path / "a")
    svg = open(os.path.join(rep.path, "layers", "rois.svg")).read()
    assert 'data-key="a&quot;&lt;b&gt;&amp;"' in svg and 'id="rois/a&quot;&lt;b&gt;&amp;"' in svg


def test_add_labels_errors():
    img = fp.fluximg(np.zeros((6, 6)))
    lab = np.zeros((6, 6), np.int16)
    lab[0, 0] = 3
    lab[1, 1] = -2
    with pytest.raises(ValueError, match="positive label; found -2"):
        img.add_labels("a", lab)
    lab[1, 1] = 5
    with pytest.raises(ValueError, match="no table row, e.g. \\[5\\]"):
        img.add_labels("a", lab, table={"label": [3]})
    with pytest.raises(ValueError, match="repeats a label"):
        img.add_labels("a", lab, table={"label": [3, 5, 5]})
    with pytest.raises(ValueError, match="integer 'label' column"):
        img.add_labels("a", lab, table={"name": ["x", "y"]}, key="name")
    with pytest.raises(ValueError, match="'label' column must hold integers"):
        img.add_labels("a", lab, table={"label": ["x", "y"]})
    with pytest.raises(ValueError, match="not a column"):
        img.add_labels("a", lab, table={"label": [3, 5]}, key="nope")
    with pytest.raises(ValueError, match="no table= was given"):
        img.add_labels("a", lab, key="source_id")
    with pytest.raises(ValueError, match="share the same"):
        img.add_labels("a", lab, table={"label": [3, 5], "k": ["x", "x"]}, key="k")
    with pytest.raises(ValueError, match="empty 'k'"):
        img.add_labels("a", lab, table={"label": [3, 5], "k": ["x", None]}, key="k")
    with pytest.raises(ValueError, match="must be integers"):
        img.add_labels("a", lab + 0.5)
    img.add_image("a_weights", np.zeros((6, 6)))
    with pytest.raises(ValueError, match="already a layer 'a_weights'"):
        img.add_labels("a", lab, weights=np.zeros((6, 6)))


def test_add_shapes_lines_from_dicts_in_row_col(tmp_path):
    lines = [{"dendrite_id": "d1", "vertices": np.array([[14.0, 14.0], [22.0, 30.0]]), "length_um": 4.5},
             {"dendrite_id": "d2", "vertices": [[31, 50], [36, 56], [40, 60]], "length_um": 2.0}]
    rep = fp.fluximg(noisy((60, 90))).add_shapes("prox_lines", lines=lines, coords="row_col",
                                                 key="dendrite_id").save(tmp_path / "a")
    L = man(rep)["layers"][0]
    assert (L["kind"], L["path"], L["table"], L["key"], L["count"], L["color"]) == (
        "vector", "layers/prox_lines.svg", "layers/prox_lines.table.json", "dendrite_id", 2, "#00e5ff")
    svg = open(os.path.join(rep.path, L["path"])).read()
    assert '<polyline id="prox_lines/d1" data-key="d1" points="14.5,14.5 30.5,22.5"/>' in svg
    assert '<polyline id="prox_lines/d2" data-key="d2" points="50.5,31.5 56.5,36.5 60.5,40.5"/>' in svg
    assert 'fill="none"' in svg
    t = json.load(open(os.path.join(rep.path, L["table"])))
    assert t == {"key": "dendrite_id", "columns": {"dendrite_id": ["d1", "d2"], "length_um": [4.5, 2.0]}}
    # the key is found when exactly one field holds strings
    rep = fp.fluximg(noisy((60, 90))).add_shapes("l", lines=lines).save(tmp_path / "b")
    assert man(rep)["layers"][0]["key"] == "dendrite_id"


def test_add_shapes_lines_and_polygons_as_arrays(tmp_path):
    img = fp.fluximg(noisy((20, 20)))
    img.add_shapes("paths", lines=[np.array([[1, 2], [3, 4]]), np.array([[5.25, 6], [7, 8.125]])])
    img.add_shapes("cells", polygons={"c1": [[1, 1], [5, 1], [5, 5]], 7: [[10, 10], [12, 10], [12, 12]]},
                   color="red", visible=False, name="Cells")
    rep = img.save(tmp_path / "a")
    paths, cells = man(rep)["layers"]
    assert paths["table"] is None and paths["key"] is None and paths["count"] == 2
    svg = open(os.path.join(rep.path, "layers", "paths.svg")).read()
    assert 'data-key="0" points="1,2 3,4"' in svg and 'data-key="1" points="5.25,6 7,8.125"' in svg
    assert (cells["color"], cells["visible"], cells["name"]) == ("#ff0000", False, "Cells")
    svg = open(os.path.join(rep.path, "layers", "cells.svg")).read()
    assert '<polygon id="cells/c1" data-key="c1" points="1,1 5,1 5,5"/>' in svg and 'data-key="7"' in svg


def test_add_shapes_circles_from_a_table(tmp_path):
    pd = pytest.importorskip("pandas")
    discs = pd.DataFrame({"disc_id": ["a", "b"], "dendrite_id": ["d1", "d1"], "disc_index": [0, 1],
                          "arc_um": [0.0, 1.5], "center_row": [15.0, 18.0], "center_col": [16.0, 22.0],
                          "n_pixels": [12, 13]})
    rep = fp.fluximg(noisy((40, 60)), pixel_size=0.25).add_shapes("discs", circles=discs, diameter=1.0,
                                                                   key="disc_id").save(tmp_path / "a")
    L = man(rep)["layers"][0]
    svg = open(os.path.join(rep.path, L["path"])).read()
    assert '<circle id="discs/a" data-key="a" cx="16.5" cy="15.5" r="2"/>' in svg
    assert '<circle id="discs/b" data-key="b" cx="22.5" cy="18.5" r="2"/>' in svg
    t = json.load(open(os.path.join(rep.path, L["table"])))
    assert t["key"] == "disc_id" and list(t["columns"]) == list(discs.columns)
    with pytest.raises(ValueError, match="string columns: disc_id, dendrite_id"):
        fp.fluximg(noisy((40, 60))).add_shapes("d", circles=discs, radius=1)


def test_add_shapes_circle_forms_and_sizes(tmp_path):
    img = fp.fluximg(noisy((40, 60)), pixel_size=(0.5, 0.25))
    img.add_shapes("e", circles=[{"cy": 3.0, "cx": 4.0, "id": "p"}], radius=1.0)         # anisotropic → ellipse
    img.add_shapes("rc", circles={"cy": [3.0], "cx": [4.0], "id": ["q"]}, radius=0.5, coords="row_col")
    img.add_shapes("arr", circles=np.array([[10.0, 20.0], [30.0, 5.0]]), diameter=[1.0, 2.0])
    img.add_shapes("col", circles={"row": [1, 2], "col": [3, 4], "name": ["u", "v"], "r_um": [0.5, 1.0]},
                   radius="r_um")
    rep = img.save(tmp_path / "a")
    read = lambda i: open(os.path.join(rep.path, "layers", f"{i}.svg")).read()  # noqa: E731
    assert '<ellipse id="e/p" data-key="p" cx="4" cy="3" rx="4" ry="2"/>' in read("e")
    assert 'cx="4.5" cy="3.5" rx="2" ry="1"' in read("rc")
    assert 'data-key="0" cx="10" cy="20" rx="2" ry="1"' in read("arr") and 'rx="4" ry="2"' in read("arr")
    assert 'data-key="u" cx="3.5" cy="1.5" rx="2" ry="1"' in read("col")
    rep = fp.fluximg(noisy((40, 60))).add_shapes("px", circles=np.array([[1, 2]]), radius=3).save(tmp_path / "b")
    assert 'cx="1" cy="2" r="3"' in open(os.path.join(rep.path, "layers", "px.svg")).read()


def test_add_shapes_with_an_explicit_table(tmp_path):
    img = fp.fluximg(noisy((20, 20)))
    img.add_shapes("p", lines={"a": [[0, 0], [1, 1]], "b": [[2, 2], [3, 3]]},
                   table={"name": ["a", "b"], "length": [1.4, 1.4]}, key="name")
    L = man(img.save(tmp_path / "a"))["layers"][0]
    assert L["key"] == "name" and L["table"] == "layers/p.table.json"
    with pytest.raises(ValueError, match="no row in the table"):
        fp.fluximg(noisy((20, 20))).add_shapes("p", lines={"zz": [[0, 0], [1, 1]]}, table={"name": ["a"]}, key="name")


def test_add_shapes_errors():
    img = fp.fluximg(np.zeros((20, 20)))
    with pytest.raises(ValueError, match="needs lines="):
        img.add_shapes("a")
    with pytest.raises(ValueError, match="coords must be"):
        img.add_shapes("a", lines=[[[0, 0], [1, 1]]], coords="ij")
    with pytest.raises(ValueError, match="at least 2 vertices"):
        img.add_shapes("a", lines=[[[0, 0]]])
    with pytest.raises(ValueError, match="at least 3 vertices"):
        img.add_shapes("a", polygons=[[[0, 0], [1, 1]]])
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        img.add_shapes("a", lines=[[0, 0, 1]])
    with pytest.raises(ValueError, match="non-finite"):
        img.add_shapes("a", lines=[[[0, 0], [np.nan, 1]]])
    with pytest.raises(ValueError, match="used twice"):
        img.add_shapes("a", lines={"k": [[0, 0], [1, 1]]}, polygons={"k": [[0, 0], [1, 1], [2, 0]]})
    with pytest.raises(ValueError, match="'vertices'"):
        img.add_shapes("a", lines=[{"id": "x"}])
    with pytest.raises(ValueError, match="key="):
        img.add_shapes("a", lines=[{"vertices": [[0, 0], [1, 1]], "a": "x", "b": "y"}])
    with pytest.raises(ValueError, match="carry no fields"):
        img.add_shapes("a", lines=[[[0, 0], [1, 1]]], key="id")
    with pytest.raises(ValueError, match="no centre columns"):
        img.add_shapes("a", circles={"u": [1], "v": [2], "k": ["x"]}, radius=1)
    with pytest.raises(ValueError, match="diameter= or radius="):
        img.add_shapes("a", circles=np.array([[1, 2]]))
    with pytest.raises(ValueError, match="not both"):
        img.add_shapes("a", circles=np.array([[1, 2]]), radius=1, diameter=2)
    with pytest.raises(ValueError, match="must be positive"):
        img.add_shapes("a", circles=np.array([[1, 2]]), radius=-1)
    with pytest.raises(ValueError, match="only apply to circles"):
        img.add_shapes("a", lines=[[[0, 0], [1, 1]]], radius=1)


def test_add_shapes_polars_table(tmp_path):
    pl = pytest.importorskip("polars")
    discs = pl.DataFrame({"disc_id": ["a", "b"], "center_row": [5.0, 6.0], "center_col": [7.0, 8.0], "n": [1, 2]})
    rep = fp.fluximg(noisy((20, 20))).add_shapes("discs", circles=discs, radius=1).save(tmp_path / "a")
    L = man(rep)["layers"][0]
    assert L["key"] == "disc_id" and L["count"] == 2


# ---------------------------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------------------------
def _scanned(shape=(40, 80)):
    data = np.full(shape, np.nan, np.float32)
    data[20:30, 50:70] = 1.0
    data[5:15, 5:25] = 2.0
    data[2:8, 30:40] = 3.0
    return data + noisy(shape, amp=0.5)


def test_auto_regions_name_valid_areas_left_to_right(tmp_path):
    m = man(save(tmp_path, _scanned()))
    assert m["regions"] == [
        {"id": "roi_1", "name": "ROI 1", "rect": {"x": 5, "y": 5, "width": 20, "height": 10}},
        {"id": "roi_2", "name": "ROI 2", "rect": {"x": 30, "y": 2, "width": 10, "height": 6}},
        {"id": "roi_3", "name": "ROI 3", "rect": {"x": 50, "y": 20, "width": 20, "height": 10}}]
    assert man(save(tmp_path, _scanned(), name="b", regions=None))["regions"] == []
    assert man(save(tmp_path, noisy((10, 10)), name="c"))["regions"] == []          # no missing data


def test_auto_regions_use_the_union_of_planes_and_give_up_past_64(tmp_path):
    stack = np.full((20, 30, 30), np.nan, np.float32)
    stack[0, 2:5, 2:5] = 1
    stack[19, 20:25, 20:25] = 2                                # only in the last plane
    stack += noisy((20, 30, 30), amp=0.2)
    m = man(save(tmp_path, stack, ("t", "y", "x"), lossless=True))
    assert [r["rect"] for r in m["regions"]] == [{"x": 2, "y": 2, "width": 3, "height": 3},
                                                 {"x": 20, "y": 20, "width": 5, "height": 5}]
    speckle = np.full((40, 40), np.nan)
    speckle[::3, ::3] = 1.0                                    # 196 separate specks
    assert man(save(tmp_path, speckle, name="s", lossless=True))["regions"] == []


def test_add_region_and_its_errors(tmp_path):
    img = fp.fluximg(_scanned())
    img.add_region("Cell body", (1, 2, 10, 5))
    img.add_region("Cell body", (0, 0, 4, 4))
    img.add_region("ROI 1", (0, 0, 80, 40), id="roi_1")
    m = man(img.save(tmp_path / "a"))
    assert [(r["id"], r["name"]) for r in m["regions"]] == [
        ("cell_body", "Cell body"), ("cell_body_2", "Cell body"), ("roi_1", "ROI 1"),
        ("roi_2", "ROI 1"), ("roi_3", "ROI 2"), ("roi_4", "ROI 3")]
    assert m["regions"][0]["rect"] == {"x": 1, "y": 2, "width": 10, "height": 5}
    with pytest.raises(ValueError, match="inside the 80 x 40 image"):
        img.add_region("big", (0, 0, 81, 10))
    with pytest.raises(ValueError, match="already a region"):
        img.add_region("again", (0, 0, 1, 1), id="roi_1")
    with pytest.raises(ValueError, match="lower-case"):
        img.add_region("x", (0, 0, 1, 1), id="R1")
    with pytest.raises(ValueError, match=r"\(x, y, width, height\)"):
        img.add_region("x", 5)
    with pytest.raises(ValueError, match="regions must be"):
        img.save(tmp_path / "b", regions="all")


# ---------------------------------------------------------------------------------------------
# borrowing layers
# ---------------------------------------------------------------------------------------------
def _mean_bundle(tmp_path):
    lab = np.zeros((20, 30), np.uint16)
    lab[2:5, 2:6] = 1
    img = fp.fluximg(noisy((2, 20, 30)), dims=("c", "y", "x"), pixel_size=0.25)
    img.add_image("activity", noisy((20, 30), seed=2), lut="magenta")
    img.add_labels("cells", lab, weights=np.where(lab > 0, 1.0, np.nan))
    img.add_shapes("lines", lines={"a": [[0, 0], [5, 5]]})
    return img.save(tmp_path / "plots" / "mean")


def test_layers_are_borrowed_by_copy(tmp_path):
    src = _mean_bundle(tmp_path)
    movie = noisy((2, 4, 30, 20))                              # given (c, t, x, y)
    rep = fp.imsave(tmp_path / "plots" / "trial", movie, dims=("c", "t", "x", "y"), pixel_size=0.25,
                    dt=1 / 80, layers=str(tmp_path / "plots" / "mean"))
    m = man(rep)
    assert [L["id"] for L in m["layers"]] == ["activity", "cells_weights", "cells", "lines"]
    assert all(L["from"] == "mean.fluximg" for L in m["layers"])
    original = {L["id"]: {k: v for k, v in L.items()} for L in man(src)["layers"]}
    for L in m["layers"]:
        assert {k: v for k, v in L.items() if k != "from"} == original[L["id"]]
    for rel in ("layers/cells.svg", "layers/cells.table.json", "layers/lines.svg", "layers/activity/zarr.json"):
        a, b = os.stat(os.path.join(src.path, rel)), os.stat(os.path.join(rep.path, rel))
        assert a.st_ino != b.st_ino and b.st_nlink == 1          # a copy, never a hard link
        assert open(os.path.join(src.path, rel), "rb").read() == open(os.path.join(rep.path, rel), "rb").read()
    back = fp.imread(rep.path)
    np.testing.assert_array_equal(back.layer("cells"), fp.imread(src.path).layer("cells"))
    assert "borrowed from mean.fluximg" in str(rep)


def test_borrowing_refuses_another_grid_or_a_clash(tmp_path):
    _mean_bundle(tmp_path)
    with pytest.raises(ValueError, match="pixel grid is 20 x 30"):
        fp.imsave(tmp_path / "x", noisy((30, 20)), layers=tmp_path / "plots" / "mean.fluximg")
    with pytest.raises(ValueError, match="axis y differs"):
        fp.imsave(tmp_path / "x", noisy((20, 30)), pixel_size=0.5, layers=tmp_path / "plots" / "mean")
    img = fp.fluximg(noisy((20, 30)), pixel_size=0.25).add_image("activity", noisy((20, 30)))
    with pytest.raises(ValueError, match="'activity' exists twice"):
        img.save(tmp_path / "x", layers=tmp_path / "plots" / "mean")
    with pytest.raises(FileNotFoundError):
        fp.imsave(tmp_path / "x", noisy((20, 30)), layers=tmp_path / "nothing")
    assert not os.path.exists(tmp_path / "x.fluximg")
    rep = fp.imsave(tmp_path / "y", noisy((20, 30)), layers=[tmp_path / "plots" / "mean"])   # no pixel size: grid only
    assert len(man(rep)["layers"]) == 4


# ---------------------------------------------------------------------------------------------
# saving: paths, replacement, preview, recipe, FLUXPLOT_ONLY, the report
# ---------------------------------------------------------------------------------------------
def test_suffix_and_safe_replacement(tmp_path):
    rep = save(tmp_path, noisy((10, 10)), name="a")
    assert rep.path.endswith("a.fluximg") and os.path.isfile(os.path.join(rep.path, "fluximg.json"))
    assert fp.imsave(tmp_path / "b.fluximg", noisy((10, 10))).path.endswith("b.fluximg")
    mode = stat.S_IMODE(os.stat(rep.path).st_mode)
    assert mode & 0o055 == 0o055 & ~_umask()                     # an ordinary directory, not mkdtemp's 0700
    rep2 = fp.fluximg(noisy((12, 14))).add_image("extra", noisy((12, 14))).save(tmp_path / "a")
    assert man(rep2)["pixels"]["shape"] == [12, 14] and [L["id"] for L in man(rep2)["layers"]] == ["extra"]
    leftovers = [f for f in os.listdir(tmp_path) if f.startswith(".")]
    assert leftovers == []
    os.makedirs(tmp_path / "c.fluximg")
    (tmp_path / "c.fluximg" / "notes.txt").write_text("mine")
    with pytest.raises(ValueError, match="not a fluximg bundle"):
        save(tmp_path, noisy((4, 4)), name="c")
    (tmp_path / "d.fluximg").write_text("a file")
    with pytest.raises(ValueError, match="not a fluximg bundle directory"):
        save(tmp_path, noisy((4, 4)), name="d")
    os.makedirs(tmp_path / "e.fluximg")                          # an empty directory may be filled
    save(tmp_path, noisy((4, 4)), name="e")


def _umask():
    m = os.umask(0)
    os.umask(m)
    return m


def test_a_failed_save_leaves_the_old_bundle(tmp_path, monkeypatch):
    rep = save(tmp_path, noisy((10, 10)), name="a")
    before = open(os.path.join(rep.path, "fluximg.json")).read()

    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(W, "_write_vector", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        fp.fluximg(noisy((12, 12))).add_shapes("s", lines=[[[0, 0], [1, 1]]]).save(tmp_path / "a")
    assert open(os.path.join(rep.path, "fluximg.json")).read() == before
    assert [f for f in os.listdir(tmp_path) if f.startswith(".")] == []


def test_the_manifest_is_written_last(tmp_path, monkeypatch):
    order = []
    real = W._write_text

    def spy(root, rel, text):
        order.append(rel)
        return real(root, rel, text)
    monkeypatch.setattr(W, "_write_text", spy)
    fp.fluximg(noisy((10, 10))).add_shapes("s", lines=[[[0, 0], [1, 1]]]).save(tmp_path / "a")
    assert order[-1] == "fluximg.json" and order.count("fluximg.json") == 1


def test_preview_png_shows_the_opening_plane(tmp_path):
    import matplotlib.image as mpimg
    stack = np.full((2, 5, 600, 700), 100, np.uint16)
    stack[:, 2] = 1000                                          # only the middle z plane is bright
    stack[:, :, :100] = 0
    rep = save(tmp_path, stack, ("c", "z", "y", "x"), nodata=0, regions=None)
    px = mpimg.imread(os.path.join(rep.path, "preview.png"))
    assert px.shape == (300, 350, 4)                            # at most 512 px: 2x block means
    assert px[..., 3][:50].max() == 0 and px[..., 3][60:].min() == 1   # no data is transparent
    assert px[60:, :, :3].max() > 0.9                           # the middle plane, not plane 0
    rep = save(tmp_path, stack, ("c", "z", "y", "x"), nodata=0, name="b", default_index={"z": 0})
    px = mpimg.imread(os.path.join(rep.path, "preview.png"))
    assert px[60:, :, :3].max() < 0.05                          # default_index decides


def test_recipe_beside_the_bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rep = save(tmp_path, noisy((8, 8)), name="none")
    assert rep.recipe is None and not os.path.exists(tmp_path / "none.recipe.json")
    assert "recipe" not in man(rep)["provenance"]
    rep = save(tmp_path, noisy((8, 8)), name="auto", recipe=True)
    rec = json.load(open(tmp_path / "auto.recipe.json"))
    assert rep.recipe == str(tmp_path / "auto.recipe.json")
    assert rec["outputs"] == {"fluximg": "auto.fluximg", "manifest": "auto.fluximg/fluximg.json"}
    assert rec["plot"] == "auto" and rec["spec"] == "fluxplot/recipe"
    assert man(rep)["provenance"]["recipe"] == "auto.recipe.json"
    rep = save(tmp_path, noisy((8, 8)), name="given", recipe={"script": "make.py", "params": {"k": 1}})
    rec = json.load(open(tmp_path / "given.recipe.json"))
    assert rec["script"] == {"path": "make.py"} and rec["params"] == {"k": 1} and rec["output"] == "given.fluximg"
    save(tmp_path, noisy((8, 8)), name="off", recipe=False)
    assert not os.path.exists(tmp_path / "off.recipe.json")


def test_fluxplot_only_skips(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FLUXPLOT_ONLY", "other*")
    rep = save(tmp_path, noisy((8, 8)), name="mine")
    assert rep.skipped and not os.path.exists(rep.path)
    assert "skipped 'mine' (FLUXPLOT_ONLY=other*)" in capsys.readouterr().err
    assert str(rep).startswith("fluximg: skipped")
    monkeypatch.setenv("FLUXPLOT_ONLY", "mi*")
    assert not save(tmp_path, noisy((8, 8)), name="mine").skipped


def test_zstd_level(tmp_path):
    rep = save(tmp_path, noisy((8, 8)), zstd_level=9)
    assert _zarr_json(rep.path, "pixels/0")["codecs"][0]["configuration"]["codecs"][1]["configuration"]["level"] == 9
    with pytest.raises(ValueError, match="zstd_level"):
        save(tmp_path, noisy((8, 8)), name="b", zstd_level=0)


def test_the_report(tmp_path, capsys):
    img = fp.fluximg(_scanned(), pixel_size=0.25, name="scan")
    img.add_image("act", noisy((40, 80)), lut="magenta")
    img.add_labels("cells", np.eye(40, 80, dtype=np.uint8))
    rep = img.save(tmp_path / "scan")
    out = capsys.readouterr().out
    assert out.strip() == str(rep)
    assert rep.path in out and "y=40 × x=80" in out and "float32 → stored linear16 (uint16)" in out
    assert "why      stored at 16 bits" in out and "lossless=True" in out
    assert "on disk" in out and "of the 12.8 kB of raw pixels" in out
    assert "act (intensity), cells (labels)" in out and "layer act: 16-bit linear16" in out
    assert "regions  ROI 1, ROI 2, ROI 3" in out
    assert (rep.encoding, rep.stored_dtype, rep.dtype, rep.raw_bytes) == ("linear16", "uint16", "float32", 40 * 80 * 4)
    assert rep.bytes == sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(rep.path) for f in fs)
    assert rep.layers == [("act", "intensity"), ("cells", "labels")] and rep.regions == ["ROI 1", "ROI 2", "ROI 3"]
    assert rep.manifest == man(rep) and rep.seconds > 0 and not rep.skipped
    assert repr(img).startswith("FluxImage(y=40 × x=80, float32, 2 layer(s)")


# ---------------------------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------------------------
def test_imread_object(tmp_path):
    rep = save(tmp_path, noisy((3, 2, 10, 12)), ("t", "c", "y", "x"), pixel_size=0.5, dt=0.1,
               channels=["a", "b"], name="movie")
    for path in (rep.path, tmp_path / "movie", os.path.join(rep.path, "fluximg.json")):
        img = fp.imread(path)
        assert img.dims == ["t", "c", "y", "x"] and img.shape == [3, 2, 10, 12] and img.channels == ["a", "b"]
    assert img.axes["x"] == {"unit": "µm", "scale": 0.5, "offset": 0.0} and img.manifest["name"] == "movie"
    assert img.layers == [] and img.nodata is None and img.level == 0 and img.downsample == 1
    assert "FluximgFile('movie': t=3 × c=2 × y=10 × x=12, float64" in repr(img)


def test_imread_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        fp.imread(tmp_path / "missing")
    os.makedirs(tmp_path / "empty.fluximg")
    with pytest.raises(FluximgFormatError, match="unfinished write"):
        fp.imread(tmp_path / "empty")
    rep = fp.fluximg(noisy((10, 10))).add_shapes("s", lines={"a": [[0, 0], [2, 2]]}).add_image(
        "i", noisy((10, 10))).save(tmp_path / "a")
    img = fp.imread(rep.path)
    with pytest.raises(KeyError, match="no layer 'zz'"):
        img.layer("zz")
    with pytest.raises(ValueError, match="vector layer and has no pixel array"):
        img.layer("s")
    with pytest.raises(ValueError, match="has no table"):
        img.table("i")
    with pytest.raises(ValueError, match="has no table"):
        img.table("s")
    m = man(rep)
    m["spec"] = "fluximg/2"
    open(os.path.join(rep.path, "fluximg.json"), "w").write(json.dumps(m))
    with pytest.raises(FluximgFormatError, match="fluximg/1 only"):
        fp.imread(rep.path)
    m["spec"] = "fluximg/1"
    m["layers"][1]["path"] = "../elsewhere"
    open(os.path.join(rep.path, "fluximg.json"), "w").write(json.dumps(m))
    with pytest.raises(FluximgFormatError, match="escapes the bundle"):
        fp.imread(rep.path).layer("i")
    del m["channels"]
    open(os.path.join(rep.path, "fluximg.json"), "w").write(json.dumps(m))
    with pytest.raises(FluximgFormatError, match="required"):
        fp.imread(rep.path)


def test_to_xarray(tmp_path):
    xr = pytest.importorskip("xarray")
    rep = save(tmp_path, noisy((3, 2, 4, 10, 12)), ("trial", "c", "z", "y", "x"), pixel_size=0.5,
               coords={"trial": ["a", "b", "c"], "z": [0, 1, 5, 6]}, channels=["g", "r"], lossless=True)
    da = fp.imread(rep.path).to_xarray()
    assert isinstance(da, xr.DataArray) and da.dims == ("trial", "c", "z", "y", "x")
    assert da.coords["trial"].values.tolist() == ["a", "b", "c"] and da.coords["c"].values.tolist() == ["g", "r"]
    assert da.coords["z"].values.tolist() == [0, 1, 5, 6] and da.coords["z"].attrs["units"] == "µm"
    assert da.coords["x"].values[:2].tolist() == [0.0, 0.5]
    assert da.name == "img" and da.attrs["encoding"] == "exact"


# ---------------------------------------------------------------------------------------------
# xarray (and anything shaped like it) as input
# ---------------------------------------------------------------------------------------------
class _Coord:
    def __init__(self, values, units=None):
        self.values = np.asarray(values)
        self.attrs = {} if units is None else {"units": units}


class FakeDataArray:
    """Just the duck type fp.fluximg reads: .dims, .values, .coords, .attrs."""

    def __init__(self, values, dims, coords, attrs=None):
        self.values, self.dims, self.coords, self.attrs = values, dims, coords, attrs or {}


def test_dataarray_like_input(tmp_path):
    vals = noisy((2, 5, 6, 8))
    da = FakeDataArray(vals, ("c", "t", "x", "y"), {
        "c": _Coord(["dapi", "gfp"]), "t": _Coord(np.arange(5) * 0.25, "s"),
        "x": _Coord(np.arange(6) * 0.2 + 1.0, "mm"), "y": _Coord(np.arange(8) * 0.2)}, {"units": "nm"})
    m = man(fp.fluximg(da).save(tmp_path / "a"))
    assert m["pixels"]["dims"] == ["t", "c", "y", "x"] and [c["name"] for c in m["channels"]] == ["dapi", "gfp"]
    assert m["axes"]["t"] == {"unit": "s", "scale": 0.25, "offset": 0.0}
    assert m["axes"]["x"] == {"unit": "mm", "scale": pytest.approx(0.2), "offset": 1.0}
    assert m["axes"]["y"]["unit"] == "nm"                       # no coordinate unit: the array's
    m = man(fp.fluximg(da, pixel_size=0.5, channels=["a", "b"]).save(tmp_path / "b"))     # explicit keywords win
    assert m["axes"]["x"] == {"unit": "µm", "scale": 0.5, "offset": 1.0} and m["channels"][0]["name"] == "a"
    m = man(fp.fluximg(da, dims=("c", "t", "y", "x")).save(tmp_path / "c"))                # dims rename in place
    assert m["pixels"]["shape"] == [5, 2, 6, 8] and m["axes"]["y"]["unit"] == "mm"


def test_real_xarray_input(tmp_path):
    xr = pytest.importorskip("xarray")
    da = xr.DataArray(noisy((2, 4, 6)), dims=("c", "y", "x"),
                      coords={"c": ["a", "b"], "y": ("y", np.arange(4) * 0.5, {"units": "µm"}),
                              "x": ("x", np.arange(6) * 0.5, {"units": "µm"})})
    rep = fp.imsave(tmp_path / "a", da)
    m = man(rep)
    assert [c["name"] for c in m["channels"]] == ["a", "b"] and m["axes"]["x"] == {"unit": "µm", "scale": 0.5, "offset": 0.0}
    back = fp.imread(rep.path).to_xarray()
    assert back.dims == da.dims and np.nanmax(np.abs(back.values - da.values)) < 0.01


def test_dataarray_layers_are_transposed_by_their_dims(tmp_path):
    lay = noisy((10, 12))
    da = FakeDataArray(lay.T, ("x", "y"), {})
    rep = fp.fluximg(noisy((10, 12))).add_image("t", da, lut="gray", lossless=True).save(tmp_path / "a")
    np.testing.assert_array_equal(fp.imread(rep.path).layer("t"), lay)


# ---------------------------------------------------------------------------------------------
# the owner-approved workflow (synthetic stand-ins of the same shapes and kinds)
# ---------------------------------------------------------------------------------------------
def approved_workflow(root):
    """The three calls of the brief, on synthetic data shaped like the owner's."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(7)
    H, W = 213, 924
    scanned = np.zeros((H, W), bool)
    scanned[20:60, 30:300] = scanned[100:180, 400:520] = scanned[30:90, 700:900] = True
    meanIM = np.where(scanned, 50 + 10 * rng.standard_normal((2, H, W)), np.nan)
    actIM = np.where(scanned, 0.02 * rng.random((H, W)), np.nan)
    syn_map = np.full((H, W), -1, dtype=np.int64)
    fp_vals = np.full((H, W), np.nan)
    for sid in range(1, 101):
        y, x = 22 + (sid % 9) * 4, 35 + sid * 2
        syn_map[y:y + 3, x:x + 2] = sid
        fp_vals[y:y + 3, x:x + 2] = rng.random()
    sources = pd.DataFrame({"source_id": np.arange(1, 101), "snr": rng.random(100)})
    prox_lines = [{"dendrite_id": f"d{i}", "vertices": np.array([[25.0 + i, 40.0], [40.0 + i, 120.0], [50.0, 200.0]])}
                  for i in range(5)]
    discs = pd.DataFrame({"disc_id": [f"disc{i}" for i in range(30)], "dendrite_id": [f"d{i % 5}" for i in range(30)],
                          "disc_index": np.arange(30), "arc_um": np.linspace(0, 30, 30),
                          "center_row": rng.uniform(25, 55, 30), "center_col": rng.uniform(40, 280, 30),
                          "n_pixels": np.full(30, 12)})
    dmd_pixel_to_sample = np.array([[0.25, 0, 0, -160.0], [0, 0.25, 0, -100.0], [0, 0, 1, -61.5], [0, 0, 0, 1]])
    plots = os.path.join(root, "plots")
    img = fp.fluximg(meanIM, dims=("c", "y", "x"), pixel_size=0.25, unit="µm",
                     channels=["channel 1", "iGluSnFR"], transform=dmd_pixel_to_sample,
                     meta=dict(subject="synthetic", exp=1, loc=2, acq=3, dmd=1))
    img.add_image("activity", actIM, lut="magenta", opacity="value")
    img.add_labels("footprints", syn_map, background=-1, weights=fp_vals, table=sources, key="source_id")
    img.add_shapes("prox_lines", lines=prox_lines, coords="row_col", key="dendrite_id")
    img.add_shapes("discs", circles=discs, diameter=1.0, key="disc_id")
    mean = img.save(os.path.join(plots, "dmd1_mean"))
    z_um = np.arange(21) * 1.5 - 61.5
    ref = np.where(rng.random((2, 21, 80, 128)) < 0.9, 100 * rng.random((2, 21, 80, 128)), -1).astype(np.float32)
    stack = fp.imsave(os.path.join(plots, "dmd1_ref_stack"), ref, dims=("c", "z", "y", "x"),
                      coords={"z": z_um}, pixel_size=0.25, nodata=-1)
    tstack = np.where(scanned.T[None, None], rng.standard_normal((2, 30, W, H)), np.nan).astype(np.float32)
    trial = fp.imsave(os.path.join(plots, "dmd1_trial1"), tstack, dims=("c", "t", "x", "y"), dt=1 / 80,
                      t0=3.25, pixel_size=0.25, layers=os.path.join(plots, "dmd1_mean"))
    return {"mean": mean, "stack": stack, "trial": trial, "meanIM": meanIM, "ref": ref, "tstack": tstack}


def test_the_approved_workflow(tmp_path):
    out = approved_workflow(str(tmp_path))
    m = man(out["mean"])
    assert m["pixels"]["dims"] == ["c", "y", "x"] and m["pixels"]["encoding"]["kind"] == "linear16"
    assert [L["id"] for L in m["layers"]] == ["activity", "footprints_weights", "footprints", "prox_lines", "discs"]
    by = {L["id"]: L for L in m["layers"]}
    assert by["footprints"]["count"] == 100 and by["footprints"]["key"] == "source_id"
    assert by["discs"]["count"] == 30 and by["prox_lines"]["count"] == 5
    assert len(m["regions"]) == 3 and m["world"]["matrix"][0][3] == -160.0
    assert [c["name"] for c in m["channels"]] == ["channel 1", "iGluSnFR"]
    s = man(out["stack"])
    assert s["pixels"]["dims"] == ["c", "z", "y", "x"] and s["axes"]["z"] == {"unit": "µm", "scale": 1.5, "offset": -61.5}
    stack = fp.imread(out["stack"].path).values
    np.testing.assert_array_equal(np.isnan(stack), out["ref"] == -1)
    t = man(out["trial"])
    assert t["pixels"]["dims"] == ["t", "c", "y", "x"] and t["pixels"]["shape"] == [30, 2, 213, 924]
    assert t["axes"]["t"] == {"unit": "s", "scale": 0.0125, "offset": 3.25}
    assert [L["from"] for L in t["layers"]] == ["dmd1_mean.fluximg"] * 5
    trial = fp.imread(out["trial"].path).values
    want = np.transpose(out["tstack"], (1, 0, 3, 2))
    np.testing.assert_array_equal(np.isnan(trial), np.isnan(want))


@pytest.mark.slow
def test_a_large_stack_is_written_plane_by_plane(tmp_path):
    import tracemalloc
    data = noisy((2, 400, 300, 213))                           # 204 MB, given (c, t, x, y)
    tracemalloc.start()
    try:
        fp.imsave(tmp_path / "big", data, dims=("c", "t", "x", "y"), regions=None)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # no second copy: what is held is the bounded statistics sample (~1M values and ~2M neighbour
    # differences per channel), a few planes and the chunks waiting for compression
    assert peak < 0.5 * data.nbytes
