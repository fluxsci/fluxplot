"""fluximg/1 building blocks: CRC-32C, the shard index, the Zarr profile, encodings and statistics,
exact label outlines, the PNG encoder, tables and the zstd choice. Synthetic data only."""
import json
import math
import struct

import numpy as np
import pytest

from fluxplot._fluximg import _encoding as E
from fluxplot._fluximg import _outlines as O
from fluxplot._fluximg import _png
from fluxplot._fluximg import _tables as T
from fluxplot._fluximg import _zarr as Z
from fluxplot._fluximg import _zstd


# ---- CRC-32C and the shard index -----------------------------------------------------------
def test_crc32c_matches_the_castagnoli_check_value():
    assert Z.crc32c(b"123456789") == 0xE3069283
    assert Z.crc32c(b"") == 0
    assert Z.crc32c(bytes(32)) == 0x8A9136AA        # RFC 3720 B.4: 32 bytes of zeros


def test_shard_index_round_trip_and_missing_marker():
    raw = Z.encode_shard_index([0, None, 10], [10, 0, 7])
    assert len(raw) == 3 * 16 + 4
    body = np.frombuffer(raw[:48], dtype="<u8").reshape(3, 2)
    assert body[1].tolist() == [2**64 - 1, 2**64 - 1]
    assert struct.unpack("<I", raw[48:])[0] == Z.crc32c(raw[:48])
    offsets, lengths = Z.decode_shard_index(raw, 3)
    assert offsets.tolist() == [0, -1, 10] and lengths.tolist() == [10, 0, 7]


def test_shard_index_refuses_bad_crc_and_length():
    raw = bytearray(Z.encode_shard_index([0], [5]))
    raw[0] ^= 1
    with pytest.raises(Z.FluximgFormatError, match="crc32c"):
        Z.decode_shard_index(bytes(raw), 1)
    Z.decode_shard_index(bytes(raw), 1, verify=False)          # interactive readers may skip the check
    with pytest.raises(Z.FluximgFormatError, match="expected 20 bytes"):
        Z.decode_shard_index(bytes(raw[:-1]), 1)


# ---- the Zarr profile ------------------------------------------------------------------------
def _meta(**over):
    m = Z.array_meta_json([4, 2, 30, 40], "uint16", [1, 1, 30, 40], [4, 2, 30, 40], 0, ["t", "c", "y", "x"], 5)
    m.update(over)
    return m


def test_array_meta_is_the_spec_profile():
    m = _meta()
    assert m["zarr_format"] == 3 and m["node_type"] == "array" and m["fill_value"] == 0
    sc = m["codecs"][0]
    assert sc["name"] == "sharding_indexed"
    assert sc["configuration"]["codecs"] == [{"name": "bytes", "configuration": {"endian": "little"}},
                                             {"name": "zstd", "configuration": {"level": 5, "checksum": False}}]
    assert sc["configuration"]["index_codecs"] == [{"name": "bytes", "configuration": {"endian": "little"}},
                                                   {"name": "crc32c"}]
    assert sc["configuration"]["index_location"] == "end"
    assert m["chunk_key_encoding"] == {"name": "default", "configuration": {"separator": "/"}}
    assert Z.array_meta_json([2, 2], "float32", [2, 2], [2, 2], math.nan, ["y", "x"], 3)["fill_value"] == "NaN"
    assert Z.array_meta_json([2, 2], "int16", [2, 2], [2, 2], -1, ["y", "x"], 3)["fill_value"] == -1
    parsed = Z.parse_array_meta(json.loads(json.dumps(m)))
    assert parsed["inner"] == [1, 1, 30, 40] and parsed["level"] == 5 and parsed["dims"] == ["t", "c", "y", "x"]


@pytest.mark.parametrize("patch, message", [
    (dict(zarr_format=2), "Zarr v3"),
    (dict(data_type="int64"), "unsupported data_type"),
    (dict(shape=[5]), "at least y and x"),
    (dict(chunk_grid={"name": "rectilinear", "configuration": {"chunk_shape": [4, 2, 30, 40]}}), "regular"),
    (dict(chunk_key_encoding={"name": "v2"}), "chunk_key_encoding"),
    (dict(codecs=[{"name": "bytes"}]), "sharding_indexed"),
    (dict(fill_value=None), "fill_value"),
    (dict(dimension_names=["t", "c", "y"]), "dimension_names"),
])
def test_parse_array_meta_refuses_anything_outside_the_profile(patch, message):
    with pytest.raises(Z.FluximgFormatError, match=message):
        Z.parse_array_meta(_meta(**patch))


def test_parse_array_meta_refuses_wrong_codecs():
    m = _meta()
    m["codecs"][0]["configuration"]["codecs"][1]["name"] = "gzip"
    with pytest.raises(Z.FluximgFormatError, match=r"\[bytes, zstd\]"):
        Z.parse_array_meta(m)
    m = _meta()
    m["codecs"][0]["configuration"]["chunk_shape"] = [1, 1, 7, 40]
    with pytest.raises(Z.FluximgFormatError, match="whole number"):
        Z.parse_array_meta(m)
    m = _meta()
    m["codecs"][0]["configuration"]["index_location"] = "start"
    with pytest.raises(Z.FluximgFormatError, match="index_location"):
        Z.parse_array_meta(m)


def test_array_writer_streams_shards_and_skips_fill(tmp_path):
    codec = _zstd.codec()
    w = Z.ArrayWriter(str(tmp_path), "a", [6, 4, 4], "float32", [1, 4, 4], math.nan, ["t", "y", "x"],
                      level=3, max_shard_bytes=200, codec=codec)
    rng = np.random.default_rng(0)
    planes = [rng.random((4, 4), dtype=np.float32) for _ in range(6)]
    planes[2][:] = np.nan                                     # an all-fill chunk is not stored
    planes[3][:] = np.nan
    for t, p in enumerate(planes):
        w.add((t, 0, 0), p)
    meta = w.finish()
    k = meta["chunk_grid"]["configuration"]["chunk_shape"][0]
    assert 1 <= k < 6                                         # the 200-byte cap split the shards along t
    reader = Z.ArrayReader(str(tmp_path / "a"), codec)
    out = reader.read()
    np.testing.assert_array_equal(out, np.stack(planes))
    stored = sorted(c[0] for c, _ in reader.chunks())
    assert stored == [0, 1, 4, 5]
    files = sorted(p.relative_to(tmp_path / "a").as_posix() for p in (tmp_path / "a" / "c").rglob("*") if p.is_file())
    assert all(f.startswith("c/") and f.endswith("/0/0") for f in files)


def test_array_writer_writes_no_shard_without_chunks(tmp_path):
    codec = _zstd.codec()
    w = Z.ArrayWriter(str(tmp_path), "e", [3, 2, 2], "uint8", [1, 2, 2], 0, ["t", "y", "x"],
                      level=1, max_shard_bytes=1 << 20, codec=codec)
    for t in range(3):
        w.add((t, 0, 0), np.zeros((2, 2), np.uint8))
    w.finish()
    assert not (tmp_path / "e" / "c").exists()
    assert Z.ArrayReader(str(tmp_path / "e"), codec).read().tolist() == np.zeros((3, 2, 2)).tolist()


def test_array_writer_with_a_thread_pool_matches_serial(tmp_path):
    import concurrent.futures
    codec = _zstd.codec()
    data = np.random.default_rng(1).integers(0, 1000, (20, 8, 8)).astype(np.uint16)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        w = Z.ArrayWriter(str(tmp_path), "p", data.shape, "uint16", [1, 8, 8], 0, ["t", "y", "x"],
                          level=5, max_shard_bytes=600, codec=codec, pool=pool, max_inflight=3)
        for t in range(20):
            w.add((t, 0, 0), data[t])
        w.finish()
    np.testing.assert_array_equal(Z.ArrayReader(str(tmp_path / "p"), codec).read(), data)


def test_array_writer_refuses_a_wrong_sized_chunk(tmp_path):
    w = Z.ArrayWriter(str(tmp_path), "w", [2, 2], "uint8", [2, 2], 0, ["y", "x"], level=1,
                      max_shard_bytes=1 << 20, codec=_zstd.codec())
    with pytest.raises(ValueError, match="expected 4"):
        w.add((0, 0), np.ones((3, 2), np.uint8))


def test_array_reader_refuses_damage(tmp_path):
    codec = _zstd.codec()
    w = Z.ArrayWriter(str(tmp_path), "d", [2, 3], "int16", [2, 3], 0, ["y", "x"], level=1,
                      max_shard_bytes=1 << 20, codec=codec)
    w.add((0, 0), np.arange(1, 7, dtype=np.int16).reshape(2, 3))
    w.finish()
    shard = tmp_path / "d" / "c" / "0" / "0"
    data = bytearray(shard.read_bytes())
    data[-1] ^= 0xFF                                           # the index CRC
    shard.write_bytes(bytes(data))
    with pytest.raises(Z.FluximgFormatError, match="crc32c"):
        Z.ArrayReader(str(tmp_path / "d"), codec).read()
    with pytest.raises(Z.FluximgFormatError, match="missing"):
        Z.ArrayReader(str(tmp_path / "nothing"), codec)


# ---- encodings, the guard, statistics -----------------------------------------------------------
def test_linear16_encode_decode_law():
    lo, hi = -2.0, 5.0
    p = E.linear16_params(lo, hi)
    assert p == {"scale": 7.0 / 65534, "offset": -2.0}
    v = np.array([lo, hi, 0.0, np.nan, np.inf, -np.inf, lo - 1, hi + 1])
    k = E.encode_linear16(v, p["scale"], p["offset"])
    assert k.dtype == np.uint16
    assert k[0] == 1 and k[1] == 65535 and k[3] == 0 and k[4] == 0 and k[5] == 0
    assert k[6] == 1 and k[7] == 65535                        # clamped
    back = E.decode_linear16(k, p["scale"], p["offset"])
    assert np.isnan(back[3])
    assert back[0] == lo
    assert abs(back[2]) <= p["scale"] / 2
    # value = offset + scale * (k - 1), in exactly that order
    assert back[1] == -2.0 + (7.0 / 65534) * (65535 - 1)
    assert E.linear16_params(3.0, 3.0) == {"scale": 1.0, "offset": 3.0}


def test_round_half_up_is_javascript_math_round():
    x = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.49999999999999994, 2.4, -2.6])
    assert E.round_half_up(x).tolist() == [1, 2, 3, 0, -1, -2, 0, 2, -3]


def test_guard_ratio_and_decision():
    assert E.guard_ratio(0.01, 1.0) == 0.01
    assert E.guard_ratio(0.01, 0.0) is None
    assert E.guard_ratio(0.01, math.nan) is None
    assert E.guard_passes([0.01, 0.05])
    assert not E.guard_passes([0.01, 0.1])
    assert not E.guard_passes([0.01, None])
    assert not E.guard_passes([])


def _acc(values2d, stride=1):
    a = E.ChannelAcc(1, stride)
    a.learn(values2d[np.isfinite(values2d)])
    a.learn_diffs(values2d)
    return a


def test_choose_encoding_paths():
    rng = np.random.default_rng(3)
    noisy = (100 + 10 * rng.standard_normal((50, 60))).astype(np.float32)
    enc, note = E.choose_encoding([_acc(noisy)], np.float32, lossless=False)
    assert enc["kind"] == "linear16" and enc["guard"]["threshold"] == 0.1 and enc["sourceDtype"] == "float32"
    assert "16 bits" in note and "lossless=True" in note
    enc, note = E.choose_encoding([_acc(noisy)], np.float32, lossless=True)
    assert enc == {"kind": "exact", "sourceDtype": "float32"} and "as asked" in note
    flat = np.full((10, 10), 4.0)
    enc, note = E.choose_encoding([_acc(flat)], np.float64, lossless=False, channel_names=["dapi"])
    assert enc["kind"] == "exact" and enc["fallback"] == "guard" and enc["guard"]["stepOverNeighbour"] == [None]
    assert "refused" in note and "dapi (no variation)" in note
    enc, _ = E.choose_encoding([_acc(noisy.astype(np.int32))], np.int32, lossless=False)
    assert enc == {"kind": "exact", "sourceDtype": "int32"}


def test_stats_are_exact_at_the_ends_and_histogram_sums():
    rng = np.random.default_rng(4)
    v = rng.standard_normal(10_000)
    a = E.ChannelAcc(1, 1)
    a.learn(v)
    st = E.finish_stats(a)
    assert st["min"] == v.min() and st["max"] == v.max() and st["count"] == v.size
    keys = list(E.PERCENTILE_KEYS)
    assert keys == ["0.1", "1", "50", "99", "99.8", "99.9"]
    seq = [st["min"]] + [st["percentiles"][k] for k in keys] + [st["max"]]
    assert seq == sorted(seq)
    assert st["percentiles"]["50"] == pytest.approx(np.percentile(v, 50))
    assert len(st["hist"]["counts"]) == 256
    assert st["hist"]["lo"] == st["percentiles"]["0.1"] and st["hist"]["hi"] == st["percentiles"]["99.9"]
    inside = ((v >= st["hist"]["lo"]) & (v <= st["hist"]["hi"])).sum()
    assert sum(st["hist"]["counts"]) == inside
    assert E.default_range(st) == [st["percentiles"]["1"], st["percentiles"]["99.8"]]
    assert E.layer_range(st) == [st["percentiles"]["50"], st["percentiles"]["99.8"]]


def test_stats_subsample_deterministically_and_handle_empty_and_flat():
    a = E.ChannelAcc(sample_stride=3)
    a.learn(np.arange(5.0))
    a.learn(np.arange(5.0, 10.0))
    assert a.sample().tolist() == [0, 3, 6, 9]                # every third valid value across planes
    empty = E.finish_stats(E.ChannelAcc())
    assert empty["count"] == 0 and empty["hist"]["counts"] == [0] * 256
    assert E.default_range(empty) == [0.0, 1.0]
    flat = E.ChannelAcc()
    flat.learn(np.full(10, 7.0))
    st = E.finish_stats(flat)
    assert st["hist"]["lo"] == 7.0 and st["hist"]["hi"] == 8.0
    assert E.default_range(st) == [7.0, 8.0]


def test_block_mean_ignores_missing_and_handles_edges():
    v = np.arange(25, dtype=float).reshape(5, 5)
    v[0, 0] = np.nan
    m = E.block_mean(v, 2)
    assert m.shape == (3, 3)
    assert m[0, 0] == np.mean([1, 5, 6])
    assert m[2, 2] == 24.0                                     # a 1 x 1 corner block
    assert m[0, 2] == np.mean([4, 9])
    v2 = np.full((4, 4), np.nan)
    assert np.isnan(E.block_mean(v2, 4)).all()


# ---- exact label outlines -----------------------------------------------------------------
def _reference_labels():
    """The reference fixture's footprint map: a rectangle, a ring, two squares touching at a corner,
    and a thin bent shape (paths below are the reference writer's output, character for character)."""
    lab = np.zeros((60, 90), np.uint16)
    lab[12:16, 14:19] = 1
    lab[20:26, 30:36] = 2
    lab[22:24, 32:34] = 0
    lab[30:33, 50:53] = 3
    lab[33:36, 53:56] = 3
    for y, x in [(40, 20), (40, 21), (41, 21), (42, 21), (42, 22)]:
        lab[y, x] = 7
    return lab


REFERENCE_PATHS = {
    1: "M14,12 L19,12 L19,16 L14,16 Z",
    2: "M30,20 L36,20 L36,26 L30,26 Z M32,22 L32,24 L34,24 L34,22 Z",
    3: "M50,30 L53,30 L53,33 L50,33 Z M53,33 L56,33 L56,36 L53,36 Z",
    7: "M20,40 L22,40 L22,42 L23,42 L23,43 L21,43 L21,41 L20,41 Z",
}


def test_outlines_reproduce_the_reference_paths():
    out = O.label_outlines(_reference_labels())
    assert [o.label for o in out] == [1, 2, 3, 7]
    assert {o.label: o.d for o in out} == REFERENCE_PATHS
    o2 = out[1]
    assert o2.pixels == 32 and o2.box == (30, 20, 6, 6)
    ys, xs = np.nonzero(_reference_labels() == 2)
    assert (o2.cx, o2.cy) == (int(xs.sum()) / 32 + 0.5, int(ys.sum()) / 32 + 0.5)


def _rasterize(d, shape):
    """Pixels whose centres the path encloses (nonzero winding; outer clockwise, holes counter-clockwise)."""
    from matplotlib.path import Path
    verts, codes = [], []
    for sub in filter(None, d.split("M")):
        pts = [tuple(map(float, p.strip().split(","))) for p in sub.replace("Z", "").split("L")]
        verts += pts + [pts[0]]
        codes += [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1) + [Path.CLOSEPOLY]
    path = Path(verts, codes)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    centres = np.column_stack([xx.ravel() + 0.5, yy.ravel() + 0.5])
    return path.contains_points(centres).reshape(shape)


@pytest.mark.parametrize("seed", range(6))
def test_outlines_enclose_exactly_the_labelled_pixels(seed):
    rng = np.random.default_rng(seed)
    lab = (rng.random((24, 30)) < 0.45).astype(np.uint32) * rng.integers(1, 5, (24, 30)).astype(np.uint32)
    out = O.label_outlines(lab)
    assert [o.label for o in out] == sorted(set(lab[lab > 0].tolist()))
    for o in out:
        mask = lab == o.label
        assert o.pixels == mask.sum()
        assert O.outline_area(o.d) == o.pixels                 # clockwise outers, counter-clockwise holes
        np.testing.assert_array_equal(_rasterize(o.d, lab.shape), mask)
        ys, xs = np.nonzero(mask)
        assert (o.cx, o.cy) == (xs.mean() + 0.5, ys.mean() + 0.5)


def test_outlines_of_an_empty_map_and_bad_input():
    assert O.label_outlines(np.zeros((3, 3), np.uint16)) == []
    with pytest.raises(ValueError, match="2-D"):
        O.label_outlines(np.zeros((2, 3, 3), np.uint16))


# ---- PNG ---------------------------------------------------------------------------------
@pytest.mark.parametrize("channels", [3, 4])
def test_png_round_trip(tmp_path, channels):
    import matplotlib.image as mpimg
    px = np.random.default_rng(5).integers(0, 256, (7, 11, channels)).astype(np.uint8)
    path = tmp_path / "a.png"
    path.write_bytes(_png.encode_png(px))
    back = np.rint(mpimg.imread(path) * 255).astype(np.uint8)
    np.testing.assert_array_equal(back, px)
    with pytest.raises(ValueError):
        _png.encode_png(np.zeros((4, 4), np.uint8))


# ---- tables --------------------------------------------------------------------------------
def test_columns_of_accepts_dicts_rows_and_dataframes():
    cols = T.columns_of({"a": np.array([1, 2]), "b": ["x", None], "c": [np.float32(1.5), np.nan]})
    assert cols == {"a": [1, 2], "b": ["x", None], "c": [1.5, None]}
    assert T.columns_of([{"a": 1}, {"a": 2, "b": True}]) == {"a": [1, 2], "b": [None, True]}
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"id": ["s1", "s2"], "n": [3, 4], "v": [0.5, np.nan]})
    assert T.columns_of(df) == {"id": ["s1", "s2"], "n": [3, 4], "v": [0.5, None]}
    with pytest.raises(ValueError, match="equal lengths"):
        T.columns_of({"a": [1], "b": [1, 2]})
    with pytest.raises(TypeError, match="DataFrame"):
        T.columns_of(42)


def test_columns_of_polars():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"disc_id": ["a", "b"], "center_row": [1.0, 2.0], "n_pixels": [3, None]})
    assert T.columns_of(df) == {"disc_id": ["a", "b"], "center_row": [1.0, 2.0], "n_pixels": [3, None]}


def test_json_values_and_key_text():
    assert T.json_value(np.int64(3)) == 3 and isinstance(T.json_value(np.int64(3)), int)
    assert T.json_value(np.bool_(True)) is True
    assert T.json_value(float("inf")) is None
    assert T.json_value(np.datetime64("2026-01-02")) == "2026-01-02"
    assert T.json_safe({"a": np.arange(2), "b": (np.float64(1.5),)}) == {"a": [0, 1], "b": [1.5]}
    assert [T.key_text(v) for v in (3, 3.0, 3.5, True, "s7")] == ["3", "3", "3.5", "true", "s7"]
    assert T.is_integer_column([1, 2.0]) and not T.is_integer_column([1, 2.5]) and not T.is_integer_column([True])
    assert not T.is_integer_column([]) and not T.is_integer_column([1, None])


# ---- zstd ----------------------------------------------------------------------------------
def test_zstd_round_trip_and_compact_results():
    c = _zstd.codec()
    raw = np.arange(10_000, dtype=np.uint16).tobytes()
    packed = c.compress(raw, 5)
    assert c.decompress(packed, len(raw)) == raw
    assert isinstance(packed, bytes)


def test_missing_zstd_says_how_to_install(monkeypatch):
    def nope():
        raise ImportError("no")
    monkeypatch.setattr(_zstd, "_cached", None)
    monkeypatch.setattr(_zstd, "_FACTORIES", (nope, nope))
    with pytest.raises(_zstd.MissingZstdError, match=r"fluxplot\[images\]"):
        _zstd.codec()
