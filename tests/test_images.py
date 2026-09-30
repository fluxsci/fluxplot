"""D1 — fp.image (channels, LUTs, display ranges, physical extent) and fp.scalebar."""
import json
import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402

import fluxplot as fp  # noqa: E402

RNG = np.random.default_rng(3)
H, W = 12, 16
CH0 = RNG.gamma(2.0, 40.0, (H, W))
CH1 = RNG.gamma(1.5, 30.0, (H, W))


def _save(fig, tmp_path, name="i.svg", **kw):
    res = fp.save(fig, str(tmp_path / name), recipe=dict(script="img.py"), **kw)
    return res, json.loads(open(res.manifest).read()), json.loads(open(res.recipe).read()), open(res.svg).read()


def test_extent_follows_the_pixel_size():
    fig, ax = plt.subplots()
    im = fp.image(ax, CH0, series="dapi", pixel_size=0.325, units="µm")
    assert im.extent == [0.0, W * 0.325, H * 0.325, 0.0]
    assert list(im.artist.get_extent()) == pytest.approx(im.extent)
    assert im.pixel_size == (0.325, 0.325) and im.shape == (H, W) and im.channels == ["ch0"]
    fig, ax = plt.subplots()
    im = fp.image(ax, CH0, series="dapi", pixel_size=(0.5, 0.25), origin="lower")
    assert im.extent == [0.0, W * 0.25, 0.0, H * 0.5]
    fig, ax = plt.subplots()
    im = fp.image(ax, CH0, series="dapi")  # pixel coordinates
    assert im.extent == [-0.5, W - 0.5, H - 0.5, -0.5] and im.pixel_size is None
    with pytest.raises(ValueError, match="positive"):
        fp.image(ax, CH0, series="x", pixel_size=0)


def test_two_channel_composite_equals_a_manual_rgb_composite():
    fig, ax = plt.subplots()
    im = fp.image(ax, np.stack([CH0, CH1]), series="cells", channels=["dapi", "gfp"], luts=["blue", "green"],
                  display_range=[(0, 200), (10, 100)])
    manual = np.zeros((H, W, 3))
    for arr, colour, (lo, hi) in ((CH0, "blue", (0, 200)), (CH1, "green", (10, 100))):
        cmap = mcolors.LinearSegmentedColormap.from_list("m", ["#000000", mcolors.to_hex(colour)], N=256)
        manual += ScalarMappable(mcolors.Normalize(lo, hi), cmap).to_rgba(arr)[..., :3]
    manual = np.clip(manual, 0, 1)
    assert np.allclose(im.rgb, manual)
    drawn = im.artist.get_array()
    assert drawn.shape == (H, W, 3) and np.allclose(np.asarray(drawn), manual)
    assert im.display_range == {"dapi": (0.0, 200.0), "gfp": (10.0, 100.0)}
    assert im.keys == {"dapi": "cells.dapi", "gfp": "cells.gfp"}
    assert im.luts["gfp"].name == "image.mono:#008000"  # matplotlib green
    # max compositing, (H, W, C) input and default names/LUTs
    fig, ax = plt.subplots()
    im2 = fp.image(ax, np.dstack([CH0, CH1]), series="cells", composite="max")
    assert im2.channels == ["ch0", "ch1"] and im2.luts["ch0"].name == "image.mono:#00ff00" and im2.luts["ch1"].name == "image.mono:#ff00ff"
    lo0, hi0 = np.percentile(CH0, [1, 99.8])
    assert im2.display_range["ch0"] == pytest.approx((lo0, hi0))
    layers = [ScalarMappable(mcolors.Normalize(*im2.display_range[c]), im2.luts[c]).to_rgba(a)[..., :3]
              for c, a in (("ch0", CH0), ("ch1", CH1))]
    assert np.allclose(im2.rgb, np.maximum(*layers))
    # a registered colormap name is a LUT too; a missing pixel is transparent (adds nothing)
    fig, ax = plt.subplots()
    holed = CH0.copy(); holed[0, 0] = np.nan
    im3 = fp.image(ax, holed, series="s", luts="magma")
    assert im3.luts["ch0"].name == "magma" and np.all(im3.rgb[0, 0] == 0)
    with pytest.raises(ValueError, match="neither a colormap name nor a colour"):
        fp.image(ax, CH0, series="bad", luts="not-a-thing")
    with pytest.raises(ValueError, match="channel axis"):
        fp.image(ax, np.zeros((5, 6, 7)), series="bad")
    with pytest.raises(ValueError, match="composite"):
        fp.image(ax, CH0, series="bad", composite="screen")


def test_manifest_and_recipe_carry_channels_scales_and_controls(tmp_path, monkeypatch):
    fig, ax = plt.subplots()
    fp.image(ax, np.stack([CH0, CH1]), series="cells", channels=["dapi", "gfp"], luts=["blue", "green"],
             display_range=[(0, 200), (10, 100)], pixel_size=0.5)
    res, man, rec, svg = _save(fig, tmp_path)
    (s,) = [s for s in man["series"] if s["id"] == "cells"]
    assert s["svg"]["image"] == "cells.image" and s["kind"] == "image" and man["plotType"] == "image"
    payload = s["image"]
    assert payload["shape"] == [H, W] and payload["pixelSize"] == [0.5, 0.5] and payload["units"] == "µm"
    assert payload["composite"] == "add" and payload["origin"] == "upper" and payload["extent"] == [0.0, W * 0.5, H * 0.5, 0.0]
    assert [c["name"] for c in payload["channels"]] == ["dapi", "gfp"]
    assert payload["channels"][1] == {"name": "gfp", "lut": "image.mono:#008000", "displayRange": [10.0, 100.0],
                                      "scale": "cells.gfp", "controlKey": "cells.gfp"}
    scales = {sc["id"]: sc for sc in man["colorScales"]}
    assert set(scales) == {"cells.dapi", "cells.gfp"}
    for sc in scales.values():
        assert sc["mappables"] == ["cells.image"] and sc["recolor"] == "regenerate" and sc["norm"]["kind"] == "linear"
    assert (scales["cells.gfp"]["norm"]["vmin"], scales["cells.gfp"]["norm"]["vmax"]) == (10.0, 100.0)
    assert scales["cells.gfp"]["colormap"]["lut"][-1].lower().startswith("#008000") and scales["cells.gfp"]["label"] == "gfp"
    assert 'id="cells.image"' in svg and 'data-color-scale="cells.dapi cells.gfp"' in svg and 'data-paint="fill"' in svg
    controls = rec["params"]["__fluxplot__"]
    assert controls["cells.gfp"] == {"cmap": "image.mono:#008000", "vmin": 10.0, "vmax": 100.0, "norm": {"kind": "linear"}}
    assert controls["cells.dapi"]["vmax"] == 200.0
    # an edit to one channel's control (display range + LUT) changes only that channel on rerun
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": {"cells.gfp": {"vmax": 50.0, "cmap": "magenta"}}}))
    fig, ax = plt.subplots()
    im = fp.image(ax, np.stack([CH0, CH1]), series="cells", channels=["dapi", "gfp"], luts=["blue", "green"],
                  display_range=[(0, 200), (10, 100)], pixel_size=0.5)
    assert im.display_range == {"dapi": (0.0, 200.0), "gfp": (10.0, 50.0)}
    assert im.luts["gfp"].name == "image.mono:#ff00ff" and im.luts["dapi"].name == "image.mono:#0000ff"
    res2, man2, rec2, _ = _save(fig, tmp_path, "j.svg")
    assert rec2["params"]["__fluxplot__"]["cells.gfp"] == {"cmap": "image.mono:#ff00ff", "vmin": 10.0, "vmax": 50.0, "norm": {"kind": "linear"}}
    # replaying the recorded controls reproduces the same bytes
    monkeypatch.setenv("FLUX_PARAMS", json.dumps({"__fluxplot__": rec2["params"]["__fluxplot__"]}))
    fig, ax = plt.subplots()
    fp.image(ax, np.stack([CH0, CH1]), series="cells", channels=["dapi", "gfp"], luts=["blue", "green"],
             display_range=[(0, 200), (10, 100)], pixel_size=0.5)
    res3 = fp.save(fig, str(tmp_path / "j.svg"), recipe=dict(script="img.py"))
    assert open(res3.svg, "rb").read() == open(res2.svg, "rb").read()


def test_single_channel_scale_is_the_series_and_colorbar_links(tmp_path):
    fig, ax = plt.subplots()
    im = fp.image(ax, CH0, series="dapi", luts="gray", display_range=(0, 150))
    fp.colorbar(im.mappables["ch0"], ax=ax, label="a.u.")
    res, man, rec, svg = _save(fig, tmp_path)
    (sc,) = man["colorScales"]
    assert sc["id"] == "dapi" and sc["mappables"] == ["dapi.image"] and sc["colormap"]["name"] == "gray"
    assert rec["params"]["__fluxplot__"]["dapi"] == {"cmap": "gray", "vmin": 0.0, "vmax": 150.0, "norm": {"kind": "linear"}}
    guides = [g for g in man["guides"] if g["role"] == "colorbar"]
    assert len(guides) == 1


def test_value_raster_writes_one_sidecar_per_channel(tmp_path):
    fig, ax = plt.subplots()
    fp.image(ax, np.stack([CH0, CH1]), series="cells", channels=["a", "b"], value_raster=True)
    res, man, rec, svg = _save(fig, tmp_path, "v.svg")
    scales = {sc["id"]: sc for sc in man["colorScales"]}
    for ch, arr in (("a", CH0), ("b", CH1)):
        sc = scales[f"cells.{ch}"]
        assert sc["recolor"] == "raster" and sc["valueRaster"] == f"v.cells.{ch}.values.json"
        payload = json.loads(open(os.path.join(tmp_path, sc["valueRaster"])).read())
        assert payload["spec"] == "fluxplot/values" and payload["shape"] == [H, W]
        assert payload["values"] == pytest.approx(list(arr.reshape(-1)))


def test_scalebar_length_in_data_units_and_manifest_payload(tmp_path):
    fig, ax = plt.subplots()
    fp.image(ax, CH0, series="dapi", pixel_size=0.5)
    bar = fp.scalebar(ax, 2.0, "µm")
    x = bar.get_xdata()
    assert abs(x[1] - x[0]) == pytest.approx(2.0)
    assert bar._fluxplot_label.get_text() == "2 µm"
    # anchored inside the lower-right corner: bar right end left of the axes' right edge
    assert x[1] < ax.get_xlim()[1] and x[0] > ax.get_xlim()[0]
    bar2 = fp.scalebar(ax, 1.0, "µm", loc="upper left", label="1 micron", color="#ff0000", thickness=4)
    assert bar2.get_linewidth() == 4 and bar2.get_color() == "#ff0000"
    res, man, rec, svg = _save(fig, tmp_path, "s.svg")
    ov = {o["id"]: o for o in man["overlays"] if o["role"] == "scalebar"}
    assert set(ov) == {"scalebar.0", "scalebar.1"}
    assert ov["scalebar.0"]["length"] == 2.0 and ov["scalebar.0"]["units"] == "µm" and ov["scalebar.0"]["label"] == "2 µm"
    assert ov["scalebar.1"]["label"] == "1 micron"
    assert 'id="scalebar.0.label"' in svg and 'id="scalebar.0"' in svg
    # themed ink: the bar and its label carry the ink token; an explicit colour does not
    import re
    el = re.search(r'<g id="scalebar.0"[^>]*>', svg).group(0)
    assert 'data-ink-stroke="ink"' in el
    el2 = re.search(r'<g id="scalebar.1"[^>]*>', svg).group(0)
    assert "data-ink-stroke" not in el2
    with pytest.raises(ValueError, match="loc must be"):
        fp.scalebar(ax, 1.0, loc="middle")
    with pytest.raises(ValueError, match="positive"):
        fp.scalebar(ax, 0)


def test_image_saves_are_byte_stable(tmp_path):
    outs = []
    for _ in range(2):
        fig, ax = plt.subplots()
        fp.image(ax, np.stack([CH0, CH1]), series="cells", channels=["a", "b"], pixel_size=0.5)
        fp.scalebar(ax, 2.0)
        res = fp.save(fig, str(tmp_path / "d.svg"), recipe=False)
        outs.append((open(res.svg, "rb").read(), open(res.manifest, "rb").read()))
        plt.close(fig)
    assert outs[0] == outs[1]
