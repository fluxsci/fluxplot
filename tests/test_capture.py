"""The captured anchors must reproduce the emitted SVG positions (linear AND log axes)."""
import math
import re

import matplotlib.pyplot as plt

import fluxplot as fp


def _interp(anchors, value, log=False):
    (d0, s0) = anchors[0]["data"], anchors[0]["svg"]
    (d1, s1) = anchors[1]["data"], anchors[1]["svg"]
    if log:
        d0, d1, value = math.log10(d0), math.log10(d1), math.log10(value)
    return s0 + (value - d0) / (d1 - d0) * (s1 - s0)


def _use_xy(svg_text, gid):
    use = re.search(rf'<use[^>]*id="{re.escape(gid)}"[^>]*/?>', svg_text).group(0)
    x = float(re.search(r'\bx="([-0-9.]+)"', use).group(1))
    y = float(re.search(r'\by="([-0-9.]+)"', use).group(1))
    return x, y


def test_linear_anchors_match_emitted_point(tmp_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    fp.line(ax, [0, 1, 2, 3], [10, 20, 15, 25], series="s", marker="o")
    res = fp.save(fig, str(tmp_path / "p.svg"))
    plt.close(fig)
    import json

    man = json.load(open(res.manifest))
    svg = open(res.svg).read()
    xa = man["axes"][0]["x"]["anchors"]
    ya = man["axes"][0]["y"]["anchors"]
    ux, uy = _use_xy(svg, "s.point.2")  # data (2, 15)
    assert abs(_interp(xa, 2) - ux) < 0.5
    assert abs(_interp(ya, 15) - uy) < 0.5


def test_log_axis_anchors_carry_base_and_map_correctly(tmp_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    fp.line(ax, [0, 1, 2, 3], [0.1, 1.0, 0.5, 2.0], series="s", marker="o")
    ax.set_yscale("log")
    res = fp.save(fig, str(tmp_path / "p.svg"))
    plt.close(fig)
    import json

    man = json.load(open(res.manifest))
    svg = open(res.svg).read()
    yaxis = man["axes"][0]["y"]
    assert yaxis["scale"] == "log"
    assert yaxis["base"] == 10.0
    _, uy = _use_xy(svg, "s.point.3")  # data y=2.0
    assert abs(_interp(yaxis["anchors"], 2.0, log=True) - uy) < 0.6
