"""The DOM-shape probe (see NOTES_matplotlib_svg.md §2).

If a matplotlib upgrade changes the marker/scatter SVG structure, this test fails LOUDLY rather than
letting the generator silently corrupt per-point ids/indices.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from lxml import etree  # noqa: E402

from fluxplot import render  # noqa: E402

SVG = "http://www.w3.org/2000/svg"


def _group(svg_bytes, gid):
    root = etree.fromstring(svg_bytes)
    matches = [e for e in root.iter() if e.get("id") == gid]
    assert matches, f"no element with id={gid!r}"
    return matches[0]


def test_plot_markers_emit_one_use_per_point_in_order():
    fig, ax = plt.subplots()
    xs = [0, 1, 2, 3, 4]
    (m,) = ax.plot(xs, [1, 2, 3, 4, 5], linestyle="none", marker="o")
    m.set_gid("probe.points")
    svg = render.render_svg(fig, "probe")
    plt.close(fig)

    group = _group(svg, "probe.points")
    uses = group.findall(f".//{{{SVG}}}use")
    assert len(uses) == len(xs), "expected exactly one <use> per data point"
    # ascending x positions == data order preserved
    x_positions = [float(u.get("x")) for u in uses]
    assert x_positions == sorted(x_positions)


def test_scatter_emits_one_use_per_point():
    fig, ax = plt.subplots()
    coll = ax.scatter([0, 2, 4], [1, 2, 3], marker="s")
    coll.set_gid("probe.scatter")
    svg = render.render_svg(fig, "probe")
    plt.close(fig)

    group = _group(svg, "probe.scatter")
    uses = group.findall(f".//{{{SVG}}}use")
    assert len(uses) == 3


def test_bar_each_rectangle_is_its_own_group():
    fig, ax = plt.subplots()
    bars = ax.bar([0, 1, 2], [3, 5, 4])
    for i, b in enumerate(bars):
        b.set_gid(f"counts.bar.{i}")
    svg = render.render_svg(fig, "probe")
    plt.close(fig)

    for i in range(3):
        grp = _group(svg, f"counts.bar.{i}")
        assert grp.findall(f".//{{{SVG}}}path"), "each bar group should contain a path"


def test_fonttype_none_keeps_text_nodes():
    fig, ax = plt.subplots()
    ax.set_xlabel("Time (h)")
    svg = render.render_svg(fig, "probe")
    plt.close(fig)
    root = etree.fromstring(svg)
    texts = [t for t in root.iter(f"{{{SVG}}}text")]
    assert any("Time (h)" in "".join(t.itertext()) for t in texts)
