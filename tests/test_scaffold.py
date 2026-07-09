"""The completed parts model (spec 0.2.0): scaffold tagging + manifest group nodes."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fluxplot as fp


def _groups(node, out):
    if node.get("role") == "group":
        out.append(node)
    for c in node.get("children", []):
        if isinstance(c, dict):
            _groups(c, out)
    return out


def test_scaffold_tags_and_groups(tmp_path):
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1, 2, 3], [1, 2, 1.5, 3], series="alpha", marker="o", label="Alpha")
    fp.bar(ax, [0, 1, 2, 3], [0.5, 1.0, 0.7, 1.2], series="beta", label="Beta")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.legend()

    res = fp.save(fig, str(tmp_path / "p.svg"))
    svg = open(res.svg).read()

    # every scaffold part is now tagged
    for role in ("axis", "gridline", "tick", "tick-label", "spine", "legend-swatch", "legend-label"):
        assert f'data-role="{role}"' in svg, f"missing data-role={role}"
    # the axis is a real wrapping group (closes the dangling-ref gap)
    assert 'id="axis.x"' in svg and 'id="axis.y"' in svg
    # data-axis carried on scaffold
    assert 'data-axis="x"' in svg and 'data-axis="y"' in svg

    man = json.load(open(res.manifest))
    assert man["schemaVersion"].startswith("0.2")

    groups = _groups(man["parts"], [])
    roles = {g["groupRole"] for g in groups}
    assert {"tick", "tick-label", "gridline", "point", "bar"} <= roles, roles
    # every group node carries a members[] of leaf ids
    for g in groups:
        assert g["members"] and all(isinstance(m, str) for m in g["members"])

    plt.close(fig)


def test_legend_entries_in_tree(tmp_path):
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1, 2], [1, 2, 3], series="ctrl", label="Control")
    ax.legend()
    res = fp.save(fig, str(tmp_path / "leg.svg"))
    man = json.load(open(res.manifest))

    def find(node, role):
        if node.get("role") == role:
            return node
        for c in node.get("children", []):
            if isinstance(c, dict):
                hit = find(c, role)
                if hit:
                    return hit
        return None

    legend = find(man["parts"], "legend")
    assert legend is not None
    entry = find(legend, "legend-entry")
    assert entry is not None and entry["id"] == "legend.entry.0"
    plt.close(fig)


def test_polar_scaffold_tagged(tmp_path):
    """Polar axes key their spines polar/start/end/inner — the rectangular side list
    silently dropped them all. The outer circle must be a tagged spine, and the theta/r
    tick labels + gridlines must tag exactly like their rectangular counterparts."""
    import re

    import numpy as np

    theta = np.linspace(0, 2 * np.pi, 60)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    fp.line(ax, theta, 1 + 0.3 * np.sin(3 * theta), series="orbit")

    res = fp.save(fig, str(tmp_path / "polar.svg"))
    svg = open(res.svg).read()
    ids = set(re.findall(r'\bid="([^"]+)"', svg))
    plt.close(fig)

    # the outer 'polar' circle runs along theta → the x-axis spine
    assert "axis.x.spine" in ids, "polar outer-circle spine untagged"
    assert 'data-role="spine"' in svg
    # theta (x) and r (y) scaffold: real axis wrappers + tick labels + gridlines
    assert "axis.x" in ids and "axis.y" in ids
    assert any(i.startswith("axis.x.ticklabel.") for i in ids)
    assert any(i.startswith("axis.y.ticklabel.") for i in ids)
    assert any(i.startswith("axis.x.gridline.") for i in ids)
    assert any(i.startswith("axis.y.gridline.") for i in ids)
    assert "orbit.line" in ids

    # manifest: the spine is a member of the x axis node, and nothing dangles
    man = json.load(open(res.manifest))

    def refs(node, out):
        if "ref" in node:
            out.add(node["ref"])
        out.update(node.get("members", []))
        for c in node.get("children", []):
            refs(c, out)
        return out

    referenced = refs(man["parts"], set())
    assert "axis.x.spine" in referenced
    dangling = sorted(r for r in referenced if r not in ids)
    assert not dangling, f"polar manifest references missing ids: {dangling}"
