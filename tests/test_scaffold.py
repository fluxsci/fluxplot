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
