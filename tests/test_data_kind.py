"""data-kind hints: authored kind (text | line | shape | container) on SVG nodes + manifest.

Flux's part editors pick a property set by kind (a tick-label edits like a text object, a
gridline like a line). Before these hints, the consumer re-derived kind from role sets —
heuristics that break on x- extension roles. The generator knows; it should say.
"""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import fluxplot as fp  # noqa: E402
from fluxplot import roles as roles_mod  # noqa: E402
from fluxplot import style as st  # noqa: E402

KINDS = {"text", "line", "shape", "container"}


def _attrs_by_id(svg: str) -> dict:
    """Map id → the full opening tag it appears in."""
    return {m.group(1): m.group(0) for m in re.finditer(r'<[^>]*\bid="([^"]+)"[^>]*>', svg)}


def _kind_of(tag: str):
    m = re.search(r'data-kind="([^"]+)"', tag)
    return m.group(1) if m else None


def test_kind_map_covers_core_vocabulary():
    """Every core role except the deliberately-heterogeneous "extra" has a static kind."""
    missing = sorted(roles_mod.CORE_ROLES - set(roles_mod.KIND_BY_ROLE) - {"extra"})
    assert not missing, f"core roles without a data-kind: {missing}"
    bad = {r: k for r, k in roles_mod.KIND_BY_ROLE.items() if k not in KINDS}
    assert not bad, f"kinds outside the text|line|shape|container vocabulary: {bad}"


def test_svg_nodes_carry_data_kind(tmp_path):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    x = np.linspace(0, 10, 20)
    fp.line(ax, x, np.sin(x), series="wave", label="wave")
    fp.scatter(ax, x, np.cos(x), series="dots", label="dots")
    fp.bar(ax, [2, 5, 8], [0.2, 0.4, 0.3], series="counts")
    ax.legend()
    st.title(ax, "Heading", "a subtitle")
    res = fp.save(fig, str(tmp_path / "k.svg"))
    plt.close(fig)
    tags = _attrs_by_id(open(res.svg).read())

    assert _kind_of(tags["figure"]) == "container"
    assert _kind_of(tags["plot-area"]) == "container"
    assert _kind_of(tags["wave.line"]) == "line"
    assert _kind_of(tags["dots.points"]) == "shape"  # the group mirrors its members
    assert _kind_of(tags["dots.point.0"]) == "shape"
    assert _kind_of(tags["counts.bar.0"]) == "shape"
    assert _kind_of(tags["axis.x.spine"]) == "line"
    assert _kind_of(tags["figure.title"]) == "text"
    # every tick label is text, every gridline/tick a line
    for gid, tag in tags.items():
        if ".ticklabel." in gid:
            assert _kind_of(tag) == "text", gid
        if ".gridline." in gid or re.search(r"\.tick\.\d", gid):
            assert _kind_of(tag) == "line", gid


def test_manifest_parts_mirror_kind(tmp_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    fp.line(ax, [0, 1, 2], [1, 3, 2], series="s", label="s")
    ax.legend()
    res = fp.save(fig, str(tmp_path / "m.svg"))
    plt.close(fig)
    man = json.load(open(res.manifest))

    by_id = {}

    def walk(node):
        key = node.get("id") or node.get("ref")
        by_id[key] = node
        for c in node.get("children", []):
            walk(c)

    walk(man["parts"])
    assert by_id["figure"]["kind"] == "container"
    assert by_id["plot-area"]["kind"] == "container"
    assert by_id["s"]["kind"] == "container"  # the series node
    assert by_id["s.line"]["kind"] == "line"  # a leaf ref
    assert by_id["axis.x.tick-labels"]["kind"] == "text"  # a group node
    assert by_id["legend"]["kind"] == "container"
    # no invented kinds anywhere in the tree
    for key, node in by_id.items():
        if "kind" in node:
            assert node["kind"] in KINDS, (key, node["kind"])


def test_x_role_kind_inferred_from_artist(tmp_path):
    """An x- extension role has no static kind — infer it from the concrete artist."""
    fig, ax = plt.subplots(figsize=(4, 3))
    fp.line(ax, [0, 1, 2], [1, 3, 2], series="s")
    blob = ax.add_patch(plt.Rectangle((0.5, 1.2), 1.0, 0.8))
    fp.tag(blob, role="x-blob", series="s")
    res = fp.save(fig, str(tmp_path / "x.svg"))
    plt.close(fig)
    tags = _attrs_by_id(open(res.svg).read())
    assert _kind_of(tags["s.x-blob"]) == "shape"
    man = json.load(open(res.manifest))
    s = next(e for e in man["series"] if e["id"] == "s")

    def find_ref(node, ref):
        if node.get("ref") == ref:
            return node
        for c in node.get("children", []):
            hit = find_ref(c, ref)
            if hit:
                return hit
        return None

    node = find_ref(man["parts"], s["svg"]["x-blob"])
    assert node is not None and node.get("kind") == "shape"


def test_extra_sweep_kind_from_artist(tmp_path):
    """Swept untagged artists get a per-artist kind (a raw line is a line, a patch a shape)."""
    fig, ax = plt.subplots(figsize=(4, 3))
    fp.line(ax, [0, 1, 2], [1, 3, 2], series="s")
    ax.plot([0, 2], [2, 1])  # raw, untagged
    ax.add_patch(plt.Rectangle((0.2, 1.0), 0.5, 0.5))  # raw, untagged
    res = fp.save(fig, str(tmp_path / "e.svg"))
    plt.close(fig)
    tags = _attrs_by_id(open(res.svg).read())
    assert _kind_of(tags["extra.line.0"]) == "line"
    assert _kind_of(tags["extra.patch.0"]) == "shape"
    man = json.load(open(res.manifest))
    kinds = {o["id"]: o.get("kind") for o in man["overlays"] if o["role"] == "extra"}
    assert kinds == {"extra.line.0": "line", "extra.patch.0": "shape"}
