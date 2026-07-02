"""Orphan-artist sweep (WS5): raw ax.plot/collections/patches become addressable "extra" content.

The free-text sweep already rescued untagged ``ax.text``. Untagged Line2D / collections / patches
(e.g. a raw ``ax.plot([m, m], [0, .5])`` median rule) still escaped the scene graph as anonymous
``<g id="line2d_N">`` — no data-role, absent from the manifest, so Flux couldn't mask or animate them.
The sweep assigns ``extra.line.N`` / ``extra.collection.N`` / ``extra.patch.N`` (role ``extra``), an SVG
``data-role="extra"``, a manifest overlay, and membership in an ``extras`` group under plot-area.
"""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fluxplot as fp


def _find_group(node, group_role):
    if node.get("groupRole") == group_role:
        return node
    for c in node.get("children", []):
        if isinstance(c, dict):
            hit = _find_group(c, group_role)
            if hit:
                return hit
    return None


def test_raw_line_swept_to_extra(tmp_path):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fp.line(ax, [0, 1, 2, 3], [1, 2, 1.5, 3], series="alpha", label="Alpha")
    # a raw, untagged vertical rule (the 08_ecdf median pattern)
    ax.plot([1.5, 1.5], [0, 2], color="red", linestyle=(0, (2, 2)))
    res = fp.save(fig, str(tmp_path / "extra.svg"))
    plt.close(fig)

    svg = open(res.svg).read()
    man = json.load(open(res.manifest))

    # SVG: the raw line got the extra.line.0 id + data-role="extra"
    assert 'id="extra.line.0"' in svg
    assert re.search(r'id="extra\.line\.0"[^>]*data-role="extra"', svg), "extra.line.0 needs data-role=extra"

    # manifest overlay
    extras = [o for o in man["overlays"] if o["role"] == "extra"]
    assert any(o["id"] == "extra.line.0" for o in extras), f"extra.line.0 not an overlay: {extras}"

    # manifest parts: an "extras" group under plot-area with the line as a member
    grp = _find_group(man["parts"], "extra")
    assert grp is not None and grp["id"] == "extras"
    assert "extra.line.0" in grp["members"]

    # build order includes it so it can animate
    assert "extra.line.0" in man["build"]["order"]


def test_raw_collection_and_patch_swept(tmp_path):
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(5, 3.2))
    fp.line(ax, [0, 1, 2, 3], [1, 2, 1.5, 3], series="alpha", label="Alpha")
    ax.scatter([0.5, 2.5], [2, 1], color="green")  # raw collection
    ax.add_patch(mpatches.Rectangle((0.2, 0.2), 0.5, 0.5, color="orange"))  # raw patch
    res = fp.save(fig, str(tmp_path / "cp.svg"))
    plt.close(fig)

    man = json.load(open(res.manifest))
    ids = {o["id"] for o in man["overlays"] if o["role"] == "extra"}
    assert "extra.collection.0" in ids
    assert "extra.patch.0" in ids


def test_tagged_artists_not_swept(tmp_path):
    """fp.* artists already carry a gid before the sweep runs — they must NOT become extras."""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fp.line(ax, [0, 1, 2, 3], [1, 2, 1.5, 3], series="alpha", marker="o", label="Alpha")
    fp.reference_line(ax, y=1.0, name="threshold", color="0.6", linestyle=":")
    res = fp.save(fig, str(tmp_path / "clean.svg"))
    plt.close(fig)

    man = json.load(open(res.manifest))
    assert not [o for o in man["overlays"] if o["role"] == "extra"], "no extras expected"
