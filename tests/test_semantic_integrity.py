"""Manifest ↔ SVG integrity: every id the manifest references must exist in the SVG.

This is the test that would have caught the title / colorbar-collision / free-text
escapes: a consumer (Flux's X-ray) can only address what the manifest declares, and a
manifest that names an id absent from the SVG is a broken scene graph. The integrity
sweep walks every reference path (series.svg, parts refs + group members, guides,
overlays, build.order) and asserts each resolves to a real ``id="…"`` in the SVG.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import fluxplot as fp
from fluxplot import style as st

# build.order may carry role-ref tokens that name a *class* of nodes rather than one
# svg element (the consumer expands them). They are intentionally not svg ids.
_ORDER_ROLE_REFS = {"gridlines"}


def _svg_ids(svg: str) -> set:
    return set(re.findall(r'\bid="([^"]+)"', svg))


def _referenced_ids(man: dict) -> set:
    """Every svg id the manifest points at, across all reference paths."""
    refs: set = set()

    for s in man.get("series", []):
        for v in s.get("svg", {}).values():
            refs.update(v if isinstance(v, list) else [v])
        for p in s.get("points", []):
            refs.add(p["svgId"])

    for g in man.get("guides", []):
        if "svgId" in g:
            refs.add(g["svgId"])
        for e in g.get("entries", []):
            refs.update(x for x in (e.get("swatch"), e.get("label")) if x)

    for o in man.get("overlays", []):
        refs.add(o["svgId"])

    def walk(node):
        if "ref" in node:
            refs.add(node["ref"])
        for m in node.get("members", []):
            refs.add(m)
        for c in node.get("children", []):
            walk(c)

    walk(man.get("parts", {}))

    for e in man.get("build", {}).get("order", []):
        if e not in _ORDER_ROLE_REFS:
            refs.add(e)

    return refs


def _assert_integrity(res):
    man = json.load(open(res.manifest))
    ids = _svg_ids(open(res.svg).read())
    dangling = sorted(r for r in _referenced_ids(man) if r not in ids)
    assert not dangling, f"manifest references ids absent from the SVG: {dangling}"
    return man, ids


def test_rich_figure_has_no_dangling_refs(tmp_path):
    """A figure exercising titles, subtitle, legend, line/scatter/area + free text."""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    x = np.linspace(0, 10, 30)
    fp.line(ax, x, np.sin(x), series="wave", label="wave")
    fp.scatter(ax, x, np.cos(x), series="dots", label="dots")
    ax.legend()
    st.title(ax, "Main heading", "a muted subtitle line")
    ax.text(0.5, 0.5, "free callout", transform=ax.transAxes)
    res = fp.save(fig, str(tmp_path / "rich.svg"))
    plt.close(fig)
    _assert_integrity(res)


def test_left_title_is_tagged(tmp_path):
    """House style writes a LEFT title (ax._left_title); it must become figure.title."""
    fig, ax = plt.subplots(figsize=(4, 3))
    fp.line(ax, [0, 1, 2], [1, 3, 2], series="s")
    st.title(ax, "Left aligned heading")
    res = fp.save(fig, str(tmp_path / "t.svg"))
    plt.close(fig)
    man, ids = _assert_integrity(res)
    assert "figure.title" in ids
    order = man["build"]["order"]
    assert "figure.title" in order, "title should be in the build order (phase 0)"


def test_subtitle_role_and_element(tmp_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    fp.line(ax, [0, 1, 2], [1, 3, 2], series="s")
    st.title(ax, "Heading", "the subtitle")
    res = fp.save(fig, str(tmp_path / "sub.svg"))
    plt.close(fig)
    man, ids = _assert_integrity(res)
    subs = [o for o in man["overlays"] if o["role"] == "subtitle"]
    assert subs, "subtitle should be an overlay with role 'subtitle'"
    assert subs[0]["svgId"] in ids
    assert subs[0].get("text") == "the subtitle", "subtitle text must travel into the manifest"


def test_free_text_swept_to_annotation(tmp_path):
    """A raw ax.text (an equation box, a value label) must not escape as text_N."""
    fig, ax = plt.subplots(figsize=(4, 3))
    fp.line(ax, [0, 1, 2], [1, 3, 2], series="s")
    ax.text(0.4, 0.8, r"$y = mx + b$", transform=ax.transAxes)
    res = fp.save(fig, str(tmp_path / "anno.svg"))
    plt.close(fig)
    man, ids = _assert_integrity(res)
    annos = [o for o in man["overlays"] if o["role"] == "annotation"]
    assert annos, "the raw text should be swept into an annotation overlay"
    assert annos[0]["svgId"] in ids


def test_colorbar_does_not_collide_axes(tmp_path):
    """fig.colorbar adds a second Axes; it must not duplicate plot-area or collide ids."""
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(np.random.default_rng(0).random((6, 6)))
    fig.colorbar(im, ax=ax)
    res = fp.save(fig, str(tmp_path / "cb.svg"))
    plt.close(fig)
    man, ids = _assert_integrity(res)
    # exactly one plot-area capture (the colorbar Axes is not captured as a plot)
    assert len(man["axes"]) == 1, f"expected 1 plot-area, got {len(man['axes'])}"
    # the collided ids the old code produced must be gone
    assert not [i for i in ids if re.match(r"axis\.[xy]-\d", i)], "colliding axis.x-2/y-2 ids"


def test_errorbar_central_line_is_tagged(tmp_path):
    """fp.errorbar previously dropped the central line + caps; the central line must
    now carry the errorbar id (no longer escaping as an anonymous line)."""
    fig, ax = plt.subplots(figsize=(4, 3))
    fp.errorbar(ax, [0, 1, 2], [1, 3, 2], series="e", yerr=[0.2, 0.3, 0.25])
    res = fp.save(fig, str(tmp_path / "eb.svg"))
    plt.close(fig)
    man, ids = _assert_integrity(res)
    e = next(s for s in man["series"] if s["id"] == "e")
    assert e["svg"].get("errorbar") in ids


def _parts_by_key(man):
    out = {}

    def walk(node):
        out[node.get("id") or node.get("ref")] = node
        for c in node.get("children", []):
            walk(c)

    walk(man["parts"])
    return out


def test_errorbar_members_grouped(tmp_path):
    """The composite's numbered siblings (caps + bar segments) must be first-class:
    svg.errorbars lists every member and the parts tree groups them per series —
    previously they were orphans only Flux's orphan-defense could reach."""
    fig, ax = plt.subplots(figsize=(5, 3))
    # the notebook's "Bar chart with error bars" pattern (mpl_bars_FLUXPLOT)
    fp.bar(ax, [0, 1, 2], [4.2, 6.8, 5.5], series="viability")
    fp.errorbar(ax, [0, 1, 2], [4.2, 6.8, 5.5], series="viability",
                yerr=[0.4, 0.6, 0.5], fmt="none", capsize=3)
    res = fp.save(fig, str(tmp_path / "bars.svg"))
    plt.close(fig)
    man, ids = _assert_integrity(res)

    s = next(e for e in man["series"] if e["id"] == "viability")
    members = s["svg"].get("errorbars")
    assert members and len(members) >= 2, "expected the composite's sibling gids"
    assert s["svg"]["errorbar"] in members, "the primary ref is one of the members"
    assert all(m in ids for m in members)

    parts = _parts_by_key(man)
    grp = parts.get("viability.errorbars")
    assert grp is not None, "per-series errorbars group node missing from the parts tree"
    assert grp["role"] == "group" and grp["groupRole"] == "errorbar"
    assert grp["members"] == members
    assert grp.get("kind") == "line"
    # the old lone {"ref": errorbar} sibling must not duplicate the group's coverage
    assert s["svg"]["errorbar"] not in parts or parts[s["svg"]["errorbar"]] is grp or \
        "ref" not in parts.get(s["svg"]["errorbar"], {}), "duplicate errorbar ref in tree"
    # every member is in the build order (revealed like bars)
    order = man["build"]["order"]
    assert all(m in order for m in members)


def test_errorbar_multi_mark_series_all_members(tmp_path):
    """Several errorbar marks on ONE series (the seaborn join emits one mark per segment)
    accumulate into a single errorbars list instead of last-mark-wins."""
    fig, ax = plt.subplots(figsize=(4, 3))
    for x in (0, 1, 2):
        (seg,) = ax.plot([x, x], [1.0, 2.0])
        fp.tag(seg, role="errorbar", series="joined")
    res = fp.save(fig, str(tmp_path / "multi.svg"))
    plt.close(fig)
    man, ids = _assert_integrity(res)
    s = next(e for e in man["series"] if e["id"] == "joined")
    assert len(s["svg"]["errorbars"]) == 3
    assert s["svg"]["errorbar"] == s["svg"]["errorbars"][0], "primary ref = first mark's gid"


# ---------------------------------------------------------------------------
# Gallery sweep: the committed example outputs are themselves under the integrity
# gate, so regenerating examples/basic_examples.ipynb re-validates every emitted
# manifest against its SVG (this is what would catch a notebook/output drift).
# ---------------------------------------------------------------------------
_GALLERY = Path(__file__).resolve().parent.parent / "examples" / "basic_example_output"
_GALLERY_MANIFESTS = sorted(_GALLERY.glob("*.fluxplot.json"))


@pytest.mark.parametrize(
    "manifest_path", _GALLERY_MANIFESTS,
    ids=lambda p: p.name.replace(".fluxplot.json", ""),
)
def test_gallery_integrity(manifest_path):
    man = json.load(open(manifest_path))
    svg_path = manifest_path.with_name(man["svg"])
    assert svg_path.exists(), f"manifest names a missing SVG: {man['svg']}"
    ids = _svg_ids(svg_path.read_text())
    dangling = sorted(r for r in _referenced_ids(man) if r not in ids)
    assert not dangling, (
        f"{manifest_path.name}: manifest references ids absent from the SVG: {dangling}"
    )
