"""Tick <use> dereference (WS5): draw-on can't animate a <use>, so every tick's marker is inlined.

matplotlib renders each tick as ``<g data-role="tick"><g><use href="#markerPath"/></g></g>`` with a
single shared ``<defs><path>``. Flux's draw-on preset measures a path's length (stroke-dashoffset) and
cannot do that through a ``<use>``. Post-processing replaces every tick ``<use>`` with a real inlined
``<path>`` and prunes the orphaned marker defs — WITHOUT touching per-point ``<use>`` (points animate
via opacity/transform and stay indirect).
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lxml import etree

import fluxplot as fp

SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"


def _root(path):
    return etree.parse(path).getroot()


def _href(el):
    return el.get(f"{{{XLINK}}}href") or el.get("href")


def _tick_groups(root):
    return [el for el in root.iter() if el.get("data-role") == "tick"]


def test_ticks_are_real_paths_no_use(tmp_path):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fp.line(ax, [0, 1, 2, 3], [1, 2, 1.5, 3], series="alpha", marker="o", label="Alpha")
    ax.grid(True)
    res = fp.save(fig, str(tmp_path / "ticks.svg"))
    plt.close(fig)

    root = _root(res.svg)
    ticks = _tick_groups(root)
    assert ticks, "expected some data-role='tick' groups"

    for tg in ticks:
        uses = list(tg.iter(f"{{{SVG}}}use"))
        paths = list(tg.iter(f"{{{SVG}}}path"))
        assert not uses, f"tick {tg.get('id')} still contains a <use>"
        assert paths, f"tick {tg.get('id')} has no real <path>"
        # the inlined path carries the marker geometry + the folded translate
        p = paths[0]
        assert p.get("d"), "inlined tick path must carry a real d"
        assert p.get("transform", "").startswith("translate"), "use x/y must fold into a translate"


def test_no_dangling_defs_after_deref(tmp_path):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fp.line(ax, [0, 1, 2, 3], [1, 2, 1.5, 3], series="alpha", marker="o", label="Alpha")
    res = fp.save(fig, str(tmp_path / "defs.svg"))
    plt.close(fig)

    root = _root(res.svg)
    # no tick-local <defs> should survive (their marker path was inlined + is now unreferenced)
    for tg in _tick_groups(root):
        assert not list(tg.iter(f"{{{SVG}}}defs")), f"tick {tg.get('id')} kept an orphan <defs>"

    # every surviving <use> reference resolves to a real <path id> (nothing dangling)
    path_ids = {p.get("id") for p in root.iter(f"{{{SVG}}}path") if p.get("id")}
    for use in root.iter(f"{{{SVG}}}use"):
        href = _href(use)
        if href and href.startswith("#"):
            assert href[1:] in path_ids, f"dangling <use> reference {href}"


def test_point_use_elements_untouched(tmp_path):
    """Points animate fine as per-point <use> — the tick pass must not touch them."""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fp.scatter(ax, [0, 1, 2, 3], [1, 2, 1.5, 3], series="dots", label="Dots")
    res = fp.save(fig, str(tmp_path / "pts.svg"))
    plt.close(fig)

    svg = open(res.svg).read()
    # per-point <use> elements are still present and still carry data-role="point"
    point_uses = re.findall(r'<use[^>]*data-role="point"', svg)
    assert len(point_uses) == 4, f"expected 4 point <use>, found {len(point_uses)}"
