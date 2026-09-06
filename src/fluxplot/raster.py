"""Automatic rasterization of pathologically heavy artists — the safety default.

Why this exists
---------------
A matplotlib artist that draws *N* primitives becomes *N* SVG nodes. At scientific data
scale that is routinely 10^4–10^5 nodes in a single panel:

- a ``LineCollection`` built from per-edge segments (the usual way to draw an SWC
  reconstruction or a graph) emits **one ``<path>`` per segment**;
- a ``scatter`` emits **one ``<use>`` per point**.

Flux inlines every panel's SVG as live DOM, so those nodes are real cost forever after.
A measured 14-panel figure carrying three neuron reconstructions plus 8.7k-point scatters
reached 260,907 nodes and ~390 ms per pan frame — roughly 2.5 fps against a 100 ms
interaction budget — while the *same figure* with its heavy layers rasterized rendered at
vsync from 5,493 nodes.

matplotlib already ships the right tool: ``Artist.set_rasterized(True)`` draws that ONE
artist through the Agg backend and embeds the result as a single ``<image>``, leaving
axes, ticks, tick labels, legend, annotations and every other artist as vector. This
module decides which artists get that treatment, so a plot that would cripple a downstream
editor is never written in the first place. ``fp.save(..., force_vectors=True)`` opts out.

The matplotlib mechanic that makes this non-trivial
---------------------------------------------------
Rasterization **drops the artist's gid**. ``MixedModeRenderer.stop_rasterizing`` draws the
composited buffer through a *fresh* ``GraphicsContext``, so the ``<image>`` lands carrying
matplotlib's generated id (``image`` + 10 hex chars) and no wrapping ``<g id="...">``.
Left alone that would silently delete the part from fluxplot's semantic contract — the
whole point of the library.

Each planned artist receives a temporary draw wrapper that opens an SVG group
before mixed-mode rendering and flushes the raster inside that group. :func:`reattach`
then transfers the identity to its actual image. Empty/clipped layers cannot shift
another layer's identity; multiple images retain a named container. Draw methods,
raster flags, axes z-order rasterization and figure compositing state are restored.
The wrapper is local to this figure's export; no Matplotlib class/global is patched.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass

SVG = "http://www.w3.org/2000/svg"

#: matplotlib's generated element id (``backend_svg._make_id``): a type prefix + 10 hex chars.
AUTO_IMAGE_ID = re.compile(r"^image[0-9a-f]{6,}$")

#: Per-artist primitive budget above which a layer is rasterized. A Flux figure composes up
#: to ~14 panels into one inlined SVG, so the per-panel allowance has to leave the *figure*
#: comfortable. 800 catches the mid-size clouds a 2,000 cutoff left vector (per-class scatters
#: of 1–2k cells, minor transgenic-line series), pushing a full 14-panel projection figure
#: toward vsync, while still leaving ordinary plots untouched — a 200-point scatter or a
#: 500-segment line stays fully vector and per-point addressable.
DEFAULT_THRESHOLD = 800

#: Resolution for the embedded PNGs. Vector content is unaffected (SVG user space is points,
#: verified in ``NOTES_matplotlib_svg.md`` §6); this only sets how many pixels a rasterized
#: layer gets. 600 dpi leaves ~2x headroom over a 300 dpi print of a typical 2–3 in panel,
#: so the layer still looks sharp zoomed in inside a figure editor.
DEFAULT_DPI = 600


@dataclass
class RasterItem:
    """One artist selected for rasterization, with the state needed to undo it."""

    artist: object
    gid: "str | None"
    count: int
    was_rasterized: bool

    @property
    def label(self) -> str:
        return self.gid or type(self.artist).__name__


def primitive_count(artist) -> int:
    """How many SVG primitives ``artist`` will emit, or 0 when that can't be determined.

    Counting is deliberately defensive: a save must never fail because a third-party artist
    confused the estimator, so anything unexpected reports 0 (= "not heavy") and is left alone.
    """
    try:
        from matplotlib.artist import Artist
        from matplotlib.collections import Collection, QuadMesh
        from matplotlib.image import _ImageBase
        from matplotlib.lines import Line2D

        if not isinstance(artist, Artist):
            return 0
        if isinstance(artist, _ImageBase):
            return 0  # already a single <image>
        if isinstance(artist, QuadMesh):
            # get_paths() would materialize every quad just to count them; read the mesh shape.
            coords = getattr(artist, "_coordinates", None)
            if coords is not None and getattr(coords, "ndim", 0) == 3:
                return max(int(coords.shape[0]) - 1, 0) * max(int(coords.shape[1]) - 1, 0)
            return 0
        if isinstance(artist, Collection):
            # scatter: one marker path cycled over N offsets → N <use>.
            # LineCollection: N paths, no meaningful offsets. max() covers both.
            offsets = artist.get_offsets()
            return max(len(artist.get_paths()), 0 if offsets is None else len(offsets))
        if isinstance(artist, Line2D):
            marker = artist.get_marker()
            n = len(artist.get_xdata())
            return (n if marker not in (None, "", " ", "None") else 0) + 1
        return 1
    except Exception:
        return 0


def _draw_order(fig) -> list:
    """Every artist in matplotlib's own draw order.

    Mirrors ``Figure.draw``/``Axes.draw``: children sorted by zorder with a *stable* sort, so
    ties keep insertion order exactly as matplotlib emits them.
    """
    out: list = []
    seen: set = set()

    def visit(artist) -> None:
        if id(artist) in seen:
            return
        seen.add(id(artist))
        out.append(artist)

    def descend(artist):
        if hasattr(artist, 'get_xaxis'):
            for child in sorted(artist.get_children(), key=_zorder):
                descend(child)
        else:
            visit(artist)
    for child in sorted(fig.get_children(), key=_zorder):
        descend(child)
    return out


def _zorder(artist) -> float:
    try:
        return float(artist.get_zorder())
    except Exception:
        return 0.0


def plan(fig, threshold: int = DEFAULT_THRESHOLD) -> list:
    """Artists heavy enough to rasterize, in the order matplotlib will emit their images.

    Only *visible* artists count: an invisible one draws nothing, so it emits no image and
    would throw off the document-order match in :func:`reattach`.
    """
    items = []
    for artist in _draw_order(fig):
        try:
            if not artist.get_visible():
                continue
        except Exception:
            continue
        n = primitive_count(artist)
        # An artist the CALLER already flagged rasterized is planned too, however few primitives it
        # has. matplotlib will emit an <image> for it either way, and reattach keeps each image in its own
        # explicit draw scope, including caller-rasterized lightweight artists. The commonest case is a
        # colorbar: its solids are rasterized by default, are far below any heaviness threshold, and
        # would otherwise silently orphan the plot they belong to.
        ax = getattr(artist, 'axes', None)
        cutoff = ax.get_rasterization_zorder() if ax is not None else None
        inherited = ax is not None and (ax.get_rasterized() or (cutoff is not None and _zorder(artist) < cutoff))
        if n > threshold or bool(artist.get_rasterized()) or inherited:
            items.append(
                RasterItem(
                    artist=artist,
                    gid=artist.get_gid(),
                    count=n,
                    was_rasterized=bool(artist.get_rasterized()),
                )
            )
    return items


@contextmanager
def rasterizing(fig, items, force_vectors=False):
    """Render ``fig`` with ``items`` rasterized, then hand the figure back untouched.

    ``suppressComposite`` is the load-bearing detail. ``Artist.allow_rasterization`` only
    ends a raster run when a NON-rasterized artist is drawn, so *consecutive* rasterized
    artists are flattened into a single ``<image>`` — two heavy layers in one panel (an
    axon and a dendrite, say) would become one element and the second would lose its
    identity entirely. Setting ``suppressComposite`` makes matplotlib stop and restart
    rasterizing around every artist ("restart rasterizing to prevent merging"), which is
    what guarantees the one-image-per-artist mapping :func:`reattach` relies on. Verified:
    2/3/4 same-zorder rasterized layers emit 1/1/1 images by default and 2/3/4 with it set.

    It is applied only when something is actually rasterized, so an ordinary save keeps
    matplotlib's default compositing and byte-identical output.
    """
    from .panels import all_axes
    previous = fig.suppressComposite
    axes_state = [(ax, ax.get_rasterization_zorder(), ax.get_rasterized()) for ax in all_axes(fig)]
    draws = []
    fig.suppressComposite = True
    for ax, _, _ in axes_state:
        ax.set_rasterization_zorder(None)
        ax.set_rasterized(False)
    try:
        for it in items:
            it.artist.set_rasterized(not force_vectors)
            if force_vectors or not it.gid: continue
            art = it.artist
            original = art.draw
            had_draw = 'draw' in art.__dict__
            saved_draw = art.__dict__.get('draw')
            draws.append((art, had_draw, saved_draw))
            def draw(renderer, original=original, gid=it.gid):
                # Scope identity around the actual draw, so empty/clipped layers
                # cannot shift another layer's identity. Matplotlib's public
                # renderer group API encloses the mixed-mode image flush.
                renderer.open_group('fluxplot-raster', gid=gid)
                try:
                    original(renderer)
                    if getattr(renderer, '_rasterizing', False) and renderer._raster_depth == 0:
                        renderer.stop_rasterizing()
                        renderer._rasterizing = False
                finally:
                    renderer.close_group('fluxplot-raster')
            draw._supports_rasterization = True
            art.draw = draw
        yield
    finally:
        fig.suppressComposite = previous
        for art, had_draw, saved_draw in draws:
            if had_draw: art.draw = saved_draw
            else: del art.__dict__['draw']
        for it in items:
            it.artist.set_rasterized(it.was_rasterized)
        for ax, zorder, rasterized in axes_state:
            ax.set_rasterization_zorder(zorder)
            ax.set_rasterized(rasterized)


def reattach(root, items, warnings) -> set:
    """Re-attach each rasterized artist's gid to the ``<image>`` matplotlib emitted for it.

    Returns the set of gids restored (empty when the match was ambiguous and skipped).
    """
    if not items:
        return set()
    by_id = {el.get('id'): el for el in root.iter() if el.get('id')}
    restored = set()
    for it in items:
        group = by_id.get(it.gid)
        if group is None: continue
        images = list(group.iter(f'{{{SVG}}}image'))
        if len(images) == 1:
            # Preserve the existing contract: the layer ID names the image.
            group.attrib.pop('id', None)
            images[0].set('id', it.gid)
            images[0].set('data-rasterized', '1')
            restored.add(it.gid)
        elif images:
            group.set('data-rasterized', '1')
            for i, image in enumerate(images): image.set('id', f'{it.gid}.raster.{i}')
            restored.add(it.gid)
        elif not len(group):
            group.getparent().remove(group)
    return restored


def describe(items, *, plot_name: str, dpi: int, rasterized: bool) -> str:
    """One human sentence about what happened (or would happen) to the heavy layers."""
    listed = ", ".join(f"{it.label} ({it.count:,} primitives)" for it in items[:4])
    if len(items) > 4:
        listed += f", +{len(items) - 4} more"
    total = sum(it.count for it in items)
    if rasterized:
        return (
            f"fluxplot: '{plot_name}' — rasterized {len(items)} layer(s) at {dpi} dpi: "
            f"{listed}. Axes, ticks, labels and legend stay vector. "
            f"Pass force_vectors=True to keep everything as vectors."
        )
    return (
        f"fluxplot: '{plot_name}' — kept {len(items)} layer(s) as vectors "
        f"(force_vectors=True): {listed}. That is ~{total:,} SVG nodes; editors that inline "
        f"large vector layers can slow inline editors."
    )
