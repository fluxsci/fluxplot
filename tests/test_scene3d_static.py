"""The notebook still of a Scene3D: HiDPI metadata, Flux furniture layout, translucency."""
import io

import numpy as np
import pytest
import fluxplot as fp
from fluxplot.glb import write_glb
from generate_scene3d_fixtures import sphere


def _capture(monkeypatch):
    from matplotlib.figure import Figure
    captured = []; original = Figure.savefig
    def capture(fig, *args, **kwargs):
        captured.append(fig)
        return original(fig, *args, **kwargs)
    monkeypatch.setattr(Figure, 'savefig', capture)
    return captured


def test_png_is_hidpi_with_display_metadata():
    from PIL import Image
    v, f = sphere(); sc = fp.scene3d(figsize=(3.5, 3)); fp.mesh3d(sc, (v, f), series='s')
    data, metadata = sc._repr_mimebundle_()
    assert metadata['image/png'] == {'width': 420, 'height': 360}
    assert Image.open(io.BytesIO(data['image/png'])).size == (840, 720)
    assert 'width="420" height="360"' in data['text/html']


def test_png_triad_and_scalebar_use_distinct_corners(monkeypatch):
    from fluxplot.scene3d_viewer import png_preview
    captured = _capture(monkeypatch)
    v, f = sphere(); sc = fp.scene3d(figsize=(3.5, 3), axes='triad', scalebar=.5, units='µm')
    fp.mesh3d(sc, (v, f), series='s'); png_preview(sc)
    over = max(captured[-1].axes, key=lambda a: a.get_zorder())
    texts = {t.get_text(): t.get_position() for t in over.texts}
    width = 3.5 * 120
    assert texts['500 nm'][0] > width / 2
    assert all(texts[k][0] < width / 2 for k in 'xyz')


def test_png_box_axes_draw_panes_behind_and_labels_outside_the_mesh(monkeypatch):
    from fluxplot.scene3d_viewer import png_preview, _layout, _Camera, _framing_bounds
    captured = _capture(monkeypatch)
    v, f = sphere(); sc = fp.scene3d(figsize=(3, 3), axes='box'); fp.mesh3d(sc, (v, f), series='s')
    png_preview(sc)
    fig = captured[-1]
    mesh_ax = fig.axes[0]
    under = min(fig.axes, key=lambda a: a.get_zorder())
    assert under.get_zorder() < mesh_ax.get_zorder()
    assert len(under.patches) >= 3  # the back panes sit behind the mesh
    from fluxplot.scene3d_manifest import build_manifest
    man = build_manifest(sc, write_glb(sc), 'x.glb')
    layout = _layout(man, 360, 360); camera = _Camera(sc._view, _framing_bounds(sc, man), layout['viewport'])
    center = camera.screen([0, 0, 0])[0][:2]; radius = camera.pixels_per_unit() * 1
    labels = [t for t in under.texts if t.get_text()]
    assert len(labels) >= 12
    for text in labels:
        assert np.linalg.norm(np.array(text.get_position()) - center) > radius


def test_png_frames_like_flux_tight_sphere_for_meshes_whole_box_for_box_axes():
    """Same rule as Flux framing.ts/glbCore: a mesh frames its tight vertex sphere,
    box axes frame the whole axes box, so the still matches the imported poster."""
    from fluxplot.scene3d_viewer import _layout, _Camera, _framing_bounds, _rotation
    from fluxplot.scene3d_manifest import build_manifest
    v, f = sphere()
    bare = fp.scene3d(figsize=(3, 3)); fp.mesh3d(bare, (v, f), series='s')
    man = build_manifest(bare, write_glb(bare), 'x.glb'); framed = _framing_bounds(bare, man)
    assert framed['radius'] == pytest.approx(np.linalg.norm(v, axis=1).max())
    assert framed['radius'] < np.linalg.norm(np.subtract(man['bounds']['max'], man['bounds']['min'])) / 2
    boxed = fp.scene3d(figsize=(3, 3), axes='box'); fp.mesh3d(boxed, (v, f), series='s')
    boxed.axis('x', lim=(-3, 3))
    man = build_manifest(boxed, write_glb(boxed), 'x.glb'); framed = _framing_bounds(boxed, man)
    assert 'radius' not in framed and framed['min'][0] == pytest.approx(-3) and framed['max'][0] == pytest.approx(3)
    vp = _layout(man, 288, 288)['viewport']
    limits = [man['axes'][k]['lim'] for k in 'xyz']
    corners = np.array([[limits[i][(m >> i) & 1] for i in range(3)] for m in range(8)]) @ _rotation(man).T
    for azimuth in (0, 30, 45, 135, 300):
        for elevation in (-35, 20, 60):
            for projection in ('orthographic', 'perspective'):
                view = {**boxed._view, 'azimuth': azimuth, 'elevation': elevation, 'projection': projection}
                xy = _Camera(view, framed, vp).screen(corners)[:, :2]
                assert (xy[:, 0] >= vp['x']).all() and (xy[:, 0] <= vp['x'] + vp['width']).all()
                assert (xy[:, 1] >= vp['y']).all() and (xy[:, 1] <= vp['y'] + vp['height']).all()


def test_png_translucent_parts_have_no_edge_lattice(monkeypatch):
    from matplotlib.collections import PolyCollection
    from fluxplot.scene3d_viewer import png_preview
    captured = _capture(monkeypatch)
    v, f = sphere(); sc = fp.scene3d()
    fp.mesh3d(sc, (v, f), series='inner'); fp.mesh3d(sc, (v * 2, f), series='shell', alpha=.3)
    png_preview(sc)
    collection = next(c for c in captured[-1].axes[0].collections if isinstance(c, PolyCollection))
    face = collection.get_facecolor(); widths = np.broadcast_to(collection.get_linewidths(), (len(face),))
    translucent = face[:, 3] < 1
    assert translucent.sum() == len(f) and (~translucent).sum() == len(f)
    assert np.all(widths[translucent] == 0) and np.all(widths[~translucent] > 0)


def test_empty_scene_display_explains_what_to_do():
    from fluxplot.scene3d_viewer import mimebundle, png_preview
    sc = fp.scene3d()
    data, metadata = sc._repr_mimebundle_()
    assert set(data) == {'text/plain'} and 'nothing to show yet' in data['text/plain'] and 'fp.mesh3d' in data['text/plain']
    assert metadata == {}
    assert 'nothing to show yet' in mimebundle(sc, static=True)['text/plain']
    with pytest.raises(ValueError, match='nothing to show yet'):
        png_preview(sc)
    assert 'empty' in repr(sc) and 'fp.mesh3d' in repr(sc)


def test_guide_wrap_preserves_tokens_ticks_and_full_text(monkeypatch):
    from fluxplot.scene3d_viewer import _layout, _text_width, _wrap_words, _nice_ticks, png_preview
    from fluxplot.scene3d_manifest import build_manifest
    token = 'LongUnbrokenScientificIdentifierαβγ'
    assert ''.join(_wrap_words(token, 12, 48)) == token
    assert all(_text_width(line, 12) <= 48 for line in _wrap_words(token, 12, 48))
    assert _nice_ticks(-1, 1) == [-1, -.5, 0, .5, 1]
    assert _nice_ticks(1.21, 3.82) == [1.5, 2, 2.5, 3, 3.5]
    assert _nice_ticks(2, 2) == [2]
    v, f = sphere(); sc = fp.scene3d(figsize=(2.5, 4))
    fp.surface3d(sc, v[:, 2], series='Height', surfaces=(v, f), colorbar=True, cbar_label=token)
    manifest = build_manifest(sc, write_glb(sc), 'preview.glb')
    field = next(p for p in manifest['parts'] if isinstance(p.get('field'), dict))['field']
    field.pop('ticks', None)
    layout = _layout(manifest, 300, 480)
    assert not layout['overflow']
    assert ''.join(layout['colorbars'][0]['titleLines']) == token
    captured = _capture(monkeypatch); png_preview(sc, _manifest=manifest)
    texts = max(captured[-1].axes, key=lambda a: a.get_zorder()).texts
    actual = [t.get_text() for t in texts]
    assert all(line in actual for line in layout['colorbars'][0]['titleLines'])
    assert '0' in actual  # absent source ticks still get the same deterministic Flux ticks
    # Actual matplotlib text bounds, in figure pixels, fit this sufficiently sized box.
    captured[-1].canvas.draw(); renderer = captured[-1].canvas.get_renderer()
    for text in texts:
        bounds = text.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= captured[-1].bbox.width and bounds.y1 <= captured[-1].bbox.height


def test_guide_legend_reflows_and_impossible_boxes_are_explicit():
    from fluxplot.scene3d_viewer import _layout, _text_width
    token = 'LongUnbrokenScientificIdentifierαβγ'
    manifest = {'parts': [{'id': 'a', 'role': 'mesh', 'label': token},
                          {'id': 'b', 'role': 'mesh', 'label': 'Hidden label', 'hidden': True},
                          {'id': 'legend', 'role': 'legend', 'entries': ['a', 'b']}]}
    layout = _layout(manifest, 200, 300)
    rows = layout['legends'][0]['legendRows']
    assert len(rows) == 1 and ''.join(rows[0]['lines']) == token
    assert all(layout['legends'][0]['x'] + layout['fs'] * 1.5 + _text_width(line, layout['fs']) <= 200 for line in rows[0]['lines'])
    tiny = _layout(manifest, 60, 20)
    assert tiny['overflow'] and tiny['overflowParts'] == ['legend']
    assert ''.join(tiny['legends'][0]['legendRows'][0]['lines']) == token


def test_png_box_axis_seen_end_on_hides_crowded_tick_labels(monkeypatch):
    """Flux furniture.ts ``tickLabelsCollide``: an axis seen almost end-on projects to a stub
    too short for its tick labels. They (and a title longer than the stub) hide, while the
    axis line, tick marks and grid stay, so the still matches Flux's vector furniture."""
    from fluxplot.scene3d_viewer import (png_preview, _tick_labels_collide, _text_width, _draw_box_axes,
                                         _layout, _Camera, _framing_bounds)
    from fluxplot.scene3d_manifest import build_manifest
    space = _text_width(' ', 12)
    assert not _tick_labels_collide([], 12, 'middle') and not _tick_labels_collide([(0, 0, 50)], 12, 'middle')
    assert _tick_labels_collide([(0, 0, 10), (10 + space * .9, 0, 10)], 12, 'start')
    assert not _tick_labels_collide([(0, 0, 10), (10 + space * 1.1, 0, 10)], 12, 'start')
    assert _tick_labels_collide([(0, 0, 10), (0, 12 * .9, 10)], 12, 'end')
    assert not _tick_labels_collide([(0, 0, 10), (0, 12 * 1.1, 10)], 12, 'end')
    assert _tick_labels_collide([(0, 0, 40), (15, 24, 4), (30, 0, 40)], 12, 'middle')  # every pair, not neighbours

    v, f = sphere(); sc = fp.scene3d(figsize=(2.8, 2.4), axes='box'); fp.mesh3d(sc, (v, f), series='s')
    for k in 'xyz':
        sc.axis(k, lim=(-1, 1), ticks=[-1, 0, 1], label=f'{k} (µm)')
    captured = _capture(monkeypatch)

    def still(azimuth, elevation):
        sc.view(azimuth=azimuth, elevation=elevation); png_preview(sc)
        under = min(captured[-1].axes, key=lambda a: a.get_zorder())
        return [t.get_text() for t in under.texts], len(under.lines)
    near, near_lines = still(5, 0)
    wide, wide_lines = still(30, 0)
    assert sorted(wide) == sorted(['-1', '0', '1'] * 3 + ['x (µm)', 'y (µm)', 'z (µm)'])
    assert sorted(near) == sorted(['-1', '0', '1'] * 2 + ['x (µm)', 'y (µm)'])  # the z stub's labels hide
    assert near_lines == wide_lines  # its axis line, tick marks and grid remain

    class Record:
        """Records _draw_box_axes output instead of drawing it."""
        muted = '#000'; lw = 1
        def __init__(self): self.texts = []; self.lines = []
        def pt(self, px): return px
        def polygon(self, *args, **kwargs): pass
        def line(self, a, b, **kwargs): self.lines.append((tuple(a[:2]), tuple(b[:2])))
        def text(self, x, y, label, **kwargs): self.texts.append(label)
    man = build_manifest(sc, write_glb(sc), 'x.glb'); layout = _layout(man, 336, 288)
    # A short title fits along the stub and stays; the long one hides with the ticks.
    short = {**man, 'parts': [dict(p, text='z') if p['id'] == 'axes.z.label' else p for p in man['parts']]}
    states = []
    for azimuth in np.arange(0, 30.5, .5):
        view = {**sc._view, 'azimuth': float(azimuth), 'elevation': 0.0}
        camera = _Camera(view, _framing_bounds(sc, man), layout['viewport'])
        record = Record(); _draw_box_axes(record, man, camera, layout['fs'])
        states.append(record.texts.count('-1') == 3)
        if azimuth == 5:
            assert 'z (µm)' not in record.texts
            short_record = Record(); _draw_box_axes(short_record, short, camera, layout['fs'])
            assert 'z' in short_record.texts
    assert not states[0] and states[-1]
    assert sum(a != b for a, b in zip(states, states[1:])) == 1  # one hide→show change, no flicker
