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
    from fluxplot.scene3d_viewer import png_preview, _layout, _Camera
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
    layout = _layout(man, 360, 360); camera = _Camera(sc._view, man['bounds'], layout['viewport'])
    center = camera.screen([0, 0, 0])[0][:2]; radius = camera.pixels_per_unit() * 1
    labels = [t for t in under.texts if t.get_text()]
    assert len(labels) >= 12
    for text in labels:
        assert np.linalg.norm(np.array(text.get_position()) - center) > radius


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
