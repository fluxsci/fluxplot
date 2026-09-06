"""Scientific identity and export lifecycle regressions from the 0.3 review."""
import gc
import json
import weakref
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from lxml import etree
import fluxplot as fp


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def save(tmp_path, fig, name='plot', **kwargs):
    result = fp.save(fig, str(tmp_path / name), recipe=False, **kwargs)
    raw = Path(result.manifest).read_text()
    manifest = json.loads(raw, parse_constant=lambda v: pytest.fail(f'illegal JSON {v}'))
    root = etree.parse(result.svg)
    ids = root.xpath('//@id')
    assert len(ids) == len(set(ids)), 'SVG IDs must be unique'
    def walk(node):
        for ref in node.get('members', []) + ([node['ref']] if 'ref' in node else []):
            assert ref in ids, ref
        for child in node.get('children', []): walk(child)
    walk(manifest['parts'])
    return manifest, root, result


def test_precision_gaps_and_atomic_invalid_recipe(tmp_path):
    fig, ax = plt.subplots()
    fp.line(ax, [1e-6, 2e-6, 3e-6], [1e-9, np.nan, 3e-9], series='small', marker='o')
    man, root, result = save(tmp_path, fig)
    assert man['series'][0]['data'] == {'x': [1e-6, 2e-6, 3e-6], 'y': [1e-9, None, 3e-9]}
    assert [p['index'] for p in man['series'][0]['points']] == [0, 2]
    assert 'small.point.1' not in root.xpath('//@id')
    before = [Path(p).read_bytes() for p in (result.svg, result.manifest, result.recipe)]
    with pytest.raises(ValueError):
        fp.save(fig, str(tmp_path / 'plot'), recipe={'params': {'dose': np.inf}})
    assert before == [Path(p).read_bytes() for p in (result.svg, result.manifest, result.recipe)]


def test_resave_live_data_removed_artists_and_repeated_roles(tmp_path):
    fig, ax = plt.subplots()
    line = fp.line(ax, [1, 2], [3, 4], series='s')
    ax.set_title('title')
    ax.text(1, 3, 'note')
    removed = fp.annotation(ax, text='remove', xy=(1, 3), name='old')
    for i in range(3): fp.tag(ax.axvspan(i, i + .2), series='s', role='x-night')
    a, _, _ = save(tmp_path, fig)
    b, _, _ = save(tmp_path, fig)
    assert a == b
    line.set_ydata([9, 10])
    removed.remove()
    c, _, _ = save(tmp_path, fig)
    assert c['series'][0]['data']['y'] == [9, 10]
    assert not any(o.get('name') == 'old' for o in c['overlays'])
    night = [c['svgId'] for c in c['series'][0]['components'] if c['role'] == 'x-night']
    assert len(night) == 3
    assert set(night) <= set(c['build']['order'])


def test_registry_does_not_retain_closed_figure():
    fig, ax = plt.subplots()
    fp.scatter(ax, [0], [1], series='s')
    ref = weakref.ref(fig)
    plt.close(fig)
    del fig, ax
    gc.collect()
    assert ref() is None


def test_marker_style_and_source_indices(tmp_path):
    fig, ax = plt.subplots()
    _, markers = fp.line(ax, range(5), range(5), series='s', marker='o', markersize=2,
        markerfacecolor='red', markeredgecolor='green', alpha=.2, zorder=9, markevery=2)
    assert markers.get_markersize() == 2
    assert markers.get_markerfacecolor() == 'red'
    assert markers.get_markeredgecolor() == 'green'
    assert markers.get_alpha() == .2 and markers.get_zorder() == 9
    man, _, _ = save(tmp_path, fig)
    assert [p['index'] for p in man['series'][0]['points']] == [0, 2, 4]


def test_units_scalars_and_horizontal_baselines(tmp_path):
    fig, axes = plt.subplots(1, 3)
    fp.barh(axes[0], ['a', 'b'], [2, 3], left=4, series='s')
    fp.scatter(axes[1], 1, 2, series='s')
    fp.errorbar(axes[1], [1, 2], [2, 3], yerr=.2, series='err')
    fp.line(axes[2], [datetime(2020, 1, 1) + timedelta(days=i) for i in range(2)], [1, 2], series='s')
    man, _, _ = save(tmp_path, fig)
    bars = next(s for s in man['series'] if s['kind'] == 'bar')
    assert bars['data']['x'] == [6, 7]
    assert bars['bar'] == {'orientation': 'horizontal', 'baseline': [4, 4], 'length': [2, 3]}
    assert man['axes'][0]['y']['units']['kind'] == 'category'
    assert man['axes'][2]['x']['units']['kind'] == 'date'


def test_named_panels_twins_insets_and_final_coordinates(tmp_path):
    fig, ax = plt.subplots(layout='constrained')
    twin = ax.twinx()
    inset = ax.inset_axes([.2, .5, .25, .3])
    for name, owner in [('main', ax), ('twin', twin), ('detail', inset)]:
        fp.panel(owner, name)
        fp.scatter(owner, [1, 2], [100, 200], series='same')
    man, root, _ = save(tmp_path, fig)
    assert len(man['axes']) == len(man['series']) == 3
    assert {s['id'] for s in man['series']} == {'panel.main.same', 'panel.twin.same', 'panel.detail.same'}
    for series in man['series']:
        axis = next(a for a in man['axes'] if a['panelId'] == series['panelId'])
        for point in series['points']:
            node = root.xpath('//*[@id=$id]', id=point['svgId'])[0]
            for coord in ('x', 'y'):
                a, b = axis[coord]['anchors'][0], axis[coord]['anchors'][-1]
                projected = a['svg'] + (point[coord] - a['data']) / (b['data'] - a['data']) * (b['svg'] - a['svg'])
                assert float(node.get(coord)) == pytest.approx(projected, abs=2e-5)


@pytest.mark.parametrize('projection', ['polar', '3d'])
def test_unsupported_projection_is_honest(tmp_path, projection):
    fig = plt.figure()
    ax = fig.add_subplot(projection=projection)
    pts = ax.scatter([1, 2, 3], [3, 2, 1], **({'zs': [4, 1, 2]} if projection == '3d' else {}))
    fp.tag_points(pts, series='s')
    man, _, _ = save(tmp_path, fig)
    assert man['axes'][0]['x']['supported'] is False
    assert man['axes'][0]['x']['anchors'] == []
    assert not man['series'][0]['capabilities']['dataMorph']
    if projection == '3d': assert not man['series'][0].get('points')


def test_field_cells_masks_colorbar_and_regeneration(tmp_path, monkeypatch):
    fig, ax = plt.subplots()
    monkeypatch.setenv('FLUX_PARAMS', json.dumps({'__fluxplot__': {'matrix': {'cmap': 'plasma', 'vmin': -2, 'vmax': 8}}}))
    image = fp.heatmap(ax, [[1, np.nan], [3, 4]], series='matrix', key='matrix', cells=True, include_values=True)
    fp.colorbar(image, name='intensity', label='Intensity')
    man, root, result = save(tmp_path, fig)
    field = man['series'][0]['field']
    assert field['values'] == [[1, None], [3, 4]]
    assert field['maskedIndices'] == [1]
    assert field['normalization']['vmin'] == -2 and field['cmap'] == 'plasma'
    assert 'matrix.x-heatmap.cell.1.0' in root.xpath('//@id')
    key = next(g for g in man['guides'] if g['role'] == 'colorbar')
    assert key['mappable'] == 'matrix.x-heatmap'
    assert key['label'] == 'Intensity'
    assert json.loads(Path(result.recipe).read_text())['params']['__fluxplot__']['matrix']['vmax'] == 8


def test_contours_and_large_field_are_bounded(tmp_path):
    fig, axes = plt.subplots(1, 2)
    x, y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    fp.contourf(axes[0], x, y, x*x + y*y, levels=[0, 1, 3, 8], series='field')
    fp.heatmap(axes[1], np.arange(10000).reshape(100, 100), series='field', cells=True)
    man, root, result = save(tmp_path, fig)
    assert man['series'][0]['field']['levels'] == [0, 1, 3, 8]
    assert any('.level.' in s for s in root.xpath('//@id'))
    assert man['series'][1]['rasterized']
    assert result.rasterized
    assert len(root.xpath('//*')) < 800


def test_raster_empty_layer_does_not_steal_identity_and_restores_state(tmp_path):
    fig, ax = plt.subplots()
    empty = fp.scatter(ax, [np.nan], [np.nan], series='empty', rasterized=True)
    visible = fp.scatter(ax, [1, 2], [3, 4], series='visible', rasterized=True)
    man, root, result = save(tmp_path, fig)
    assert result.rasterized == ['visible.points']
    assert [s['id'] for s in man['series']] == ['visible']
    assert visible.get_rasterized() and empty.get_rasterized()
    assert 'draw' not in visible.__dict__
    man, root, result = save(tmp_path, fig, force_vectors=True)
    assert not root.xpath('//*[local-name()="image"]')
    assert visible.get_rasterized() and not result.rasterized


def test_seaborn_hue_order_and_capped_horizontal_errors(tmp_path):
    import seaborn as sns
    import pandas as pd
    data = pd.DataFrame({'value': [1, 2, 2, 3, 11, 12, 12, 13], 'group': ['low']*4 + ['high']*4})
    fig, ax = plt.subplots()
    sns.histplot(data=data, x='value', hue='group', bins=[0, 5, 10, 15], ax=ax)
    fp.tag_seaborn(ax, plot='histplot')
    man, _, _ = save(tmp_path, fig)
    counts = {s['name']: s['bar']['length'] for s in man['series']}
    assert counts['low'] == [4, 0, 0] and counts['high'] == [0, 0, 4]
    fig, ax = plt.subplots()
    sns.barplot(data=data, x='value', y='group', capsize=.3, errorbar='sd', ax=ax)
    fp.tag_seaborn(ax, plot='barplot')
    man, _, _ = save(tmp_path, fig)
    assert sum(c['role'] == 'errorbar' for s in man['series'] for c in s['components']) == 2
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x='value', y='value', hue='group', ax=ax, errorbar=None)
    with pytest.warns(UserWarning, match='multi-hue identity'):
        assert fp.tag_seaborn(ax) == {}


def test_explicit_recipe_invocation_and_targeted_skip(tmp_path, monkeypatch):
    fig, ax = plt.subplots()
    fp.scatter(ax, 1, 2, series='s')
    result = fp.save(fig, str(tmp_path / 'target'), recipe={
        'script': __file__, 'command': 'python-custom', 'args': ['run.py', '--input', 'data.csv'],
        'cwd': '..', 'output': 'custom.svg', 'params': {'dose': 1e-7}})
    recipe = json.loads(Path(result.recipe).read_text())
    assert recipe['args'] == ['run.py', '--input', 'data.csv']
    assert recipe['cwd'] == '..' and recipe['output'] == 'custom.svg'
    assert recipe['params']['dose'] == 1e-7
    before = Path(result.svg).read_bytes()
    monkeypatch.setenv('FLUXPLOT_ONLY', 'another-plot')
    assert fp.save(fig, str(tmp_path / 'target')).skipped
    assert Path(result.svg).read_bytes() == before


def test_surface_shading_survives_render(tmp_path):
    from test_surface import _sphere
    verts, faces = _sphere()
    colors = []
    for shading in (0, .8):
        fig, ax = plt.subplots()
        artists = fp.surface(ax, verts[:, 2], series='brain', surfaces={'left': (verts, faces)},
            hemispheres=['left'], views=['lateral'], kind='continuous', shading=shading,
            cmap='viridis', colorbar=True)
        before = artists[0].get_facecolors().copy()
        save(tmp_path, fig, name=f'surface-{shading}', force_vectors=True)
        np.testing.assert_allclose(artists[0].get_facecolors(), before)
        colors.append(before)
    assert not np.allclose(colors[0], colors[1])


def test_failed_export_restores_artist_and_layout_state(tmp_path, monkeypatch):
    from fluxplot import render
    fig, ax = plt.subplots(layout='constrained')
    cloud = fp.scatter(ax, [1, 2], [3, 4], series='s', rasterized=True)
    cloud.set_gid('caller-id')
    ax.set_rasterization_zorder(3)
    before = fig.canvas, fig.dpi, fig.get_layout_engine(), fig.suppressComposite
    def fail(*args, **kwargs): raise RuntimeError('render failed')
    monkeypatch.setattr(render, 'render_svg', fail)
    with pytest.raises(RuntimeError, match='render failed'): save(tmp_path, fig)
    assert (fig.canvas, fig.dpi, fig.get_layout_engine(), fig.suppressComposite) == before
    assert cloud.get_gid() == 'caller-id' and cloud.get_rasterized()
    assert ax.get_rasterization_zorder() == 3
    assert 'draw' not in cloud.__dict__
    assert not list(tmp_path.iterdir())


def test_full_precision_svg_metadata_and_duplicate_ids(tmp_path):
    fig, ax = plt.subplots()
    x = 1.234567890123456e-7
    fp.scatter(ax, [x], [x / 2], series='s')
    _, root, _ = save(tmp_path, fig)
    assert float(root.xpath('//*[@id="s.point.0"]')[0].get('data-x')) == x
    for y in (0, 1): ax.axhline(y).set_gid('duplicate')
    with pytest.raises(ValueError, match='duplicate SVG IDs'):
        save(tmp_path, fig, name='bad')
    assert not (tmp_path / 'bad.svg').exists()
