"""3D fluxplot API review fixes: colours, references, labels, states, views, errors, units."""
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import matplotlib as mpl
import fluxplot as fp
from fluxplot.glb import write_glb
from fluxplot.tagger import registry_for
from generate_scene3d_fixtures import sphere
from test_scene3d import read_glb


def cycle():
    return [mpl.colors.to_hex(c) for c in mpl.rcParams['axes.prop_cycle'].by_key()['color']]


def manifest(sc, tmp_path, name='scene'):
    return json.loads(Path(fp.save(sc, tmp_path / name, recipe=False).manifest).read_text())


# ---- 1. colour cycle ---------------------------------------------------------------------
def test_named_parts_without_palette_advance_the_colour_cycle():
    v, f = sphere(); sc = fp.scene3d()
    parts = fp.mesh3d(sc, {'soma': (v, f), 'axon': (v + 2, f), 'dendrites': (v + 4, f)}, series='neuron', legend=True)
    colors = [p.color for p in parts]
    assert colors == cycle()[:3]
    assert len(set(colors)) == 3
    later = fp.mesh3d(sc, (v, f), series='next')
    assert later[0].color == cycle()[3]
    # A partial palette fixes only its keys; the rest keep their cycle position.
    sc = fp.scene3d()
    parts = fp.mesh3d(sc, {'a': (v, f), 'b': (v, f)}, series='s', palette={'a': '#123456'})
    assert [p.color for p in parts] == ['#123456', cycle()[1]]


# ---- 2. share_topology_with a returned part list -----------------------------------------
def test_share_topology_with_the_list_mesh3d_returned():
    v, f = sphere()
    a = fp.scene3d(); reference = fp.mesh3d(a, {'left': (v, f), 'right': (v + [3, 0, 0], f)}, series='cortex')
    b = fp.scene3d()
    fp.mesh3d(b, {'right': (v * 1.2 + [3, 0, 0], f), 'left': (v * 2, f)}, series='cortex', share_topology_with=reference)
    assert fp.can_morph(a, b)
    single = fp.mesh3d(fp.scene3d(), (v, f), series='cell')
    fp.mesh3d(fp.scene3d(), (v * 3, f), series='cell', share_topology_with=single)
    with pytest.raises(ValueError, match=r'has no part with this id.*same series='):
        fp.mesh3d(fp.scene3d(), (v, f), series='other', share_topology_with=single)


# ---- 3. label semantics ------------------------------------------------------------------
def test_label_names_a_single_part_but_the_series_of_a_multi_part_call(tmp_path):
    v, f = sphere(); sc = fp.scene3d()
    fp.mesh3d(sc, (v, f), series='cell', label='Cell 1', legend=True)
    fp.mesh3d(sc, {'soma': (v + 3, f), 'axon': (v + 6, f)}, series='neuron', label='Neuron 7', legend=True)
    parts = {p['id']: p for p in manifest(sc, tmp_path)['parts']}
    assert parts['cell.mesh']['label'] == 'Cell 1'
    assert parts['neuron.soma']['label'] == 'soma' and parts['neuron.axon']['label'] == 'axon'
    marks = {m.gid: m for m in registry_for(sc).marks}
    assert marks['neuron.soma'].label == marks['neuron.axon'].label == 'Neuron 7'
    labels = np.where(v[:, 1] > 0, 0, 1)
    atlas = fp.scene3d()
    fp.surface3d(atlas, labels, series='atlas', surfaces=(v, f), kind='label', categories={0: 'frontal', 1: 'parietal'}, label='Atlas')
    assert [p['label'] for p in manifest(atlas, tmp_path, 'atlas')['parts'] if p.get('series') == 'atlas'] == ['frontal', 'parietal']
    values = v[:, 1].copy(); values[0] = np.nan
    field = fp.scene3d()
    fp.surface3d(field, values, series='h', surfaces=(v, f), kind='continuous', label='Height')
    got = {p['id']: p['label'] for p in manifest(field, tmp_path, 'field')['parts']}
    assert got == {'h.field': 'Height', 'h.missing': 'missing'}


# ---- 4. negative category codes (the 2D half lives in test_surface.py) -------------------
def test_negative_category_codes_never_collide_or_shift():
    v, f = sphere()
    both = np.where(v[:, 1] > 0, -1, 1)
    only_positive = np.ones(len(v))
    sc = fp.scene3d(); fp.surface3d(sc, both, series='atlas', surfaces=(v, f), kind='label')
    assert sorted(p.id for p in sc.parts) == ['atlas.category-1', 'atlas.category-m1']
    alone = fp.scene3d(); fp.surface3d(alone, only_positive, series='atlas', surfaces=(v, f), kind='label')
    assert [p.id for p in alone.parts] == ['atlas.category-1']


# ---- 5. continuous fields stay out of the legend -----------------------------------------
def test_continuous_field_never_gets_a_legend_swatch():
    v, f = sphere(); values = v[:, 1].copy(); values[0] = np.nan
    sc = fp.scene3d()
    with pytest.warns(UserWarning, match='no legend swatch'):
        fp.surface3d(sc, values, series='h', surfaces=(v, f), kind='continuous', legend=True, colorbar=True)
    assert sc._legend_entries == []
    import warnings
    other = fp.scene3d()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fp.surface3d(other, values, series='h', surfaces=(v, f), kind='continuous', legend=True, legend_missing=True)
    assert other._legend_entries == ['h.missing']


# ---- 7. error messages ------------------------------------------------------------------
def test_state_errors_name_the_state_part_and_counts():
    v, f = sphere()
    with pytest.raises(ValueError, match=rf"state 'inflated' of neuron\.mesh has {len(v) - 1} vertices but its base mesh has {len(v)}"):
        fp.mesh3d(fp.scene3d(), (v, f), series='neuron', states={'inflated': v[:-1]})
    with pytest.raises(ValueError, match=r"shape state 'inflated' has no shape for part 'axon'"):
        fp.mesh3d(fp.scene3d(), {'soma': (v, f), 'axon': (v + 3, f)}, series='neuron',
                  states={'inflated': {'soma': v * 2}})


def test_same_mesh_with_different_labels_says_the_labels_differ():
    v, f = sphere()
    a = fp.scene3d(); fp.surface3d(a, np.where(v[:, 1] > 0, 0, 1), series='atlas', surfaces=(v, f), kind='label')
    with pytest.raises(ValueError, match='category labels differ'):
        fp.surface3d(fp.scene3d(), np.where(v[:, 0] > 0, 0, 1), series='atlas', surfaces=(v, f), kind='label',
                     share_topology_with=a)


# ---- 8. numeric input ------------------------------------------------------------------
def test_numpy_integer_budgets_and_positive_scalebar():
    v, f = sphere()
    sc = fp.scene3d(preview_max_faces=np.int64(5000))
    fp.mesh3d(sc, (v, f), series='s', max_faces=np.int32(len(f)))
    fp.surface3d(sc, v[:, 1], series='h', surfaces=(v, f), kind='continuous', max_faces=np.int64(len(f)))
    for bad in (True, 2.5, 0):
        with pytest.raises(ValueError, match='positive integer'):
            fp.mesh3d(fp.scene3d(), (v, f), series='s', max_faces=bad)
    with pytest.raises(ValueError, match='scalebar length must be positive'):
        fp.scene3d(scalebar=0)


# ---- A1. Copy view round trip -----------------------------------------------------------
def test_view_accepts_every_copy_view_form(tmp_path):
    v, f = sphere(); sc = fp.scene3d(); fp.mesh3d(sc, (v, f), series='s', states={'inflated': v * 1.3})
    sc.view({'azimuth': 12, 'elevation': 34, 'zoom': 1.1})
    assert (sc._view['azimuth'], sc._view['elevation'], sc._view['zoom']) == (12, 34, 1.1)
    sc.view(view=dict(azimuth=18.5, elevation=24, zoom=0.99, states={'inflated': 0.25}))
    assert sc._view['azimuth'] == 18.5 and sc._view['states'] == {'inflated': .25}
    sc.view({'azimuth': 5}, azimuth=7)  # keywords win over the mapping
    assert sc._view['azimuth'] == 7
    code = 'sc.view(azimuth=40, elevation=10, zoom=0.8, states={"inflated": 0.5})'
    eval(code, {'sc': sc})
    assert manifest(sc, tmp_path)['view']['states'] == {'inflated': .5}
    with pytest.raises(ValueError, match='unknown view keys'):
        sc.view({'azimut': 3})
    seq = fp.scene3d(); fp.mesh3d(seq, (v, f), series='cell', states=[v * (1 + i / 10) for i in range(1, 5)], sequence=True)
    seq.view(azimuth=10, elevation=5, zoom=1, frame=np.int64(3))
    assert seq._view['states'] == {'frame-3': 1.0}


def test_scene3d_view_mapping_defers_states_until_meshes_exist(tmp_path):
    v, f = sphere()
    sc = fp.scene3d(view=dict(azimuth=90, elevation=10, zoom=1.2, states={'inflated': .4}))
    assert sc._view['azimuth'] == 90
    fp.mesh3d(sc, (v, f), series='s', states={'inflated': v * 1.2})
    assert sc._view['states'] == {'inflated': .4}
    seq = fp.scene3d(view={'frame': 2.5})
    fp.mesh3d(seq, (v, f), series='s', states=[v * 1.1, v * 1.2, v * 1.3], sequence=True)
    assert seq._view['states'] == {'frame-2': .5, 'frame-3': .5}
    orphan = fp.scene3d(view={'states': {'missing': 1}}); fp.mesh3d(orphan, (v, f), series='s')
    with pytest.raises(ValueError, match="unknown shape states \\['missing'\\]"):
        fp.save(orphan, tmp_path / 'orphan', recipe=False)


# ---- A2. bare vertex-array states -------------------------------------------------------
def test_states_accept_bare_vertex_arrays():
    v, f = sphere()
    sc = fp.scene3d(); fp.mesh3d(sc, (v, f), series='s', states={'inflated': v * 1.5, 'mesh': (v * 2, f)})
    np.testing.assert_allclose(sc.parts[0].states['inflated'], v * 1.5)
    seq = fp.scene3d(); fp.mesh3d(seq, (v, f), series='s', states=np.stack([v * 1.1, v * 1.2]), sequence=True)
    assert seq.state_names == ['frame-1', 'frame-2']
    named = fp.scene3d()
    fp.mesh3d(named, {'a': (v, f), 'b': (v + 3, f)}, series='n', states={'grown': {'a': v * 2, 'b': (v + 3) * 2}})
    np.testing.assert_allclose(named.parts[1].states['grown'], (v + 3) * 2)
    field = fp.scene3d()
    fp.surface3d(field, v[:, 1], series='h', surfaces=(v, f), kind='continuous', states={'inflated': v * 1.4})
    np.testing.assert_allclose(field.parts[0].states['inflated'], v * 1.4)


# ---- A3. discoverable signatures --------------------------------------------------------
def test_scene3d_signature_matches_scene_class_and_docs_exist():
    ours = inspect.signature(fp.scene3d).parameters
    theirs = dict(inspect.signature(fp.Scene3D.__init__).parameters); theirs.pop('self')
    assert list(ours) == list(theirs) and 'kwargs' not in ours
    assert all(ours[k].default == theirs[k].default for k in ours)
    for func in (fp.scene3d, fp.mesh3d, fp.surface3d):
        assert 'Parameters\n----------' in inspect.getdoc(func)
    assert 'build the reference first' in ' '.join(inspect.getdoc(fp.mesh3d).lower().split())
    assert 'Scene3D' in fp.save.__doc__ or 'scene3d' in fp.save.__doc__


# ---- A4. scale bar labels ---------------------------------------------------------------
@pytest.mark.parametrize('length,units,text', [(10_000, 'nm', '10 µm'), (1e6, 'nm', '1 mm'), (.5, 'µm', '500 nm'),
                                              (250, 'nm', '250 nm'), (5, 'cm', '5 cm'), (3, 'voxels', '3 voxels'),
                                              (1500, 'um', '1.5 mm'), (.25, '', '0.25')])
def test_scalebar_labels_use_friendly_metric_units(length, units, text):
    from fluxplot.scene3d import scalebar_text
    assert scalebar_text(length, units) == text


def test_scalebar_manifest_keeps_data_length_and_accepts_override(tmp_path):
    v, f = sphere()
    sc = fp.scene3d(units='nm', scalebar=10_000); fp.mesh3d(sc, (v * 5000, f), series='s')
    bar = next(p for p in manifest(sc, tmp_path)['parts'] if p['role'] == 'scalebar')
    assert bar['length'] == 10_000 and bar['label'] == '10 µm'
    custom = fp.scene3d(units='nm', scalebar=10_000, scalebar_label='10 µm (XY)'); fp.mesh3d(custom, (v, f), series='s')
    assert next(p for p in manifest(custom, tmp_path, 'c')['parts'] if p['role'] == 'scalebar')['label'] == '10 µm (XY)'


# ---- A5. alpha --------------------------------------------------------------------------
def test_alpha_sets_part_opacity_for_meshes_and_fields(tmp_path):
    v, f = sphere(); sc = fp.scene3d()
    fp.mesh3d(sc, (v, f), series='shell', color='#4385BE', alpha=.25)
    fp.surface3d(sc, v[:, 1], series='h', surfaces=(v + 3, f), kind='continuous', alpha=.5)
    doc, _ = read_glb(write_glb(sc))
    shell, field = doc['materials']
    assert shell['alphaMode'] == 'BLEND' and shell['pbrMetallicRoughness']['baseColorFactor'][3] == pytest.approx(.25, abs=1/255)
    assert field['alphaMode'] == 'BLEND' and field['pbrMetallicRoughness']['baseColorFactor'][:3] == [1, 1, 1]
    assert field['pbrMetallicRoughness']['baseColorFactor'][3] == pytest.approx(.5, abs=1/255)
    colors = {p['id']: p['color'] for p in manifest(sc, tmp_path)['parts'] if 'color' in p}
    assert colors['shell.mesh'] == '#4385be40' and colors['h.field'] == '#ffffff80'
    with pytest.raises(ValueError, match='alpha'):
        fp.mesh3d(fp.scene3d(), (v, f), series='s', alpha=1.5)


# ---- A6. palette keys -------------------------------------------------------------------
def test_unknown_palette_keys_warn():
    v, f = sphere()
    with pytest.warns(UserWarning, match=r"palette keys \['axom'\]"):
        fp.mesh3d(fp.scene3d(), {'soma': (v, f), 'axon': (v + 2, f)}, series='n', palette={'soma': 'red', 'axom': 'blue'})
    labels = np.where(v[:, 1] > 0, 0, 1)
    with pytest.warns(UserWarning, match='palette codes need categories'):
        fp.surface3d(fp.scene3d(), labels, series='a', surfaces=(v, f), kind='label', palette={0: 'red', 7: 'blue'},
                     categories={0: 'frontal', 1: 'parietal'})
    with pytest.warns(UserWarning, match='use cmap='):
        fp.surface3d(fp.scene3d(), v[:, 1], series='h', surfaces=(v, f), kind='continuous', palette={'field': 'red'})


# ---- A7 / A12 ---------------------------------------------------------------------------
def test_preview_budget_default():
    assert fp.scene3d().preview_max_faces == 100_000


def test_colormap_names_resolve_through_fluxplot_collections():
    v, f = sphere(); sc = fp.scene3d()
    fp.surface3d(sc, v[:, 1], series='h', surfaces=(v, f), kind='continuous', cmap='emerald')
    spec = registry_for(sc).marks[0].data['scene3d']['field']
    assert spec['cmap']['name'] == 'cmasher.emerald'
    with pytest.raises(ValueError, match='unknown colormap'):
        fp.surface3d(fp.scene3d(), v[:, 1], series='h', surfaces=(v, f), kind='continuous', cmap='not-a-map')


def test_trimesh_scene_becomes_named_parts():
    trimesh = pytest.importorskip('trimesh')
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(), node_name='Soma', geom_name='g1')
    scene.add_geometry(trimesh.creation.icosphere(1), node_name='Axon', geom_name='g2',
                       transform=trimesh.transformations.translation_matrix([5, 0, 0]))
    sc = fp.scene3d(); parts = fp.mesh3d(sc, scene, series='neuron', legend=True)
    assert [p.id for p in parts] == ['neuron.soma', 'neuron.axon']
    assert parts[1].vertices[:, 0].mean() == pytest.approx(5)
    assert parts[0].color != parts[1].color


def test_repr_axis_warning_and_clean_ticks(tmp_path):
    v, f = sphere(); sc = fp.scene3d(units='µm', scalebar=.5)
    fp.mesh3d(sc, {'a': (v, f), 'b': (v + 2, f)}, series='n', states={'grown': {'a': v * 2, 'b': v * 2 + 2}})
    text = repr(sc)
    assert 'n.a' in text and 'n.b' in text and '2 parts' in text and '1 states' in text and '500 nm' in text
    with pytest.warns(UserWarning, match="axes='none'"):
        sc.axis('x', label='width')
    box = fp.scene3d(axes='box'); fp.mesh3d(box, (v * [.35, 1, 1] + [.35, 0, 0], f), series='s')
    ticks = manifest(box, tmp_path)['axes']['x']['ticks']
    assert ticks == [round(t, 10) for t in ticks] and 0.6 in ticks


# ---- C7. FLUXPLOT_ONLY reports 3D skips like 2D ------------------------------------------
def test_only_filter_reports_3d_skip_on_stderr(tmp_path, monkeypatch, capsys):
    v, f = sphere(); sc = fp.scene3d(); fp.mesh3d(sc, (v, f), series='s')
    monkeypatch.setenv('FLUXPLOT_ONLY', 'keep*')
    assert fp.save(sc, tmp_path / 'skipme', recipe=False).skipped
    assert "fluxplot: skipped 'skipme' (FLUXPLOT_ONLY=keep*)" in capsys.readouterr().err
