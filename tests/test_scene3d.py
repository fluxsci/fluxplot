"""Production writer behavior, cross-engine fixture semantics and adverse input checks."""
import json
import struct
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import matplotlib.pyplot as plt
import fluxplot as fp
from fluxplot.glb import write_glb, vertex_normals
from fluxplot.mesh3d import load_mesh
from fluxplot.tagger import registry_for
from generate_scene3d_fixtures import sphere


def read_glb(data):
    n=struct.unpack_from('<I',data,12)[0]
    doc=json.loads(data[20:20+n]); binary=data[28+n:]
    def accessor(index):
        a=doc['accessors'][index]; view=doc['bufferViews'][a['bufferView']]
        dtype={5121:'u1',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]
        count={'SCALAR':1,'VEC3':3,'VEC4':4}[a['type']]
        item=np.dtype(dtype).itemsize
        out=np.ndarray((a['count'],count),dtype=dtype,buffer=binary,offset=view.get('byteOffset',0)+a.get('byteOffset',0),strides=(view.get('byteStride',count*item),item)).copy()
        return out[:,0] if count==1 else out
    return doc,accessor


def scene(**kw):
    sc=fp.scene3d(**kw); v,f=sphere(); fp.mesh3d(sc,(v,f),series='neuron'); return sc


def test_save_roundtrip_determinism_and_registry(tmp_path):
    sc=scene(title='Neuron',axes='box',scalebar=.5)
    a=fp.save(sc,tmp_path/'neuron',recipe=False,_now='2000-01-01T00:00:00Z')
    first={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    fp.save(sc,tmp_path/'neuron.glb',recipe=False,_now='2000-01-01T00:00:00Z')
    assert first=={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    man=json.loads(Path(a.manifest).read_text()); recipe=json.loads(Path(a.recipe).read_text())
    assert recipe['outputs']=={'glb':'neuron.glb','manifest':'neuron.fluxplot.json'}
    assert registry_for(sc).marks[0].gid=='neuron.mesh'
    assert {'neuron.mesh','title','scalebar','axes.x.label'} <= {p['id'] for p in man['parts']}
    assert man['size']=={'width':3.5,'height':3.,'unit':'in'}


@pytest.mark.parametrize('up,expected',[('y',[1,2,3]),('z',[1,3,-2]),('-y',[1,-2,-3]),('x',[-2,1,3]),('-z',[1,-3,2]),('-x',[2,-1,3])])
def test_up_normals_and_unreferenced_vertex_compaction(up,expected):
    v=np.array([[1,2,3],[2,2,3],[1,3,3],[999,999,999]],float); f=np.array([[0,1,2]])
    sc=fp.scene3d(up=up); fp.mesh3d(sc,(v,f),series='s')
    doc,read=read_glb(write_glb(sc)); attrs=doc['meshes'][0]['primitives'][0]['attributes']
    np.testing.assert_allclose(read(attrs['POSITION'])[0],expected)
    assert len(read(attrs['POSITION']))==3
    np.testing.assert_allclose(np.linalg.norm(read(attrs['NORMAL']),axis=1),1)
    assert np.linalg.det(sc.to_world[:3,:3])==1


def test_shape_targets_reconstruct_defaults_and_union_bounds(tmp_path):
    v,f=sphere(); sc=fp.scene3d(up='z')
    targets={'inflated':v*2,'shifted':v+[2,0,0]}
    fp.mesh3d(sc,(v,f),series='s',states={k:(a,f) for k,a in targets.items()})
    sc.view(states={'inflated':.25,'shifted':1.2})
    data=write_glb(sc); doc,read=read_glb(data); mesh=doc['meshes'][0]; prim=mesh['primitives'][0]
    assert mesh['extras']['targetNames']==list(targets) and mesh['weights']==[.25,1.2]
    base=read(prim['attributes']['POSITION'])
    for target,gl_target in zip(targets.values(),prim['targets']): np.testing.assert_allclose(base+read(gl_target['POSITION']),target@sc.to_world[:3,:3].T,atol=1e-6)
    result=fp.save(sc,tmp_path/'shape',recipe=False); man=json.loads(Path(result.manifest).read_text())
    assert man['bounds']['max'][0]==3
    assert man['view']['states']['shifted']==1.2


def test_sequence_frame_and_failed_state_is_atomic():
    v,f=sphere(); sc=fp.scene3d()
    fp.mesh3d(sc,(v,f),series='cell',states=[(v*(1+i/20),f) for i in range(1,8)],sequence=True)
    sc.view(frame=2.25)
    assert sc._view['states']=={'frame-2':.75,'frame-3':.25}
    sc.view(frame=0); assert sc._view['states']=={}
    before=len(sc.parts)
    with pytest.raises(ValueError,match='shared topology'): fp.mesh3d(sc,(v,f),series='bad',states={'wrong':(v[:-1],f[:-12])})
    assert len(sc.parts)==before


def test_raw_values_valid_mask_and_real_zero(tmp_path):
    v,f=sphere(); values=v[:,1].copy(); values[0]=np.nan; values[1]=0
    sc=fp.scene3d(); fp.surface3d(sc,values,series='height',surfaces=(v,f),kind='continuous',colorbar=True,cmap='viridis')
    doc,read=read_glb(write_glb(sc))
    saw_zero=saw_missing=False
    for mesh in doc['meshes']:
        attrs=mesh['primitives'][0]['attributes']; raw=read(attrs['_VALUE'])
        assert np.isfinite(raw).all()
        valid=read(attrs['_VALID']) if '_VALID' in attrs else np.ones(len(raw))
        saw_zero |= bool(((raw==0)&(valid==1)).any()); saw_missing |= bool((valid==0).any())
        if '_VALID' in attrs:
            acc=doc['accessors'][attrs['_VALID']]; assert doc['bufferViews'][acc['bufferView']]['byteStride']==4
    assert saw_zero and saw_missing
    m=json.loads(Path(fp.save(sc,tmp_path/'field',recipe=False).manifest).read_text())
    assert any(p.get('field')=='height.field' for p in m['parts'])


def test_continuous_mapping_same_as_surface():
    v,f=sphere(); values=np.linspace(-2,5,len(v))
    fig,ax=plt.subplots()
    fp.surface(ax,values,series='v',surfaces={'left':(v,f)},hemispheres=('left',),views=('lateral',),kind='continuous',cmap='viridis',percentile=(2,98))
    summary=next(m.data['surface'] for m in registry_for(fig).marks if m.role=='surface')
    sc=fp.scene3d(); fp.surface3d(sc,values,series='v',surfaces=(v,f),kind='continuous',cmap='viridis',percentile=(2,98))
    spec=registry_for(sc).marks[0].data['scene3d']['field']
    assert spec['range']==[summary['vmin'],summary['vmax']]
    assert spec['cmap']['name']==summary['cmap']
    coll=next(m.artists[0] for m in registry_for(fig).marks if m.role=='surface-field')
    np.testing.assert_allclose(sc.parts[0].colors,coll.cmap(coll.norm(sc.parts[0].values)))
    plt.close(fig)


def test_category_mapping_missing_and_legend_same_as_surface():
    v,f=sphere(); values=np.where(v[:,1]>0,0,1); values[0]=-1
    kw=dict(series='atlas',kind='label',categories={0:'frontal',1:'parietal'},palette={0:'#D14D41',1:'#4385BE'},missing_below=0)
    fig,ax=plt.subplots(); fp.surface(ax,values,surfaces={'left':(v,f)},hemispheres=('left',),views=('lateral',),**kw)
    sc=fp.scene3d(); fp.surface3d(sc,values,surfaces=(v,f),**kw)
    colors={m.name:m.data['surface']['color'] for m in registry_for(fig).marks if m.role=='surface-region'}
    for p in sc.parts:
        name=p.id.split('.')[-1]
        if name!='missing': assert p.color==colors[name]
    assert 'atlas.missing' in {p.id for p in sc.parts}
    assert sc._legend_entries==['atlas.frontal','atlas.parietal']
    plt.close(fig)


def test_same_topology_morph_and_mismatch():
    v,f=sphere(); a=fp.scene3d(); b=fp.scene3d()
    fp.mesh3d(a,{'left':(v,f),'right':(v+[3,0,0],f)},series='cortex')
    fp.mesh3d(b,{'right':(v*[1,2,1]+[3,0,0],f),'left':(v*2,f)},series='cortex',share_topology_with=a,morph_group='cortex')
    assert fp.can_morph(a,b) and len(fp.can_morph(a,b)['pairs'])==2
    with pytest.raises(ValueError,match='indices differ'): fp.mesh3d(fp.scene3d(),(v,f[:,::-1]),series='cortex',share_topology_with=a.parts[0])
    c=fp.scene3d(); fp.mesh3d(c,sphere(4,6),series='cortex')
    assert not fp.can_morph(a,c)


def test_duck_adapters_and_input_errors():
    v,f=sphere()
    for obj in [SimpleNamespace(vertices=v,faces=f),SimpleNamespace(points=v,faces=np.column_stack([np.full(len(f),3),f]).ravel())]:
        vv,ff=load_mesh(obj); np.testing.assert_array_equal(vv,v); np.testing.assert_array_equal(ff,f)
    for bad in [(v,[[0,1,.5]]),(v,[[0,1,10000]]),(v*float('nan'),f)]:
        with pytest.raises(ValueError): load_mesh(bad)
    sc=scene()
    with pytest.raises(ValueError): sc.view(zoom=0)
    with pytest.raises(ValueError): sc.view(states={'unknown':1})
    with pytest.raises(ValueError): fp.save(fp.scene3d(),'/tmp/no-scene-empty',recipe=False)


def test_decimation_replay_states_values_and_parts():
    pytest.importorskip('fast_simplification')
    v,f=sphere(20,30)
    a=fp.scene3d(); b=fp.scene3d()
    fp.mesh3d(a,(v,f),series='cortex',max_faces=200,states={'inflated':(v*1.2,f)})
    fp.mesh3d(b,(v*[1,1.3,.8],f),series='cortex',share_topology_with=a)
    assert len(a.parts[0].faces)<=200 and fp.can_morph(a,b)
    assert a.parts[0].states['inflated'].shape==a.parts[0].vertices.shape
    sc=fp.scene3d(); fp.surface3d(sc,v[:,1],series='field',surfaces=(v,f),kind='continuous',cmap='viridis',max_faces=200)
    assert len(sc.parts[0].values)==len(sc.parts[0].vertices)
    import matplotlib as mpl
    np.testing.assert_allclose(sc.parts[0].colors,mpl.colormaps['viridis']((sc.parts[0].values+1)/2))


def test_style_and_data_axes_survive_up_conversion(tmp_path):
    fp.use_paper(); sc=scene(up='z',axes='box'); sc.axis('x',lim=(-2,2),ticks=[-2,0,2],label='width')
    man=json.loads(Path(fp.save(sc,tmp_path/'axes',recipe=False).manifest).read_text())
    assert man['axes']['x']=={'lim':[-2.,2.],'ticks':[-2.,0.,2.],'label':'width'}
    assert man['style']['fontSizePt']==plt.rcParams['font.size']


def test_only_filter_and_recipe_output(tmp_path,monkeypatch):
    monkeypatch.setenv('FLUXPLOT_ONLY','keep*')
    result=fp.save(scene(),tmp_path/'skip',recipe=False)
    assert result.skipped and not list(tmp_path.iterdir())
    monkeypatch.delenv('FLUXPLOT_ONLY')
    result=fp.save(scene(),tmp_path/'keep',recipe={'script':str(tmp_path/'build.py')})
    recipe=json.loads(Path(result.recipe).read_text()); assert recipe['output']=='keep.glb'


@pytest.mark.parametrize('scale',[1e30,1e-20])
def test_large_and_tiny_finite_positions_produce_valid_normals(scale):
    vertices=np.array([[0,0,0],[scale,0,0],[0,scale,0]],dtype=np.float32)
    sc=fp.scene3d(); fp.mesh3d(sc,(vertices,[[0,1,2]]),series='large')
    doc,read=read_glb(write_glb(sc)); normals=read(doc['meshes'][0]['primitives'][0]['attributes']['NORMAL'])
    assert np.isfinite(normals).all()
    np.testing.assert_allclose(normals,[[0,0,1]]*3)


def test_polygon_adapter_uses_triangulation_and_tiny_part_budget_rejected():
    vertices=np.array([[0,0,0],[2,0,0],[1,1,0],[2,2,0],[0,2,0]],float)
    raw=SimpleNamespace(points=vertices,faces=np.array([5,0,1,2,3,4]))
    with pytest.raises(ValueError,match='triangulated'): load_mesh(raw)
    result=np.array([[0,1,2],[0,2,4],[2,3,4]])
    raw.triangulate=lambda:SimpleNamespace(points=vertices,faces=np.column_stack([np.full(3,3),result]).ravel())
    _,faces=load_mesh(raw); np.testing.assert_array_equal(faces,result)
    with pytest.raises(ValueError,match='one triangle per named part'):
        fp.mesh3d(fp.scene3d(),{'one':(vertices,result),'two':(vertices,result)},series='tiny',max_faces=1)


def test_static_preview_and_preview_copy_preserve_source():
    from fluxplot.scene3d_viewer import mimebundle,preview_scene
    from matplotlib.colors import LinearSegmentedColormap
    sc=fp.scene3d(title='Still');v,f=sphere(20,30)
    custom=LinearSegmentedColormap.from_list('not-registered',['#123456','#fedcba'])
    fp.surface3d(sc,v[:,1],series='custom',surfaces=(v,f),kind='continuous',cmap=custom)
    original=write_glb(sc);sc.preview_max_faces=200
    png=mimebundle(sc,static=True)
    assert set(png)=={'image/png'} and png['image/png'].startswith(b'\x89PNG\r\n\x1a\n')
    assert write_glb(sc)==original
    if __import__('importlib.util').util.find_spec('fast_simplification'):
        reduced=preview_scene(sc);assert sum(len(p.faces) for p in reduced.parts)<=200


def test_production_handshake_fixtures_reproduce(tmp_path):
    from generate_scene3d_library_fixtures import generate,ROOT
    assert generate(tmp_path)==json.loads((ROOT/'SHA256SUMS.json').read_text())
    for path in ROOT.iterdir(): assert path.read_bytes()==(tmp_path/path.name).read_bytes()


def test_repeated_series_keeps_each_allocated_field_colorbar(tmp_path):
    v,f=sphere();sc=fp.scene3d()
    for offset in (0,3):
        fp.surface3d(sc,v[:,1],series='repeat',surfaces=(v+[offset,0,0],f),kind='continuous',colorbar=True)
    man=json.loads(Path(fp.save(sc,tmp_path/'repeat',recipe=False).manifest).read_text())
    assert {p['field'] for p in man['parts'] if p['role']=='colorbar'}=={'repeat.field','repeat.field-2'}


def test_surface_budget_cannot_drop_semantic_parts():
    v,f=sphere();values=np.where(v[:,1]>0,0,1)
    with pytest.raises(ValueError,match='one triangle per semantic part'):
        fp.surface3d(fp.scene3d(),values,series='tiny',surfaces=(v,f),kind='label',max_faces=1)
