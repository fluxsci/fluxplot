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
    from PIL import Image
    import io
    assert Image.open(io.BytesIO(png['image/png'])).convert('RGBA').getpixel((0,0))==(255,255,255,255)
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


def test_notebook_bundle_stamp_license_and_safe_html():
    import hashlib
    from importlib.resources import files
    from fluxplot.scene3d_viewer import mimebundle,viewer_bundle
    folder=files('fluxplot').joinpath('_viewer')
    js=viewer_bundle();stamp=json.loads(folder.joinpath('stamp.json').read_text())
    assert hashlib.sha256(js.encode()).hexdigest()==stamp['sha256']
    assert stamp['version'].startswith('m3d-')
    assert 'MIT' in folder.joinpath('THIRD-PARTY.txt').read_text()
    sc=scene(title='</script><script>alert(1)</script>')
    bundle=mimebundle(sc)
    assert set(bundle)=={'text/html','image/png'}
    html=bundle['text/html']
    assert 'document.currentScript' in html and 'script.parentElement' in html
    assert 'data:image/png;base64,' in html and 'FluxModel3dViewer.mount(host,' in html
    assert '</script><script>alert(1)' not in html
    assert html.count('</script>')==1


def test_static_perspective_uses_camera_image_plane(monkeypatch):
    from matplotlib.figure import Figure
    from fluxplot.scene3d_viewer import png_preview
    captured=[]; original=Figure.savefig
    def capture(fig,*args,**kwargs):
        captured.append(fig)
        return original(fig,*args,**kwargs)
    monkeypatch.setattr(Figure,'savefig',capture)
    sc=fp.scene3d(figsize=(3,3),axes='none',lighting='unlit')
    fp.mesh3d(sc,([[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]],[[0,1,2],[0,2,3]]),series='plane')
    sc.view(azimuth=0,elevation=0,projection='perspective',fov=90,zoom=1)
    png_preview(sc)
    # Radius sqrt(2), fitted camera distance 2; a 90-degree lens spans +/-2 at z=0.
    np.testing.assert_allclose(captured[-1].axes[0].get_xlim(),[-2,2])
    sc.view(projection='orthographic');png_preview(sc)
    np.testing.assert_allclose(captured[-1].axes[0].get_xlim(),[-np.sqrt(2),np.sqrt(2)])


def test_uneven_part_budgets_and_production_reduction():
    from fluxplot._mesh_reduce import face_budgets
    from fluxplot.scene3d_viewer import preview_scene
    assert face_budgets([9900]+[1]*99,100)==[1]*100
    for counts in ([9900]+[1]*99,[1,3,8,100],[1,1]):
        for budget in (len(counts),len(counts)+1,sum(counts)//2,sum(counts),sum(counts)+100):
            if budget<len(counts): continue
            quotas=face_budgets(counts,budget)
            assert sum(quotas)==min(budget,sum(counts))
            assert all(1<=q<=n for q,n in zip(quotas,counts))
    pytest.importorskip('fast_simplification')
    v,f=sphere(20,30);triangle=(np.array([[0,0,0],[1,0,0],[0,1,0]]),np.array([[0,1,2]]))
    parts={'large':(v,f),**{f'tiny{i}':triangle for i in range(99)}}
    sc=fp.scene3d();fp.mesh3d(sc,parts,series='uneven',max_faces=199)
    assert len(sc.parts)==100 and sum(len(p.faces) for p in sc.parts)<=199
    original=fp.scene3d(preview_max_faces=199);fp.mesh3d(original,parts,series='uneven')
    preview=preview_scene(original)
    assert len(preview.parts)==100 and sum(len(p.faces) for p in preview.parts)<=199
    assert sum(len(p.faces) for p in original.parts)==len(f)+99
    original.preview_max_faces=99
    with pytest.warns(UserWarning,match='one triangle per part'):
        assert preview_scene(original) is original
    # Categorical surface groups exercise the same total quota allocation path.
    vertices=np.concatenate([v]+[triangle[0]+[3+i*2,0,0] for i in range(99)])
    faces=np.concatenate([f]+[triangle[1]+len(v)+i*3 for i in range(99)])
    values=np.concatenate([np.zeros(len(v))]+[np.full(3,i+1) for i in range(99)])
    sc=fp.scene3d();fp.surface3d(sc,values,series='labels',surfaces=(vertices,faces),kind='label',max_faces=199)
    assert len(sc.parts)==100 and sum(len(p.faces) for p in sc.parts)<=199


def test_face_cap_accounts_for_state_and_part_costs(tmp_path, monkeypatch):
    import re
    from fluxplot import _scene3d_size as sizing
    # Disjoint triangles exercise the worst possible referenced-vertex ratio.
    f=np.arange(900).reshape(-1,3)
    v=np.column_stack([np.arange(900),np.arange(900)%3,np.arange(900)%7]).astype(float)
    def make(cap, count=3, states=24):
        sc=fp.scene3d()
        for i in range(count):
            values=v[:,1].copy(); values[0]=np.nan
            fp.surface3d(sc,values,series=f'part{i}',surfaces=(v,f[:cap]),kind='continuous',
                         states={f'state{n}':(v+[n*.123456789,0,0],f[:cap]) for n in range(states)})
        return sc
    sc=make(300); original=write_glb(sc)
    limit=480_000
    hint=sizing.face_cap_recommendation(sc,original,max_bytes=limit)
    cap=int(re.search(r'max_faces=(\d+)',hint)[1])
    assert 1<=cap<300 and 'each mesh3d/surface3d call' in hint
    # Apply the recommendation independently to every call, not as a total
    # divided among them. Serialize with the real writer and all field masks.
    reduced=make(cap)
    assert len(write_glb(reduced))<=limit
    simpler=make(300,states=0)
    simple_cap=int(re.search(r'max_faces=(\d+)',sizing.face_cap_recommendation(simpler,write_glb(simpler),max_bytes=limit))[1])
    assert simple_cap>cap
    more_parts=make(300,count=5)
    more_cap=int(re.search(r'max_faces=(\d+)',sizing.face_cap_recommendation(more_parts,write_glb(more_parts),max_bytes=limit))[1])
    assert more_cap<cap and len(write_glb(make(more_cap,count=5)))<=limit
    monkeypatch.setattr(sizing,'WARN_BYTES',limit)
    with pytest.warns(UserWarning): saved=fp.save(sc,tmp_path/'large',recipe=False)
    assert any(f'max_faces={cap}' in warning for warning in saved.warnings)
    assert all('400000' not in warning for warning in saved.warnings)


def test_face_cap_impossible_metadata_and_part_floor():
    import re
    from fluxplot._scene3d_size import face_cap_recommendation
    v=np.array([[0,0,0],[1,0,0],[0,1,0]],float); f=np.array([[0,1,2]])
    sc=fp.scene3d()
    fp.mesh3d(sc,{f'part{i}':(v,f) for i in range(12)},series='many',
              states={f'state{i}':{f'part{j}':(v+i,f) for j in range(12)} for i in range(25)})
    data=write_glb(sc)
    hint=face_cap_recommendation(sc,data,max_bytes=len(data)-1)
    assert 'No max_faces cap can fit' in hint and 'one triangle' in hint and 'max_faces=' not in hint
    assert 'preserving all 12 parts' in face_cap_recommendation(sc,data,max_triangles=11)
    # Large fixed state-name metadata cannot be reduced by changing faces.
    large=fp.scene3d(); vf=np.concatenate([v,v+2]); ff=np.array([[0,1,2],[3,4,5]])
    fp.mesh3d(large,(vf,ff),series='named',states={'a'*5000:(vf,ff)})
    hint=face_cap_recommendation(large,write_glb(large),max_bytes=4000)
    assert 'fixed GLB part/state metadata' in hint and 'max_faces=' not in hint
    # A successful multi-part call cannot receive a cap below its part census.
    assert sc._max_call_parts==12
    permitted=face_cap_recommendation(sc,data,max_bytes=10_000_000)
    assert int(re.search(r'max_faces=(\d+)',permitted)[1])>=12


def test_recommended_cap_reduces_real_state_mesh_below_limit():
    import re
    pytest.importorskip('fast_simplification')
    from fluxplot._scene3d_size import face_cap_recommendation
    v,f=sphere(20,30)
    states={f'shape{i}':(v*(1+i*.01),f) for i in range(24)}
    source=fp.scene3d(); fp.mesh3d(source,(v,f),series='cortex',states=states)
    limit=100_000
    assert len(write_glb(source))>limit
    hint=face_cap_recommendation(source,write_glb(source),max_bytes=limit)
    cap=int(re.search(r'max_faces=(\d+)',hint)[1])
    reduced=fp.scene3d(); fp.mesh3d(reduced,(v,f),series='cortex',states=states,max_faces=cap)
    assert len(reduced.parts[0].faces)<=cap and len(write_glb(reduced))<=limit
    # Recommendations for an already-reduced scene still describe rerunning
    # the authored source, not just its smaller current vertex/face arrays.
    assert face_cap_recommendation(reduced,write_glb(reduced),max_bytes=limit)==hint
    follower=fp.scene3d(); fp.mesh3d(follower,(v*2,f),series='cortex',share_topology_with=reduced)
    assert fp.can_morph(reduced,follower)


def test_reduction_compacts_backend_orphans_in_every_channel(monkeypatch):
    import sys
    from fluxplot._mesh_reduce import reduce_part
    from fluxplot.scene3d import MeshPart
    vertices=np.arange(15,dtype=float).reshape(5,3)
    faces=np.array([[0,2,4]])
    values=np.arange(5,dtype=float); values[2]=np.nan
    colors=np.tile(np.arange(5,dtype=float)[:,None]/4,(1,4))
    part=MeshPart('mesh',vertices.copy(),faces.copy(),'#123456',{'large':vertices*2},values.copy(),colors.copy())
    def replay(points,_faces,_collapses): return points.copy(),faces.copy(),np.arange(5)
    monkeypatch.setitem(sys.modules,'fast_simplification',SimpleNamespace(replay_simplification=replay,
        simplify=lambda *_args,**_kw:(None,None,np.empty((0,2),dtype=np.int32))))
    reduce_part(part,collapses=np.empty((0,2),dtype=np.int32))
    np.testing.assert_array_equal(part.vertices,vertices[[0,2,4]])
    np.testing.assert_array_equal(part.faces,[[0,1,2]])
    np.testing.assert_array_equal(part.states['large'],vertices[[0,2,4]]*2)
    np.testing.assert_array_equal(part.values,[0,np.nan,4])
    np.testing.assert_array_equal(part.colors,colors[[0,2,4]])
    field_scene=fp.scene3d(); field_scene.parts.append(part)
    doc,read=read_glb(write_glb(field_scene)); attrs=doc['meshes'][0]['primitives'][0]['attributes']
    np.testing.assert_array_equal(read(attrs['_VALID']),[1,0,1])
    np.testing.assert_array_equal(read(attrs['_VALUE']),[0,0,4])
    # Public reference/follower replay takes the same surviving-index map even
    # when the backend leaves two orphan vertices in both shape outputs.
    original_faces=np.array([[0,1,2],[2,3,4]])
    reference=fp.scene3d(); follower=fp.scene3d()
    fp.mesh3d(reference,(vertices,original_faces),series='pair',max_faces=1,
              states={'large':(vertices*2,original_faces)})
    fp.mesh3d(follower,(vertices*3,original_faces),series='pair',share_topology_with=reference,
              states={'large':(vertices*6,original_faces)})
    assert fp.can_morph(reference,follower) and len(reference.parts[0].vertices)==3
    np.testing.assert_array_equal(follower.parts[0].states['large'],vertices[[0,2,4]]*6)
    for current in (reference,follower):
        doc,read=read_glb(write_glb(current))
        np.testing.assert_array_equal(read(doc['meshes'][0]['primitives'][0]['indices']),[0,1,2])
