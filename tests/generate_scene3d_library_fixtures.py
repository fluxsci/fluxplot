"""F5 handshake fixtures authored through the public production API (no private writer).

The P0.0 contract oracle stays independent in fixtures/scene3d. Run this generator with
``uv run --extra dev python tests/generate_scene3d_library_fixtures.py [output-dir]``.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import fluxplot as fp
from generate_scene3d_fixtures import sphere

ROOT=Path(__file__).parent/'fixtures'/'scene3d-library'


def scenes():
    v,f=sphere()
    def new(**kw): return fp.scene3d(units='µm',**kw)
    plain=new();fp.mesh3d(plain,(v,f),series='sample');yield 'plain',plain
    parts={'soma':(v*.45+[-.75,0,0],f),'dendrites':(v*[.3,1.1,.3]+[.7,0,0],f)}
    named=new();fp.mesh3d(named,parts,series='neuron',legend=True,palette={'soma':'#D14D41','dendrites':'#3AA99F'});yield 'named-parts',named
    continuous=new();fp.surface3d(continuous,v[:,1],series='height',surfaces=(v,f),kind='continuous',cmap=LinearSegmentedColormap.from_list('contract-blue-red',['#0000FF','#FF0000']),colorbar=True,label='Height');yield 'continuous',continuous
    labels=np.where(v[:,1]>0,0,1);labels[0]=-1
    categorical=new();fp.surface3d(categorical,labels,series='atlas',surfaces=(v,f),kind='label',categories={0:'frontal',1:'parietal'},palette={0:'#D14D41',1:'#4385BE'},missing_below=0);yield 'categorical-missing',categorical
    box=new(axes='box');fp.mesh3d(box,(v,f),series='sample');yield 'box-axes',box
    scale=new(scalebar=.5);fp.mesh3d(scale,(v,f),series='sample');yield 'scalebar',scale
    meshes={'left':(v+[-1.2,0,0],f),'right':(v+[1.2,0,0],f)}
    a=new();fp.mesh3d(a,meshes,series='cortex',morph_group='cortex');yield 'morph-a',a
    b=new();fp.mesh3d(b,{k:(p*[1,1.35,.65]+[0,.1,0],ff) for k,(p,ff) in meshes.items()},series='cortex',share_topology_with=a,morph_group='cortex');yield 'morph-b',b
    vv,ff=sphere(6,10);bad=new();fp.mesh3d(bad,{'left':(vv+[-1.2,0,0],ff),'right':meshes['right']},series='cortex',morph_group='cortex');yield 'morph-incompatible',bad
    state=new();fp.mesh3d(state,(v,f),series='sample',states={'inflated':(v*[1.35,1.2,1.1],f),'bent':(v+np.column_stack([.3*v[:,1]**2,np.zeros(len(v)),np.zeros(len(v))]),f)});state.view(states={'inflated':.25,'bent':0});yield 'states',state
    sequence=new();fp.mesh3d(sequence,(v,f),series='sample',states=[(v*[1+i*.06,1,1-i*.04],f) for i in range(1,8)],sequence=True);yield 'sequence',sequence


def generate(root=ROOT):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with mpl.rc_context():
        fp.use_paper()
        for name,scene in scenes(): fp.save(scene,root/name,recipe=False,_now='2000-01-01T00:00:00Z')
    receipts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.suffix in ('.glb','.json') and p.name!='SHA256SUMS.json'}
    (root/'SHA256SUMS.json').write_text(json.dumps(receipts,indent=2,sort_keys=True)+'\n')
    return receipts


if __name__=='__main__': generate(sys.argv[1] if len(sys.argv)>1 else ROOT)
