"""Generate a scratch neuron, named states, a morph pair and an eight-frame sequence.

uv run --extra mesh python examples/scene3d_demo.py

Defaults to test-results/model3d/demo/plots; --out selects another scratch folder.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import fluxplot as fp


def sphere(rings=12,segments=20):
    verts=[[0,1,0]]
    for j in range(1,rings):
        phi=np.pi*j/rings
        for i in range(segments):
            t=2*np.pi*i/segments;verts.append([np.sin(phi)*np.cos(t),np.cos(phi),np.sin(phi)*np.sin(t)])
    verts.append([0,-1,0]); faces=[]
    for i in range(segments): faces.append([0,1+(i+1)%segments,1+i])
    for j in range(rings-2):
        a=1+j*segments;b=a+segments
        for i in range(segments):
            k=(i+1)%segments;faces.extend([[a+i,a+k,b+i],[a+k,b+k,b+i]])
    a=1+(rings-2)*segments
    for i in range(segments): faces.append([a+i,a+(i+1)%segments,len(verts)-1])
    return np.asarray(verts),np.asarray(faces)


def tube(points,radius=.1,sides=8):
    points=np.asarray(points);vertices=[]; faces=[]
    for i,p in enumerate(points):
        tangent=points[min(i+1,len(points)-1)]-points[max(i-1,0)]; tangent=tangent/np.linalg.norm(tangent)
        ref=np.array([0,0,1.]) if abs(tangent[2])<.9 else np.array([1.,0,0])
        u=np.cross(tangent,ref);u/=np.linalg.norm(u);w=np.cross(tangent,u)
        r=radius*(1-.7*i/(len(points)-1))
        vertices.extend(p+r*(np.cos(t)*u+np.sin(t)*w) for t in np.arange(sides)*2*np.pi/sides)
    for i in range(len(points)-1):
        for j in range(sides):
            a=i*sides+j;b=i*sides+(j+1)%sides;c=a+sides;d=b+sides
            faces.extend([[a,b,c],[b,d,c]])
    vertices.extend([points[0],points[-1]])
    for j in range(sides):
        faces.append([len(vertices)-2,(j+1)%sides,j]);a=(len(points)-1)*sides;faces.append([len(vertices)-1,a+j,a+(j+1)%sides])
    return np.asarray(vertices),np.asarray(faces)


def combine(meshes):
    v=[];f=[];n=0
    for vv,ff in meshes: v.append(vv);f.append(ff+n);n+=len(vv)
    return np.concatenate(v),np.concatenate(f)


def neuron_scene(*, states=False):
    """Named branched neuron with a scale bar, optionally a notebook shape control."""
    v,f=sphere(); soma=(v*[.55,.7,.5],f)
    axon=tube([[0,-.3,0],[.05,-1.4,0],[-.3,-2.5,.1],[.1,-3.7,.3]],.13)
    branches=[]
    for angle in np.linspace(0,2*np.pi,6,endpoint=False):
        d=np.array([np.cos(angle),.5,np.sin(angle)])
        branches.append(tube([d*.3,d*1.1,d*1.8+[0,.6,0],d*2.3+[.2,.9,0]],.11))
    neuron=fp.scene3d(figsize=(3.5,3),units='µm',title='Neuron-like mesh',scalebar=1)
    parts={'soma':soma,'axon':axon,'dendrites':combine(branches)}
    shape_states={'expanded':{name:(vv*[1.18,1,1.18],ff) for name,(vv,ff) in parts.items()}} if states else None
    fp.mesh3d(neuron,parts,series='neuron',states=shape_states,palette={'soma':'#D14D41','axon':'#4385BE','dendrites':'#3AA99F'},legend=True)
    return neuron


def build(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);fp.use_paper()
    neuron=neuron_scene()
    fp.save(neuron,out/'neuron',recipe=False)
    v,f=sphere(24,40)
    # Deterministic folded surface: the same connectivity supports states and pairs.
    folds=1+.10*np.sin(12*v[:,0])*np.cos(10*v[:,1])*np.sin(8*v[:,2])
    v=v*folds[:,None]*[1.15,.9,.8]
    base=fp.scene3d(figsize=(3,3),units='mm',title='Cortex shape states',axes='triad')
    fp.surface3d(base,v[:,1],series='cortex',surfaces=(v,f),kind='continuous',cmap='viridis',colorbar=True,cbar_label='Value',states={'inflated':(v*[1.4,1.2,.8],f),'bent':(v+np.column_stack([.35*v[:,1]**2,np.zeros(len(v)),np.zeros(len(v))]),f)})
    fp.save(base,out/'cortex-states',recipe=False)
    field=fp.scene3d(figsize=(3,3),units='mm',title='Continuous field',axes='box')
    fp.surface3d(field,v[:,1]+.3*v[:,0],series='height',surfaces=(v,f),kind='continuous',cmap='viridis',colorbar=True,cbar_label='Value')
    fp.save(field,out/'continuous-field',recipe=False)
    pial=fp.scene3d();fp.mesh3d(pial,{'left':(v+[-1.2,0,0],f),'right':(v+[1.2,0,0],f)},series='cortex')
    inflated=fp.scene3d();fp.mesh3d(inflated,{'left':(v*[1,1.4,.6]+[-1.2,0,0],f),'right':(v*[1,1.4,.6]+[1.2,0,0],f)},series='cortex',share_topology_with=pial,morph_group='cortex')
    fp.save(pial,out/'cortex-pial',recipe=False);fp.save(inflated,out/'cortex-inflated',recipe=False)
    sequence=fp.scene3d();fp.mesh3d(sequence,(v,f),series='cell',states=[(v*[1+i*.06,1,1-i*.04],f) for i in range(1,8)],sequence=True)
    fp.save(sequence,out/'cell-sequence',recipe=False)
    return neuron

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=Path('test-results/model3d/demo/plots'))
    build(parser.parse_args().out)
