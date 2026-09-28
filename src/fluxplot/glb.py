"""Deterministic numpy-only binary glTF 2.0 writer (scene3d contract 0.1.0)."""
from __future__ import annotations
import json
import struct
import numpy as np
from matplotlib.colors import to_rgba
from .version import __version__


def linear_rgb(rgb):
    rgb=np.asarray(rgb,dtype=float)
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)


def vertex_normals(vertices, faces):
    out=np.zeros_like(vertices,dtype=np.float64)
    # Convert before subtraction/cross products: finite float32 coordinates can
    # overflow float32 area products while their float64 products remain finite.
    t=np.asarray(vertices,dtype=np.float64)[faces]
    area=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0])
    for i in range(3): np.add.at(out,faces[:,i],area)
    size=np.linalg.norm(out,axis=1)
    out[size==0]=(0,1,0)
    return (out/np.linalg.norm(out,axis=1,keepdims=True)).astype('<f4')


def _finite32(a, name):
    with np.errstate(over='ignore'): a=np.asarray(a,dtype='<f4')
    if not np.isfinite(a).all(): raise ValueError(f'{name} cannot be represented by finite float32 values')
    return a


def write_glb(scene):
    """Serialize one GLB. No timestamps, external resources, textures, or vertex welding."""
    if not scene.parts: raise ValueError('cannot save an empty Scene3D; add a mesh first')
    binary=bytearray()
    doc=dict(asset={'version':'2.0','generator':f'fluxplot {__version__}'},buffers=[{}],bufferViews=[],accessors=[],meshes=[],nodes=[],materials=[],scenes=[{'nodes':list(range(len(scene.parts)))}],scene=0)
    def accessor(a, kind, component, *, bounds=False, normalized=False, target=None, stride=None):
        a=np.ascontiguousarray(a); binary.extend(b'\0'*(-len(binary)%4))
        view={'buffer':0,'byteOffset':len(binary),'byteLength':a.nbytes}
        if target is not None: view['target']=target
        if stride is not None: view['byteStride']=stride
        doc['bufferViews'].append(view); binary.extend(a.tobytes())
        entry={'bufferView':len(doc['bufferViews'])-1,'componentType':component,'count':len(a),'type':kind}
        if bounds: entry.update(min=a.min(axis=0).tolist(),max=a.max(axis=0).tolist())
        if normalized: entry['normalized']=True
        doc['accessors'].append(entry)
        return len(doc['accessors'])-1
    rotation=scene.to_world[:3,:3]
    for part in scene.parts:
        v=_finite32(part.vertices@rotation.T,'positions')
        f=part.faces.astype('<u2' if part.faces.max()<65536 else '<u4')
        n=vertex_normals(v,f)
        attrs={'POSITION':accessor(v,'VEC3',5126,bounds=True,target=34962),'NORMAL':accessor(n,'VEC3',5126,target=34962)}
        if part.values is not None:
            valid=np.isfinite(part.values)
            value=_finite32(np.where(valid,part.values,0),'values')
            attrs['_VALUE']=accessor(value,'SCALAR',5126,target=34962)
            if not valid.all(): attrs['_VALID']=accessor(np.column_stack([valid,np.zeros((len(valid),3),dtype='u1')]).astype('u1'),'SCALAR',5121,target=34962,stride=4)
        if part.colors is not None:
            rgba=np.asarray(part.colors,dtype=float).copy(); rgba[:,:3]=linear_rgb(rgba[:,:3])
            attrs['COLOR_0']=accessor(np.rint(np.clip(rgba,0,1)*255).astype('u1'),'VEC4',5121,normalized=True,target=34962)
        rgba=to_rgba(part.color); base=[*linear_rgb(rgba[:3]).tolist(),rgba[3]]
        # Vertex colors already carry the complete map; white avoids tinting them twice.
        if part.colors is not None: base=[1,1,1,1]
        material={'name':part.id,'pbrMetallicRoughness':{'baseColorFactor':base,'metallicFactor':0,'roughnessFactor':.6},'doubleSided':True}
        if rgba[3]<1 or (part.colors is not None and (part.colors[:,3]<1).any()): material['alphaMode']='BLEND'
        doc['materials'].append(material)
        prim={'attributes':attrs,'indices':accessor(f.ravel(),'SCALAR',5123 if f.dtype.itemsize==2 else 5125,target=34963),'material':len(doc['materials'])-1,'mode':4}
        mesh={'name':part.id,'primitives':[prim]}
        if part.states:
            prim['targets']=[]
            for name,target in part.states.items():
                world=_finite32(target@rotation.T,'shape positions')
                prim['targets'].append({'POSITION':accessor(_finite32(world-v,'shape deltas'),'VEC3',5126,bounds=True,target=34962),'NORMAL':accessor(vertex_normals(world,f)-n,'VEC3',5126,target=34962)})
            mesh['extras']={'targetNames':list(part.states)}
            mesh['weights']=[scene._view.get('states',{}).get(name,0) for name in part.states]
        doc['meshes'].append(mesh); doc['nodes'].append({'name':part.id,'mesh':len(doc['meshes'])-1})
    doc['buffers'][0]['byteLength']=len(binary)
    js=json.dumps(doc,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')
    js+=b' '*(-len(js)%4); binary.extend(b'\0'*(-len(binary)%4))
    return struct.pack('<4sII',b'glTF',2,28+len(js)+len(binary))+struct.pack('<I4s',len(js),b'JSON')+js+struct.pack('<I4s',len(binary),b'BIN\0')+binary
