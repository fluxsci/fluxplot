"""Duck-typed mesh input, stable part identity and checked topology correspondence."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import warnings
import numpy as np
from matplotlib.colors import to_hex, to_rgba
from .scene3d import Scene3D, MeshPart
from .ids import series_id, slugify
from .descriptors import Mark
from .tagger import Registry, registry_for


def load_mesh(mesh):
    """Read arrays, vertices/faces objects, PyVista PolyData, GIFTI, or trimesh files."""
    if isinstance(mesh,MeshPart): v,f=mesh.vertices,mesh.faces
    elif hasattr(mesh,'points') and hasattr(mesh,'faces'):
        v,f=mesh.points,np.asarray(mesh.faces)
        if f.ndim==1:
            def polygons(flat):
                result=[]; offset=0
                while offset<len(flat):
                    count=int(flat[offset]); poly=flat[offset+1:offset+1+count]
                    if count<3 or count!=flat[offset] or len(poly)!=count:
                        raise ValueError('malformed PolyData polygon faces')
                    result.append(poly); offset+=count+1
                return result
            faces=polygons(f)
            if any(len(poly)!=3 for poly in faces):
                # A triangle fan is incorrect for concave polygons. Delegate to the
                # adapter's triangulator, which returns a new mesh in PyVista.
                if not callable(getattr(mesh,'triangulate',None)):
                    raise ValueError('PolyData polygons must be triangulated; call mesh.triangulate() first')
                triangle_mesh=mesh.triangulate()
                v=triangle_mesh.points; faces=polygons(np.asarray(triangle_mesh.faces))
                if any(len(poly)!=3 for poly in faces):
                    raise ValueError('PolyData triangulate() did not produce triangle faces')
            f=faces
    elif hasattr(mesh,'vertices') and hasattr(mesh,'faces'): v,f=mesh.vertices,mesh.faces
    elif isinstance(mesh,(tuple,list)) and len(mesh)==2: v,f=mesh
    elif isinstance(mesh,(str,Path)):
        if str(mesh).lower().endswith('.gii'):
            from .surface import _load_surface
            v,f=_load_surface(mesh)
        else:
            try: import trimesh
            except ImportError as exc: raise ImportError('mesh file input needs fluxplot[mesh] or trimesh; arrays need no optional package') from exc
            loaded=trimesh.load(str(mesh),force='mesh',process=False)
            if not hasattr(loaded,'vertices'): raise ValueError(f'no triangle mesh in {mesh}')
            v,f=loaded.vertices,loaded.faces
    else: raise TypeError('mesh must be (vertices, faces), a mesh object, or a mesh file path')
    v=np.asarray(v,dtype=float)
    raw=np.asarray(f)
    if v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all(): raise ValueError('vertices must be a nonempty finite N×3 array')
    if raw.ndim!=2 or raw.shape[1]!=3 or not len(raw): raise ValueError('faces must be a nonempty M×3 triangle array')
    if not np.issubdtype(raw.dtype,np.integer) and (not np.isfinite(raw).all() or not np.equal(raw,np.floor(raw)).all()): raise ValueError('face indices must be integers')
    if raw.min()<0 or raw.max()>=len(v): raise ValueError('face index outside vertex array')
    return v.copy(),raw.astype(np.int64,copy=True)


def _states(states, sequence):
    if states is None: return {}
    if isinstance(states,dict):
        result={str(k):v for k,v in states.items()}
        if any(not k for k in result) or len(result)!=len(states): raise ValueError('state names must be nonempty and unique')
        return result
    if not sequence: raise ValueError('a list of states requires sequence=True')
    return {f'frame-{i+1}':v for i,v in enumerate(states)}


def _same_topology(vertices, faces, ref_vertices, ref_faces, label):
    if len(vertices)!=len(ref_vertices): raise ValueError(f'{label}: {len(vertices)} vs {len(ref_vertices)} vertices; shared topology required')
    if faces.shape!=ref_faces.shape or not np.array_equal(faces,ref_faces): raise ValueError(f'{label}: face indices differ; shared topology required')


def _reference(ref, part_id, *, single):
    if ref is None: return None
    if isinstance(ref,Scene3D):
        found=next((p for p in ref.parts if p.id==part_id),None)
        if found is None: raise ValueError(f'{part_id}: reference scene has no matching part')
        return found
    if isinstance(ref,MeshPart): return ref
    if not single and isinstance(ref,dict):
        for key,value in ref.items():
            if part_id.endswith('.'+slugify(key)): return value
        raise ValueError(f'{part_id}: reference mapping has no matching part')
    return ref


def _prepare_part(part_id, mesh, color, states=None, *, reference=None, max_faces=None,
                  values=None, colors=None):
    v,f=load_mesh(mesh)
    state_arrays={}
    for name,shape in (states or {}).items():
        sv,sf=load_mesh(shape); _same_topology(v,f,sv,sf,f'state {name}')
        state_arrays[name]=sv
    if reference is not None:
        if isinstance(reference,MeshPart):
            if reference.source_count!=len(v) or not np.array_equal(reference.source_faces,f): raise ValueError(f'{part_id}: reference vertex count or face indices differ')
        else:
            rv,rf=load_mesh(reference); _same_topology(v,f,rv,rf,part_id)
    used=np.unique(f)
    faces=np.searchsorted(used,f)
    part=MeshPart(part_id,v[used],faces,to_hex(color,keep_alpha=to_rgba(color)[3]!=1),
                  {name:a[used] for name,a in state_arrays.items()},
                  None if values is None else np.asarray(values,dtype=float)[used].copy(),
                  None if colors is None else np.asarray(colors,dtype=float)[used].copy(),
                  source_faces=f,source_count=len(v),source_indices=used,compact_faces=faces.copy())
    if isinstance(reference,MeshPart) and reference.collapses is not None:
        from ._mesh_reduce import reduce_part
        reduce_part(part,collapses=reference.collapses)
    elif max_faces is not None and len(part.faces)>max_faces:
        if reference is not None: raise ValueError('shared reference was not decimated; apply max_faces to the reference first')
        from ._mesh_reduce import reduce_part
        reduce_part(part,max_faces=max_faces)
    if isinstance(reference,MeshPart): _same_topology(part.vertices,part.faces,reference.vertices,reference.faces,part_id)
    return part


def _publish(scene, parts, descriptors, alloc, *, legend=False, sequence=False, morph_group=None):
    reg=registry_for(scene)
    candidate=Registry(); candidate.marks=list(reg.marks); candidate._series_slugs=dict(reg._series_slugs)
    for mark in descriptors: candidate.add(mark)
    if sequence and scene.parts and not scene.sequence: raise ValueError('cannot mix named states and a sequence in one scene')
    if scene.sequence or sequence:
        existing=scene.state_names
        for p in parts:
            if existing and list(p.states)!=existing: raise ValueError('all sequence parts must have the same ordered frame names')
            existing=list(p.states)
    if morph_group is not None and scene.morph_group not in (None,str(morph_group)): raise ValueError('one scene cannot belong to two morph groups')
    reg.marks=candidate.marks; reg._series_slugs=candidate._series_slugs
    scene.parts.extend(parts); scene._alloc=alloc
    if legend: scene._legend_entries.extend(p.id for p in parts)
    scene.sequence=scene.sequence or sequence
    if morph_group is not None: scene.morph_group=str(morph_group)
    return parts


def mesh3d(scene, mesh, *, series, color=None, palette=None, legend=False, label=None,
           states=None, sequence=False, share_topology_with=None, morph_group=None, max_faces=None):
    """Add one mesh or a named mapping of meshes; return its addressable MeshParts.

    Optional ``max_faces`` simplifies each part within an apportioned total budget. Save
    the reference first with that budget, then use ``share_topology_with=reference_scene``
    to replay its exact collapses for another shape. ``states`` obeys the same face check.
    """
    import matplotlib as mpl
    if not isinstance(scene,Scene3D): raise TypeError('first argument must be fp.scene3d()')
    if max_faces is not None and (not isinstance(max_faces,int) or max_faces<1): raise ValueError('max_faces must be a positive integer')
    named=isinstance(mesh,dict)
    meshes=mesh if named else {'mesh':mesh}
    if not meshes: raise ValueError('mesh mapping is empty')
    if max_faces is not None and max_faces<len(meshes):
        raise ValueError('max_faces must allow at least one triangle per named part')
    shapes=_states(states,sequence)
    default=mpl.rcParams['axes.prop_cycle'].by_key()['color'][len(scene.parts)%len(mpl.rcParams['axes.prop_cycle'].by_key()['color'])]
    alloc=deepcopy(scene._alloc); parts=[]; marks=[]
    loaded={name:load_mesh(spec) for name,spec in meshes.items()}
    from ._mesh_reduce import face_budgets
    budgets=face_budgets([len(f) for v,f in loaded.values()],max_faces)
    for (name,(v,f)),budget in zip(loaded.items(),budgets):
        pid=alloc.take(series_id(series,slugify(name)))
        pcolor=(palette or {}).get(name,color or default)
        partstates={k:(val[name] if named and isinstance(val,dict) else val) for k,val in shapes.items()}
        ref=_reference(share_topology_with,pid,single=not named)
        part=_prepare_part(pid,(v,f),pcolor,partstates,reference=ref,max_faces=budget)
        parts.append(part)
        marks.append(Mark(role='mesh',series=str(series),name=str(name),kind='mesh',label=label or str(name),gid=pid,
                          data={'scene3d':{'id':pid,'role':'mesh','kind':'mesh','node':pid,'series':str(series),'label':label or str(name),'color':part.color}}))
    return _publish(scene,parts,marks,alloc,legend=legend,sequence=sequence,morph_group=morph_group)


class MorphResult(dict):
    """Mapping with ``ok``, ``pairs`` and optional ``reason``; truth tests use ``ok``."""
    def __bool__(self): return self['ok']


def can_morph(a,b):
    """Check ordered faces and vertex counts, pairing named parts as Flux does."""
    def parts(value):
        if isinstance(value,Scene3D): return value.parts
        if isinstance(value,MeshPart): return [value]
        if isinstance(value,list) and value and isinstance(value[0],MeshPart): return value
        v,f=load_mesh(value); used=np.unique(f)
        return [MeshPart('',v[used],np.searchsorted(used,f),'#4385BE')]
    aa,bb=parts(a),parts(b)
    if len(aa)!=len(bb): return MorphResult(ok=False,pairs=[],reason=f'{len(aa)} vs {len(bb)} parts')
    byname=all(p.id for p in aa+bb); lookup={p.id:p for p in bb}; pairs=[]
    for i,ap in enumerate(aa):
        bp=lookup.get(ap.id) if byname else bb[i]
        if bp is None: return MorphResult(ok=False,pairs=[],reason=f'{ap.id}: missing partner')
        if len(ap.vertices)!=len(bp.vertices): return MorphResult(ok=False,pairs=[],reason=f'{ap.id or i}: {len(ap.vertices)} vs {len(bp.vertices)} vertices')
        if not np.array_equal(ap.faces,bp.faces): return MorphResult(ok=False,pairs=[],reason=f'{ap.id or i}: face indices differ')
        pairs.append({'nodeA':ap.id,'nodeB':bp.id})
    return MorphResult(ok=True,pairs=pairs)
