"""Duck-typed mesh input, stable part identity and checked topology correspondence."""
from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import warnings
import numpy as np
from matplotlib.colors import to_hex, to_rgba
from .scene3d import Scene3D, MeshPart, _number, _positive_int
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
    elif _is_trimesh_scene(mesh):
        raise TypeError('a trimesh.Scene holds several meshes; pass it as the whole mesh argument '
                        'of fp.mesh3d (one named part per scene node), not as a single shape')
    else: raise TypeError('mesh must be (vertices, faces), a mesh object, or a mesh file path')
    v=np.asarray(v,dtype=float)
    raw=np.asarray(f)
    if v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all(): raise ValueError('vertices must be a nonempty finite N×3 array')
    if raw.ndim!=2 or raw.shape[1]!=3 or not len(raw): raise ValueError('faces must be a nonempty M×3 triangle array')
    if not np.issubdtype(raw.dtype,np.integer) and (not np.isfinite(raw).all() or not np.equal(raw,np.floor(raw)).all()): raise ValueError('face indices must be integers')
    if raw.min()<0 or raw.max()>=len(v): raise ValueError('face index outside vertex array')
    return v.copy(),raw.astype(np.int64,copy=True)


def _is_trimesh_scene(obj):
    graph = getattr(obj, 'graph', None)
    return isinstance(getattr(obj, 'geometry', None), Mapping) and hasattr(graph, 'nodes_geometry')


def _trimesh_scene_parts(scene):
    """``{node name: (vertices, faces)}`` with each node's scene transform applied."""
    parts = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        geometry = scene.geometry[geometry_name]
        if not (hasattr(geometry, 'vertices') and hasattr(geometry, 'faces')) or not len(geometry.faces):
            continue  # point clouds and paths have no triangles to draw
        transform = np.asarray(transform, dtype=float)
        vertices = np.asarray(geometry.vertices, dtype=float) @ transform[:3, :3].T + transform[:3, 3]
        parts[str(node)] = (vertices, np.asarray(geometry.faces))
    if not parts:
        raise ValueError('the trimesh.Scene contains no triangle meshes')
    return parts


def _named_meshes(mesh):
    """Return ``(meshes, named)``: a mapping of part name to mesh spec."""
    if isinstance(mesh, Mapping):
        return dict(mesh), True
    if _is_trimesh_scene(mesh):
        return _trimesh_scene_parts(mesh), True
    return {'mesh': mesh}, False


def _states(states, sequence):
    if states is None: return {}
    if isinstance(states,Mapping):
        result={str(k):v for k,v in states.items()}
        if any(not k for k in result) or len(result)!=len(states): raise ValueError('state names must be nonempty and unique')
        return result
    if not sequence: raise ValueError('a list of states requires sequence=True (or pass a dict of named states)')
    return {f'frame-{i+1}':v for i,v in enumerate(states)}


def _is_bare_vertices(shape):
    """True for an N×3 vertex array; a 2-item tuple/list is a ``(vertices, faces)`` pair."""
    if isinstance(shape, (tuple, list)) and len(shape) == 2:
        return False
    if not isinstance(shape, (np.ndarray, tuple, list)):
        return False
    try:
        array = np.asarray(shape, dtype=float)
    except (TypeError, ValueError):
        return False
    return array.ndim == 2 and array.shape[1] == 3


def _state_for_part(shape, state, part, named):
    """Pick one part's shape out of a named-part state mapping, with a clear error."""
    if not (named and isinstance(shape, Mapping)):
        return shape
    if part not in shape:
        have = ', '.join(map(str, shape)) or 'no parts'
        raise ValueError(f'shape state {state!r} has no shape for part {part!r} (it has {have}); '
                         'give every named part a shape in each state')
    return shape[part]


def _state_vertices(shape, vertices, faces, state, part_id):
    """Vertices of one shape state. Bare N×3 arrays reuse the base faces."""
    if _is_bare_vertices(shape):
        state_vertices = np.asarray(shape, dtype=float)
        if not np.isfinite(state_vertices).all():
            raise ValueError(f'shape state {state!r} of {part_id}: vertices must be finite')
        state_faces = faces
    else:
        state_vertices, state_faces = load_mesh(shape)
    if len(state_vertices) != len(vertices):
        raise ValueError(f'shape state {state!r} of {part_id} has {len(state_vertices):,} vertices but its '
                         f'base mesh has {len(vertices):,}; each state must be the same mesh moved '
                         '(same vertex count and faces), shared topology required')
    if state_faces.shape != faces.shape or not np.array_equal(state_faces, faces):
        raise ValueError(f'shape state {state!r} of {part_id} has the base vertex count but different '
                         'face indices; shared topology required (pass just its N×3 vertex array to reuse '
                         'the base faces)')
    return state_vertices


def _same_topology(vertices, faces, ref_vertices, ref_faces, label):
    if len(vertices)!=len(ref_vertices):
        raise ValueError(f'{label}: this mesh has {len(vertices):,} vertices but the reference has '
                         f'{len(ref_vertices):,}; shared topology required')
    if faces.shape!=ref_faces.shape or not np.array_equal(faces,ref_faces):
        raise ValueError(f'{label}: same vertex count as the reference but the face indices differ; '
                         'shared topology required')


def _match_part(parts, part_id, what):
    found = next((p for p in parts if p.id == part_id), None)
    if found is None:
        have = ', '.join(p.id for p in parts) or 'no parts'
        raise ValueError(f'{part_id}: the {what} has no part with this id (it has {have}); use the '
                         'same series= and part names as the reference so Flux can pair them by name')
    return found


def _reference(ref, part_id, *, single):
    """The reference MeshPart (or raw mesh) that ``part_id`` must share topology with."""
    if ref is None: return None
    if isinstance(ref, Scene3D):
        return _match_part(ref.parts, part_id, 'reference scene')
    if isinstance(ref, MeshPart): return ref
    if isinstance(ref, (list, tuple)) and ref and all(isinstance(p, MeshPart) for p in ref):
        # The list returned by fp.mesh3d / fp.surface3d; pair by part id.
        return _match_part(ref, part_id, 'reference part list')
    if not single and isinstance(ref, Mapping):
        for key,value in ref.items():
            if part_id.endswith('.'+slugify(key)): return value
        raise ValueError(f'{part_id}: reference mapping has no matching part')
    return ref


def _prepare_part(part_id, mesh, color, states=None, *, reference=None, max_faces=None,
                  values=None, colors=None):
    v,f=load_mesh(mesh)
    state_arrays={name:_state_vertices(shape,v,f,name,part_id) for name,shape in (states or {}).items()}
    if reference is not None:
        if isinstance(reference,MeshPart):
            if reference.source_count!=len(v):
                raise ValueError(f'{part_id}: this mesh has {len(v):,} vertices but its reference '
                                 f'{reference.id} has {reference.source_count:,}; share_topology_with '
                                 'needs the same mesh topology')
            if not np.array_equal(reference.source_faces,f):
                raise ValueError(f'{part_id}: same vertex count as its reference {reference.id} but the '
                                 'face indices differ; share_topology_with needs identical faces')
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
    scene._max_call_parts=max(scene._max_call_parts,len(parts))
    if legend: scene._legend_entries.extend(p.id for p in parts)
    scene.sequence=scene.sequence or sequence
    if morph_group is not None: scene.morph_group=str(morph_group)
    scene._resolve_pending_view(strict=False)
    return parts


def _check_max_faces(max_faces):
    return None if max_faces is None else _positive_int(max_faces, 'max_faces')


def _check_alpha(alpha):
    return None if alpha is None else _number(alpha, 'alpha', 0, 1)


def _with_alpha(color, alpha):
    return color if alpha is None else to_rgba(color, alpha)


def _warn_unknown_palette(palette, known, *, hint=''):
    if not palette:
        return
    unknown = [key for key in palette if key not in known]
    if unknown:
        names = ', '.join(repr(k) for k in known) or 'none'
        warnings.warn(f'palette keys {unknown!r} match no part and are ignored (parts: {names}){hint}',
                      stacklevel=3)


def mesh3d(scene, mesh, *, series, color=None, palette=None, alpha=None, legend=False, label=None,
           states=None, sequence=False, share_topology_with=None, morph_group=None, max_faces=None):
    """Add one mesh, or a named mapping of meshes, as addressable parts of a 3D scene.

    Parameters
    ----------
    scene
        The :func:`fp.scene3d <fluxplot.scene3d>` to add to.
    mesh
        One mesh, or ``{name: mesh}`` for named parts (``"soma"``, ``"axon"``, …), each
        of which Flux can recolour, fade or hide separately. A mesh is a
        ``(vertices, faces)`` pair, an object with ``.vertices``/``.faces`` (trimesh,
        meshparty, cloudvolume), PyVista ``PolyData``, a GIFTI path (needs ``nibabel``)
        or a ``.ply/.obj/.stl/.off`` path (needs ``trimesh``). A ``trimesh.Scene``
        becomes one named part per node.
    series
        Name of this series; parts are ``<series>.mesh`` or ``<series>.<name>``. Ids are
        stable across reruns, so edits made in Flux survive regeneration.
    color, palette
        ``color`` colours every part; ``palette={name: color}`` colours named parts. Parts
        without either take successive colours of the house cycle, so a legend can tell
        them apart.
    alpha
        Opacity in [0, 1] for every part of this call (overrides any alpha in the
        colours). Flux can still change it per part.
    legend
        Add these parts to the scene legend.
    label
        Display label. For a single mesh it labels the part (e.g. in the legend). For a
        named mapping it names the whole series, as in :func:`fp.surface
        <fluxplot.surface>`; each part keeps its mapping key as its label.
    states
        Other shapes of the same mesh, shown as weight sliders in Flux:
        ``{"inflated": inflated}``. A state is the base mesh moved, so it may be a bare
        N×3 vertex array or a mesh with identical faces (checked). For named parts give
        ``{state: {part: shape}}``.
    sequence
        With a list of states (``states=frames[1:]``), mark a time sequence: frame 0 is
        this mesh and each state is the next frame (``sc.view(frame=3)``).
    share_topology_with
        A reference scene, the list of parts a previous ``mesh3d`` call returned, or a
        mesh with the same faces. The faces are checked, and a decimated reference's
        exact collapses are replayed, so the two stay morphable in Flux. Build the
        reference first (with any ``max_faces``), use the same ``series`` and part
        names, then pass it here.
    morph_group
        A label (e.g. ``"cortex"``) suggesting morph partners in Flux.
    max_faces
        Optional triangle budget for this call, apportioned over its parts (needs the
        ``fluxplot[mesh]`` extra). Save warnings recommend a value for large scenes.

    Returns
    -------
    list of MeshPart
        The new parts, in order.

    Examples
    --------
    >>> sc = fp.scene3d(units="nm", scalebar=10_000)
    >>> fp.mesh3d(sc, {"soma": soma, "axon": axon}, series="neuron", legend=True,
    ...           palette={"soma": fp.colors.red400, "axon": fp.colors.blue400})
    >>> fp.mesh3d(sc, pial, series="cortex", states={"inflated": inflated_vertices})
    """
    import matplotlib as mpl
    if not isinstance(scene,Scene3D): raise TypeError('first argument must be fp.scene3d()')
    max_faces=_check_max_faces(max_faces)
    alpha=_check_alpha(alpha)
    meshes,named=_named_meshes(mesh)
    if not meshes: raise ValueError('mesh mapping is empty')
    if max_faces is not None and max_faces<len(meshes):
        raise ValueError('max_faces must allow at least one triangle per named part')
    _warn_unknown_palette(palette, list(meshes))
    shapes=_states(states,sequence)
    cycle=mpl.rcParams['axes.prop_cycle'].by_key()['color']
    alloc=deepcopy(scene._alloc); parts=[]; marks=[]
    loaded={name:load_mesh(spec) for name,spec in meshes.items()}
    from ._mesh_reduce import face_budgets
    budgets=face_budgets([len(f) for v,f in loaded.values()],max_faces)
    for index,((name,(v,f)),budget) in enumerate(zip(loaded.items(),budgets)):
        pid=alloc.take(series_id(series,slugify(name)))
        # Each uncoloured part advances the house cycle, deterministically by scene order.
        default=cycle[(len(scene.parts)+index)%len(cycle)]
        pcolor=_with_alpha((palette or {}).get(name,color or default),alpha)
        partstates={state:_state_for_part(shape,state,name,named) for state,shape in shapes.items()}
        ref=_reference(share_topology_with,pid,single=not named)
        part=_prepare_part(pid,(v,f),pcolor,partstates,reference=ref,max_faces=budget)
        parts.append(part)
        part_label=str(name) if named or label is None else str(label)
        series_label=part_label if label is None else str(label)
        spec={'id':pid,'role':'mesh','kind':'mesh','node':pid,'series':str(series),'label':part_label,'color':part.color}
        marks.append(Mark(role='mesh',series=str(series),name=str(name),kind='mesh',label=series_label,gid=pid,
                          data={'scene3d':spec}))
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
