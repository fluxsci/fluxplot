"""Named categorical and continuous 3D surface fields with the 2D surface color law."""
from __future__ import annotations
from copy import deepcopy
import warnings
import numpy as np
from matplotlib.colors import to_hex, to_rgba
from matplotlib.ticker import MaxNLocator
from ._fieldmap import _normalise_missing, infer_kind, categorical_colors, continuous_mapping
from .surface import _face_labels, _face_values
from .mesh3d import load_mesh, _states, _prepare_part, _reference, _publish
from .scene3d import Scene3D
from .ids import series_id, slugify
from .descriptors import Mark


def surface3d(scene,values,*,series,surfaces,kind='auto',categories=None,palette=None,cmap=None,
              color_range=None,percentile=None,missing_below=None,missing_values=(),missing_color='#D8D8D8',
              colorbar=False,cbar_label=None,cbar_ticks=None,legend=None,legend_missing=False,label=None,
              states=None,sequence=False,share_topology_with=None,morph_group=None,max_faces=None):
    """Add a value map on meshes, preserving raw vertex values for live Flux remapping.

    Geometry may be one mesh or a named mesh mapping. Values may likewise be a mapping or
    one concatenated array in mesh order. Categorical boundaries use majority-face labels;
    a face touching missing data becomes a separate missing part. Continuous mesh colors
    interpolate per vertex; 2D surface colors its per-face mean with the same color mapping.
    """
    if not isinstance(scene,Scene3D): raise TypeError('first argument must be fp.scene3d()')
    if max_faces is not None and (not isinstance(max_faces,int) or max_faces<1): raise ValueError('max_faces must be positive')
    meshes=surfaces if isinstance(surfaces,dict) else {'mesh':surfaces}
    if not meshes: raise ValueError('surfaces mapping is empty')
    shapes=_states(states,sequence)
    verts=[]; faces=[]; vals=[]; state_parts={k:[] for k in shapes}; offset=0
    flat=None if isinstance(values,dict) else np.asarray(values,dtype=float)
    for name,spec in meshes.items():
        v,f=load_mesh(spec); n=len(v)
        val=np.asarray(values[name],dtype=float) if isinstance(values,dict) else flat[offset:offset+n]
        if val.shape!=(n,): raise ValueError(f'{name}: exactly one value per vertex is required')
        vals.append(_normalise_missing(val,missing_below,missing_values))
        verts.append(v); faces.append(f+offset)
        for state,target in shapes.items():
            sv,sf=load_mesh(target[name] if isinstance(target,dict) else target)
            if sv.shape!=v.shape or not np.array_equal(sf,f): raise ValueError(f'state {state}: face indices or vertex count differ')
            state_parts[state].append(sv)
        offset+=n
    if flat is not None and flat.shape!=(offset,): raise ValueError('values length must equal the total vertex count')
    v=np.concatenate(verts); f=np.concatenate(faces); data=np.concatenate(vals)
    finite=data[np.isfinite(data)]; kind=infer_kind(finite,kind)
    if not finite.size: warnings.warn('surface3d(): every vertex is missing; drawing no-data',stacklevel=2)
    # Make all scientific non-finites missing before face assignment and serialization.
    data[~np.isfinite(data)]=np.nan
    fv=_face_labels(data,f) if kind=='label' else _face_values(data,f)
    missing=~np.isfinite(fv)
    groups={}
    field_spec=None; rgba=None
    if kind=='label':
        for code in np.unique(fv[~missing]):
            name=str((categories or {}).get(int(code),f'category-{int(code)}'))
            if name=='missing': raise ValueError("category name 'missing' is reserved for no-data")
            groups.setdefault(name,np.zeros(len(f),dtype=bool))
            groups[name]|=fv==code
        resolved=categorical_colors(groups,palette,categories)
    else:
        cmap_obj,norm=continuous_mapping(finite,cmap,color_range,percentile)
        rgba=np.asarray(cmap_obj(norm(data)))
        rgba[~np.isfinite(data)]=to_rgba(missing_color)
        # Dense stops preserve the used mapping as portable data; no cmap package in Flux.
        points=np.linspace(0,1,256)
        field_spec={'cmap':{'name':cmap_obj.name,'stops':[[float(x),to_hex(cmap_obj(x),keep_alpha=False)] for x in points]},
                    'range':[float(norm.vmin),float(norm.vmax)],'missingColor':to_hex(missing_color)}
        if percentile is not None: field_spec['rule']={'percentile':list(percentile)}
        if cbar_label is not None: field_spec['label']=str(cbar_label)
        ticks=list(cbar_ticks) if cbar_ticks is not None else MaxNLocator(nbins=5).tick_values(norm.vmin,norm.vmax)
        field_spec['ticks']=[float(x) for x in ticks if norm.vmin<=x<=norm.vmax]
        groups['field']=~missing; resolved={'field':'#FFFFFF'}
    if missing.any(): groups['missing']=missing; resolved['missing']=to_hex(missing_color)
    if max_faces is not None and max_faces<sum(bool(mask.any()) for mask in groups.values()):
        raise ValueError('max_faces must allow at least one triangle per semantic part')
    allstates={k:np.concatenate(a) for k,a in state_parts.items()}
    alloc=deepcopy(scene._alloc); parts=[]; marks=[]; legend_ids=[]
    for name,mask in sorted(groups.items()):
        if not mask.any(): continue
        pid=alloc.take(series_id(series,slugify(name)))
        ref=_reference(share_topology_with,pid,single=len(groups)==1)
        # States and reference must use this exact face subset so category boundaries persist.
        localstates={k:(a,f[mask]) for k,a in allstates.items()}
        budget=max(1,int(max_faces*mask.sum()/len(f))) if max_faces is not None else None
        part=_prepare_part(pid,(v,f[mask]),resolved[name],localstates,reference=ref,max_faces=budget,
                           values=data if kind=='continuous' or name=='missing' else None,
                           colors=rgba if kind=='continuous' and name=='field' else None)
        if kind=='continuous' and name=='field':
            part.colors=np.asarray(cmap_obj(norm(part.values)))
            part.colors[~np.isfinite(part.values)]=to_rgba(missing_color)
        partkind='missing' if name=='missing' else ('field' if kind=='continuous' else 'mesh')
        spec={'id':pid,'role':'surface-field' if partkind=='field' else 'surface-region','kind':partkind,
              'node':pid,'series':str(series),'label':str(label or name),'color':part.color}
        if partkind=='field': spec['field']=field_spec
        parts.append(part); marks.append(Mark(role=spec['role'],series=str(series),name=name,kind='surface',label=label,gid=pid,data={'scene3d':spec}))
        if partkind!='missing' or legend_missing: legend_ids.append(pid)
    field_ids=[m.gid for m in marks if m.data['scene3d']['kind']=='field']
    if colorbar and field_ids:
        pid=alloc.take(series_id(series,'colorbar')); field_id=field_ids[0]
        marks.append(Mark(role='colorbar',series=str(series),name='colorbar',gid=pid,data={'scene3d':{'id':pid,'role':'colorbar','kind':'furniture','field':field_id}}))
    result=_publish(scene,parts,marks,alloc,sequence=sequence,morph_group=morph_group)
    if legend if legend is not None else kind=='label': scene._legend_entries.extend(legend_ids)
    return result
