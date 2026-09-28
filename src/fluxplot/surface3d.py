"""Named categorical and continuous 3D surface fields with the 2D surface color law."""
from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
import warnings
import numpy as np
from matplotlib.colors import to_hex, to_rgba
from matplotlib.ticker import MaxNLocator
from ._fieldmap import _normalise_missing, infer_kind, categorical_colors, category_name, continuous_mapping
from .surface import _face_labels, _face_values
from .mesh3d import (load_mesh, _states, _prepare_part, _reference, _publish, _named_meshes,
                     _state_for_part, _state_vertices, _check_max_faces, _check_alpha, _with_alpha,
                     _warn_unknown_palette)
from .scene3d import Scene3D, MeshPart, _clean_float
from .ids import series_id, slugify
from .descriptors import Mark


def _face_rows_subset(subset, faces):
    """True when every triangle of ``subset`` is also a triangle of ``faces``."""
    def rows(a):
        a = np.ascontiguousarray(a, dtype=np.int64)
        return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).ravel()
    return bool(np.isin(rows(subset), rows(faces)).all())


def surface3d(scene,values,*,series,surfaces,kind='auto',categories=None,palette=None,cmap=None,
              color_range=None,percentile=None,missing_below=None,missing_values=(),missing_color='#D8D8D8',
              colorbar=False,cbar_label=None,cbar_ticks=None,legend=None,legend_missing=False,label=None,
              alpha=None,states=None,sequence=False,share_topology_with=None,morph_group=None,max_faces=None):
    """Add a per-vertex value map on a mesh: a categorical atlas or a continuous field.

    The value-to-colour law is the one :func:`fp.surface <fluxplot.surface>` uses (same
    arguments and missing-data rules). The raw values are saved too, so Flux can change
    the colormap and limits without rerunning Python.

    Parameters
    ----------
    scene
        The :func:`fp.scene3d <fluxplot.scene3d>` to add to.
    values
        One value per vertex: an array in mesh order (concatenated when ``surfaces`` is a
        mapping) or ``{name: array}`` matching ``surfaces``.
    series
        Part-id prefix: label maps give ``<series>.<category>``, continuous maps
        ``<series>.field``; faces touching missing data form ``<series>.missing``.
    surfaces
        One mesh (any :func:`mesh3d` input) or ``{name: mesh}``, e.g. both hemispheres.
        A mapping is drawn as one combined mesh; for separately hideable hemispheres,
        call ``surface3d`` once per hemisphere with its own series and a shared
        ``color_range``.
    kind
        ``"label"`` (``"categorical"``), ``"continuous"``, or ``"auto"`` (label when the
        data are integral and have at most 32 distinct values).
    categories
        ``{code: name}`` for label maps. Unnamed codes become ``category-<code>``
        (negative codes ``category-m<abs>``).
    palette
        ``{name_or_code: color}`` for label maps; fixed, never remapped. Codes need
        ``categories``.
    cmap, color_range, percentile
        Continuous styling. ``cmap`` is a Colormap or a name: matplotlib's, or fluxplot's
        collections (``"emerald"``, ``"crameri.batlow"``). ``percentile=(2, 98)`` clips
        to those percentiles; an explicit ``color_range=(lo, hi)`` wins. Pass the same
        ``color_range`` to several calls (e.g. two hemispheres) for identical colours.
    missing_below, missing_values, missing_color
        Values below ``missing_below`` or in ``missing_values`` are missing, like NaN
        and infinities; zero stays a real value. Missing faces use ``missing_color``.
    colorbar, cbar_label, cbar_ticks
        Add a colorbar for a continuous map, with an optional title and fixed ticks.
    legend, legend_missing
        Add the category parts to the scene legend (default: on for label maps). A
        continuous field has no legend swatch; its key is the colorbar.
        ``legend_missing=True`` adds the missing-data part.
    label
        For a continuous map, the field's display label. For a label map it names the
        whole series (as in ``fp.surface``); categories keep their names.
    alpha
        Opacity in [0, 1] for every part of this call.
    states, sequence, share_topology_with, morph_group, max_faces
        As in :func:`mesh3d`. States are other shapes of the same mesh (bare N×3
        vertex arrays are fine), e.g. ``states={"inflated": inflated}``.

    Returns
    -------
    list of MeshPart
        The new parts, in order.

    Examples
    --------
    >>> cx = fp.scene3d(figsize=(3, 3), units="mm")
    >>> fp.surface3d(cx, thickness, series="thickness", surfaces=pial, kind="continuous",
    ...              cmap="emerald", percentile=(2, 98), colorbar=True,
    ...              cbar_label="Thickness (mm)", states={"inflated": inflated})
    """
    if not isinstance(scene,Scene3D): raise TypeError('first argument must be fp.scene3d()')
    max_faces=_check_max_faces(max_faces)
    alpha=_check_alpha(alpha)
    meshes,named=_named_meshes(surfaces)
    if not meshes: raise ValueError('surfaces mapping is empty')
    shapes=_states(states,sequence)
    verts=[]; faces=[]; vals=[]; state_parts={k:[] for k in shapes}; offset=0
    flat=None if isinstance(values,Mapping) else np.asarray(values,dtype=float)
    for name,spec in meshes.items():
        v,f=load_mesh(spec); n=len(v)
        if isinstance(values,Mapping):
            if name not in values: raise ValueError(f'values has no array for surface {name!r}')
            val=np.asarray(values[name],dtype=float)
        else:
            val=flat[offset:offset+n]
        if val.shape!=(n,): raise ValueError(f'{name}: exactly one value per vertex is required ({len(val)} values for {n} vertices)')
        vals.append(_normalise_missing(val,missing_below,missing_values))
        verts.append(v); faces.append(f+offset)
        for state,target in shapes.items():
            shape=_state_for_part(target,state,name,named)
            state_parts[state].append(_state_vertices(shape,v,f,state,f'{series}.{name}' if named else str(series)))
        offset+=n
    if flat is not None and flat.shape!=(offset,): raise ValueError(f'values length must equal the total vertex count ({flat.size} values for {offset} vertices)')
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
            name=str(category_name(code,categories))
            if name=='missing': raise ValueError("category name 'missing' is reserved for no-data")
            groups.setdefault(name,np.zeros(len(f),dtype=bool))
            groups[name]|=fv==code
        known=[*groups,*(categories or {}),*(categories or {}).values()]
        _warn_unknown_palette(palette,list(dict.fromkeys(known)),
                              hint='; palette codes need categories={code: name}, or key it by part name')
        resolved=categorical_colors(groups,palette,categories)
    else:
        if palette:
            warnings.warn('palette= applies to label maps and is ignored for a continuous field; use cmap=',stacklevel=2)
        cmap_obj,norm=continuous_mapping(finite,cmap,color_range,percentile)
        rgba=np.asarray(cmap_obj(norm(data)))
        rgba[~np.isfinite(data)]=to_rgba(missing_color)
        # Dense stops preserve the used mapping as portable data; no cmap package in Flux.
        points=np.linspace(0,1,256)
        field_spec={'cmap':{'name':cmap_obj.name,'stops':[[float(x),to_hex(cmap_obj(x),keep_alpha=False)] for x in points]},
                    'range':[float(norm.vmin),float(norm.vmax)],'missingColor':to_hex(missing_color)}
        if percentile is not None: field_spec['rule']={'percentile':list(percentile)}
        if cbar_label is not None: field_spec['label']=str(cbar_label)
        if cbar_ticks is not None:
            ticks=[float(x) for x in cbar_ticks]
        else:
            ticks=[_clean_float(x) for x in MaxNLocator(nbins=5).tick_values(norm.vmin,norm.vmax)]
        field_spec['ticks']=[x for x in ticks if norm.vmin<=x<=norm.vmax]
        groups['field']=~missing; resolved={'field':'#FFFFFF'}
    if missing.any(): groups['missing']=missing; resolved['missing']=to_hex(missing_color)
    if max_faces is not None and max_faces<sum(bool(mask.any()) for mask in groups.values()):
        raise ValueError('max_faces must allow at least one triangle per semantic part')
    allstates={k:np.concatenate(a) for k,a in state_parts.items()}
    alloc=deepcopy(scene._alloc); parts=[]; marks=[]; legend_ids=[]
    from ._mesh_reduce import face_budgets
    nonempty=[(name,mask) for name,mask in sorted(groups.items()) if mask.any()]
    budgets=face_budgets([mask.sum() for name,mask in nonempty],max_faces)
    for (name,mask),budget in zip(nonempty,budgets):
        pid=alloc.take(series_id(series,slugify(name)))
        ref=_reference(share_topology_with,pid,single=len(groups)==1)
        if (isinstance(ref,MeshPart) and ref.source_count==len(v) and not np.array_equal(ref.source_faces,f[mask])
                and _face_rows_subset(ref.source_faces,f)):
            what='category labels' if kind=='label' and name!='missing' else 'missing-data vertices'
            raise ValueError(f'{pid}: same mesh as its reference {ref.id}, but the {what} differ, so this part '
                             'covers different faces; morph partners need identical labels and missing data')
        # States and reference must use this exact face subset so category boundaries persist.
        part=_prepare_part(pid,(v,f[mask]),_with_alpha(resolved[name],alpha),allstates,reference=ref,max_faces=budget,
                           values=data if kind=='continuous' or name=='missing' else None,
                           colors=rgba if kind=='continuous' and name=='field' else None)
        if kind=='continuous' and name=='field':
            part.colors=np.asarray(cmap_obj(norm(part.values)))
            part.colors[~np.isfinite(part.values)]=to_rgba(missing_color)
        partkind='missing' if name=='missing' else ('field' if kind=='continuous' else 'mesh')
        # A continuous field is the call's one data part, so label= names it; category
        # parts keep their names and label= names the series (as in 2D surface).
        part_label=str(label) if partkind=='field' and label is not None else str(name)
        spec={'id':pid,'role':'surface-field' if partkind=='field' else 'surface-region','kind':partkind,
              'node':pid,'series':str(series),'label':part_label,'color':part.color}
        if partkind=='field': spec['field']=field_spec
        parts.append(part); marks.append(Mark(role=spec['role'],series=str(series),name=name,kind='surface',label=label,gid=pid,data={'scene3d':spec}))
        if partkind=='mesh' or (partkind=='missing' and legend_missing): legend_ids.append(pid)
    field_ids=[m.gid for m in marks if m.data['scene3d']['kind']=='field']
    if colorbar and field_ids:
        pid=alloc.take(series_id(series,'colorbar')); field_id=field_ids[0]
        marks.append(Mark(role='colorbar',series=str(series),name='colorbar',gid=pid,data={'scene3d':{'id':pid,'role':'colorbar','kind':'furniture','field':field_id}}))
    if legend and kind=='continuous' and not legend_missing:
        warnings.warn('a continuous field has no legend swatch; use colorbar=True for its key '
                      '(legend_missing=True still lists the missing-data part)',stacklevel=2)
    result=_publish(scene,parts,marks,alloc,sequence=sequence,morph_group=morph_group)
    if legend if legend is not None else kind=='label': scene._legend_entries.extend(legend_ids)
    return result
