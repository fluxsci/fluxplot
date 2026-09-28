"""Semantic scene manifests assembled from the shared Mark/Registry machinery."""
from __future__ import annotations
import hashlib
import json
import os
import sys
from importlib.resources import files
from copy import deepcopy
from dataclasses import dataclass, field
import warnings
import numpy as np
from matplotlib.ticker import MaxNLocator
from .scene3d import SCENE3D_SPEC_VERSION, _clean_float, scalebar_text
from .tagger import registry_for, Registry
from .descriptors import Mark
from .ids import IdAllocator
from .canonical_json import dumps
from .version import __version__, SPEC_VERSION


def build_manifest(scene, glb, filename):
    scene._resolve_pending_view(strict=True)
    parts=[deepcopy(m.data['scene3d']) for m in registry_for(scene).marks if 'scene3d' in m.data]
    allv=np.concatenate([v for p in scene.parts for v in [p.vertices,*p.states.values()]])
    world=allv@scene.to_world[:3,:3].T
    reg=Registry(); alloc=IdAllocator()
    for p in parts: alloc.take(p['id'])
    def guide(pid,role,**data):
        gid=alloc.take(pid); spec=dict(id=gid,role=role,kind='furniture',**data)
        reg.add(Mark(role=role,name=pid,gid=gid,data={'scene3d':spec}))
    axes={'kind':scene.axes}
    if scene.axes!='none':
        axes['grid']=True
        for i,k in enumerate('xyz'):
            lo,hi=float(allv[:,i].min()),float(allv[:,i].max())
            if lo==hi: lo-=.5; hi+=.5
            spec={'lim':[lo,hi],'label':f'{k} ({scene.units})' if scene.units else k}
            spec.update(scene._axis_specs.get(k,{}))
            if 'ticks' not in spec:
                ticks=[_clean_float(x) for x in MaxNLocator(nbins=4).tick_values(*spec['lim'])]
                spec['ticks']=[x for x in ticks if spec['lim'][0]<=x<=spec['lim'][1]]
            axes[k]=spec
            for suffix,role in [('axis','axis'),('pane','pane'),('grid','gridline'),('ticks','tick-label')]: guide(f'axes.{k}.{suffix}',role)
            guide(f'axes.{k}.label','axis-title',text=spec['label'])
    if scene.scalebar is not None:
        # length stays in data units; only the label is simplified (10000 nm -> 10 µm).
        text=scene.scalebar_label if scene.scalebar_label is not None else scalebar_text(scene.scalebar,scene.units)
        guide('scalebar','scalebar',length=scene.scalebar,label=text)
    layout={}
    if scene.title:
        guide('title','title',text=str(scene.title)); layout['title']='top'
    if scene._legend_entries:
        guide('legend','legend',entries=list(scene._legend_entries)); layout['legend']='right'
    if any(p['role']=='colorbar' for p in parts): layout['colorbar']='right'
    parts.extend(m.data['scene3d'] for m in reg.marks)
    out={'spec':'fluxplot/scene3d','schemaVersion':SCENE3D_SPEC_VERSION,'plotType':'scene3d','glb':filename,
         'glbSha256':hashlib.sha256(glb).hexdigest(),'size':{'width':scene.figsize[0],'height':scene.figsize[1],'unit':'in'},
         'units':scene.units,'toWorld':scene.to_world.ravel(order='F').tolist(),
         'bounds':{'min':world.min(axis=0).tolist(),'max':world.max(axis=0).tolist()},
         'view':deepcopy(scene._view),'lighting':scene.lighting,'style':dict(scene.style),'axes':axes,
         'parts':parts,'order':[p['id'] for p in parts],'layout':layout,'build':{'generator':f'fluxplot {__version__}'}}
    if scene.state_names: out.update(states=[{'name':n,'label':n} for n in scene.state_names],sequence=scene.sequence)
    if scene.morph_group is not None: out['morphGroup']=scene.morph_group
    return out


@dataclass
class Scene3DSaveResult:
    glb: str
    manifest: str
    recipe: str
    warnings: list = field(default_factory=list)
    skipped: bool = False


def save_scene3d(scene,path,*,recipe=None,validate=True,_now=None):
    import fnmatch
    from .glb import write_glb
    from .recipe import build_recipe
    from .api import _write_staged
    path=os.fspath(path)
    # A dotted stem is still a stem; only recognized output suffixes are stripped.
    base=path[:-4] if path.lower().endswith('.glb') else path
    if base.lower().endswith(('.svg','.png','.pdf')): raise ValueError('save a Scene3D to a stem or .glb path')
    result=Scene3DSaveResult(base+'.glb',base+'.fluxplot.json',base+'.recipe.json')
    only=os.environ.get('FLUXPLOT_ONLY','').strip()
    patterns=[p.strip() for p in only.split(',') if p.strip()]
    plot_name=os.path.basename(base)
    if patterns and not any(fnmatch.fnmatchcase(plot_name,p) for p in patterns):
        # Same targeted-rerun report as the 2D save.
        print(f"fluxplot: skipped '{plot_name}' (FLUXPLOT_ONLY={only})",file=sys.stderr)
        result.skipped=True; result.warnings.append(f'skipped by FLUXPLOT_ONLY={only}'); return result
    data=write_glb(scene)
    triangles=sum(len(p.faces) for p in scene.parts)
    from ._scene3d_size import WARN_BYTES, WARN_TRIANGLES, face_cap_recommendation
    if len(data)>WARN_BYTES or triangles>WARN_TRIANGLES:
        message=f'3D plot has {triangles:,} triangles and {len(data)/1024**2:.1f} MiB; '+face_cap_recommendation(scene,data)
        result.warnings.append(message); warnings.warn(message,stacklevel=2)
    if len(scene.state_names)>20 or (scene.state_names and len(data)>100*1024**2):
        message=f'{len(scene.state_names)} shape states add 24 bytes per vertex per state; use fewer frames to reduce saved size'
        result.warnings.append(message); warnings.warn(message,stacklevel=2)
    man=build_manifest(scene,data,os.path.basename(result.glb))
    rec=build_recipe(recipe,plot_name=os.path.basename(base),glb_filename=os.path.basename(result.glb),
                     manifest_filename=os.path.basename(result.manifest),spec_version=SPEC_VERSION,
                     base_dir=os.getcwd(),recipe_dir=os.path.dirname(os.path.abspath(base)),now=_now)
    if validate:
        import jsonschema
        jsonschema.validate(man,json.loads(files('fluxplot').joinpath('schemas/scene3d.schema.json').read_text()))
        jsonschema.validate(rec,json.loads(files('fluxplot').joinpath('schemas/recipe.schema.json').read_text()))
    os.makedirs(os.path.dirname(os.path.abspath(base)),exist_ok=True)
    _write_staged([(result.glb,data),(result.manifest,dumps(man).encode()),(result.recipe,dumps(rec).encode())])
    return result
