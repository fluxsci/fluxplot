"""Notebook HTML using Flux's vendored renderer, bundled with a static PNG alternative."""
from __future__ import annotations
import base64
from copy import deepcopy
import hashlib
import io
import json
from importlib.resources import files
import warnings
import numpy as np


def preview_scene(scene):
    """Reduce a private notebook copy only; saving the original remains full resolution."""
    total=sum(len(p.faces) for p in scene.parts)
    if total<=scene.preview_max_faces: return scene
    try: import fast_simplification  # noqa: F401
    except ImportError: return scene
    from ._mesh_reduce import reduce_part,face_budgets
    from ._fieldmap import continuous_mapping
    from .tagger import registry_for
    from matplotlib.colors import to_rgba, Normalize, LinearSegmentedColormap
    if scene.preview_max_faces<len(scene.parts):
        warnings.warn('preview_max_faces cannot preserve one triangle per part; showing the full scene',stacklevel=3)
        return scene
    out=deepcopy(scene)
    specs={m.gid:m.data.get('scene3d',{}) for m in registry_for(out).marks}
    budgets=face_budgets([len(p.faces) for p in out.parts],out.preview_max_faces)
    for p,budget in zip(out.parts,budgets):
        if len(p.faces)>budget:
            reduce_part(p,max_faces=budget)
            field=specs.get(p.id,{}).get('field')
            if isinstance(field,dict) and p.colors is not None:
                try:
                    cmap,norm=continuous_mapping(p.values[np.isfinite(p.values)],field['cmap']['name'],field['range'],None)
                except (KeyError, ValueError):
                    cmap=LinearSegmentedColormap.from_list(field['cmap']['name'],field['cmap']['stops'])
                    norm=Normalize(*field['range'])
                p.colors=np.asarray(cmap(norm(p.values)))
                p.colors[~np.isfinite(p.values)]=to_rgba(field.get('missingColor','#D8D8D8'))
    return out


def png_preview(scene, *, _manifest=None):
    """Static orthographic/perspective painter preview using surface's projection/shading law.

    The interactive notebook and Flux views use a depth buffer. This fallback sorts all
    faces together; intersecting transparent surfaces remain an approximate painter view.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgba, Normalize, LinearSegmentedColormap
    from matplotlib.patches import Patch
    from .surface import _face_shading, _shade_rgba
    from .scene3d_manifest import build_manifest
    from .glb import write_glb
    if not scene.parts: raise ValueError('add a mesh before displaying a Scene3D')
    man=_manifest if _manifest is not None else build_manifest(scene,write_glb(scene),'preview.glb')
    view=scene._view; az,el,roll=np.deg2rad([view['azimuth']%360,view['elevation'],view.get('roll',0)])
    direction=np.array([np.sin(az)*np.cos(el),np.sin(el),np.cos(az)*np.cos(el)])
    right=np.array([np.cos(az),0,-np.sin(az)]); up=np.cross(direction,right)
    right,up=right*np.cos(roll)+up*np.sin(roll),up*np.cos(roll)-right*np.sin(roll)
    bounds=np.array([man['bounds']['min'],man['bounds']['max']]); center=bounds.mean(axis=0)
    radius=max(np.linalg.norm((bounds[1]-bounds[0])/2),1e-9)
    target=center+radius*(view.get('panX',0)*right+view.get('panY',0)*up)
    half=radius/view['zoom']; half_fov=np.deg2rad(view.get('fov',30))/2
    distance=half/np.sin(half_fov)
    if view['projection']=='perspective': half=distance*np.tan(half_fov)
    def project(points):
        delta=points-target; xy=np.column_stack([delta@right,delta@up]); depth=delta@direction
        if view['projection']=='perspective': xy*=distance/np.maximum(distance-depth,1e-9)[:,None]
        return xy,depth
    fig=Figure(figsize=scene.figsize,dpi=120,layout="none"); FigureCanvasAgg(fig); fig.patch.set_facecolor("white")
    # Keep margins in physical inches: axes text never grows when the scene is resized.
    has_key=bool(scene._legend_entries or any(p.get('role')=='colorbar' for p in man['parts']))
    ax=fig.add_axes([.08,.08,.67 if has_key else .84,.8 if scene.title else .84]); ax.patch.set_alpha(0)
    polygons=[]; colors=[]; depths=[]
    for part in scene.parts:
        vertices=part.vertices.copy()
        for name,weight in view.get('states',{}).items():
            if name in part.states: vertices+=weight*(part.states[name]-part.vertices)
        world=vertices@scene.to_world[:3,:3].T
        xy,depth=project(world)
        polygons.append(xy[part.faces]); depths.append(depth[part.faces].mean(axis=1))
        rgba=part.colors[part.faces].mean(axis=1) if part.colors is not None else np.tile(to_rgba(part.color),(len(part.faces),1))
        if scene.lighting=='studio':
            # Reframe into surface's (toward viewer, right, up) convention.
            camera_vertices=np.column_stack([world@direction,world@right,world@up])
            rgba=_shade_rgba(rgba,_face_shading(camera_vertices,part.faces,1,(.8,-.4,.6),.35))
        colors.append(rgba)
    polygons=np.concatenate(polygons); colors=np.concatenate(colors); depths=np.concatenate(depths)
    order=np.argsort(depths,kind='stable')
    # The camera sets both limits below. Avoid scanning every triangle a second
    # time to derive automatic limits that would immediately be overwritten.
    ax.add_collection(PolyCollection(polygons[order],facecolors=colors[order],edgecolors='face',linewidths=.1,antialiased=True),autolim=False)
    w,h=scene.figsize[0]*(.67 if has_key else .84),scene.figsize[1]*(.8 if scene.title else .84)
    ax.set_xlim(-half*max(w/h,1),half*max(w/h,1)); ax.set_ylim(-half*max(h/w,1),half*max(h/w,1)); ax.set_aspect('equal'); ax.set_axis_off()
    ink=scene.style['ink']; font=dict(fontsize=scene.style['fontSizePt'],color=ink,fontfamily=scene.style['font'])
    if scene.title: ax.set_title(scene.title,fontsize=scene.style['titleSizePt'],color=ink,fontfamily=scene.style['font'])
    if scene.axes=='box':
        low=np.array([man['axes'][k]['lim'][0] for k in 'xyz'])
        for i,k in enumerate('xyz'):
            spec=man['axes'][k]; start=low.copy(); end=low.copy(); end[i]=spec['lim'][1]
            line,_=project(np.array([start,end])@scene.to_world[:3,:3].T)
            ax.plot(line[:,0],line[:,1],color=ink,lw=scene.style['lineWidthPt'],zorder=0)
            ax.text(*line[1],spec['label'],ha='center',va='top',**font)
            for tick in spec['ticks']:
                p=low.copy();p[i]=tick; pos,_=project((p@scene.to_world[:3,:3].T)[None,:])
                ax.annotate(f'{tick:g}',pos[0],xytext=(0,-5),textcoords='offset points',ha='center',**font)
    elif scene.axes=='triad':
        for i,k in enumerate('xyz'):
            axis=scene.to_world[:3,i]; dx,dy=float(axis@right),float(axis@up)
            ax.annotate('',xy=(.12+.08*dx,.12+.08*dy),xytext=(.12,.12),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':ink,'lw':scene.style['lineWidthPt']})
            ax.text(.12+.10*dx,.12+.10*dy,k,transform=ax.transAxes,ha='center',va='center',**font)
    if scene.scalebar is not None and view['projection']=='orthographic':
        left=ax.get_xlim()[0]+half*.15; bottom=ax.get_ylim()[0]+half*.15
        ax.plot([left,left+scene.scalebar],[bottom,bottom],color=ink,lw=scene.style['lineWidthPt']*2)
        ax.annotate(f'{scene.scalebar:g} {scene.units}'.strip(),(left+scene.scalebar/2,bottom),xytext=(0,4),textcoords='offset points',ha='center',**font)
    byid={p['id']:p for p in man['parts']}
    if scene._legend_entries:
        handles=[Patch(facecolor=byid[pid].get('color','#4385BE'),label=byid[pid].get('label',pid)) for pid in scene._legend_entries]
        ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(1,1),frameon=False,fontsize=scene.style['fontSizePt'],labelcolor=ink)
    bars=[p for p in man['parts'] if p['role']=='colorbar']
    if bars:
        import matplotlib.cm as cm
        field=byid[bars[0]['field']]['field']; cmap=LinearSegmentedColormap.from_list('preview',field['cmap']['stops'])
        cax=fig.add_axes([.81,.2,.035,.55]); cb=fig.colorbar(cm.ScalarMappable(norm=Normalize(*field['range']),cmap=cmap),cax=cax,ticks=field['ticks'])
        cb.ax.tick_params(labelsize=scene.style['fontSizePt'],colors=ink)
        if field.get('label'): cb.set_label(field['label'],fontsize=scene.style['fontSizePt'],color=ink)
    stream=io.BytesIO();fig.savefig(stream,format='png',transparent=False,facecolor='white',dpi=120)
    return stream.getvalue()


def viewer_bundle():
    folder=files('fluxplot').joinpath('_viewer')
    source=folder.joinpath('flux-model3d-viewer.min.js').read_bytes()
    stamp=json.loads(folder.joinpath('stamp.json').read_text())
    if hashlib.sha256(source).hexdigest()!=stamp['sha256']: raise RuntimeError('Flux notebook viewer stamp mismatch; run scripts/sync_flux_viewer.py')
    return source.decode('utf8')


def mimebundle(scene,*,static=False):
    from .glb import write_glb
    from .scene3d_manifest import build_manifest
    preview=preview_scene(scene)
    # One immutable preparation per representation; Scene3D remains mutable
    # between calls, so this must never become a cross-call cache.
    data=write_glb(preview)
    manifest=build_manifest(preview,data,'preview.glb')
    png=png_preview(preview,_manifest=manifest)
    bundle={'image/png':png}
    if static: return bundle
    if len(data)>30*1024**2: warnings.warn(f'Notebook preview is {len(data)/1024**2:.1f} MiB; install fluxplot[mesh] or lower preview_max_faces',stacklevel=3)
    try: runtime=viewer_bundle()
    except FileNotFoundError:
        warnings.warn('Interactive 3D viewer is not bundled yet; displaying PNG fallback',stacklevel=3)
        return bundle
    if '</script' in runtime.lower(): raise RuntimeError('viewer bundle contains an unsafe script terminator')
    payload=json.dumps({'glb':base64.b64encode(data).decode(),'manifest':manifest,'width':round(scene.figsize[0]*120),'height':round(scene.figsize[1]*120)},allow_nan=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    fallback=base64.b64encode(png).decode()
    # currentScript belongs to each output even in a renderer's shadow root. Capture it
    # synchronously; a document-global id lookup can select another notebook output.
    bundle['text/html']=f'''<div class="fluxplot-scene3d"><div data-fluxplot-scene3d-host><img alt="3D plot preview" src="data:image/png;base64,{fallback}" /></div><script>(()=>{{const script=document.currentScript;const host=script.parentElement.querySelector('[data-fluxplot-scene3d-host]');const fallback=host.innerHTML;{runtime}\nFluxModel3dViewer.mount(host,{payload}).then(view=>{{if(view.available===false)host.innerHTML=fallback;}}).catch(error=>{{host.innerHTML=fallback;host.title=String(error);}});}})();</script></div>'''
    return bundle


def display_size(scene):
    """CSS size of the notebook viewer and of its PNG still."""
    return round(scene.figsize[0] * 120), round(scene.figsize[1] * 120)


def mimebundle_metadata(scene, bundle):
    """Show the PNG at the viewer's CSS size (Jupyter/VS Code honour image width/height)."""
    if 'image/png' not in bundle:
        return {}
    width, height = display_size(scene)
    return {'image/png': {'width': width, 'height': height}}
