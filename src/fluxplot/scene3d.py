"""Mesh scenes sharing FluxPlot's semantic registry, styles and save contract."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np

from .ids import IdAllocator
from .tagger import registry_for

SCENE3D_SPEC_VERSION = '0.1.0'


def up_matrix(up):
    """Proper rotation from a signed data-up axis to glTF +Y; no winding reflection."""
    matrices = {
        'y': [[1,0,0],[0,1,0],[0,0,1]],
        '-y': [[1,0,0],[0,-1,0],[0,0,-1]],
        'z': [[1,0,0],[0,0,1],[0,-1,0]],
        '-z': [[1,0,0],[0,0,-1],[0,1,0]],
        'x': [[0,-1,0],[1,0,0],[0,0,1]],
        '-x': [[0,1,0],[-1,0,0],[0,0,1]],
    }
    if up not in matrices:
        raise ValueError("up must be 'x', '-x', 'y', '-y', 'z', or '-z'")
    result = np.eye(4); result[:3,:3] = matrices[up]
    return result


def _number(value, name, lo=None, hi=None):
    try: value = float(value)
    except (TypeError, ValueError): raise ValueError(f'{name} must be a finite number') from None
    if not math.isfinite(value) or (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ValueError(f'{name} must be finite' + (f' in [{lo}, {hi}]' if lo is not None else ''))
    return value


@dataclass(eq=False)
class MeshPart:
    """One addressable mesh; arrays are copied on construction and retain vertex correspondence."""
    id: str
    vertices: np.ndarray
    faces: np.ndarray
    color: str
    states: dict = field(default_factory=dict)
    values: np.ndarray | None = None
    colors: np.ndarray | None = None
    source_faces: np.ndarray | None = None
    source_count: int = 0
    source_indices: np.ndarray | None = None
    collapses: np.ndarray | None = None
    # Collapse replay works on the compacted pre-decimation topology.
    compact_faces: np.ndarray | None = None


class Scene3D:
    """A scientific 3D scene. Add meshes with :func:`mesh3d`, save with ``fp.save``.

    ``figsize`` is inches; ``up`` describes the input coordinates. Axes, scales and values
    retain data units while GLB writing rotates positions into glTF Y-up.
    """
    def __init__(self, *, figsize=(3.5,3), units='', up='y', title=None, axes='none',
                 scalebar=None, preview_max_faces=300_000, lighting='studio'):
        import matplotlib as mpl
        from matplotlib.colors import to_hex
        from matplotlib.font_manager import FontProperties
        if len(figsize) != 2 or not all(math.isfinite(float(x)) and x > 0 for x in figsize):
            raise ValueError('figsize must contain two positive finite inch dimensions')
        if axes not in ('none','box','triad'): raise ValueError('axes must be none, box, or triad')
        if lighting not in ('studio','unlit'): raise ValueError('lighting must be studio or unlit')
        if not isinstance(preview_max_faces,int) or preview_max_faces <= 0: raise ValueError('preview_max_faces must be positive')
        self.figsize = tuple(float(x) for x in figsize)
        self.units, self.up, self.title, self.axes = str(units), up, title, axes
        self.to_world = up_matrix(up)
        self.scalebar = None if scalebar is None else _number(scalebar,'scalebar',np.finfo(float).tiny)
        self.preview_max_faces, self.lighting = preview_max_faces, lighting
        self.parts: list[MeshPart] = []
        self._alloc = IdAllocator()
        self._view = dict(azimuth=30.,elevation=20.,roll=0.,zoom=.9,panX=0.,panY=0.,projection='orthographic',fov=30.)
        self._legend_entries: list[str] = []
        self._axis_specs = {}
        self.sequence = False
        self.morph_group = None
        self.style = dict(font=FontProperties().get_name(),fontSizePt=float(mpl.rcParams['font.size']),
                          titleSizePt=FontProperties(size=mpl.rcParams['axes.titlesize']).get_size_in_points(),
                          ink=to_hex(mpl.rcParams['text.color']),muted=to_hex(mpl.rcParams['axes.labelcolor']),
                          lineWidthPt=float(mpl.rcParams['axes.linewidth']))
        registry_for(self)

    def view(self, *, azimuth=None, elevation=None, roll=None, zoom=None, panX=None, panY=None,
             projection=None, fov=None, states=None, frame=None):
        """Set the saved Home view; return this scene for notebook-friendly chaining.

        Azimuth is unwrapped. ``frame=0`` is the base, frames 1..N are sequence targets.
        A fractional frame blends its neighbors. ``states`` accepts finite unclamped weights.
        """
        next_view=dict(self._view)
        ranges={'elevation':(-90,90),'zoom':(.02,50),'fov':(5,120)}
        for key,val in dict(azimuth=azimuth,elevation=elevation,roll=roll,zoom=zoom,panX=panX,panY=panY,fov=fov).items():
            if val is not None: next_view[key]=_number(val,key,*ranges.get(key,(None,None)))
        if projection is not None:
            if projection not in ('orthographic','perspective'): raise ValueError('unknown projection')
            next_view['projection']=projection
        if states is not None and frame is not None: raise ValueError('pass states or frame, not both')
        if states is not None:
            names=self.state_names
            unknown=set(states)-set(names)
            if unknown: raise ValueError(f'unknown shape states: {sorted(unknown)}')
            next_view['states']={str(k):_number(v,f'state {k}') for k,v in states.items()}
        if frame is not None:
            if not self.sequence: raise ValueError('frame requires a sequence scene')
            frame=_number(frame,'frame',0,len(self.state_names))
            low=int(frame); frac=frame-low; weights={}
            if low>0: weights[self.state_names[low-1]]=1-frac
            if frac>0: weights[self.state_names[low]]=frac
            next_view['states']=weights
        self._view=next_view
        return self

    @property
    def state_names(self):
        return list(dict.fromkeys(name for p in self.parts for name in p.states))

    def axis(self, which, *, lim=None, ticks=None, label=None):
        """Override the authored data-unit ticks, limits or label of x/y/z."""
        if which not in 'xyz' or len(which)!=1: raise ValueError('axis must be x, y or z')
        spec=dict(self._axis_specs.get(which,{}))
        if lim is not None:
            if len(lim)!=2: raise ValueError('lim must be a pair')
            spec['lim']=[_number(x,'limit') for x in lim]
            if spec['lim'][1]<=spec['lim'][0]: raise ValueError('axis limits must increase')
        if ticks is not None: spec['ticks']=[_number(x,'tick') for x in ticks]
        if label is not None: spec['label']=str(label)
        self._axis_specs[which]=spec
        return self

    def _repr_mimebundle_(self, include=None, exclude=None):
        from .scene3d_viewer import mimebundle
        bundle=mimebundle(self)
        return {k:v for k,v in bundle.items() if (include is None or k in include) and (exclude is None or k not in exclude)}

    def show(self, *, static=False):
        from IPython.display import display
        from .scene3d_viewer import mimebundle
        bundle=mimebundle(self,static=static)
        return display(bundle,raw=True)


def scene3d(**kwargs):
    """Create a :class:`Scene3D`; helpers take the scene as their first argument."""
    return Scene3D(**kwargs)
