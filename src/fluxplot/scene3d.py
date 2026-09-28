"""Mesh scenes sharing FluxPlot's semantic registry, styles and save contract."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import numbers
import warnings
import numpy as np

from .ids import IdAllocator
from .tagger import registry_for

SCENE3D_SPEC_VERSION = '0.1.0'

#: Default notebook preview budget. Only the private notebook copy is simplified
#: (with the optional ``fluxplot[mesh]`` extra); ``fp.save`` keeps full resolution.
PREVIEW_MAX_FACES = 100_000

_VIEW_RANGES = {'elevation': (-90, 90), 'zoom': (.02, 50), 'fov': (5, 120)}
_VIEW_NUMBERS = ('azimuth', 'elevation', 'roll', 'zoom', 'panX', 'panY', 'fov')
_VIEW_KEYS = (*_VIEW_NUMBERS, 'projection', 'states', 'frame')


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


def _positive_int(value, name):
    """Accept Python and numpy integers (not bools, not floats) that are at least one."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer (got {value!r})')
    return int(value)


def _clean_float(value):
    """Round away binary noise from computed values (0.30000000000000004 -> 0.3)."""
    value = float(f'{float(value):.12g}')
    return 0.0 if value == 0 else value


# Metric length ladder for scale-bar labels; 'um' and 'micron' are accepted spellings.
_SI_LENGTHS = {'pm': -12, 'nm': -9, 'µm': -6, 'um': -6, 'μm': -6, 'micron': -6, 'microns': -6,
               'mm': -3, 'cm': -2, 'm': 0, 'km': 3}
_SI_LADDER = [('pm', -12), ('nm', -9), ('µm', -6), ('mm', -3), ('m', 0), ('km', 3)]


def scalebar_text(length, units):
    """Friendly scale-bar label: ``10000 nm`` -> ``10 µm``, ``1e6 nm`` -> ``1 mm``.

    Metric lengths move to the unit that keeps the number in [1, 1000); a value that
    already reads well in its own unit (``5 cm``, ``250 nm``) is left alone. Other
    units (``voxels``, ``a.u.``) are printed as given.
    """
    units = str(units or '')
    exponent = _SI_LENGTHS.get(units)
    value = float(length)
    if exponent is not None and not 1 <= abs(value) < 1000:
        metres = _clean_float(value * 10.0 ** exponent)
        for unit, power in reversed(_SI_LADDER):
            # Tolerate binary noise at the threshold so 1000 nm reads 1 µm, not 1000 nm.
            if abs(metres) >= 10.0 ** power * (1 - 1e-9) or unit == 'pm':
                units, value = unit, metres / 10.0 ** power
                break
    number = f'{_clean_float(value):g}'
    return f'{number} {units}' if units else number


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

    def __repr__(self):
        states = f', states={list(self.states)}' if self.states else ''
        return (f'MeshPart({self.id!r}, {len(self.vertices):,} vertices, '
                f'{len(self.faces):,} triangles, color={self.color!r}{states})')


class Scene3D:
    """A scientific 3D scene: meshes with named parts, value fields and shape states.

    Create one with :func:`fp.scene3d <scene3d>` (same parameters), add meshes with
    :func:`fp.mesh3d <fluxplot.mesh3d>` / :func:`fp.surface3d <fluxplot.surface3d>`,
    choose the saved angle with :meth:`view`, and write it with ``fp.save``.
    """
    def __init__(self, *, figsize=(3.5, 3), units='', up='y', title=None, axes='none',
                 scalebar=None, scalebar_label=None, view=None, lighting='studio',
                 preview_max_faces=PREVIEW_MAX_FACES):
        import matplotlib as mpl
        from matplotlib.colors import to_hex
        from matplotlib.font_manager import FontProperties
        try:
            size = [float(x) for x in figsize]
        except (TypeError, ValueError):
            size = []
        if len(size) != 2 or not all(math.isfinite(x) and x > 0 for x in size):
            raise ValueError('figsize must contain two positive finite inch dimensions')
        if axes not in ('none', 'box', 'triad'):
            raise ValueError("axes must be 'none', 'box', or 'triad'")
        if lighting not in ('studio', 'unlit'):
            raise ValueError("lighting must be 'studio' or 'unlit'")
        self.figsize = tuple(size)
        self.units = str(units)
        self.up = up
        self.title = title
        self.axes = axes
        self.to_world = up_matrix(up)
        self.scalebar = None
        if scalebar is not None:
            self.scalebar = _number(scalebar, 'scalebar')
            if self.scalebar <= 0:
                raise ValueError(f'scalebar length must be positive (got {scalebar!r}); '
                                 'it is a length in your data units, e.g. scalebar=10_000 with units="nm"')
        self.scalebar_label = None if scalebar_label is None else str(scalebar_label)
        self.preview_max_faces = _positive_int(preview_max_faces, 'preview_max_faces')
        self.lighting = lighting
        self.parts: list[MeshPart] = []
        self._max_call_parts = 1
        self._alloc = IdAllocator()
        self._view = dict(azimuth=30., elevation=20., roll=0., zoom=.9, panX=0., panY=0.,
                          projection='orthographic', fov=30.)
        # Shape-state weights (or a frame) requested before any mesh exists; applied
        # as soon as the scene has those states, and required by save/display time.
        self._pending_view = None
        self._legend_entries: list[str] = []
        self._axis_specs = {}
        self.sequence = False
        self.morph_group = None
        self.style = dict(font=FontProperties().get_name(), fontSizePt=float(mpl.rcParams['font.size']),
                          titleSizePt=FontProperties(size=mpl.rcParams['axes.titlesize']).get_size_in_points(),
                          ink=to_hex(mpl.rcParams['text.color']), muted=to_hex(mpl.rcParams['axes.labelcolor']),
                          lineWidthPt=float(mpl.rcParams['axes.linewidth']))
        registry_for(self)
        if view is not None:
            self.view(view)

    # ---- view ------------------------------------------------------------------------------
    def view(self, mapping=None, /, *, view=None, azimuth=None, elevation=None, roll=None,
             zoom=None, panX=None, panY=None, projection=None, fov=None, states=None, frame=None):
        """Set the saved Home view; return this scene so a notebook cell can end with it.

        Accepts keywords, a mapping, or both (keywords win), so every Copy view form
        pastes back unchanged::

            sc.view(azimuth=30, elevation=15, zoom=0.9)
            sc.view({"azimuth": 30, "elevation": 15, "zoom": 0.9})
            sc.view(view=dict(azimuth=30, elevation=15, zoom=0.9))
            sc.view(frame=3)                    # sequences: frame 0 is the base mesh
            sc.view(states={"inflated": 0.5})   # named shape states

        Parameters
        ----------
        azimuth, elevation, roll
            Camera angles in degrees; azimuth is unwrapped, elevation is in [-90, 90].
        zoom
            Magnification in [0.02, 50]; 1 fits the scene's bounding sphere.
        panX, panY
            Pan in bounding-sphere radii.
        projection, fov
            ``"orthographic"`` (default; scale bars need it) or ``"perspective"`` with a
            vertical field of view ``fov`` in [5, 120] degrees.
        states
            ``{state_name: weight}`` default weights of named shape states. Weights are
            stored unclamped; 0 is the base shape and 1 the full state.
        frame
            For ``sequence=True`` scenes: frame 0 is the base, 1..N the sequence
            targets; a fractional frame blends its two neighbours.
        """
        params = {}
        for source, what in ((mapping, 'view mapping'), (view, 'view=')):
            if source is None:
                continue
            if not isinstance(source, Mapping):
                raise TypeError(f'{what} must be a mapping such as dict(azimuth=30, elevation=15)')
            unknown = sorted(str(k) for k in source if k not in _VIEW_KEYS)
            if unknown:
                raise ValueError(f'unknown view keys {unknown}; valid keys are {", ".join(_VIEW_KEYS)}')
            params.update(source)
        explicit = dict(azimuth=azimuth, elevation=elevation, roll=roll, zoom=zoom, panX=panX,
                        panY=panY, projection=projection, fov=fov, states=states, frame=frame)
        params.update({k: v for k, v in explicit.items() if v is not None})

        next_view = dict(self._view)
        for key in _VIEW_NUMBERS:
            if params.get(key) is not None:
                next_view[key] = _number(params[key], key, *_VIEW_RANGES.get(key, (None, None)))
        if params.get('projection') is not None:
            if params['projection'] not in ('orthographic', 'perspective'):
                raise ValueError("projection must be 'orthographic' or 'perspective'")
            next_view['projection'] = params['projection']

        states, frame = params.get('states'), params.get('frame')
        if states is not None and frame is not None:
            raise ValueError('pass states or frame, not both')
        pending = None
        if states is not None or frame is not None:
            if states is not None and not isinstance(states, Mapping):
                raise TypeError('states must be a mapping of state name to weight')
            if not self.parts:
                # Nothing to validate against yet (e.g. fp.scene3d(view=...)); check the
                # numbers now and resolve the names when meshes with states arrive.
                if states is not None:
                    pending = {'states': {str(k): _number(v, f'state {k}') for k, v in states.items()}}
                else:
                    pending = {'frame': _number(frame, 'frame', 0)}
            else:
                next_view['states'] = self._state_weights(states, frame)
        self._view = next_view
        if states is not None or frame is not None:
            self._pending_view = pending
        return self

    def _state_weights(self, states, frame):
        names = self.state_names
        if states is not None:
            unknown = sorted(set(map(str, states)) - set(names))
            if unknown:
                have = ', '.join(names) if names else 'no shape states'
                raise ValueError(f'unknown shape states {unknown}; this scene has {have}')
            return {str(k): _number(v, f'state {k}') for k, v in states.items()}
        if not self.sequence:
            raise ValueError('frame= needs a sequence scene (mesh3d(..., states=[...], sequence=True)); '
                             'use states={...} for named shape states')
        frame = _number(frame, 'frame', 0, len(names))
        low = int(frame); frac = frame - low; weights = {}
        if low > 0: weights[names[low-1]] = 1 - frac
        if frac > 0: weights[names[low]] = frac
        return weights

    def _resolve_pending_view(self, *, strict):
        """Apply a view requested before its shape states existed (see ``view``)."""
        pending = self._pending_view
        if pending is None:
            return
        try:
            weights = self._state_weights(pending.get('states'), pending.get('frame'))
        except ValueError as exc:
            if strict:
                raise ValueError(f'the view set before adding meshes cannot be applied: {exc}') from None
            return
        self._view = {**self._view, 'states': weights}
        self._pending_view = None

    @property
    def state_names(self):
        return list(dict.fromkeys(name for p in self.parts for name in p.states))

    # ---- furniture -------------------------------------------------------------------------
    def axis(self, which, *, lim=None, ticks=None, label=None):
        """Override the authored data-unit ticks, limits or label of x/y/z (box axes)."""
        if which not in ('x', 'y', 'z'): raise ValueError("axis must be 'x', 'y' or 'z'")
        if self.axes == 'none':
            warnings.warn("sc.axis() has no visible effect with axes='none'; create the scene "
                          "with fp.scene3d(axes='box') to draw ticks and labels", stacklevel=2)
        spec = dict(self._axis_specs.get(which, {}))
        if lim is not None:
            if len(lim) != 2: raise ValueError('lim must be a pair')
            spec['lim'] = [_number(x, 'limit') for x in lim]
            if spec['lim'][1] <= spec['lim'][0]: raise ValueError('axis limits must increase')
        if ticks is not None: spec['ticks'] = [_number(x, 'tick') for x in ticks]
        if label is not None: spec['label'] = str(label)
        self._axis_specs[which] = spec
        return self

    # ---- display ---------------------------------------------------------------------------
    def __repr__(self):
        width, height = self.figsize
        head = f'Scene3D({width:g}×{height:g} in'
        if self.units: head += f", units={self.units!r}"
        if self.title: head += f', title={str(self.title)!r}'
        if not self.parts:
            return head + ', empty: add a mesh with fp.mesh3d(sc, ...) or fp.surface3d(sc, ...))'
        triangles = sum(len(p.faces) for p in self.parts)
        ids = [p.id for p in self.parts]
        shown = ', '.join(ids[:6]) + (f', … (+{len(ids) - 6})' if len(ids) > 6 else '')
        text = f'{head}, {len(ids)} part{"s" if len(ids) != 1 else ""} [{shown}], {triangles:,} triangles'
        if self.state_names:
            kind = 'frames' if self.sequence else 'states'
            text += f', {len(self.state_names)} {kind}'
        if self.axes != 'none': text += f', axes={self.axes!r}'
        if self.scalebar is not None: text += f', scalebar={scalebar_text(self.scalebar, self.units)!r}'
        return text + ')'

    def _repr_mimebundle_(self, include=None, exclude=None):
        from .scene3d_viewer import mimebundle, mimebundle_metadata
        bundle = mimebundle(self)
        data = {k: v for k, v in bundle.items()
                if (include is None or k in include) and (exclude is None or k not in exclude)}
        return data, mimebundle_metadata(self, data)

    def show(self, *, static=False):
        """Display in a notebook; ``static=True`` shows only the PNG still."""
        from IPython.display import display
        from .scene3d_viewer import mimebundle, mimebundle_metadata
        bundle = mimebundle(self, static=static)
        return display(bundle, metadata=mimebundle_metadata(self, bundle), raw=True)


def scene3d(*, figsize=(3.5, 3), units='', up='y', title=None, axes='none', scalebar=None,
            scalebar_label=None, view=None, lighting='studio', preview_max_faces=PREVIEW_MAX_FACES):
    """Create a 3D scene; add meshes with :func:`mesh3d` / :func:`surface3d`, save with ``fp.save``.

    ``fp.save(sc, "plots/name")`` writes ``name.glb`` + ``name.fluxplot.json`` +
    ``name.recipe.json``. Flux imports it at its physical size and can re-angle,
    restyle, recolour and animate it without rerunning Python.

    Parameters
    ----------
    figsize
        ``(width, height)`` in inches: the default physical size in Flux.
    units
        Data length unit (``"nm"``, ``"µm"``, ``"mm"``, …) used by axis labels and the
        scale bar. Coordinates are never rescaled.
    up
        Which data axis points up: ``"x"``, ``"-x"``, ``"y"`` (default), ``"-y"``, ``"z"``
        or ``"-z"``. ``"z"`` suits most neuroimaging/connectomics data, ``"-y"`` image
        coordinates. Only the stored orientation changes; axes and ticks keep your units.
    title
        Optional title above the scene.
    axes
        ``"none"`` (default), ``"box"`` (back panes, grid, ticks in data units; tune with
        :meth:`Scene3D.axis`) or ``"triad"`` (a small x/y/z orientation gizmo).
    scalebar
        Scale-bar length in data units (e.g. ``10_000`` with ``units="nm"``); shown in
        orthographic views. Its label is simplified to a friendly metric unit
        (``10 µm``).
    scalebar_label
        Replace the automatic scale-bar label text.
    view
        Initial saved view, the same mapping :meth:`Scene3D.view` accepts, e.g. the
        notebook's Copy view output: ``view=dict(azimuth=30, elevation=15, zoom=0.9)``.
    lighting
        ``"studio"`` (default) or ``"unlit"`` (flat colours, e.g. for label maps).
    preview_max_faces
        Triangle budget of the notebook preview only (default 100,000). With the
        optional ``fluxplot[mesh]`` extra the preview is simplified to this budget;
        ``fp.save`` always keeps the scene's full resolution.

    Returns
    -------
    Scene3D
        Display it as the last line of a notebook cell to orbit it interactively.

    Examples
    --------
    >>> sc = fp.scene3d(figsize=(3.5, 3), units="nm", up="-y", scalebar=10_000)
    >>> fp.mesh3d(sc, (vertices, faces), series="neuron")
    >>> sc.view(azimuth=30, elevation=15, zoom=0.9)
    >>> fp.save(sc, "plots/neuron")
    """
    return Scene3D(figsize=figsize, units=units, up=up, title=title, axes=axes, scalebar=scalebar,
                   scalebar_label=scalebar_label, view=view, lighting=lighting,
                   preview_max_faces=preview_max_faces)
