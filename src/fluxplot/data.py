"""Exact public artist data adapters shared by wrappers and save-time capture."""
from __future__ import annotations
import numpy as np


def values(seq):
    """Numeric data with null gaps; preserve length and original observation indices."""
    if seq is None:
        return None
    arr = np.ma.asarray(seq, dtype=float).filled(np.nan)
    return [float(v) if np.isfinite(v) else None for v in np.atleast_1d(arr).reshape(-1)]


def converted(ax, x, y):
    return values(ax.convert_xunits(x)), values(ax.convert_yunits(y))


def artist_xy(artist):
    from matplotlib.collections import PathCollection
    from matplotlib.lines import Line2D
    # 3D collections sort by depth; their offsets are not in source order.
    if getattr(getattr(artist, 'axes', None), 'name', None) == '3d':
        return None, None
    if isinstance(artist, Line2D):
        return values(artist.get_xdata(orig=False)), values(artist.get_ydata(orig=False))
    if isinstance(artist, PathCollection):
        off = np.ma.asarray(artist.get_offsets(), dtype=float)
        if off.ndim == 2 and off.shape[1] == 2:
            return values(off[:, 0]), values(off[:, 1])
    return None, None


def bar_data(patches, orientation='vertical'):
    horizontal = orientation == 'horizontal'
    centers = [p.get_y() + p.get_height()/2 if horizontal else p.get_x() + p.get_width()/2 for p in patches]
    lengths = [p.get_width() if horizontal else p.get_height() for p in patches]
    bases = [p.get_x() if horizontal else p.get_y() for p in patches]
    ends = np.asarray(bases) + np.asarray(lengths)
    x, y = (ends, centers) if horizontal else (centers, ends)
    return values(x), values(y), {'orientation': orientation, 'baseline': values(bases), 'length': values(lengths)}


def refresh(mark):
    """Refresh the save snapshot only; registration keeps its stable identity."""
    if not mark.artists:
        return
    art = mark.artists[0]
    from .fields import capture_mark
    capture_mark(mark)
    if mark.data.get('field_config', {}).get('kind') == 'scatter' and hasattr(art, 'get_array'):
        mark.data['c'] = values(art.get_array())  # the colour-mapped values, as drawn
    if mark.live_data:
        if mark.role == 'bar':
            mark.x, mark.y, meta = bar_data(mark.artists, mark.data.get('bar', {}).get('orientation', 'vertical'))
            mark.data['bar'] = meta
        else:
            x, y = artist_xy(art)
            if x is not None and y is not None:
                mark.x, mark.y = x, y
            elif getattr(getattr(art, 'axes', None), 'name', None) == '3d':
                mark.x = mark.y = None
    if mark.x is not None:
        mark.x, mark.y = values(mark.x), values(mark.y)
    from matplotlib.text import Text
    if isinstance(art, Text):
        mark.data['text'] = art.get_text()
    if mark.role == 'area' and hasattr(art, 'get_paths') and 'x' not in (mark.data.get('band') or {}):
        # a promoted fill_between: only its polygon is known (fp.area / fp.band record their inputs)
        mark.data['band'] = {'paths': [[values(v) for v in p.vertices] for p in art.get_paths()]}
    if mark.series is not None:
        mark.data['color'] = primary_paint(mark)


def primary_paint(mark):
    """The one colour a series mark is painted with (``{hex, alpha, token?, palette?}``), or
    ``{"hex": "varies"}`` when its elements differ (a colour-mapped collection). A Line2D's line
    colour, a collection's or a bar's face colour; ``token`` is the exact palette token
    (``flexoki.blue-600``) and ``palette`` the position in the active cycle, when either holds."""
    from matplotlib.collections import Collection
    from matplotlib.colors import to_hex, to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from .colors import palette_of, token_of
    arts = mark.artists
    if not arts:
        return None
    rgba = None
    art = arts[0]
    try:
        if isinstance(art, Line2D):
            rgba = to_rgba(art.get_color())
        elif isinstance(art, Collection):
            faces = art.get_facecolor()
            if getattr(art, 'get_array', None) is not None and art.get_array() is not None:
                return {'hex': 'varies'}
            if len(faces) == 0:
                edges = art.get_edgecolor()
                if len(edges) == 0:
                    return None
                faces = edges
            first = tuple(faces[0])
            if any(tuple(f) != first for f in faces):
                return {'hex': 'varies'}
            rgba = first
        elif isinstance(art, Patch):
            first = to_rgba(art.get_facecolor())
            if any(to_rgba(a.get_facecolor()) != first for a in arts if isinstance(a, Patch)):
                return {'hex': 'varies'}
            rgba = first
    except (ValueError, TypeError, AttributeError):
        return None
    if rgba is None:
        return None
    out = {'hex': to_hex(rgba, keep_alpha=False), 'alpha': round(float(rgba[3]), 6)}
    token = token_of(rgba)
    if token:
        out['token'] = token
    palette = palette_of(rgba)
    if palette:
        out['palette'] = palette
    return out


def point_indices(mark):
    """Eligible source indices, before the exact SVG count guard."""
    if mark.x is None or mark.y is None:
        return []
    indices = np.arange(min(len(mark.x), len(mark.y)))
    art = mark.artists[0]
    every = getattr(art, 'get_markevery', lambda: None)()
    if every is not None:
        if isinstance(every, int):
            indices = indices[::every]
        elif isinstance(every, slice):
            indices = indices[every]
        elif isinstance(every, tuple) and len(every) == 2 and all(isinstance(v, int) for v in every):
            indices = indices[every[0]::every[1]]
        elif isinstance(every, (list, np.ndarray)):
            indices = indices[every]
        else:
            return []  # display-distance subsampling has no stable source-index contract
    return [int(i) for i in indices if mark.x[i] is not None and mark.y[i] is not None]
