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
    if mark.role == 'area' and hasattr(art, 'get_paths'):
        mark.data['band'] = {'paths': [[values(v) for v in p.vertices] for p in art.get_paths()]}


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
