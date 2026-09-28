"""One value-to-color law shared by 2D surface and Scene3D; no geometry or rendering."""
from __future__ import annotations
import numpy as np
from matplotlib.colors import Normalize, to_hex


def _normalise_missing(values, missing_below=None, missing_values=()):
    """Normalize explicit sentinel codes; zero remains a real observation."""
    v=np.asarray(values,dtype=float).copy()
    if missing_below is not None: v[v<missing_below]=np.nan
    for mv in missing_values: v[v==mv]=np.nan
    return v


def _resolve_range(finite,color_range,percentile):
    if color_range is not None: return float(color_range[0]),float(color_range[1])
    if percentile is not None: return float(np.percentile(finite,percentile[0])),float(np.percentile(finite,percentile[1]))
    return float(np.min(finite)),float(np.max(finite))


def categorical_colors(names,palette=None,categories=None):
    """Resolve the fixed category-name/code palette in stable sorted-name order."""
    resolved={}
    for name in sorted(names):
        color=None
        if palette:
            color=palette.get(name)
            if color is None and categories:
                for code,nm in categories.items():
                    if nm==name and code in palette:
                        color=palette[code]; break
        if color is None: color=f'C{len(resolved)%10}'
        resolved[name]=to_hex(color)
    return resolved


def continuous_mapping(finite,cmap=None,color_range=None,percentile=None):
    """Return the shared matplotlib colormap and Normalize, including all-missing fallback."""
    import matplotlib as mpl
    if finite.size==0 and color_range is None: lo,hi=0.,1.
    else: lo,hi=_resolve_range(finite,color_range,percentile)
    if not np.isfinite([lo,hi]).all() or lo>hi: raise ValueError('color_range must be finite and nondecreasing')
    obj=mpl.colormaps[mpl.rcParams['image.cmap']] if cmap is None else (mpl.colormaps[cmap] if isinstance(cmap,str) else cmap)
    return obj,Normalize(vmin=lo,vmax=hi)


def infer_kind(finite,kind):
    if kind=='categorical': kind='label'
    if kind=='auto':
        if not finite.size: raise ValueError("every vertex is missing and kind='auto' cannot infer label vs continuous; pass kind explicitly")
        uniq=np.unique(finite)
        kind='label' if uniq.size<=32 and np.allclose(uniq,np.round(uniq)) else 'continuous'
    if kind not in ('label','continuous'): raise ValueError("kind must be 'label', 'continuous', or 'auto'")
    return kind
