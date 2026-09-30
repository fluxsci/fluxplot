"""One value-to-color law shared by 2D surface and Scene3D; no geometry or rendering."""
from __future__ import annotations
import numpy as np
from matplotlib.colors import Normalize, to_hex, to_rgba


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


def category_name(code, categories=None):
    """Name of one integer label code: ``categories[code]``, else ``category-<code>``.

    Negative codes are spelled ``category-m<abs>`` (``-1`` -> ``category-m1``). A plain
    ``category--1`` would slugify to ``category-1`` and collide with code ``+1``, so the
    id of one category would shift depending on whether the other is present.
    """
    code = int(code)
    if categories and code in categories:
        return categories[code]
    return f'category-{code}' if code >= 0 else f'category-m{-code}'


def categorical_colors(names,palette=None,categories=None):
    """Resolve the fixed category-name/code palette in stable sorted-name order.

    A category the palette does not name takes its colour from the registry
    (``fp.colors.categories``): pinned, or the next free slot of the active cycle, remembered —
    so ``'b'`` is the same colour whether or not ``'a'`` is present, in every figure.
    """
    from .colors import categories as _registry
    resolved={}
    for name in sorted(names):
        color=None
        if palette:
            color=palette.get(name)
            if color is None and categories:
                for code,nm in categories.items():
                    if nm==name and code in palette:
                        color=palette[code]; break
        if color is None: color=_registry.get(name)
        rgba=to_rgba(color)
        resolved[name]=to_hex(rgba, keep_alpha=rgba[3] < 1)  # an opaque colour stays #rrggbb
    return resolved


def continuous_mapping(finite,cmap=None,color_range=None,percentile=None):
    """Return the shared matplotlib colormap and Normalize, including all-missing fallback."""
    if finite.size==0 and color_range is None: lo,hi=0.,1.
    else: lo,hi=_resolve_range(finite,color_range,percentile)
    if not np.isfinite([lo,hi]).all() or lo>hi: raise ValueError('color_range must be finite and nondecreasing')
    return resolve_colormap(cmap),Normalize(vmin=lo,vmax=hi)


def resolve_colormap(cmap=None):
    """A Colormap from ``None`` (the style default), a Colormap, or a name.

    Names resolve through matplotlib first (so existing names keep their exact maps),
    then through fluxplot's shipped collections: ``'emerald'``, ``'crameri.batlow'``,
    ``'batlow_r'``, or a map added with ``fp.colors.maps.register``.
    """
    import matplotlib as mpl
    if cmap is None:
        return mpl.colormaps[mpl.rcParams['image.cmap']]
    if not isinstance(cmap, str):
        return cmap
    try:
        return mpl.colormaps[cmap]
    except KeyError:
        pass
    from .colors import maps
    try:
        return maps.get(cmap)
    except KeyError:
        raise ValueError(f"unknown colormap {cmap!r}: use a matplotlib name ('viridis') or a "
                         "fluxplot map ('emerald', 'crameri.batlow'); list them with "
                         "fp.colors.maps.collections()") from None


def infer_kind(finite,kind):
    if kind=='categorical': kind='label'
    if kind=='auto':
        if not finite.size: raise ValueError("every vertex is missing and kind='auto' cannot infer label vs continuous; pass kind explicitly")
        uniq=np.unique(finite)
        kind='label' if uniq.size<=32 and np.allclose(uniq,np.round(uniq)) else 'continuous'
    if kind not in ('label','continuous'): raise ValueError("kind must be 'label', 'continuous', or 'auto'")
    return kind
