"""Matrix, contour and color-key helpers; the returned objects are Matplotlib artists.

Raw cell values are opt-in. Geometry, masks, exact normalization and level identity
are always recorded, while large fields use the normal bounded raster pipeline.
"""
from __future__ import annotations
import numpy as np
from . import ids, tagger
from .descriptors import Mark, GuideTag
from .data import values


def _options(ax, series, key, kwargs):
    from .recipe import params
    from .panels import all_axes
    panel = getattr(ax, '_fluxplot_panel_name', None)
    prefix = 'panel.' + ids.slugify(panel) if panel else 'axes.' + str(all_axes(ax.figure).index(ax) + 1)
    key = str(key or prefix + '.' + ids.series_root(series))
    overrides = params().get('__fluxplot__', {}).get(key, {})
    for option in ('cmap', 'vmin', 'vmax'):
        if option in overrides:
            kwargs[option] = overrides[option]
    # A caller's Normalize object may carry a nonlinear scale. Change its limits
    # without replacing the scale or mutating the caller-owned instance.
    if kwargs.get('norm') is not None and not isinstance(kwargs['norm'], str):
        from copy import copy
        norm = copy(kwargs['norm'])
        for option in ('vmin', 'vmax'):
            if option in kwargs:
                setattr(norm, option, kwargs.pop(option))
        kwargs['norm'] = norm
    return key


def normalization(artist):
    from matplotlib import colors
    norm = artist.norm
    out = {'kind': type(norm).__name__, 'vmin': float(norm.vmin) if norm.vmin is not None else None,
           'vmax': float(norm.vmax) if norm.vmax is not None else None, 'clip': bool(norm.clip)}
    for attr in ('vcenter', 'gamma', 'linthresh', 'linscale'):
        if hasattr(norm, attr): out[attr] = float(getattr(norm, attr))
    if isinstance(norm, colors.BoundaryNorm): out['boundaries'] = values(norm.boundaries)
    return out


def _capture(artist, config):
    out = dict(config)
    out['normalization'] = normalization(artist)
    out['cmap'] = artist.get_cmap().name
    out['missingColor'] = artist.get_cmap().get_bad().tolist()
    out['underColor'] = artist.get_cmap().get_under().tolist()
    out['overColor'] = artist.get_cmap().get_over().tolist()
    # imshow / pcolormesh can be edited after helper construction.
    if out['kind'] == 'heatmap':
        arr = np.ma.masked_invalid(np.ma.asarray(artist.get_array(), dtype=float))
        shape = out['shape']
        arr = arr.reshape(shape)
        out['maskedIndices'] = np.flatnonzero(np.ma.getmaskarray(arr)).tolist()
        if out.pop('includeValues', False):
            out['values'] = [values(row) for row in arr]
        if hasattr(artist, 'get_extent'):
            out['extent'] = [float(x) for x in artist.get_extent()]
            out['origin'] = artist.origin
        else:
            coords = artist.get_coordinates()
            xx, yy = coords[:, :, 0], coords[:, :, 1]
            regular = np.array_equal(xx, np.broadcast_to(xx[0], xx.shape)) and np.array_equal(yy, np.broadcast_to(yy[:, :1], yy.shape))
            out['grid'] = {'x': xx[0].tolist() if regular else xx.tolist(),
                           'y': yy[:, 0].tolist() if regular else yy.tolist()}
    return out


def heatmap(ax, data, *, series, x=None, y=None, cells=False, include_values=False,
            key=None, **kwargs):
    """Draw a scalar matrix. ``x``/``y`` select pcolormesh (including irregular grids).

    ``cells=True`` gives modest meshes row/column cell IDs; above the raster
    threshold only the layer is addressable. ``include_values`` stores raw values.
    ``key`` names the recipe color controls; defaults to the owning axes and series.
    """
    arr = np.ma.masked_invalid(np.ma.asarray(data, dtype=float))
    if arr.ndim != 2 or not arr.size:
        raise ValueError('heatmap data must be a nonempty 2D scalar matrix')
    if (x is None) != (y is None):
        raise ValueError('heatmap x and y must be supplied together')
    key = _options(ax, series, key, kwargs)
    if x is not None or cells:
        if x is None:
            x, y = np.arange(arr.shape[1] + 1), np.arange(arr.shape[0] + 1)
        artist = ax.pcolormesh(x, y, arr, **kwargs)
    else:
        artist = ax.imshow(arr, **kwargs)
    config = {'kind': 'heatmap', 'shape': list(arr.shape), 'includeValues': bool(include_values),
              'controlKey': key}
    mark = Mark(role='x-heatmap', series=series, kind='heatmap', artists=[artist],
                data={'field_config': config, 'field_artist': artist, 'cells': bool(cells)})
    tagger.registry_for(ax.figure).add(mark)
    return artist


def _contour(ax, args, series, filled, include_values, key, kwargs):
    key = _options(ax, series, key, kwargs)
    artist = (ax.contourf if filled else ax.contour)(*args, **kwargs)
    levels = values(artist.levels)
    config = {'kind': 'contourf' if filled else 'contour', 'levels': levels,
              'extend': artist.extend, 'controlKey': key}
    z = np.ma.masked_invalid(np.ma.asarray(args[0] if len(args) < 3 else args[2], dtype=float))
    config['shape'] = list(z.shape)
    config['maskedIndices'] = np.flatnonzero(np.ma.getmaskarray(z)).tolist()
    if include_values: config['values'] = [values(row) for row in z]
    if len(args) >= 3:
        config['grid'] = {'x': np.asarray(args[0]).tolist(), 'y': np.asarray(args[1]).tolist()}
    # 3.8 made ContourSet itself an Artist; 3.7 has one Collection per level.
    if hasattr(artist, 'set_gid'):
        artists = [artist]
        split = True
    else:
        artists = list(artist.collections)
        split = False
    mark = Mark(role='x-contourf' if filled else 'x-contour', series=series,
                kind=config['kind'], artists=artists, axes=ax,
                data={'field_config': config, 'field_artist': artist, 'contour_paths': split, 'contour_legacy': not split})
    tagger.registry_for(ax.figure).add(mark)
    return artist


def contour(ax, *args, series, include_values=False, key=None, **kwargs):
    """Matplotlib contour with exact levels and addressable level paths."""
    return _contour(ax, args, series, False, include_values, key, kwargs)


def contourf(ax, *args, series, include_values=False, key=None, **kwargs):
    """Matplotlib filled contours with exact boundaries and band identities."""
    return _contour(ax, args, series, True, include_values, key, kwargs)


def colorbar(mappable, *, name='color', ax=None, **kwargs):
    """Create a named, linked color key using Figure.colorbar's usual options."""
    owner = ax if ax is not None else getattr(mappable, 'axes', None)
    if owner is None:
        raise ValueError('colorbar needs ax when the mappable has no owning axes')
    cb = owner.figure.colorbar(mappable, ax=owner, **kwargs)
    cb.ax._fluxplot_colorbar_name = str(name)
    cb.ax._fluxplot_owner_axes = owner
    return cb


def capture_mark(mark):
    if 'field_config' not in mark.data: return
    mark.data['field'] = _capture(mark.data['field_artist'], mark.data['field_config'])


def colorbar_guides(fig, owner, alloc):
    from .panels import all_axes
    guides = []
    for ax in all_axes(fig):
        cb = getattr(ax, '_colorbar', None)
        if cb is None: continue
        parent = getattr(ax, '_fluxplot_owner_axes', getattr(cb.mappable, 'axes', None))
        if parent is not owner: continue
        gid = alloc.take('colorbar.' + ids.slugify(getattr(ax, '_fluxplot_colorbar_name', 'color')))
        ax.set_gid(gid)
        axis = ax.yaxis if cb.orientation == 'vertical' else ax.xaxis
        parts = []
        artists = [('label', axis.label), ('outline', cb.outline)]
        primary_side = 2 if axis.get_ticks_position() in ('right', 'top') else 1
        artists.extend((suffix.replace('ticklabel', 'tick-label'), art)
                       for suffix, role, _, art in tagger.axis_tick_artists(axis, primary_side)
                       if role in ('tick', 'tick-label', 'gridline'))
        if cb.solids is not None: artists.append(('solids', cb.solids))
        for suffix, art in artists:
            if not art.get_visible(): continue
            part = gid + '.' + suffix
            # Surface keys may already have explicit registered identity.
            if art.get_gid(): part = art.get_gid()
            else: art.set_gid(part)
            role = next(token for token in suffix.split('.')
                        if token not in ('minor', 'secondary'))
            kind = {'label': 'text', 'tick-label': 'text', 'tick': 'line',
                    'gridline': 'line', 'outline': 'line', 'solids': 'shape'}[role]
            parts.append({'svgId': part, 'role': 'colorbar-' + role, 'kind': kind})
            guides.append(GuideTag(gid=part, role=parts[-1]['role'], kind=kind))
        lo, hi = sorted(axis.get_view_interval())
        visible_ticks = [t.get_loc() for t in axis.get_major_ticks()
                         if np.isfinite(t.get_loc()) and lo <= t.get_loc() <= hi
                         and t.get_visible() and any(a.get_visible() for a in
                             (t.tick1line, t.tick2line, t.label1, t.label2))]
        guides.append(GuideTag(gid=gid, role='colorbar', data={
            'orientation': cb.orientation, 'label': axis.label.get_text(),
            'ticks': values(visible_ticks), 'normalization': normalization(cb.mappable),
            'cmap': cb.mappable.get_cmap().name, 'parts': parts,
            'mappable': cb.mappable.get_gid() if hasattr(cb.mappable, 'get_gid') else None}))
    return guides
