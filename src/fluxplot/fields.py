"""Matrix, contour and color-key helpers; the returned objects are Matplotlib artists.

Raw cell values are opt-in. Geometry, masks, exact normalization and level identity
are always recorded, while large fields use the normal bounded raster pipeline.
"""
from __future__ import annotations
import numpy as np
from . import colorscale, ids, tagger
from ._fieldmap import resolve_colormap
from .descriptors import Mark, GuideTag
from .data import values


#: ``__fluxplot__`` entries that are not colour scales: the theme, the palette, per-series colours.
RESERVED_CONTROL_KEYS = frozenset({"theme", "palette", "series"})


def control_key(ax, series, key=None):
    """``(key, legacy_key)`` naming a colour-mapped series' entry in ``recipe.params.__fluxplot__``.

    The key is what Flux edits, so it has to survive the figure being rearranged. In order:

    1. the explicit ``key``;
    2. the bare series root (``rates``) when no other colour-controlled series of the figure has
       claimed it yet — the common case, and independent of where the axes sits;
    3. ``panel.<slug>.<root>`` when the axes was named with :func:`fluxplot.panel`;
    4. ``axes.<n>.<root>`` with ``n`` the axes' 1-based position among the figure's axes.

    ``legacy_key`` is what rule 3/4 alone would have produced (the rule before 0.3.1); overrides
    saved under it by an older Flux still apply. Claimed keys are remembered per figure in the
    registry so two same-named series never share a key.
    """
    from . import tagger
    from .panels import all_axes
    root = ids.series_root(series)
    panel = getattr(ax, '_fluxplot_panel_name', None)
    positional = ('panel.' + ids.slugify(panel) if panel
                  else 'axes.' + str(all_axes(ax.figure).index(ax) + 1)) + '.' + root
    claimed = tagger.registry_for(ax.figure)._color_keys
    if key is not None:
        chosen = str(key)
    elif root not in claimed and root not in RESERVED_CONTROL_KEYS:
        chosen = root
    else:
        chosen = positional
        n = 2
        while chosen in claimed:  # the same series twice on one axes: deterministic, never silent
            chosen = f'{positional}-{n}'
            n += 1
    claimed.add(chosen)
    return chosen, positional


def _options(ax, series, key, kwargs, *, resolve=None):
    """Apply the recipe's colour controls for one colour-mapped series; return its control key.

    ``kwargs`` are the colour keywords the helper is about to pass to matplotlib (``cmap``,
    ``vmin``, ``vmax``, ``norm``). Flux edits them through ``recipe.params.__fluxplot__[key]``
    (see :func:`control_key`). An override is applied only where it differs from what the script
    itself asked for: the controls are written back on every save, so replaying them must
    reproduce the script's own objects (a ``ListedColormap`` has no resolvable name) rather than
    fail on them. A colormap override that resolves to nothing raises, naming the key, instead of
    letting matplotlib fail later on an unnamed map. ``resolve`` turns a colormap name into a
    ``Colormap`` (default: :func:`fluxplot._fieldmap.resolve_colormap`); helpers with their own
    name spaces (the hexmatrix's single-colour ramps) pass theirs.
    """
    from ._fieldmap import resolve_colormap
    from .recipe import params
    resolve = resolve or resolve_colormap
    key, legacy = control_key(ax, series, key)
    controls = params().get('__fluxplot__') or {}
    overrides = controls.get(key) if key not in RESERVED_CONTROL_KEYS else None
    if overrides is None and legacy != key:
        overrides = controls.get(legacy)
    overrides = dict(overrides or {})
    if overrides:
        colorscale.apply_override(kwargs, overrides, key, resolve=resolve)
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
    """The norm as ``field.normalization`` records it: ``kind`` is the matplotlib class name (kept
    as an alias of the portable ``colorScales[].norm.kind``), the parameters are the record's."""
    record = colorscale.norm_record(artist.norm)
    out = {'kind': type(artist.norm).__name__, 'vmin': record['vmin'], 'vmax': record['vmax'], 'clip': record['clip']}
    for attr in ('vcenter', 'halfrange', 'gamma', 'linthresh', 'linscale', 'base', 'boundaries'):
        if record.get(attr) is not None:
            out[attr] = record[attr]
    return out


def _capture(artist, config, resolve=None):
    out = dict(config)
    out['normalization'] = normalization(artist)
    out['cmap'] = artist.get_cmap().name
    out['missingColor'] = artist.get_cmap().get_bad().tolist()
    out['underColor'] = artist.get_cmap().get_under().tolist()
    out['overColor'] = artist.get_cmap().get_over().tolist()
    # the portable colour scale (colorscale.py): the recipe names the map the way it can rebuild
    # it, the manifest's colorScales[] entry carries the exact LUT, and the field points at it
    out['cmapSpec'] = colorscale.cmap_spec(artist.get_cmap(), resolve or resolve_colormap)
    out['colorScale'] = out['controlKey']
    if not out.get('extend'):
        cb = getattr(artist, 'colorbar', None)
        out['extend'] = getattr(cb, 'extend', None) or getattr(artist, 'extend', None) or 'neither'
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
            key=None, value_raster=False, **kwargs):
    """Draw a scalar matrix. ``x``/``y`` select pcolormesh (including irregular grids).

    ``cells=True`` gives modest meshes row/column cell IDs; above the raster
    threshold only the layer is addressable. ``include_values`` stores raw values.
    ``key`` names the recipe color controls; defaults to the owning axes and series.
    ``value_raster=True`` writes the matrix as ``<plot>.<key>.values.json`` beside the SVG
    (row-major, ``null`` for missing) and points the colour scale at it, so a consumer can
    repaint an image layer from its values instead of regenerating.
    """
    arr = np.ma.masked_invalid(np.ma.asarray(data, dtype=float))
    if arr.ndim != 2 or not arr.size:
        raise ValueError('heatmap data must be a nonempty 2D scalar matrix')
    if (x is None) != (y is None):
        raise ValueError('heatmap x and y must be supplied together')
    key = _options(ax, series, key, kwargs)
    extend = kwargs.pop('_extend', None)
    if isinstance(kwargs.get('cmap'), str):
        kwargs['cmap'] = resolve_colormap(kwargs['cmap'])  # 'emerald', 'crameri.batlow', 'batlow_r'
    if x is not None or cells:
        if x is None:
            x, y = np.arange(arr.shape[1] + 1), np.arange(arr.shape[0] + 1)
        artist = ax.pcolormesh(x, y, arr, **kwargs)
    else:
        artist = ax.imshow(arr, **kwargs)
    config = {'kind': 'heatmap', 'shape': list(arr.shape), 'includeValues': bool(include_values),
              'controlKey': key}
    if extend:
        config['extend'] = extend
    data = {'field_config': config, 'field_artist': artist, 'cells': bool(cells)}
    if value_raster:
        data['value_raster'] = {}  # filename and payload are fixed at save time
    mark = Mark(role='x-heatmap', series=series, kind='heatmap', artists=[artist], data=data)
    tagger.registry_for(ax.figure).add(mark)
    return artist


def _contour(ax, args, series, filled, include_values, key, kwargs):
    key = _options(ax, series, key, kwargs)
    extend = kwargs.pop('_extend', None)
    if extend:
        kwargs['extend'] = extend  # an edited extend replaces the script's: the bands change with it
    if isinstance(kwargs.get('cmap'), str):
        kwargs['cmap'] = resolve_colormap(kwargs['cmap'])
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


def _field_mark(fig, artist):
    return next((m for m in tagger.registry_for(fig).marks if m.data.get('field_artist') is artist), None)


def colorbar(mappable, *, name='color', ax=None, **kwargs):
    """Create a named, linked color key using Figure.colorbar's usual options.

    The key follows its scale's recipe controls: an ``extend`` edited in Flux is applied here
    (and recorded) unless the call names its own.
    """
    owner = ax if ax is not None else getattr(mappable, 'axes', None)
    if owner is None:
        raise ValueError('colorbar needs ax when the mappable has no owning axes')
    mark = _field_mark(owner.figure, mappable)
    if mark is not None and mark.data['field_config'].get('extend') and 'extend' not in kwargs:
        kwargs['extend'] = mark.data['field_config']['extend']
    cb = owner.figure.colorbar(mappable, ax=owner, **kwargs)
    if mark is not None:
        mark.data['field_config']['extend'] = cb.extend
    cb.ax._fluxplot_colorbar_name = str(name)
    cb.ax._fluxplot_owner_axes = owner
    return cb


def paint_of(artist) -> str:
    """Which paint properties a colour scale drives on an artist's elements: ``"fill"`` for filled
    shapes and images, ``"stroke"`` for line contours and line collections, ``"fill stroke"`` for a
    collection whose edges take the face colour (``edgecolor="face"``: hexagons, scatter markers)."""
    from matplotlib.collections import Collection, LineCollection
    from matplotlib.contour import ContourSet
    from matplotlib.image import AxesImage
    if isinstance(artist, ContourSet):
        return 'fill' if artist.filled else 'stroke'
    if isinstance(artist, AxesImage):
        return 'fill'
    if isinstance(artist, LineCollection):
        return 'stroke'
    if isinstance(artist, Collection):
        ec, fc = artist.get_edgecolor(), artist.get_facecolor()
        if len(fc) and len(ec) and np.array_equal(ec, fc):
            return 'fill stroke'
        if not len(fc):
            return 'stroke'
    return 'fill'


def capture_mark(mark):
    if 'field_config' not in mark.data: return
    artist, resolve = mark.data['field_artist'], mark.data.get('field_resolve')
    field = _capture(artist, mark.data['field_config'], resolve)
    mark.data['field'] = field
    from matplotlib.image import AxesImage
    recolor = 'regenerate' if isinstance(artist, AxesImage) else 'live'
    raster = mark.data.get('value_raster')
    if recolor == 'regenerate' and raster:
        recolor = 'raster'  # the values travel beside the SVG: a canvas can repaint them
    mark.data['color_scale'] = colorscale.scale_record(
        field['controlKey'], artist, label=field.get('label'), extend=field['extend'], recolor=recolor)
    mark.data['color_paint'] = paint_of(artist)
    if raster:
        arr = np.ma.masked_invalid(np.ma.asarray(artist.get_array(), dtype=float))
        raster['payload'] = {'spec': 'fluxplot/values', 'scale': field['controlKey'],
                             'shape': list(arr.shape), 'values': values(arr.reshape(-1))}
        mark.data['color_scale']['valueRaster'] = raster['filename']


def anonymous_scales(ax, reg):
    """Colour scales for the raw colour-mapped artists on ``ax`` no helper tagged: every image
    or collection with a data array (``imshow``, ``pcolormesh``, ``scatter(c=…)``, a contour set)
    gets a scale named after its own gid, so its colour key links and its colours are editable.
    Returns ``[{"gid", "record", "paint"}]``; there is no series — just the scale."""
    from matplotlib.image import AxesImage
    tagged = {id(a) for m in reg.marks for a in m.artists}
    out = []
    for art in list(ax.images) + list(ax.collections):
        if id(art) in tagged or not art.get_gid() or not art.get_visible():
            continue
        if getattr(art, 'get_array', None) is None or art.get_array() is None:
            continue
        gid = art.get_gid()
        record = colorscale.scale_record(gid, art, recolor='regenerate' if isinstance(art, AxesImage) else 'live')
        out.append({'gid': gid, 'record': record, 'paint': paint_of(art)})
    return out


class vector_colorbars:
    """Render colour-key solids as vectors (matplotlib rasterizes them by default) so postprocess
    can replace the quads with one exact gradient rect; restores each colorbar's setting after."""

    def __init__(self, fig):
        from .panels import all_axes
        self.solids = [ax._colorbar.solids for ax in all_axes(fig)
                       if getattr(ax, '_colorbar', None) is not None and ax._colorbar.solids is not None]

    def __enter__(self):
        self.before = [(s, s.get_rasterized()) for s in self.solids]
        for s in self.solids:
            s.set_rasterized(False)

    def __exit__(self, *exc):
        for s, was in self.before:
            s.set_rasterized(was)


def _tick_kinds(axis):
    """Portable names for the colour key's tick locator and formatter classes."""
    from matplotlib import ticker
    loc, fmt = axis.get_major_locator(), axis.get_major_formatter()
    if isinstance(loc, ticker.LogLocator):
        locator = 'log'
    elif isinstance(loc, ticker.FixedLocator):
        locator = 'fixed'
    else:
        locator = 'auto'
    if isinstance(fmt, ticker.LogFormatter):
        formatter = 'log'
    elif isinstance(fmt, ticker.PercentFormatter):
        formatter = 'percent'
    elif isinstance(fmt, ticker.ScalarFormatter):
        formatter = 'sci' if fmt.get_offset() else 'plain'
    else:
        formatter = 'custom'
    return locator, formatter


def _gradient(cb, fig):
    """The exact vector form of a colour key's solids: one hard-stepped gradient along its long
    axis. Offsets are the quad boundaries (``cb._y``) in SVG user units along the axis — uniform
    in the axis' own scale, whatever the norm — and the colours are the quads' own."""
    from matplotlib.colors import to_hex, to_rgba
    from .capture import data_to_svg
    vertical = cb.orientation == 'vertical'
    y = np.asarray(cb._y, dtype=float)
    along = np.array([(data_to_svg(cb.ax, fig, 0.5, v) if vertical else data_to_svg(cb.ax, fig, v, 0.5))
                      [1 if vertical else 0] for v in y])
    span = along[-1] - along[0]
    offsets = (along - along[0]) / span if span else np.linspace(0, 1, len(along))
    values = np.asarray(cb._values)[cb._inside]
    rgba = cb.cmap(cb.norm(values))
    colours = [to_hex(c, keep_alpha=False) for c in rgba]
    alpha = float(cb.alpha) if cb.alpha is not None else None
    opacities = [round(float(c[3]) * (alpha if alpha is not None else 1.0), 6) for c in rgba]
    if len(colours) != len(offsets) - 1:
        return None
    return {'axis': 'y' if vertical else 'x', 'start': float(along[0]), 'end': float(along[-1]),
            'offsets': [float(o) for o in offsets], 'colors': colours, 'opacities': opacities}


def colorbar_guides(fig, owner, alloc):
    from .panels import all_axes
    from .capture import data_to_svg
    guides = []
    for ax in all_axes(fig):
        cb = getattr(ax, '_colorbar', None)
        if cb is None: continue
        parent = getattr(ax, '_fluxplot_owner_axes', getattr(cb.mappable, 'axes', None))
        if parent is not owner: continue
        gid = alloc.take('colorbar.' + ids.slugify(getattr(ax, '_fluxplot_colorbar_name', 'color')))
        ax.set_gid(gid)
        vertical = cb.orientation == 'vertical'
        axis = ax.yaxis if vertical else ax.xaxis
        parts = []
        artists = [('label', axis.label), ('outline', cb.outline)]
        primary_side = 2 if axis.get_ticks_position() in ('right', 'top') else 1
        artists.extend((suffix.replace('ticklabel', 'tick-label'), art)
                       for suffix, role, _, art in tagger.axis_tick_artists(axis, primary_side)
                       if role in ('tick', 'tick-label', 'gridline'))
        if cb.solids is not None: artists.append(('solids', cb.solids))
        # the extend triangles / rectangles beyond the ends: lower first, then upper
        extend_parts = {}
        sides = [side for side, on in (('min', cb._extend_lower()), ('max', cb._extend_upper())) if on]
        for side, patch in zip(sides, getattr(cb, '_extend_patches', [])):
            artists.append((f'extend-{side}', patch))
        for suffix, art in artists:
            if not art.get_visible(): continue
            part = gid + '.' + suffix
            # Surface keys may already have explicit registered identity.
            if art.get_gid(): part = art.get_gid()
            else: art.set_gid(part)
            role = next(token for token in suffix.split('.')
                        if token not in ('minor', 'secondary'))
            if role.startswith('extend-'):
                extend_parts[role[len('extend-'):]] = part
                role = 'extend'
            kind = {'label': 'text', 'tick-label': 'text', 'tick': 'line',
                    'gridline': 'line', 'outline': 'line', 'solids': 'shape', 'extend': 'shape'}[role]
            parts.append({'svgId': part, 'role': 'colorbar-' + role, 'kind': kind})
            guides.append(GuideTag(gid=part, role=parts[-1]['role'], kind=kind))
        lo, hi = sorted(axis.get_view_interval())
        visible_ticks = [t.get_loc() for t in axis.get_major_ticks()
                         if np.isfinite(t.get_loc()) and lo <= t.get_loc() <= hi
                         and t.get_visible() and any(a.get_visible() for a in
                             (t.tick1line, t.tick2line, t.label1, t.label2))]
        # data ↔ SVG anchors along the long axis, exactly as capture_axes does for plot axes
        anchors = []
        norm = cb.norm
        for value in (norm.vmin, getattr(norm, 'vcenter', None), norm.vmax):
            if value is None or not np.isfinite(value): continue
            sx, sy = data_to_svg(ax, fig, 0.5, value) if vertical else data_to_svg(ax, fig, value, 0.5)
            anchors.append({'value': float(value), 'svg': sy if vertical else sx})
        locator, formatter = _tick_kinds(axis)
        data = {'orientation': cb.orientation, 'label': axis.label.get_text(),
                'ticks': values(visible_ticks), 'normalization': normalization(cb.mappable),
                'cmap': cb.mappable.get_cmap().name, 'parts': parts,
                'mappable': cb.mappable.get_gid() if hasattr(cb.mappable, 'get_gid') else None,
                'anchors': anchors, 'axisLength': abs(anchors[-1]['svg'] - anchors[0]['svg']) if len(anchors) > 1 else None,
                'tickLocator': locator, 'tickFormatter': formatter, 'extend': cb.extend}
        if extend_parts:
            data['extendParts'] = extend_parts
        if cb.solids is not None and cb.solids.get_visible():
            gradient = _gradient(cb, fig)
            if gradient is not None:
                data['_gradient'] = dict(gradient, solids=cb.solids.get_gid())  # consumed by postprocess
        guides.append(GuideTag(gid=gid, role='colorbar', data=data))
    return guides
