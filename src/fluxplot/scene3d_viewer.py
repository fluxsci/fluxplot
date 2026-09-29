"""Notebook HTML using Flux's vendored renderer, bundled with a static PNG alternative."""
from __future__ import annotations
import base64
from copy import deepcopy
from functools import cmp_to_key
import hashlib
import io
import json
import math
from importlib.resources import files
import warnings
import numpy as np

#: CSS pixels per figure inch of the notebook viewer (its HTML payload uses the same).
PX_PER_INCH = 120
#: The static PNG is rendered at this multiple of its CSS size for HiDPI screens.
HIDPI = 2

EMPTY_MESSAGE = ('Scene3D: nothing to show yet; add a mesh with fp.mesh3d(sc, (vertices, faces), '
                 "series='name') or a value map with fp.surface3d(sc, values, series=..., surfaces=...)")


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


# ---- static still ------------------------------------------------------------------------
# The PNG mirrors the notebook viewer: the same CSS-pixel layout rules as Flux's
# furnitureLayout.ts / furniture.ts (text in physical points, 4/3 CSS px per point),
# a camera matching orbit.ts, and a painter's sort instead of a depth buffer.

def _tick_label(value):
    """Flux ticks.ts tickLabel: compact decimals, exponent outside [1e-3, 1e5)."""
    if value == 0:
        return '0'
    if abs(value) >= 1e5 or abs(value) < 1e-3:
        mantissa, exponent = f'{value:.2e}'.split('e')
        if set(mantissa.split('.')[-1]) == {'0'}:
            mantissa = mantissa.split('.')[0]
        exponent = int(exponent)
        return f'{mantissa}e{"+" if exponent >= 0 else "-"}{abs(exponent)}'
    return f'{value:.6g}'


# Deterministic Arial advance estimates shared with Flux textMetrics.ts; unknown
# code points use 600/1000 em. The shared 20% reserve covers common sans fonts;
# custom fonts remain estimates. Physical rendered sizes do not change.
_ADVANCE = tuple(map(int, '278,278,355,556,556,889,667,191,333,333,389,584,278,333,278,278,556,556,556,556,556,556,556,556,556,556,278,278,584,584,584,556,1015,667,667,722,722,667,611,778,722,278,500,667,556,833,722,778,667,778,722,667,611,722,667,944,667,667,611,278,278,278,469,556,333,556,556,500,556,556,278,556,556,222,222,500,222,833,556,556,556,556,333,500,278,556,500,722,500,500,500,334,260,334,584'.split(',')))
_EXTRA = {'µ':576,'°':400,'±':549,'−':584,'×':584,'÷':549,'²':333,'³':333,'¹':333,'·':278,'–':556,'—':1000,'λ':500,'Å':667,'π':690,'σ':617,'Δ':668,'α':578,'β':575,'γ':500,'θ':556,'…':1000,'’':222}


def _text_width(text, font_px):
    return sum(_ADVANCE[ord(ch)-32] if 32 <= ord(ch) < 127 else _EXTRA.get(ch, 600) for ch in text) * font_px / 1000 * 1.2


def _wrap_words(text, font_px, max_width):
    lines = []; line = ''
    for word in text.split():
        if line and _text_width(line + ' ' + word, font_px) <= max_width:
            line += ' ' + word
            continue
        if line:
            lines.append(line); line = ''
        for ch in word:
            if line and _text_width(line + ch, font_px) > max_width:
                lines.append(line); line = ''
            line += ch
    if line:
        lines.append(line)
    return lines or ['']


def _nice_ticks(lo, hi, count=5):
    if lo == hi:
        return [lo]
    raw = (hi-lo)/(count-1); power = 10 ** math.floor(math.log10(raw)); error = raw/power
    step = (10 if error >= math.sqrt(50) else 5 if error >= math.sqrt(10) else 2 if error >= math.sqrt(2) else 1)*power
    first = math.ceil(lo/step-1e-10); last = math.floor(hi/step+1e-10)
    return [float(f'{i*step:.12g}') or 0 for i in range(first, min(last+1, first+1000))]


def _field_ticks(field):
    lo, hi = field['range']
    return [v for v in field.get('ticks', _nice_ticks(lo, hi)) if lo <= v <= hi]


def _layout(manifest, width, height):
    """Source-layout twin of Flux furnitureLayout (CSS px, y down; physical fonts)."""
    width, height = max(1, width), max(1, height)
    style = manifest.get('style', {}); parts = manifest.get('parts', [])
    by_id = {p['id']: p for p in parts}
    def hidden(part):
        return part.get('hidden', False) or bool(part.get('parent') in by_id and hidden(by_id[part['parent']]))
    def visible(role):
        return [p for p in parts if p['role'] == role and not hidden(p)]
    fs = style.get('fontSizePt', 7)*4/3; line_height = fs*1.4; pad = fs*.5
    rules = manifest.get('layout', {})
    titles = [] if rules.get('title') == 'none' else visible('title')
    bars = [] if rules.get('colorbar') == 'none' else visible('colorbar')
    legends = [] if rules.get('legend') == 'none' else visible('legend')
    scales = visible('scalebar')
    title_h = max(line_height, style.get('titleSizePt', 8)*4/3*1.5)+4 if titles else 0
    def field_of(part):
        return by_id.get(part.get('field'), {}).get('field')
    def entries_of(part):
        return [pid for pid in part.get('entries', []) if pid in by_id and not hidden(by_id[pid])]
    need_w = 0
    for part in bars:
        field = field_of(part)
        if not isinstance(field, dict):
            continue
        tick_w = max([0] + [_text_width(_tick_label(v), fs) for v in _field_ticks(field)])
        need_w = max(need_w, fs + max(_text_width(field.get('label', ''), fs), max(6, fs)+6+tick_w)+pad)
    for part in legends:
        need_w = max(need_w, fs*2.5+max([0]+[_text_width(by_id[pid].get('label', pid), fs) for pid in entries_of(part)])+pad)
    guide_w = min(width*.4, max(64, fs*9, need_w)) if bars or legends else 0
    margin = min(3*fs, width*.15, height*.15) if manifest.get('axes', {}).get('kind') == 'box' else 0
    vp = dict(x=margin, y=title_h+margin, width=max(1, width-guide_w-2*margin), height=max(1, height-title_h-2*margin))
    out = dict(width=width, height=height, fs=fs, line_height=line_height, viewport=vp,
               colorbars=[], legends=[], scalebars=[], title=None, overflow=False, overflowParts=[])
    def overflow(pid):
        if pid not in out['overflowParts']:
            out['overflowParts'].append(pid)
    guides = []
    for part in bars:
        field = field_of(part) or {}; label = field.get('label')
        lines = _wrap_words(label, fs, guide_w-fs-pad) if label else []
        top = fs*(2.1+1.25*max(0, len(lines)-1)); bottom = 2*line_height
        if any(_text_width(line, fs) > guide_w-fs-pad for line in lines) or any(fs+max(6, fs)+6+_text_width(_tick_label(v), fs)+pad > guide_w for v in (_field_ticks(field) if field else [])):
            overflow(part['id'])
        guides.append(dict(part=part, top=top, bottom=bottom, minHeight=top+bottom+max(fs*2, (len(_field_ticks(field))-1)*fs*1.4 if field else 0), titleLines=lines))
    for part in legends:
        rows = []; offset = 0
        for pid in entries_of(part):
            lines = _wrap_words(by_id[pid].get('label', pid), fs, guide_w-fs*2.5-pad)
            rows.append(dict(id=pid, lines=lines, offset=offset)); offset += len(lines)*line_height
            if any(_text_width(line, fs) > guide_w-fs*2.5-pad for line in lines):
                overflow(part['id'])
        guides.append(dict(part=part, top=line_height, bottom=line_height, minHeight=line_height+max(line_height, offset), legendRows=rows))
    available = max(0, height-title_h); equal = available/max(1, len(guides))
    heights = [max(equal, g['minHeight']) for g in guides]; deficit = sum(heights)-available
    for i, g in enumerate(guides):
        if deficit <= 0:
            break
        take = min(deficit, heights[i]-g['minHeight']); heights[i] -= take; deficit -= take
    offset = 0
    for g, slot_h in zip(guides, heights):
        if offset+slot_h > available+1e-9:
            overflow(g['part']['id'])
        slot = dict(part=g['part'], x=width-guide_w+fs, y=title_h+offset+g['top'])
        if 'titleLines' in g:
            slot.update(width=max(6, fs), height=max(1, slot_h-g['top']-g['bottom']), titleLines=g['titleLines'])
            out['colorbars'].append(slot)
        else:
            slot.update(width=max(1, guide_w-fs*2), height=max(g['top'], slot_h-g['top']), legendRows=g['legendRows'])
            out['legends'].append(slot)
        offset += slot_h
    for i, part in enumerate(scales):
        out['scalebars'].append(dict(part=part, x=vp['x']+12, y=vp['y']+vp['height']-12-i*(line_height+10)))
    if titles:
        out['title'] = dict(part=titles[0], x=0, y=0, width=width, height=title_h)
    out['overflow'] = bool(out['overflowParts'])
    return out


class _Camera:
    """Orbit camera of Flux ``orbitPose`` over the manifest's world bounds."""

    def __init__(self, view, bounds, viewport):
        az, el, roll = np.deg2rad([view['azimuth'] % 360, view['elevation'], view.get('roll', 0)])
        direction = np.array([np.sin(az)*np.cos(el), np.sin(el), np.cos(az)*np.cos(el)])  # toward the camera
        right = np.array([np.cos(az), 0, -np.sin(az)]); up = np.cross(direction, right)
        right, up = right*np.cos(roll) + up*np.sin(roll), up*np.cos(roll) - right*np.sin(roll)
        # Framing (Flux orbit.ts boundsSphere/orbitPose): the tight ``radius`` when the
        # bounds carry one (see _framing_bounds), else the circumscribed sphere;
        # radius / zoom across the viewport's smaller side.
        tight = bounds.get('radius')
        bounds = np.array([bounds['min'], bounds['max']]); center = bounds.mean(axis=0)
        radius = max(tight if tight is not None and np.isfinite(tight) and tight >= 0
                     else np.linalg.norm((bounds[1] - bounds[0]) / 2), 1e-9)
        target = center + radius * (view.get('panX', 0) * right + view.get('panY', 0) * up)
        half = radius / view['zoom']; half_fov = np.deg2rad(view.get('fov', 30)) / 2
        distance = half / np.sin(half_fov)
        self.perspective = view['projection'] == 'perspective'
        if self.perspective:
            half = distance * np.tan(half_fov)
            self.near = max(.001 * distance, distance - 1.2 * radius * max(1, 1 / view['zoom']))
        else:
            distance = 3 * radius
            self.near = -math.inf
        aspect = viewport['width'] / viewport['height']
        self.half_width, self.half_height = half * max(aspect, 1), half * max(1 / aspect, 1)
        self.direction, self.right, self.up, self.target = direction, right, up, target
        self.distance, self.viewport = distance, viewport

    def project(self, points):
        """Camera-plane coordinates (image plane at the target) and depth toward the camera."""
        delta = np.asarray(points, dtype=float) - self.target
        xy = np.column_stack([delta @ self.right, delta @ self.up]); toward = delta @ self.direction
        if self.perspective:
            xy *= self.distance / np.maximum(self.distance - toward, 1e-9)[:, None]
        return xy, toward

    def screen(self, points):
        """CSS-pixel positions (y down) and distance from the camera, like Flux ``project``."""
        xy, toward = self.project(np.atleast_2d(points))
        vp = self.viewport
        x = vp['x'] + vp['width'] * (.5 + xy[:, 0] / (2 * self.half_width))
        y = vp['y'] + vp['height'] * (.5 - xy[:, 1] / (2 * self.half_height))
        return np.column_stack([x, y, self.distance - toward])

    def visible(self, point):
        return bool(np.isfinite(point[:2]).all() and point[2] >= self.near)

    def pixels_per_unit(self):
        return None if self.perspective else self.viewport['height'] / (2 * self.half_height)


class _Furniture:
    """Draws Flux furniture nodes into one matplotlib axes whose data units are CSS px."""

    def __init__(self, ax, style, fs):
        self.ax, self.style, self.fs = ax, style, fs
        self.ink = style.get('ink', '#100F0F'); self.muted = style.get('muted', '#6F6E69')
        self.lw = style.get('lineWidthPt', .6) * 4 / 3
        self.font = style.get('font')

    @staticmethod
    def pt(px):
        """CSS px -> matplotlib points at the viewer's PX_PER_INCH."""
        return px * 72 / PX_PER_INCH

    def line(self, a, b, *, color=None, alpha=1, width=None):
        self.ax.plot([a[0], b[0]], [a[1], b[1]], color=color or self.ink, alpha=alpha,
                     lw=self.pt(width or self.lw), solid_capstyle='butt')

    def text(self, x, y, label, *, anchor='middle', size=None, rotation=0, center=None):
        """SVG-style text: (x, y) is the baseline point; optional rotation about ``center``."""
        ha = {'start': 'left', 'middle': 'center', 'end': 'right'}[anchor]
        kwargs = dict(ha=ha, va='baseline', fontsize=self.pt(size or self.fs), color=self.ink)
        if self.font:
            kwargs['fontfamily'] = self.font
        if rotation:
            cx, cy = center
            theta = np.deg2rad(rotation); dx, dy = x - cx, y - cy
            x = cx + dx * np.cos(theta) - dy * np.sin(theta)
            y = cy + dx * np.sin(theta) + dy * np.cos(theta)
            kwargs.update(rotation=-rotation, rotation_mode='anchor')
        self.ax.text(x, y, label, **kwargs)

    def polygon(self, points, **kwargs):
        from matplotlib.patches import Polygon
        self.ax.add_patch(Polygon(points, closed=True, **kwargs))


def _rotation(manifest):
    return np.asarray(manifest['toWorld'], dtype=float).reshape(4, 4, order='F')[:3, :3]


def _box_limits(manifest):
    """Data-space box-axes limits: an axis's ``lim``, else the world bounds carried back
    into data space (Flux framing.ts ``axesBoxLimits``)."""
    world_bounds = np.array([manifest['bounds']['min'], manifest['bounds']['max']])
    corners = np.array([[world_bounds[(mask >> i) & 1, i] for i in range(3)] for mask in range(8)])
    data_corners = corners @ _rotation(manifest)  # inverse of a proper rotation is its transpose
    axes = [manifest['axes'].get(k, {}) for k in 'xyz']
    return [a.get('lim', [data_corners[:, i].min(), data_corners[:, i].max()]) for i, a in enumerate(axes)]


def _framing_bounds(scene, manifest):
    """The bounds the camera frames, exactly as Flux frames the saved GLB.

    A bare mesh frames its tight sphere: the largest distance from the AABB centre to
    any vertex, base shape and every state (Flux glbCore ``framingRadius``). Box axes
    are part of the figure, so the frame grows to hold the whole axes box and uses its
    circumscribed sphere (Flux framing.ts ``framingBounds``).
    """
    lo, hi = np.array(manifest['bounds']['min'], dtype=float), np.array(manifest['bounds']['max'], dtype=float)
    rotation = _rotation(manifest)
    if manifest.get('axes', {}).get('kind') == 'box':
        limits = _box_limits(manifest)
        corners = np.array([[limits[i][(mask >> i) & 1] for i in range(3)] for mask in range(8)]) @ rotation.T
        return {'min': np.minimum(lo, corners.min(axis=0)).tolist(), 'max': np.maximum(hi, corners.max(axis=0)).tolist()}
    world = np.concatenate([v for p in scene.parts for v in [p.vertices, *p.states.values()]]) @ rotation.T
    radius = float(np.linalg.norm(world - (lo + hi) / 2, axis=1).max())
    return {'min': lo.tolist(), 'max': hi.tolist(), 'radius': radius}


def _tick_labels_collide(labels, font_px, anchor):
    """Flux furniture.ts ``tickLabelsCollide``: do any two of one axis's tick labels collide?

    ``labels`` are ``(x, y, width)`` label points (vertical centre) sharing an anchor and a
    font size; each is a box of its deterministic width by one font size. Side by side they
    must sit at least a word space apart. A pure function of the pose (no hysteresis).
    """
    space = _text_width(' ', font_px)
    boxes = []
    for x, y, width in labels:
        x0 = x - width if anchor == 'end' else x if anchor == 'start' else x - width / 2
        boxes.append((x0 - space / 2, x0 + width + space / 2, y - font_px / 2, y + font_px / 2))
    return any(a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]
               for i, a in enumerate(boxes) for b in boxes[i + 1:])


def _draw_box_axes(draw, manifest, camera, fs):
    """Back panes and grid behind the mesh; axis lines on silhouette edges, labels outward."""
    rotation = _rotation(manifest)
    axes = [manifest['axes'].get(k, {}) for k in 'xyz']
    limits = _box_limits(manifest)
    center = np.array([sum(lim) / 2 for lim in limits])
    screen = lambda p: camera.screen(np.asarray(p, dtype=float) @ rotation.T)[0]
    back = [0 if rotation[:, axis] @ camera.direction >= 0 else 1 for axis in range(3)]
    parts = {p['id']: p for p in manifest.get('parts', [])}
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        quad = []
        for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
            p = center.copy(); p[axis] = limits[axis][back[axis]]
            p[other[0]] = limits[other[0]][a]; p[other[1]] = limits[other[1]][b]
            quad.append(screen(p))
        if all(camera.visible(q) for q in quad):
            points = [q[:2] for q in quad]
            draw.polygon(points, facecolor=draw.muted, alpha=.06, edgecolor='none')
            draw.polygon(points, facecolor='none', edgecolor=draw.muted, alpha=.25, lw=draw.pt(draw.lw))
    box = [screen([limits[i][(mask >> i) & 1] for i in range(3)]) for mask in range(8)]
    box = [c for c in box if camera.visible(c)]
    labelled = []
    for axis, name in enumerate('xyz'):
        other = [i for i in range(3) if i != axis]
        candidates = []
        for k in range(4):
            a = center.copy(); b = center.copy(); a[axis] = limits[axis][0]; b[axis] = limits[axis][1]
            for j in range(2):
                a[other[j]] = b[other[j]] = limits[other[j]][(k >> j) & 1]
            mid = screen((a + b) / 2)
            if camera.visible(mid) and camera.visible(screen(a)) and camera.visible(screen(b)):
                candidates.append(dict(a=a, b=b, y=mid[1], depth=mid[2]))
        if not candidates:
            continue

        def silhouette(edge):
            sa, sb = screen(edge['a']), screen(edge['b'])
            length = math.hypot(sb[0] - sa[0], sb[1] - sa[1])
            if length < 1:
                return False
            d = [((sb[0]-sa[0]) * (p[1]-sa[1]) - (sb[1]-sa[1]) * (p[0]-sa[0])) / length for p in box]
            return all(v >= -1e-6 for v in d) or all(v <= 1e-6 for v in d)

        def overlaps(edge):
            sa, sb = screen(edge['a']), screen(edge['b'])
            dx, dy = sb[0] - sa[0], sb[1] - sa[1]; length = math.hypot(dx, dy)
            for e in labelled:
                ex, ey = e[2] - e[0], e[3] - e[1]
                if (abs(dx*ey - dy*ex) < 1e-6 * length * math.hypot(ex, ey)
                        and abs(dx*(e[1]-sa[1]) - dy*(e[0]-sa[0])) < fs * length):
                    return 1
            return 0

        def order(e1, e2):
            first = overlaps(e1) - overlaps(e2)
            if first:
                return first
            return e2['y'] - e1['y'] if abs(e2['y'] - e1['y']) > 1e-6 else e1['depth'] - e2['depth']

        edges = [e for e in candidates if silhouette(e)] or candidates
        edge = sorted(edges, key=cmp_to_key(order))[0]
        sa, sb = screen(edge['a']), screen(edge['b'])
        length = math.hypot(sb[0] - sa[0], sb[1] - sa[1])
        if length < 1:
            continue
        mid = (sa[:2] + sb[:2]) / 2; origin = screen(center)
        ox, oy = -(sb[1] - sa[1]) / length, (sb[0] - sa[0]) / length
        if ox * (mid[0] - origin[0]) + oy * (mid[1] - origin[1]) < 0:
            ox, oy = -ox, -oy
        labelled.append((sa[0], sa[1], sb[0], sb[1]))
        draw.line(sa, sb)
        anchor = 'end' if ox < -.5 else 'start' if ox > .5 else 'middle'
        spec = axes[axis]
        drawn = []
        for value in spec.get('ticks', []):
            if not limits[axis][0] <= value <= limits[axis][1]:
                continue
            p = edge['a'].copy(); p[axis] = value; at = screen(p)
            if camera.visible(at):
                drawn.append((value, at, _tick_label(value)))
        # Seen nearly end-on, an axis projects to a stub too short for its labels.
        # Hide them (and a title longer than the stub) instead of stacking them.
        crowded = _tick_labels_collide(
            [(at[0] + ox * (fs*.9 + 4), at[1] + oy * (fs*.9 + 4), _text_width(label, fs)) for _, at, label in drawn], fs, anchor)
        title = parts.get(f'axes.{name}.label', {}).get('text') or spec.get('label') or name
        title_hidden = crowded and _text_width(title, fs) > length
        for value, at, label in drawn:
            draw.line(at, (at[0] + ox * 4, at[1] + oy * 4))
            if not crowded:
                draw.text(at[0] + ox * (fs*.9 + 4), at[1] + oy * (fs*.9 + 4) + fs*.3, label, anchor=anchor)
            if manifest['axes'].get('grid') is not False:
                for plane in other:
                    across = next(i for i in other if i != plane)
                    p1 = center.copy(); p2 = center.copy(); p1[axis] = p2[axis] = value
                    p1[plane] = p2[plane] = limits[plane][back[plane]]
                    p1[across] = limits[across][0]; p2[across] = limits[across][1]
                    g1, g2 = screen(p1), screen(p2)
                    if camera.visible(g1) and camera.visible(g2):
                        draw.line(g1, g2, color=draw.muted, alpha=.3)
        if title_hidden:
            continue
        title_x, title_y = mid[0] + ox * fs * 3.4, mid[1] + oy * fs * 3.4
        angle = math.degrees(math.atan2(sb[1] - sa[1], sb[0] - sa[0]))
        upright = angle - 180 if angle > 90 else angle + 180 if angle < -90 else angle
        draw.text(title_x, title_y + fs * .3, title, rotation=upright, center=(title_x, title_y))


def _draw_mesh(ax, scene, camera):
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgba
    from .surface import _face_shading, _shade_rgba
    view = scene._view
    polygons = []; colors = []; depths = []
    for part in scene.parts:
        vertices = part.vertices.copy()
        for name, weight in view.get('states', {}).items():
            if name in part.states: vertices += weight * (part.states[name] - part.vertices)
        world = vertices @ scene.to_world[:3, :3].T
        xy, depth = camera.project(world)
        polygons.append(xy[part.faces]); depths.append(depth[part.faces].mean(axis=1))
        if part.colors is not None:
            rgba = part.colors[part.faces].mean(axis=1)
            rgba[:, 3] *= to_rgba(part.color)[3]  # alpha= on a coloured field
        else:
            rgba = np.tile(to_rgba(part.color), (len(part.faces), 1))
        if scene.lighting == 'studio':
            # Reframe into surface's (toward viewer, right, up) convention.
            camera_vertices = np.column_stack([world @ camera.direction, world @ camera.right, world @ camera.up])
            rgba = _shade_rgba(rgba, _face_shading(camera_vertices, part.faces, 1, (.8, -.4, .6), .35))
        colors.append(rgba)
    polygons = np.concatenate(polygons); colors = np.concatenate(colors); depths = np.concatenate(depths)
    order = np.argsort(depths, kind='stable')
    colors = colors[order]
    # Opaque faces draw hairline edges in their own colour to close raster seams.
    # Translucent faces must not: each edge would be blended twice and etch a
    # triangle lattice into the part. (Aliased fills double-cover shared edges too.)
    opaque = colors[:, 3] >= 1
    edges = colors.copy(); edges[~opaque] = 0
    # The camera sets both limits. Avoid scanning every triangle a second time to
    # derive automatic limits that would immediately be overwritten.
    ax.add_collection(PolyCollection(polygons[order], facecolors=colors, edgecolors=edges,
                                     linewidths=np.where(opaque, .1, 0.), antialiased=True),
                      autolim=False)


def _draw_guides(draw, manifest, layout, camera):
    """Title, legend, colorbar, scale bar and triad in Flux's slots."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgba
    fs, line_height, vp = layout['fs'], layout['line_height'], layout['viewport']
    style = manifest.get('style', {})
    parts = {p['id']: p for p in manifest.get('parts', [])}
    rotation = np.asarray(manifest['toWorld'], dtype=float).reshape(4, 4, order='F')[:3, :3]
    triad = manifest.get('axes', {}).get('kind') == 'triad'
    if triad:
        # Bottom-left orientation gizmo (Flux: viewport corner inset by 30 px).
        origin = (vp['x'] + 30, vp['y'] + vp['height'] - 30)
        for axis, name in enumerate('xyz'):
            direction = rotation[:, axis]
            dx, dy = float(direction @ camera.right), -float(direction @ camera.up)
            draw.line(origin, (origin[0] + dx * 22, origin[1] + dy * 22))
            draw.text(origin[0] + dx * 32, origin[1] + dy * 32 + fs * .3, name)
    ppu = camera.pixels_per_unit()
    for slot in layout['scalebars']:
        part = slot['part']
        if not part.get('length') or ppu is None:
            continue
        length = part['length'] * ppu * float(np.linalg.norm(rotation[:, 0]))
        x, y = slot['x'], slot['y']
        if triad:
            # The triad owns the bottom-left corner; the scale bar takes the bottom-right.
            x = vp['x'] + vp['width'] - 12 - length
        draw.line((x, y), (x + length, y), width=max(draw.lw, 1.5))
        draw.text(x + length / 2, y - fs * .7, part.get('label') or _tick_label(part['length']))
    for slot in layout['legends']:
        for row in slot['legendRows']:
            pid = row['id']
            entry = parts.get(pid)
            if entry is None:
                continue
            y = slot['y'] + row['offset']
            color = to_rgba(entry.get('color', '#4385BE'))
            color = (*color[:3], color[3] * entry.get('opacity', 1))
            draw.polygon([(slot['x'], y - fs*.7), (slot['x'] + fs, y - fs*.7), (slot['x'] + fs, y + fs*.3),
                          (slot['x'], y + fs*.3)], facecolor=color, edgecolor='none')
            for i, label in enumerate(row['lines']):
                draw.text(slot['x'] + fs * 1.5, y + fs * .2 + i*line_height, label, anchor='start')
    for slot in layout['colorbars']:
        field = parts.get(slot['part'].get('field'), {}).get('field')
        if not isinstance(field, dict):
            continue
        lo, hi = field['range']
        x, y, w, h = slot['x'], slot['y'], slot['width'], slot['height']
        cmap = LinearSegmentedColormap.from_list('preview', field['cmap']['stops'])
        ramp = cmap(np.linspace(1, 0, 256))[:, None, :]
        draw.ax.imshow(ramp, extent=(x, x + w, y + h, y), aspect='auto', interpolation='bilinear', zorder=1)
        draw.polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], facecolor='none',
                     edgecolor=draw.muted, lw=draw.pt(draw.lw), zorder=2)
        for value in _field_ticks(field):
            if not lo <= value <= hi:
                continue
            ty = y + h * (.5 if hi == lo else 1 - (value - lo) / (hi - lo))
            draw.line((x + w, ty), (x + w + 3, ty))
            draw.text(x + w + 6, ty + fs * .3, _tick_label(value), anchor='start')
        for i, label in enumerate(slot['titleLines']):
            draw.text(x, y-fs*(1.05+1.25*(len(slot['titleLines'])-1-i)), label, anchor='start')
    if layout['title']:
        slot = layout['title']; part = slot['part']
        draw.text(slot['x'] + slot['width'] / 2, slot['y'] + slot['height'] * .7,
                  part.get('text') or part.get('label') or '', size=style.get('titleSizePt', 8) * 4 / 3)


def png_preview(scene, *, _manifest=None):
    """Static orthographic/perspective painter still of the notebook viewer, at 2× (HiDPI).

    Furniture follows Flux's layout rules. The interactive notebook and Flux views
    use a depth buffer; this fallback sorts all faces together, so intersecting
    transparent surfaces remain an approximate painter view.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from .scene3d_manifest import build_manifest
    from .glb import write_glb
    if not scene.parts: raise ValueError(EMPTY_MESSAGE)
    man = _manifest if _manifest is not None else build_manifest(scene, write_glb(scene), 'preview.glb')
    width, height = scene.figsize[0] * PX_PER_INCH, scene.figsize[1] * PX_PER_INCH
    layout = _layout(man, width, height)
    if layout['overflow']:
        warnings.warn('Enlarge scene3d figsize to fit guide labels at the current font size', stacklevel=2)
    vp = layout['viewport']
    camera = _Camera(scene._view, _framing_bounds(scene, man), vp)
    dpi = PX_PER_INCH * HIDPI
    fig = Figure(figsize=scene.figsize, dpi=dpi, layout='none'); FigureCanvasAgg(fig)
    fig.patch.set_facecolor('white')
    # Mesh axes first (fig.axes[0]): camera-plane units over exactly the viewport.
    ax = fig.add_axes([vp['x'] / width, 1 - (vp['y'] + vp['height']) / height,
                       vp['width'] / width, vp['height'] / height], zorder=1)
    ax.set_xlim(-camera.half_width, camera.half_width); ax.set_ylim(-camera.half_height, camera.half_height)
    ax.set_axis_off(); ax.patch.set_alpha(0)
    # Furniture layers in CSS px (y down): 'under' sits behind the mesh, 'over' in front.
    layers = []
    for zorder in (0, 2):
        layer = fig.add_axes([0, 0, 1, 1], zorder=zorder)
        layer.set_xlim(0, width); layer.set_ylim(height, 0); layer.set_axis_off(); layer.patch.set_alpha(0)
        layers.append(_Furniture(layer, man.get('style', scene.style), layout['fs']))
    under, over = layers
    _draw_mesh(ax, scene, camera)
    if man.get('axes', {}).get('kind') == 'box':
        _draw_box_axes(under, man, camera, layout['fs'])
    _draw_guides(over, man, layout, camera)
    stream = io.BytesIO(); fig.savefig(stream, format='png', transparent=False, facecolor='white', dpi=dpi)
    return stream.getvalue()


def viewer_bundle():
    folder=files('fluxplot').joinpath('_viewer')
    source=folder.joinpath('flux-model3d-viewer.min.js').read_bytes()
    stamp=json.loads(folder.joinpath('stamp.json').read_text())
    if hashlib.sha256(source).hexdigest()!=stamp['sha256']: raise RuntimeError('Flux notebook viewer stamp mismatch; run scripts/sync_flux_viewer.py')
    return source.decode('utf8')


def display_size(scene):
    """CSS size of the notebook viewer and of its PNG still."""
    return round(scene.figsize[0] * PX_PER_INCH), round(scene.figsize[1] * PX_PER_INCH)


def mimebundle_metadata(scene, bundle):
    """Show the 2× PNG at its CSS size (Jupyter/VS Code honour image width/height)."""
    if 'image/png' not in bundle:
        return {}
    width, height = display_size(scene)
    return {'image/png': {'width': width, 'height': height}}


def mimebundle(scene,*,static=False):
    from .glb import write_glb
    from .scene3d_manifest import build_manifest
    if not scene.parts:
        return {'text/plain': EMPTY_MESSAGE}
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
    width,height=display_size(scene)
    payload=json.dumps({'glb':base64.b64encode(data).decode(),'manifest':manifest,'width':width,'height':height},allow_nan=False).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    fallback=base64.b64encode(png).decode()
    image=f'<img alt="3D plot preview" width="{width}" height="{height}" style="max-width:100%;height:auto" src="data:image/png;base64,{fallback}" />'
    # currentScript belongs to each output even in a renderer's shadow root. Capture it
    # synchronously; a document-global id lookup can select another notebook output.
    bundle['text/html']=f'''<div class="fluxplot-scene3d"><div data-fluxplot-scene3d-host>{image}</div><script>(()=>{{const script=document.currentScript;const host=script.parentElement.querySelector('[data-fluxplot-scene3d-host]');const fallback=host.innerHTML;{runtime}\nFluxModel3dViewer.mount(host,{payload}).then(view=>{{if(view.available===false)host.innerHTML=fallback;}}).catch(error=>{{host.innerHTML=fallback;host.title=String(error);}});}})();</script></div>'''
    return bundle
