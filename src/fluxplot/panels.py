"""Panel ownership and ID namespacing, shared by every save pipeline stage."""
from __future__ import annotations
from dataclasses import dataclass, replace
from types import SimpleNamespace
from . import ids
from .autotag import is_colorbar_axes


def all_axes(fig):
    out, seen = [], set()
    def visit(ax):
        if id(ax) in seen:
            return
        seen.add(id(ax)); out.append(ax)
        for child in ax.child_axes:
            visit(child)
    for ax in fig.axes:
        visit(ax)
    return out


def panel(ax, name):
    """Give an axes a durable panel identity, independent of subplot layout/order."""
    name = str(name).strip()
    if not name:
        raise ValueError('panel name cannot be empty')
    slug = ids.slugify(name)
    for other in all_axes(ax.figure):
        old = getattr(other, '_fluxplot_panel_name', None)
        if other is not ax and old is not None and ids.slugify(old) == slug:
            raise ValueError(f'panel name {name!r} collides with {old!r}')
    ax._fluxplot_panel_name = name
    return ax


def _letters(n):
    out = ''
    while n >= 0:
        out = chr(97 + n % 26) + out
        n = n // 26 - 1
    return out


@dataclass
class Panel:
    axes: object
    id: str | None
    label: str
    index: int

    @property
    def prefix(self):
        return self.id + '.' if self.id else ''

    @property
    def svg_id(self):
        return self.prefix + 'plot-area'


class ScopedAllocator:
    def __init__(self, allocator, prefix):
        self.allocator, self.prefix = allocator, prefix

    def take(self, candidate):
        return self.allocator.take(candidate if self.prefix and candidate.startswith(self.prefix) else self.prefix + candidate)


def plan(fig):
    axes = [ax for ax in all_axes(fig) if not is_colorbar_axes(ax)]
    axes.sort(key=lambda a: (-round(a.get_position().y1, 8), round(a.get_position().x0, 8)))
    multi = len(axes) > 1 or any(getattr(a, '_fluxplot_panel_name', None) for a in axes)
    reserved = {ids.slugify(a._fluxplot_panel_name) for a in axes if getattr(a, '_fluxplot_panel_name', None)}
    result, n = [], 0
    for index, ax in enumerate(axes):
        label = getattr(ax, '_fluxplot_panel_name', None)
        if label is None:
            while _letters(n) in reserved:
                n += 1
            label = _letters(n); n += 1
        result.append(Panel(ax, 'panel.' + ids.slugify(label) if multi else None, label, index))
    return result


def local_mark(mark, prefix):
    def local(gid):
        return gid[len(prefix):] if gid and prefix and gid.startswith(prefix) else gid
    data = dict(mark.data)
    if data.get('field_members'):
        data['field_members'] = [local(g) for g in data['field_members']]
    if data.get('label_gid'):
        data['label_gid'] = local(data['label_gid'])
    return replace(mark, gid=local(mark.gid), member_gids=[local(g) for g in mark.member_gids], data=data)


def namespace(man, prefix):
    """Prefix semantic references only; data, display names, roles and paths stay literal."""
    if not prefix:
        return man
    def ref(value):
        return prefix + value
    def walk(node):
        for key in ('id', 'ref', 'svgId'):
            if node.get(key):
                node[key] = ref(node[key])
        if 'members' in node:
            node['members'] = [ref(v) for v in node['members']]
        for c in node.get('children', []):
            walk(c)
    for axis in man['axes']:
        axis['id'], axis['svgId'] = ref(axis['id']), ref(axis['svgId'])
    for s in man['series']:
        s['id'] = ref(s['id'])
        s['svg'] = {k: [ref(g) for g in v] if isinstance(v, list) else ref(v) for k, v in s['svg'].items()}
        for c in s.get('components', []):
            walk(c)
        for p in s.get('points', []):
            p['svgId'] = ref(p['svgId'])
        for part in s.get('surface', {}).get('parts', []):
            if part.get('ref'): part['ref'] = ref(part['ref'])
    for g in man.get('guides', []):
        walk(g)
        if g.get('mappable'): g['mappable'] = ref(g['mappable'])
        for part in g.get('parts', []): walk(part)
        for entry in g.get('entries', []):
            for key in ('series', 'part', 'swatch', 'label'):
                if entry.get(key): entry[key] = ref(entry[key])
    for o in man.get('overlays', []):
        walk(o)
    walk(man['parts'])
    # The one legacy role token has figure-wide meaning and is retained once.
    man['build']['order'] = [ref(v) if v != 'gridlines' else v for v in man['build']['order']]
    return man


def manifest(fig, reg, guides_by_panel, panels, axes_capture, present, rasterized, **kwargs):
    from .manifest import build_manifest
    documents = []
    for panel, guides, capture in zip(panels, guides_by_panel, axes_capture):
        prefix = panel.prefix
        marks = [local_mark(m, prefix) for m in reg.marks if getattr(m, '_panel_axes', m.axes) is panel.axes]
        local_guides = []
        for g in guides:
            def local(v):
                return v[len(prefix):] if v and prefix and v.startswith(prefix) else v
            gd = dict(g.data)
            if 'mappable' in gd: gd['mappable'] = local(gd['mappable'])
            if 'parts' in gd:
                gd['parts'] = [{**v, 'svgId': local(v['svgId'])} for v in gd['parts']]
            local_guides.append(replace(g, gid=local(g.gid), data=gd))
        kept = {g[len(prefix):] if prefix else g for g in present if not prefix or g.startswith(prefix)}
        rasters = {g[len(prefix):] if prefix else g for g in rasterized if not prefix or g.startswith(prefix)}
        doc = build_manifest(fig, SimpleNamespace(marks=marks), local_guides, [capture],
                             present=kept, rasterized=rasters, **kwargs)
        namespace(doc, prefix)
        if panel.id:
            for s in doc['series']:
                s['panelId'] = panel.id
            doc['axes'][0]['panelId'] = panel.id
        documents.append(doc)
    if not documents:
        return build_manifest(fig, reg, [], [], present=present, rasterized=rasterized, **kwargs)
    out = documents[0]
    if panels[0].id is None:
        return out
    parts, descriptions = [], []
    for p, d in zip(panels, documents):
        parts.append({'id': p.id, 'role': 'panel', 'label': p.label, 'kind': 'container',
                      'children': d['parts']['children']})
        descriptions.append({'id': p.id, 'svgId': p.svg_id, 'label': p.label, 'index': p.index})
    out['panels'] = descriptions
    out['parts'] = {'id': 'figure', 'role': 'figure', 'kind': 'container', 'children': parts}
    for key in ('axes', 'series', 'guides', 'overlays'):
        out[key] = [entry for d in documents for entry in d[key]]
    out['build'] = {'order': list(dict.fromkeys(v for d in documents for v in d['build']['order'])),
                    'presets': {k: v for d in documents for k, v in d['build']['presets'].items()}}
    return out
