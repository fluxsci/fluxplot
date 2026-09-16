"""Post-render SVG pass (lxml): inject ``data-*`` + canonicalize.

matplotlib emits only ``id`` (from the gids we set) — never arbitrary ``data-*``. So this pass is
*mechanically required*, and it is NOT the rejected "post-hoc heuristic surgery": the meaning was
captured at birth (we set ``id="control.line"`` before rendering), so this is an **exact structural
join on ids we authored** + lossless annotation. It never infers anything from pixels.

It: (1) joins by our gids and injects ``data-role``/``data-series``/``data-index``/``data-x``/
``data-y``; (2) splits the per-series points group into addressable per-point ``<use>``; (3) renames
matplotlib's ``figure_1``/``axes_1`` wrappers to ``figure``/``plot-area``; (4) strips volatile
metadata (comments, ``<metadata>``); (5) deterministically serializes.
"""
from __future__ import annotations

from lxml import etree

from . import raster as _raster
from .descriptors import Mark, mark_kind
from .roles import kind_for_role

SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"


def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return repr(v)
    return str(v)


def _set(el, **attrs) -> None:
    for k, v in attrs.items():
        if v is None:
            continue
        el.set(k.replace("_", "-"), _fmt(v))


def postprocess(svg_bytes: bytes, reg, guides, plot_type: str, raster_items=(), check_ids=True):
    """Return ``(processed_svg_bytes, warnings, present)``."""
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(svg_bytes, parser)
    warnings: list[str] = []

    # 1. strip volatile metadata
    etree.strip_tags(root, etree.Comment)
    for md in root.findall(f"{{{SVG}}}metadata"):
        md.getparent().remove(md)

    # 1b. give rasterized layers their gids back BEFORE anything joins on ids — matplotlib
    # drops the gid when it rasterizes an artist (see raster.py), and every step below is an
    # exact join on the ids we authored.
    _raster.reattach(root, raster_items, warnings)

    id_map = {el.get("id"): el for el in root.iter() if el.get("id")}

    # 2. rename matplotlib's structural wrappers → semantic roots
    _rename(id_map, "figure_1", "figure", "figure")
    if not any(g == "plot-area" or g.endswith(".plot-area") for g in id_map):
        _rename(id_map, "axes_1", "plot-area", "plot-area")
    for gid, el in id_map.items():
        if gid == "plot-area" or gid.endswith(".plot-area"):
            _set(el, data_role="plot-area", data_kind="container")

    # 3. root markers
    root.set("data-fluxplot", "1")
    root.set("data-plot-type", plot_type)

    # 4. inject data-* per Mark
    for m in reg.marks:
        if m.data.get('contour_legacy'):
            _group_legacy_contour(m, id_map)
        if m.data.get('cells') or m.data.get('contour_paths'):
            _inject_field(m, id_map, warnings)
        if m.role == "point":
            _inject_points(m, id_map, warnings)
        elif m.role == "bar":
            _inject_indexed(m, id_map, role="bar")
        elif m.series is not None:
            # every member (a composite like errorbar renders the centre line + caps +
            # bar segments as numbered siblings) carries role/series/kind, not just the first
            kind = mark_kind(m)
            for gid in m.member_gids or ([m.gid] if m.gid else []):
                el = id_map.get(gid)
                if el is not None:
                    _set(el, data_role=m.role, data_series=m.series, data_kind=kind)
        else:
            _inject_overlay(m, id_map)

    # 5. inject data-role (+ data-kind hint) on scaffold/guides
    for g in guides:
        el = id_map.get(g.gid)
        if el is not None:
            _set(
                el,
                data_role=g.role,
                data_axis=g.axis,
                data_index=g.index,
                data_series=g.series,
                data_kind=g.kind,
            )

    # 6. dereference tick <use> → real <path> so Flux's draw-on preset can measure/animate
    # them (a <use> has no measurable path length). Scoped to axis/colorbar tick
    # groups; point <use> elements (which animate via opacity/transform) are untouched.
    _deref_ticks(root)

    # The set of ids that actually survived into the SVG (computed AFTER injection
    # so per-point <use> ids are included). matplotlib culls boundary ticks/
    # gridlines at draw and omits empty axis titles even though the artists carry a
    # gid — the manifest must reference only what's really here, so callers prune
    # guides/members against this set (else the X-Ray shows dead nodes).
    all_ids = [el.get("id") for el in root.iter() if el.get("id")]
    present = set(all_ids)
    if check_ids and len(all_ids) != len(present):
        from collections import Counter
        duplicates = [gid for gid, count in Counter(all_ids).items() if count > 1]
        raise ValueError('duplicate SVG IDs: ' + ', '.join(duplicates[:10]))

    return _serialize(root), warnings, present


def _rename(id_map, old, new, role) -> None:
    el = id_map.pop(old, None)
    if el is not None:
        el.set("id", new)
        el.set("data-role", role)
        _set(el, data_kind=kind_for_role(role))
        id_map[new] = el


def _inject_points(m: Mark, id_map, warnings) -> None:
    group = id_map.get(m.gid)
    if group is None:
        return
    # the group's kind mirrors its members (edit the group ⇒ restyle every point)
    _set(group, data_series=m.series, data_kind=kind_for_role("point"))
    if group.get("data-rasterized") == "1":
        # A rasterized point cloud IS one <image> — there are no per-point <use> nodes to
        # split, and that is the intended outcome, not a shortfall. The series stays
        # addressable as a whole; no warning (see raster.py).
        _set(group, data_role="point")
        return
    uses = list(group.iter(f"{{{SVG}}}use"))
    n = len(m.member_gids)
    if len(uses) != n:
        warnings.append(
            f"points '{m.gid}': {len(uses)} <use> vs {n} data points — skipping per-point ids"
        )
        _set(group, data_role="point")
        return
    xs = list(m.x) if m.x is not None else [None] * n
    ys = list(m.y) if m.y is not None else [None] * n
    for k, use_el in enumerate(uses):
        use_el.set("id", m.member_gids[k])
        _set(
            use_el,
            data_role="point",
            data_series=m.series,
            data_index=m.member_indices[k],
            data_x=xs[m.member_indices[k]],
            data_y=ys[m.member_indices[k]],
            data_kind=kind_for_role("point"),
        )


def _inject_indexed(m: Mark, id_map, role: str) -> None:
    xs = list(m.x) if m.x is not None else [None] * len(m.member_gids)
    ys = list(m.y) if m.y is not None else [None] * len(m.member_gids)
    for k, gid in enumerate(m.member_gids):
        el = id_map.get(gid)
        if el is None:
            continue
        _set(
            el,
            data_role=role,
            data_series=m.series,
            data_index=k,
            data_x=xs[k] if k < len(xs) else None,
            data_y=ys[k] if k < len(ys) else None,
            data_kind=kind_for_role(role),
        )


def _inject_overlay(m: Mark, id_map) -> None:
    el = id_map.get(m.gid)
    if el is not None:
        _set(el, data_role=m.role, data_name=m.name, data_kind=mark_kind(m))
    label_gid = m.data.get("label_gid")
    if label_gid is not None:
        lab = id_map.get(label_gid)
        if lab is not None:
            _set(lab, data_role="label", data_name=m.name, data_kind=kind_for_role("label"))


def _href(el):
    """Read an SVG reference from either ``xlink:href`` or a plain ``href`` attribute."""
    return el.get(f"{{{XLINK}}}href") or el.get("href")


def _parse_style(s: str | None) -> dict:
    out: dict = {}
    if not s:
        return out
    for decl in s.split(";"):
        decl = decl.strip()
        if not decl:
            continue
        key, _, val = decl.partition(":")
        key = key.strip()
        if key:
            out[key] = val.strip()
    return out


def _merge_style(target_style: str | None, use_style: str | None) -> str | None:
    """Merge the referenced path's style *under* the ``<use>``'s own style (use wins).

    Deterministic: target declarations first (insertion order), then any the use adds/overrides.
    """
    merged = _parse_style(target_style)
    merged.update(_parse_style(use_style))
    if not merged:
        return None
    return "; ".join(f"{k}: {v}" for k, v in merged.items())


def _deref_ticks(root) -> None:
    """Inline ``<use>`` elements in axis and colorbar tick groups as measurable paths.

    matplotlib renders a tick as ``<g data-role="tick"><g><use href="#markerPath"/></g></g>`` where
    only the first tick of an axis carries the shared ``<defs><path>``. A ``<use>`` has no measurable
    geometry, so Flux's draw-on (stroke-dashoffset over path length) can't animate it. We resolve the
    referenced path, inline it (folding the use's x/y offset into a ``translate`` transform and merging
    styles, use wins), unwrap the now-bare anonymous ``<g>``, and drop any tick-local ``<defs>`` whose
    path id is no longer referenced anywhere in the document.
    """
    # Document-wide map of path id → element (the target may live in a *different* tick's defs).
    path_by_id = {p.get("id"): p for p in root.iter(f"{{{SVG}}}path") if p.get("id")}

    tick_groups = [el for el in root.iter() if el.get("data-role") in ("tick", "colorbar-tick")]

    for tg in tick_groups:
        for use in list(tg.iter(f"{{{SVG}}}use")):
            href = _href(use)
            if not href or not href.startswith("#"):
                continue
            target = path_by_id.get(href[1:])
            if target is None:
                continue
            new = etree.Element(f"{{{SVG}}}path")
            new.set("d", target.get("d", ""))
            transforms = []
            if use.get("transform"):
                transforms.append(use.get("transform"))
            x, y = use.get("x"), use.get("y")
            if x is not None or y is not None:
                transforms.append(f"translate({x or 0} {y or 0})")
            if transforms:
                new.set("transform", " ".join(transforms))
            style = _merge_style(target.get("style"), use.get("style"))
            if style:
                new.set("style", style)
            parent = use.getparent()
            idx = parent.index(use)
            parent.remove(use)
            parent.insert(idx, new)
            # Unwrap an anonymous <g> wrapper with no attributes and no other children.
            if parent.tag == f"{{{SVG}}}g" and not parent.attrib and len(parent) == 1:
                gp = parent.getparent()
                if gp is not None:
                    gidx = gp.index(parent)
                    gp.remove(parent)
                    gp.insert(gidx, new)

    # Drop tick-local <defs> paths that nothing references anymore (careful: a <use> elsewhere
    # in the document — e.g. a point cloud — must keep its defs).
    referenced = set()
    for use in root.iter(f"{{{SVG}}}use"):
        href = _href(use)
        if href and href.startswith("#"):
            referenced.add(href[1:])
    for tg in tick_groups:
        for defs in list(tg.iter(f"{{{SVG}}}defs")):
            for p in list(defs):
                pid = p.get("id")
                if p.tag == f"{{{SVG}}}path" and pid is not None and pid not in referenced:
                    defs.remove(p)
            if len(defs) == 0:
                dp = defs.getparent()
                if dp is not None:
                    dp.remove(defs)


def _serialize(root) -> bytes:
    body = etree.tostring(root, pretty_print=True, encoding="unicode")
    return ('<?xml version="1.0" encoding="utf-8" standalone="no"?>\n' + body).encode("utf-8")


def _inject_field(mark, id_map, warnings):
    group = id_map.get(mark.gid)
    if group is None or group.get('data-rasterized') == '1': return
    # Direct paths are emitted in row-major QuadMesh order or ContourSet level
    # order. Exclude definitions/clip paths; reject any backend count mismatch.
    paths = group.findall(f'{{{SVG}}}path')
    field = mark.data['field']
    if mark.data.get('cells'):
        rows, cols = field['shape']
        count = rows * cols
        names = [f'cell.{i // cols}.{i % cols}' for i in range(count)]
    else:
        artist = mark.data['field_artist']
        count = len(artist.get_paths())
        names = [f'level.{i}' for i in range(count)]
    if len(paths) != count:
        warnings.append(f"field '{mark.gid}': backend path count differs; keeping layer identity")
        return
    members = []
    for i, (path, name) in enumerate(zip(paths, names)):
        gid = mark.gid + '.' + name
        path.set('id', gid)
        _set(path, data_role='cell' if mark.data.get('cells') else 'contour-level',
             data_index=i, data_series=mark.series, data_kind='shape')
        if mark.data.get('cells'):
            _set(path, data_row=i // cols, data_column=i % cols)
        members.append(gid)
        id_map[gid] = path
    mark.data['field_members'] = members


def _group_legacy_contour(mark, id_map):
    nodes = [id_map[g] for g in mark.member_gids if g in id_map]
    if not nodes: return
    parent = nodes[0].getparent()
    positions = [parent.index(n) for n in nodes if n.getparent() is parent]
    if len(positions) != len(nodes) or positions != list(range(positions[0], positions[0] + len(nodes))):
        return  # Keep truthful per-level references if callers interleaved artists.
    group = etree.Element(f'{{{SVG}}}g', id=mark.gid)
    parent.insert(positions[0], group)
    for node in nodes: group.append(node)
    id_map[mark.gid] = group
    mark.data['field_members'] = [node.get('id') for node in nodes]
