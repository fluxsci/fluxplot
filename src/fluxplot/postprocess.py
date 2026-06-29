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

from .descriptors import Mark

SVG = "http://www.w3.org/2000/svg"


def _fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return "%.10g" % v
    return str(v)


def _set(el, **attrs) -> None:
    for k, v in attrs.items():
        if v is None:
            continue
        el.set(k.replace("_", "-"), _fmt(v))


def postprocess(svg_bytes: bytes, reg, guides, plot_type: str):
    """Return ``(processed_svg_bytes, warnings)``."""
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(svg_bytes, parser)
    warnings: list[str] = []

    # 1. strip volatile metadata
    etree.strip_tags(root, etree.Comment)
    for md in root.findall(f"{{{SVG}}}metadata"):
        md.getparent().remove(md)

    id_map = {el.get("id"): el for el in root.iter() if el.get("id")}

    # 2. rename matplotlib's structural wrappers → semantic roots
    _rename(id_map, "figure_1", "figure", "figure")
    _rename(id_map, "axes_1", "plot-area", "plot-area")

    # 3. root markers
    root.set("data-fluxplot", "1")
    root.set("data-plot-type", plot_type)

    # 4. inject data-* per Mark
    for m in reg.marks:
        if m.role == "point":
            _inject_points(m, id_map, warnings)
        elif m.role == "bar":
            _inject_indexed(m, id_map, role="bar")
        elif m.series is not None:
            el = id_map.get(m.gid)
            if el is not None:
                _set(el, data_role=m.role, data_series=m.series)
        else:
            _inject_overlay(m, id_map)

    # 5. inject data-role on scaffold/guides
    for g in guides:
        el = id_map.get(g.gid)
        if el is not None:
            _set(el, data_role=g.role, data_axis=g.axis, data_index=g.index, data_series=g.series)

    # The set of ids that actually survived into the SVG (computed AFTER injection
    # so per-point <use> ids are included). matplotlib culls boundary ticks/
    # gridlines at draw and omits empty axis titles even though the artists carry a
    # gid — the manifest must reference only what's really here, so callers prune
    # guides/members against this set (else the X-Ray shows dead nodes).
    present = {el.get("id") for el in root.iter() if el.get("id")}

    return _serialize(root), warnings, present


def _rename(id_map, old, new, role) -> None:
    el = id_map.pop(old, None)
    if el is not None:
        el.set("id", new)
        el.set("data-role", role)
        id_map[new] = el


def _inject_points(m: Mark, id_map, warnings) -> None:
    group = id_map.get(m.gid)
    if group is None:
        return
    _set(group, data_series=m.series)
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
            data_index=k,
            data_x=xs[k],
            data_y=ys[k],
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
        )


def _inject_overlay(m: Mark, id_map) -> None:
    el = id_map.get(m.gid)
    if el is not None:
        _set(el, data_role=m.role, data_name=m.name)
    label_gid = m.data.get("label_gid")
    if label_gid is not None:
        lab = id_map.get(label_gid)
        if lab is not None:
            _set(lab, data_role="label", data_name=m.name)


def _serialize(root) -> bytes:
    body = etree.tostring(root, pretty_print=True, encoding="unicode")
    return ('<?xml version="1.0" encoding="utf-8" standalone="no"?>\n' + body).encode("utf-8")
