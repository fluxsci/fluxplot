"""Assemble the ``*.fluxplot.json`` manifest — the semantic index (spec §4–§8).

Authority rule: geometry stays in the SVG (we never copy paths/bboxes here); the manifest owns
data, semantics, the coordinate mapping, and choreography. Every node points *into* the SVG by id.
"""
from __future__ import annotations

from . import ids as _ids
from . import presets as _presets
from . import roles as _roles
from .capture import svg_viewbox
from .descriptors import mark_kind


def _floats(seq):
    from .data import values
    return values(seq)


# Composite sub-part roles → the plural series-svg key listing every member. An errorbar/box/
# violin renders its statistics as numbered sibling groups; collecting EVERY member across the
# series' marks makes each whisker/cap/median addressable (the shipped errorbar pattern,
# generalized per plan §4). The singular key keeps the first gid as the primary ref (compat).
COMPOSITE_ROLES = {
    "errorbar": "errorbars",
    "whisker": "whiskers",
    "cap": "caps",
    "median": "medians",
    "flier": "fliers",
    "mean": "means",
    "segment": "segments",
    # A surface (brain) map draws one collection per category, so every region is separately
    # addressable; ``regions`` lists them all while ``surface-region`` keeps the first as the
    # primary ref (compat, as for the other composites).
    "surface-region": "regions",
}


def build_manifest(
    fig,
    reg,
    guides,
    axes_capture: list,
    plot_type: str,
    svg_filename: str,
    spec_version: str,
    fluxplot_version: str,
    mpl_version: str,
    present: set | None = None,
    svg_sha256: str | None = None,
    rasterized: set | None = None,
) -> dict:
    vbw, vbh = svg_viewbox(fig)
    # gids rendered as a single embedded <image> instead of vector primitives (raster.py).
    # Additive: consumers that ignore it are unaffected; those that read it know why a
    # point cloud has no per-point ids and can label the layer honestly.
    rasterized = rasterized or set()

    # `present` = the gids that actually survived into the SVG (matplotlib culls
    # boundary ticks and renders point clouds as collections where per-point ids
    # can't be assigned). Reference only what's really there so the parts tree /
    # group members stay honest. None = keep everything (direct/legacy callers).
    def _keep(gid) -> bool:
        return present is None or gid in present

    # group series marks by series name (insertion order preserved)
    by_series: dict[str, list] = {}
    overlays = []
    for m in reg.marks:
        if m.series is not None:
            by_series.setdefault(m.series, []).append(m)
        else:
            overlays.append(m)

    series_entries = []
    # per-series {role → data-kind hint} for the parts tree (artist-informed for x- roles)
    series_kinds: dict = {}
    for series, marks in by_series.items():
        svg: dict = {}
        data: dict = {}
        points = None
        kind = None
        label = None
        roles = sorted({m.role for m in marks})
        kinds: dict = {}
        composite_members: dict[str, list] = {}
        components = []
        for m in marks:
            actual = [g for g in m.member_gids or ([m.gid] if m.gid else []) if _keep(g)]
            if m.data.get('field_members') and _keep(m.gid):
                components.append({'role': m.role, 'svgId': m.gid,
                                   'members': [g for g in m.data['field_members'] if _keep(g)]})
            elif m.role == "point" and _keep(m.gid):
                components.append({"role": "point", "svgId": m.gid, "members": actual})
            else:
                components.extend({"role": m.role, "svgId": g} for g in actual)
            if not actual and not _keep(m.gid):
                continue
            kind = kind or m.kind
            label = label or m.label
            if m.role not in kinds:
                mk = mark_kind(m)
                if mk is not None:
                    kinds[m.role] = mk
            if m.x is not None and not data:
                data = {"x": _floats(m.x), "y": _floats(m.y)}
            if m.role == "line":
                if _keep(m.gid):
                    svg.setdefault("line", m.gid)
            elif m.role == "point":
                if _keep(m.gid):
                    svg.setdefault("points", m.gid)
                points = (points or []) + [
                    {
                        "index": m.member_indices[k],
                        "svgId": m.member_gids[k],
                        "x": m.x[m.member_indices[k]],
                        "y": m.y[m.member_indices[k]],
                    }
                    for k in range(len(m.member_gids))
                    if _keep(m.member_gids[k])
                ]
            elif m.role == "bar":
                bars = [g for g in m.member_gids if _keep(g)]
                if bars:
                    svg.setdefault("bars", []).extend(bars)
            elif m.role in COMPOSITE_ROLES:
                if _keep(m.gid) and m.role not in svg:
                    svg[m.role] = m.gid  # primary ref stays (compat); first mark wins
                composite_members.setdefault(m.role, []).extend(
                    g for g in m.member_gids if _keep(g)
                )
            elif _keep(m.gid):
                svg.setdefault(m.role, m.gid)
        for crole, members in composite_members.items():
            if members:
                svg[COMPOSITE_ROLES[crole]] = members
        # Escape-hatch series (fp.tag) carry no helper kind; fall back to the first
        # tagged role so `kind` is always a string (Flux's validator rejects null).
        if kind is None:
            kind = marks[0].role
        if not svg:
            continue
        datasets = [(m.x, m.y) for m in marks if m.x is not None and _keep(m.gid)]
        if (datasets and any(pair != datasets[0] for pair in datasets[1:])) or sum(m.role == 'point' for m in marks) > 1:
            # Several components may share one semantic series while depicting
            # different observations. Keep part identity without one false data table.
            data, points = {}, None
        entry = {
            "id": _ids.series_root(series),
            "name": str(series),
            "kind": kind,
            "roles": roles,
            "svg": svg,
            "data": data,
            "components": components,
        }
        for field in ("bar", "band", "uncertainty", "field"):
            payload = next((m.data[field] for m in marks if m.data.get(field)), None)
            if payload is not None:
                entry[field] = payload
        ordinary = all(m.role in ('line', 'point') for m in marks)
        ordinary = ordinary and len([m for m in marks if m.role == 'line']) <= 1
        ordinary = ordinary and len([m for m in marks if m.role == 'point']) <= 1
        for m in marks:
            for a in m.artists:
                if hasattr(a, 'get_drawstyle') and a.get_drawstyle() != 'default': ordinary = False
                if getattr(getattr(a, 'axes', None), 'name', None) not in (None, 'rectilinear'): ordinary = False
                if hasattr(a, 'get_transform') and m.axes is not None and a.get_transform() != m.axes.transData:
                    # Collections carry data through their offset transform.
                    if not hasattr(a, 'get_offset_transform') or a.get_offset_transform() != m.axes.transData:
                        ordinary = False
        entry['capabilities'] = {'dataMorph': bool(ordinary and data and
            not any(c['svgId'] in rasterized for c in components) and
            (svg.get('line') or points))}
        if label:
            entry["label"] = label
        if points:
            entry["points"] = points
        if any(c["svgId"] in rasterized for c in components):
            entry["rasterized"] = True
        # additive provenance for auto-promoted series: how identity/data were captured
        # (identity=artist-label, data=artist — see autotag.py)
        cap = next((m.data["capture"] for m in marks if m.data.get("capture")), None)
        if cap:
            entry["capture"] = cap
        # additive exact-distribution payload (fp.hist bin edges/counts; opt-in raw values) —
        # deliberately separate from data{x,y}: bar heights are not original observations
        dist = next((m.data["distribution"] for m in marks if m.data.get("distribution")), None)
        if dist:
            entry["distribution"] = dist
        # additive surface payload — the complete value→colour contract of a surface (brain) map.
        # The point of the primitive is that this mapping is DATA, not baked pixels: each part is
        # listed with its own id and style, so recolouring a region, swapping a colormap or moving a
        # threshold is a declarative edit that re-renders deterministically.
        surf_marks = [m for m in marks if m.data.get("surface")]
        if surf_marks:
            summary = next((m.data["surface"] for m in surf_marks if m.role == "surface"), {})
            parts_payload = []
            for m in surf_marks:
                if m.role == "surface":
                    continue
                p = {k: v for k, v in m.data["surface"].items()
                     if k not in ("views", "hemispheres", "missingRule", "missingColor")}
                if _keep(m.gid):
                    p["ref"] = m.gid
                p.setdefault("part", m.name or m.role)
                parts_payload.append(p)
            entry["surface"] = {**summary, "parts": parts_payload}
        series_entries.append(entry)
        series_kinds[entry["id"]] = kinds

    # organize the scaffold guides per axis (+ legend entries + titles + swept text/artists)
    axes_parts, legend_entries, figure_titles, scaffold_annotations, extras = _organize_guides(guides)

    # guides → manifest guides (axis refs + legend with per-entry svg ids)
    guide_entries = []
    legend_present = any(g.role == "legend" for g in guides)
    for g in guides:
        if g.role == "axis":
            guide_entries.append({"id": g.gid, **({} if g.virtual else {"svgId": g.gid}), "role": "axis", "axis": g.axis})
    if legend_present:
        # entry ↔ series joined by exact, UNIQUE label text — positional order is not
        # identity (plan §7). An entry whose text matches no series label (or an ambiguous
        # duplicated label) keeps its swatch/label as addressable guides, with no series claim.
        by_label: dict[str, list] = {}
        for s in series_entries:
            if s.get("label"):
                by_label.setdefault(s["label"], []).append(s["id"])
        # A surface map's legend keys its PARTS, not its series (all its regions live under one
        # series), so an entry also joins on a unique part name — giving it a ref to the very
        # element it describes, which is what makes "recolour the block this swatch names" possible.
        by_part: dict[str, list] = {}
        for s in series_entries:
            for prt in (s.get("surface") or {}).get("parts", []):
                if prt.get("part") and prt.get("ref"):
                    by_part.setdefault(prt["part"], []).append((s["id"], prt["ref"]))
        entries = []
        for k in sorted(legend_entries):
            ent = legend_entries[k]
            e = {}
            matches = by_label.get(ent.get("text"), [])
            if len(matches) == 1:
                e["series"] = matches[0]
            part_matches = by_part.get(ent.get("text"), [])
            if len(part_matches) == 1:
                e["series"], e["part"] = part_matches[0]
            if ent.get("text"):
                e["text"] = ent["text"]
            if ent.get("swatch"):
                e["swatch"] = ent["swatch"]
            if ent.get("label"):
                e["label"] = ent["label"]
            entries.append(e)
        guide_entries.append({"id": "legend", "svgId": "legend", "role": "legend", "entries": entries})

    for g in guides:
        if g.role == 'colorbar':
            payload = dict(g.data)
            payload['parts'] = [part for part in payload.get('parts', []) if _keep(part['svgId'])]
            if not _keep(payload.get('mappable')): payload.pop('mappable', None)
            guide_entries.append({'id': g.gid, 'svgId': g.gid, 'role': g.role, **payload})

    overlay_entries = []
    for m in overlays:
        if not _keep(m.gid):
            continue
        oe = {"id": m.gid, "svgId": m.gid, "role": m.role}
        mk = mark_kind(m)
        if mk is not None:
            oe["kind"] = mk
        if m.name is not None:
            oe["name"] = m.name
        for key in ("label", "between", "p", "text"):  # carry the annotation text too
            if key in m.data:
                oe[key] = m.data[key]
        if m.gid in rasterized:
            oe["rasterized"] = True
        overlay_entries.append(oe)
    # swept free text → annotation overlays (addressable + animatable like fp.annotation)
    for a in scaffold_annotations:
        oe = {"id": a["id"], "svgId": a["id"], "role": "annotation", "kind": "text"}
        if a.get("text"):
            oe["text"] = a["text"]
        overlay_entries.append(oe)

    # swept untagged artists (raw ax.plot lines / collections / patches) → "extra" overlays.
    # Kept separate from overlay_entries so they group under a single "extras" node in the parts
    # tree (rather than each becoming a loose ref) while still appearing in the manifest overlays.
    extra_entries = []
    for e in extras:
        ee = {"id": e["id"], "svgId": e["id"], "role": "extra"}
        if e.get("kind"):
            ee["kind"] = e["kind"]
        if e["id"] in rasterized:
            ee["rasterized"] = True
        extra_entries.append(ee)

    parts = _build_parts_tree(
        series_entries, axes_parts, legend_entries, figure_titles, overlay_entries,
        legend_present, extra_entries, series_kinds,
    )
    for guide in guide_entries:
        if guide['role'] == 'colorbar':
            children = []
            grouped = {'colorbar-tick': ('ticks', 'tick'),
                       'colorbar-tick-label': ('tick-labels', 'tick-label'),
                       'colorbar-gridline': ('gridlines', 'gridline')}
            for role, (suffix, group_role) in grouped.items():
                members = [p['svgId'] for p in guide.get('parts', []) if p['role'] == role]
                if members:
                    children.append(_group(guide['id'] + '.' + suffix, group_role, members))
            children.extend(_ref(p['svgId'], p.get('kind')) for p in guide.get('parts', [])
                            if p['role'] not in grouped)
            parts['children'].append({'id': guide['id'], 'role': 'colorbar', 'kind': 'container',
                                      'children': children})
    build = _build_order(series_entries, guide_entries, overlay_entries, reg, figure_titles, extra_entries)

    out = {
        "spec": "fluxplot/manifest",
        "schemaVersion": spec_version,
        "generator": {
            "name": "fluxplot",
            "version": fluxplot_version,
            "matplotlib": mpl_version,
        },
        "plotType": plot_type,
        "svg": svg_filename,
        "size": {"width": vbw, "height": vbh, "unit": "pt"},
        "axes": axes_capture,
        "series": series_entries,
        "guides": guide_entries,
        "overlays": overlay_entries + extra_entries,
        "parts": parts,
        "build": build,
    }
    if svg_sha256 is not None:
        # checksum of the FINAL postprocessed SVG bytes: deterministic (the SVG is
        # byte-stable) and acyclic (the SVG does not contain the manifest). Consumers use it
        # to detect a stale/mismatched sidecar pair instead of silently degrading (plan §5).
        out["artifact"] = {"svgSha256": svg_sha256}
    return out


def _organize_guides(guides):
    """Bucket the flat GuideTag list into per-axis parts + legend entries + the figure title."""
    axes: dict = {}
    legend_entries: dict = {}
    figure_titles: list = []  # left/center/right + suptitle can coexist (no last-wins)
    annotations: list = []  # swept free text → addressable annotation overlays
    extras: list = []  # swept untagged artists → addressable "extra" content
    for g in guides:
        if g.role == "axis":
            axes.setdefault(g.axis, {})["gid"] = g.gid
        elif g.role == "axis-title":
            axes.setdefault(g.axis, {})["title"] = g.gid
        elif g.role == "tick-label":
            axes.setdefault(g.axis, {}).setdefault("ticklabels", []).append(g.gid)
        elif g.role == "tick":
            axes.setdefault(g.axis, {}).setdefault("ticks", []).append(g.gid)
        elif g.role == "gridline":
            axes.setdefault(g.axis, {}).setdefault("gridlines", []).append(g.gid)
        elif g.role in ("spine", "background"):
            axes.setdefault(g.axis, {}).setdefault("spines", []).append(g.gid)
        elif g.role == "legend-swatch":
            legend_entries.setdefault(g.index, {})["swatch"] = g.gid
        elif g.role == "legend-label":
            legend_entries.setdefault(g.index, {})["label"] = g.gid
            legend_entries[g.index]["text"] = g.text
        elif g.role == "title":
            figure_titles.append(g.gid)
        elif g.role == "annotation":
            annotations.append({"id": g.gid, "text": g.text})
        elif g.role == "extra":
            extras.append({"id": g.gid, "kind": g.kind})
    return axes, legend_entries, figure_titles, annotations, extras


def _group(gid: str, group_role: str, members: list) -> dict:
    """A manifest-only node grouping sibling leaves so a consumer can act on all at once.

    The node mirrors its members' data-kind (a group of tick labels edits like text, a
    group of gridlines like a line) so consumers can pick property sets without the DOM.
    """
    node = {"id": gid, "role": "group", "groupRole": group_role, "members": list(members)}
    k = _roles.kind_for_role(group_role)
    if k is not None:
        node["kind"] = k
    return node


def _ref(gid: str, kind=None) -> dict:
    """A leaf reference into the SVG, carrying the data-kind hint when known."""
    node = {"ref": gid}
    if kind is not None:
        node["kind"] = kind
    return node


def _build_parts_tree(
    series_entries, axes_parts, legend_entries, figure_titles, overlay_entries, legend_present,
    extra_entries=(), series_kinds=None,
) -> dict:
    series_kinds = series_kinds or {}
    plot_children = []

    # axes → real <g id="axis.x"> nodes, each with spine + grouped ticks/labels/gridlines + title
    for which in ("x", "y", "z"):
        ap = axes_parts.get(which)
        if not ap:
            continue
        kids = []
        for sp in ap.get("spines", []):
            kids.append(_ref(sp, _roles.kind_for_role("spine")))
        if ap.get("ticks"):
            kids.append(_group(f"axis.{which}.ticks", "tick", ap["ticks"]))
        if ap.get("ticklabels"):
            kids.append(_group(f"axis.{which}.tick-labels", "tick-label", ap["ticklabels"]))
        if ap.get("gridlines"):
            kids.append(_group(f"axis.{which}.gridlines", "gridline", ap["gridlines"]))
        if ap.get("title"):
            kids.append(_ref(ap["title"], _roles.kind_for_role("axis-title")))
        plot_children.append(
            {
                "id": ap.get("gid", f"axis.{which}"),
                "role": "axis",
                "axis": which,
                "kind": "container",
                "children": kids,
            }
        )

    # series → line + grouped points/bars + ANY other tagged role. The generic
    # tail covers area/errorbar/box AND custom plot kinds (x-violin, x-heatmap-cell,
    # x-stem, x-trajectory, x-contour, …) so every drawn series part is addressable
    # and no series becomes a childless phantom node.
    for s in series_entries:
        kinds = series_kinds.get(s["id"], {})
        kids = []
        by_role = {}
        for component in s["components"]:
            by_role.setdefault(component["role"], []).append(component)
        for role, components in by_role.items():
            kind = kinds.get(role, _roles.kind_for_role(role))
            if role == "point" or any(c.get("members") for c in components):
                for c in components:
                    kids.append(_group(c["svgId"], role, c["members"]) if c.get("members")
                                else _ref(c["svgId"], kind))
            elif role == "bar" or role in COMPOSITE_ROLES or len(components) > 1:
                plural = "bars" if role == "bar" else COMPOSITE_ROLES.get(role, role + "-parts")
                kids.append(_group(s["id"] + "." + plural, role, [c["svgId"] for c in components]))
            else:
                kids.append(_ref(components[0]["svgId"], kind))
        plot_children.append({"id": s["id"], "role": "series", "kind": "container", "children": kids})

    for o in overlay_entries:
        plot_children.append(_ref(o["svgId"], o.get("kind")))

    # untagged user-drawn artists → one "extras" group so a consumer can act on all at once
    if extra_entries:
        plot_children.append(_group("extras", "extra", [e["svgId"] for e in extra_entries]))

    figure_children = [
        {"id": "plot-area", "role": "plot-area", "kind": "container", "children": plot_children}
    ]
    if legend_present:
        leg_kids = []
        for k in sorted(legend_entries):
            ent = legend_entries[k]
            ek = []
            if ent.get("swatch"):
                ek.append(_ref(ent["swatch"], _roles.kind_for_role("legend-swatch")))
            if ent.get("label"):
                ek.append(_ref(ent["label"], _roles.kind_for_role("legend-label")))
            leg_kids.append(
                {"id": f"legend.entry.{k}", "role": "legend-entry", "kind": "container", "children": ek}
            )
        figure_children.append(
            {"id": "legend", "role": "legend", "kind": "container", "children": leg_kids}
        )
    for t in figure_titles:
        figure_children.append(_ref(t, _roles.kind_for_role("title")))
    return {"id": "figure", "role": "figure", "kind": "container", "children": figure_children}


def _build_order(series_entries, guide_entries, overlay_entries, reg, figure_titles=(), extra_entries=()) -> dict:
    order = []
    for g in guide_entries:
        if g["role"] == "axis":
            order.append(g.get("svgId", g["id"]))
    order.extend(g['svgId'] for g in guide_entries if g['role'] == 'colorbar')
    order.extend(figure_titles)  # titles reveal with the axes (phase 0)
    order.append("gridlines")
    # Use the same component inventory as the tree. Unknown/extension roles are
    # ordinary drawable parts, not exceptions silently excluded from animation.
    for s in series_entries:
        order.extend(c["svgId"] for c in s["components"] if c["role"] == "line")
    for s in series_entries:
        order.extend(c["svgId"] for c in s["components"] if c["role"] != "line")
    if any(g["role"] == "legend" for g in guide_entries):
        order.append("legend")
    for o in overlay_entries:
        order.append(o["svgId"])
    for e in extra_entries:
        order.append(e["svgId"])

    roles_present = {m.role for m in reg.marks} | {"axis", "gridline"}
    if extra_entries:
        roles_present.add("extra")
    return {"order": list(dict.fromkeys(order)), "presets": _presets.presets_for(sorted(roles_present))}
