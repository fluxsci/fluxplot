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
    return [float(v) for v in seq] if seq is not None else None


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
) -> dict:
    vbw, vbh = svg_viewbox(fig)

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
        for m in marks:
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
                    svg["line"] = m.gid
            elif m.role == "point":
                if _keep(m.gid):
                    svg["points"] = m.gid
                points = [
                    {
                        "index": k,
                        "svgId": m.member_gids[k],
                        "x": float(m.x[k]),
                        "y": float(m.y[k]),
                    }
                    for k in range(len(m.member_gids))
                    if _keep(m.member_gids[k])
                ]
            elif m.role == "bar":
                bars = [g for g in m.member_gids if _keep(g)]
                if bars:
                    svg["bars"] = bars
            elif m.role in COMPOSITE_ROLES:
                if _keep(m.gid) and m.role not in svg:
                    svg[m.role] = m.gid  # primary ref stays (compat); first mark wins
                composite_members.setdefault(m.role, []).extend(
                    g for g in m.member_gids if _keep(g)
                )
            elif _keep(m.gid):
                svg[m.role] = m.gid
        for crole, members in composite_members.items():
            if members:
                svg[COMPOSITE_ROLES[crole]] = members
        # Escape-hatch series (fp.tag) carry no helper kind; fall back to the first
        # tagged role so `kind` is always a string (Flux's validator rejects null).
        if kind is None:
            kind = marks[0].role
        entry = {
            "id": _ids.series_root(series),
            "name": str(series),
            "kind": kind,
            "roles": roles,
            "svg": svg,
            "data": data,
        }
        if label:
            entry["label"] = label
        if points:
            entry["points"] = points
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
        series_entries.append(entry)
        series_kinds[entry["id"]] = kinds

    # organize the scaffold guides per axis (+ legend entries + titles + swept text/artists)
    axes_parts, legend_entries, figure_titles, scaffold_annotations, extras = _organize_guides(guides)

    # guides → manifest guides (axis refs + legend with per-entry svg ids)
    guide_entries = []
    legend_present = any(g.role == "legend" for g in guides)
    for g in guides:
        if g.role == "axis":
            guide_entries.append({"id": g.gid, "svgId": g.gid, "role": "axis", "axis": g.axis})
    if legend_present:
        labeled = [s for s in series_entries if s.get("label")]
        entries = []
        for k, s in enumerate(labeled):
            e = {"series": s["id"]}
            ent = legend_entries.get(k, {})
            if ent.get("swatch"):
                e["swatch"] = ent["swatch"]
            if ent.get("label"):
                e["label"] = ent["label"]
            entries.append(e)
        guide_entries.append({"id": "legend", "svgId": "legend", "role": "legend", "entries": entries})

    overlay_entries = []
    for m in overlays:
        oe = {"id": m.gid, "svgId": m.gid, "role": m.role}
        mk = mark_kind(m)
        if mk is not None:
            oe["kind"] = mk
        if m.name is not None:
            oe["name"] = m.name
        for key in ("label", "between", "p", "text"):  # carry the annotation text too
            if key in m.data:
                oe[key] = m.data[key]
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
        extra_entries.append(ee)

    parts = _build_parts_tree(
        series_entries, axes_parts, legend_entries, figure_titles, overlay_entries,
        legend_present, extra_entries, series_kinds,
    )
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
        elif g.role == "spine":
            axes.setdefault(g.axis, {}).setdefault("spines", []).append(g.gid)
        elif g.role == "legend-swatch":
            legend_entries.setdefault(g.index, {})["swatch"] = g.gid
        elif g.role == "legend-label":
            legend_entries.setdefault(g.index, {})["label"] = g.gid
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
    for which in ("x", "y"):
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
        svg = s["svg"]
        kinds = series_kinds.get(s["id"], {})
        kids = []
        if "line" in svg:
            kids.append(_ref(svg["line"], kinds.get("line", _roles.kind_for_role("line"))))
        if "points" in svg:
            members = [p["svgId"] for p in s.get("points", [])]
            kids.append(
                _group(svg["points"], "point", members)
                if members
                else _ref(svg["points"], _roles.kind_for_role("point"))
            )
        if svg.get("bars"):
            kids.append(_group(f'{s["id"]}.bars', "bar", svg["bars"]))
        # one group node per composite role (errorbars/whiskers/caps/medians/fliers/means/
        # segments): every statistic addressable individually AND as a group
        for crole, plural in COMPOSITE_ROLES.items():
            if svg.get(plural):
                kids.append(_group(f'{s["id"]}.{plural}', crole, svg[plural]))
        plurals = set(COMPOSITE_ROLES.values())
        for role, val in svg.items():
            if role in ("line", "points", "bars") or role in plurals:
                continue
            if role in COMPOSITE_ROLES and svg.get(COMPOSITE_ROLES[role]):
                continue  # already covered by its composite group (avoid a duplicate ref)
            k = kinds.get(role, _roles.kind_for_role(role))
            if isinstance(val, list):
                grp = _group(f'{s["id"]}.{role}', role, val)
                if k is not None and "kind" not in grp:
                    grp["kind"] = k
                kids.append(grp)
            else:
                kids.append(_ref(val, k))
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
            order.append(g["svgId"])
    order.extend(figure_titles)  # titles reveal with the axes (phase 0)
    order.append("gridlines")
    for s in series_entries:
        if "line" in s["svg"]:
            order.append(s["svg"]["line"])
    for s in series_entries:
        if "points" in s["svg"]:
            order.append(s["svg"]["points"])
        if "bars" in s["svg"]:
            order.extend(s["svg"]["bars"])
        # bodies first, then composite statistics — every sub-part reveals, like bars
        for key in ("area", "box", "violin", "errorbar", "whisker", "cap", "median", "flier", "mean", "segment"):
            plural = COMPOSITE_ROLES.get(key)
            if plural and s["svg"].get(plural):
                order.extend(s["svg"][plural])
            elif key in s["svg"]:
                order.append(s["svg"][key])
    if any(g["role"] == "legend" for g in guide_entries):
        order.append("legend")
    for o in overlay_entries:
        order.append(o["svgId"])
    for e in extra_entries:
        order.append(e["svgId"])

    roles_present = {m.role for m in reg.marks} | {"axis", "gridline"}
    if extra_entries:
        roles_present.add("extra")
    return {"order": order, "presets": _presets.presets_for(sorted(roles_present))}
