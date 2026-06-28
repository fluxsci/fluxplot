"""Assemble the ``*.fluxplot.json`` manifest — the semantic index (spec §4–§8).

Authority rule: geometry stays in the SVG (we never copy paths/bboxes here); the manifest owns
data, semantics, the coordinate mapping, and choreography. Every node points *into* the SVG by id.
"""
from __future__ import annotations

from . import ids as _ids
from . import presets as _presets
from .capture import svg_viewbox


def _floats(seq):
    return [float(v) for v in seq] if seq is not None else None


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
) -> dict:
    vbw, vbh = svg_viewbox(fig)

    # group series marks by series name (insertion order preserved)
    by_series: dict[str, list] = {}
    overlays = []
    for m in reg.marks:
        if m.series is not None:
            by_series.setdefault(m.series, []).append(m)
        else:
            overlays.append(m)

    series_entries = []
    for series, marks in by_series.items():
        svg: dict = {}
        data: dict = {}
        points = None
        kind = None
        label = None
        roles = sorted({m.role for m in marks})
        for m in marks:
            kind = kind or m.kind
            label = label or m.label
            if m.x is not None and not data:
                data = {"x": _floats(m.x), "y": _floats(m.y)}
            if m.role == "line":
                svg["line"] = m.gid
            elif m.role == "point":
                svg["points"] = m.gid
                points = [
                    {
                        "index": k,
                        "svgId": m.member_gids[k],
                        "x": float(m.x[k]),
                        "y": float(m.y[k]),
                    }
                    for k in range(len(m.member_gids))
                ]
            elif m.role == "bar":
                svg["bars"] = list(m.member_gids)
            else:
                svg[m.role] = m.gid
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
        if points is not None:
            entry["points"] = points
        series_entries.append(entry)

    # guides → manifest guides (+ legend entries from series labels)
    guide_entries = []
    legend_present = any(g.role == "legend" for g in guides)
    for g in guides:
        if g.role == "axis":
            guide_entries.append(
                {"id": g.gid, "svgId": g.gid, "role": "axis", "axis": g.axis}
            )
    if legend_present:
        guide_entries.append(
            {
                "id": "legend",
                "svgId": "legend",
                "role": "legend",
                "entries": [
                    {"series": s["id"]} for s in series_entries if s.get("label")
                ],
            }
        )

    overlay_entries = []
    for m in overlays:
        oe = {"id": m.gid, "svgId": m.gid, "role": m.role}
        if m.name is not None:
            oe["name"] = m.name
        for key in ("label", "between", "p"):
            if key in m.data:
                oe[key] = m.data[key]
        overlay_entries.append(oe)

    parts = _build_parts_tree(series_entries, guide_entries, overlay_entries, legend_present)
    build = _build_order(series_entries, guide_entries, overlay_entries, reg)

    return {
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
        "overlays": overlay_entries,
        "parts": parts,
        "build": build,
    }


def _build_parts_tree(series_entries, guide_entries, overlay_entries, legend_present) -> dict:
    plot_children = []
    for g in guide_entries:
        if g["role"] == "axis":
            plot_children.append({"ref": g["svgId"]})
    for s in series_entries:
        kids = []
        for key in ("line", "points", "area", "errorbar", "box"):
            if key in s["svg"]:
                kids.append({"ref": s["svg"][key]})
        if "bars" in s["svg"]:
            kids.extend({"ref": b} for b in s["svg"]["bars"])
        plot_children.append({"id": s["id"], "role": "series", "children": kids})
    for o in overlay_entries:
        plot_children.append({"ref": o["svgId"]})

    figure_children = [{"id": "plot-area", "role": "plot-area", "children": plot_children}]
    if legend_present:
        figure_children.append({"ref": "legend"})
    return {"id": "figure", "role": "figure", "children": figure_children}


def _build_order(series_entries, guide_entries, overlay_entries, reg) -> dict:
    order = []
    for g in guide_entries:
        if g["role"] == "axis":
            order.append(g["svgId"])
    order.append("gridlines")
    for s in series_entries:
        if "line" in s["svg"]:
            order.append(s["svg"]["line"])
    for s in series_entries:
        if "points" in s["svg"]:
            order.append(s["svg"]["points"])
        if "bars" in s["svg"]:
            order.extend(s["svg"]["bars"])
        for key in ("area", "errorbar", "box"):
            if key in s["svg"]:
                order.append(s["svg"][key])
    if any(g["role"] == "legend" for g in guide_entries):
        order.append("legend")
    for o in overlay_entries:
        order.append(o["svgId"])

    roles_present = {m.role for m in reg.marks} | {"axis", "gridline"}
    return {"order": order, "presets": _presets.presets_for(sorted(roles_present))}
