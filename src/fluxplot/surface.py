"""Surface (brain) maps as first-class, part-addressable FluxPlot marks.

Why this module exists
----------------------
FluxPlot's contract is that a plot's *science* — which value became which colour — stays **data,
not pixels**. Every 2D chart honours that: named marks in the SVG, a ``.fluxplot.json`` manifest,
a ``.recipe.json`` for provenance, all editable in the Flux interface.

**Surface maps** (a per-vertex scalar or label field painted on a triangulated mesh — the standard
way neuroimaging shows a cortical result, but equally any scalar field on any mesh) had no such
primitive. They were produced by external VTK-backed tools as flat raster PNGs, so the one thing a
reviewer most wants to correct — *this region should be that colour; this threshold is wrong* — was
the one thing baked irreversibly into pixels.

``surface()`` closes that gap. It draws the mesh with matplotlib primitives only (numpy +
matplotlib; ``nibabel`` is optional and used only to read GIFTI files), so:

* **Categorical / label maps** are split into **one collection per category** — each becomes a
  named, selectable, recolourable part (``atlas.frontal``, ``atlas.parietal``, …). Recolouring a
  region, or hiding it, is a style edit on a named part, exactly like restyling a bar series. This
  is the full part-addressability tier, not a raster with a legend.
* **Continuous maps** carry their complete value→colour mapping (colormap, vmin/vmax, the
  percentile rule that produced them) in the manifest, with the colorbar as its own named part, so
  range/threshold edits are declarative and re-render deterministically.
* The **medial wall / missing data** is its own named part, never conflated with a real value.

Rendering is vector (one ``PolyCollection`` per part). A typical cortical mesh is tens of thousands
of triangles per hemisphere, which is exactly the node count :mod:`fluxplot.raster` exists to tame:
the collections are handed to the standard rasterisation planner, which composites each *part* to a
single ``<image>`` **while preserving its gid**. So the file stays light and every part stays
addressable — the mesh is drawn honestly as geometry, and the id contract survives.

Correctness rules made first-class (each is a documented failure mode of naive surface plots):

* vertices with no data (a mesh's medial wall / non-surface region) are **missing** (NaN) → their
  own grey part, never a data value;
* a value of **exactly 0 is real** and must stay distinguishable from missing — there is no
  "zero is transparent" behaviour here;
* a **sentinel code** (e.g. −1 to blank one hemisphere) greys out via ``missing_below`` /
  ``missing_values`` rather than becoming a spurious extra category;
* categorical colour is a **fixed id→colour map** with no silent remapping;
* back-facing geometry is culled, so a far-side face can never paint over the visible surface.

Example
-------
>>> import fluxplot as fp, matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(figsize=(6.4, 1.6))
>>> fp.surface(ax, labels, series="atlas",                     # per-vertex integer labels
...            surfaces={"left": "lh.surf.gii", "right": "rh.surf.gii"},
...            kind="label",
...            categories={0: "frontal", 1: "parietal", 2: "temporal"},
...            palette={"frontal": "#4C78A8", "parietal": "#F58518",
...                     "temporal": "#54A24B"})
>>> fp.save(fig, "plots/atlas.svg")

Meshes may also be passed as ``(vertices, faces)`` arrays, so the core has no I/O dependency;
``nibabel`` is imported only when a GIFTI path is given.
"""
from __future__ import annotations

import warnings

import numpy as np

# Views are named by what the viewer sees. For each (hemisphere, view) we look down ±x and keep
# (y, z); the sign flip keeps anterior consistently to the same side of the page across views.
_VIEW_SPEC = {
    ("left", "lateral"): (-1, +1),
    ("left", "medial"): (+1, -1),
    ("right", "lateral"): (+1, -1),
    ("right", "medial"): (-1, +1),
}


def _load_surface(spec):
    """``spec`` → ``(vertices[V,3], faces[F,3])``.

    Accepts an already-loaded ``(vertices, faces)`` pair (keeps fluxplot dependency-free) or a path
    to a GIFTI surface, which needs ``nibabel``.
    """
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        v, f = spec
        return np.asarray(v, dtype=float), np.asarray(f, dtype=int)
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "reading a GIFTI surface needs nibabel (pip install nibabel); alternatively pass "
            "surfaces={'left': (vertices, faces), ...} directly so fluxplot stays dependency-free"
        ) from exc
    g = nib.load(str(spec))
    verts = g.agg_data("pointset")
    faces = g.agg_data("triangle")
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)


def _project(verts, hemi, view):
    """Orthographic projection of ``verts`` for one (hemisphere, view), plus a depth per vertex.

    Returns ``(xy[V,2], depth[V], sign_x)`` where larger depth is nearer the viewer and ``sign_x``
    is the axis sign pointing at the camera (used for back-face culling).
    """
    try:
        sign_x, sign_y = _VIEW_SPEC[(hemi, view)]
    except KeyError:
        raise ValueError(
            f"unknown (hemisphere, view) = ({hemi!r}, {view!r}); "
            f"expected one of {sorted(_VIEW_SPEC)}") from None
    xy = np.column_stack([sign_y * verts[:, 1], verts[:, 2]])
    depth = sign_x * verts[:, 0]
    return xy, depth, sign_x


def _front_facing(verts, faces, sign_x):
    """Boolean mask of faces whose outward normal points at the camera.

    Back-face culling — not merely an optimisation, it is what makes per-region collections
    *correct*. Splitting a map into one collection per category means matplotlib draws the
    categories in sequence, so a painter's-algorithm depth sort inside each collection cannot stop a
    far-side face of one category from painting over a near-side face of another (the far wall of
    the opposite bank would bleed through, and the medial wall would be hidden by whatever is drawn
    after it). Culling the hemisphere's far half first leaves a single visible layer, after which
    drawing order between categories is irrelevant.
    """
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return sign_x * normals[:, 0] > 0


def _face_values(values, faces):
    """Per-face value = mean of its vertices; a face touching missing data is itself missing.

    Propagating NaN (rather than averaging around it) keeps the medial-wall boundary crisp instead
    of smearing a halo of interpolated colour across it.
    """
    # A plain mean already propagates NaN, which IS the rule we want: a face touching missing data
    # is itself missing, so the medial-wall boundary stays crisp instead of smearing a halo of
    # interpolated colour across it.
    return values[faces].mean(axis=1)


def _face_labels(values, faces):
    """Per-face label = the label of its vertices when they agree, else missing.

    A face straddling a block boundary has no single honest identity, so it is left unassigned
    rather than silently attributed to one block (which would fatten that block along every border).
    """
    tri = values[faces]
    same = (tri[:, 0] == tri[:, 1]) & (tri[:, 1] == tri[:, 2])
    out = np.where(same, tri[:, 0], np.nan)
    out[np.isnan(tri).any(axis=1)] = np.nan
    return out


def _normalise_missing(values, missing_below=None, missing_values=()):
    """Return a float copy with every 'missing' convention collapsed to NaN.

    ``missing_below`` catches sentinel encodings (the common one is −1 for the off-hemisphere half of
    a single-hemisphere map); ``missing_values`` catches explicit codes. Zero is never treated as
    missing — an on-cortex 0 is a real measurement.
    """
    v = np.asarray(values, dtype=float).copy()
    if missing_below is not None:
        v[v < missing_below] = np.nan
    for mv in missing_values:
        v[v == mv] = np.nan
    return v


def _resolve_range(finite, color_range, percentile):
    if color_range is not None:
        return float(color_range[0]), float(color_range[1])
    if percentile is not None:
        lo = float(np.percentile(finite, percentile[0]))
        hi = float(np.percentile(finite, percentile[1]))
        return lo, hi
    return float(np.min(finite)), float(np.max(finite))


def surface(ax, values, *, series, surfaces, kind="auto", categories=None, palette=None,
            cmap=None, color_range=None, percentile=None, views=("lateral", "medial"),
            hemispheres=("left", "right"), missing_below=None, missing_values=(),
            missing_color="#D8D8D8", gap=0.06, edgecolor="none", linewidth=0.0,
            colorbar=False, cbar_label=None, label=None):
    """Draw a per-vertex surface map as named, addressable parts.

    Parameters
    ----------
    ax
        Target axes. Every requested (hemisphere, view) is tiled left→right inside it; the axes is
        set to equal aspect with its frame hidden, because a cortical projection has no meaningful
        data axes.
    values
        Per-vertex array over the concatenated hemispheres (left then right), or per hemisphere via
        a ``{"left": ..., "right": ...}`` mapping.
    series
        Addressable name for this map — parts are ``<series>.<category>`` (label maps) or
        ``<series>.field`` (continuous), plus ``<series>.missing``.
    surfaces
        ``{"left": spec, "right": spec}`` where each spec is a GIFTI path or a ``(vertices, faces)``
        pair.
    kind
        ``"label"``, ``"continuous"``, or ``"auto"`` (label when the data are integral and few-valued).
    categories
        ``{code: name}`` for label maps. Codes absent from the data are ignored; data codes absent
        here get a ``category-<code>`` name so nothing is silently dropped.
    palette
        ``{name_or_code: colour}`` for label maps — a fixed mapping, never remapped.
    cmap, color_range, percentile
        Continuous styling. ``percentile=(2, 98)`` clips to those percentiles of the finite data;
        ``color_range`` wins if both are given. The resolved range is recorded in the manifest.
    missing_below, missing_values
        Sentinel handling (e.g. ``missing_below=0`` turns a −1 off-hemisphere sentinel into missing).
    colorbar
        Add a colorbar for continuous maps as its own addressable part.

    Returns the list of matplotlib collections drawn (in draw order).
    """
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize, to_hex

    from . import tagger as _tagger
    from .descriptors import Mark

    hemispheres = [h for h in hemispheres if h in surfaces]
    if not hemispheres:
        raise ValueError("no requested hemisphere present in `surfaces`")

    # ---- per-hemisphere geometry + values -------------------------------------------------
    geom, vals = {}, {}
    offset = 0
    for h in hemispheres:
        v, f = _load_surface(surfaces[h])
        geom[h] = (v, f)
        if isinstance(values, dict):
            hv = np.asarray(values[h], dtype=float)
        else:
            flat = np.asarray(values, dtype=float)
            hv = flat[offset:offset + v.shape[0]]
            offset += v.shape[0]
        if hv.shape[0] != v.shape[0]:
            raise ValueError(
                f"{h}: {hv.shape[0]} values for {v.shape[0]} vertices — a surface map must carry "
                f"exactly one value per vertex")
        vals[h] = _normalise_missing(hv, missing_below, missing_values)

    all_vals = np.concatenate([vals[h] for h in hemispheres])
    finite = all_vals[np.isfinite(all_vals)]
    if finite.size == 0:
        raise ValueError("every vertex is missing — nothing to draw")

    if kind == "auto":
        uniq = np.unique(finite)
        kind = "label" if (uniq.size <= 32 and np.allclose(uniq, np.round(uniq))) else "continuous"
    if kind not in ("label", "continuous"):
        raise ValueError(f"kind must be 'label', 'continuous' or 'auto' (got {kind!r})")

    # ---- lay the views out side by side in axes coordinates --------------------------------
    panes = [(h, view) for h in hemispheres for view in views]
    spans = []
    for h, view in panes:
        xy, _, _ = _project(geom[h][0], h, view)
        spans.append((np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
    unit_w = max(s[0] for s in spans)
    unit_h = max(s[1] for s in spans)
    pitch = unit_w * (1.0 + gap)

    # ---- assemble faces, grouped by the part they belong to --------------------------------
    # parts: name -> list of (polygon vertex arrays); continuous also collects per-face values.
    parts, part_vals = {}, {}
    for i, (h, view) in enumerate(panes):
        verts, faces = geom[h]
        xy, depth, sign_x = _project(verts, h, view)
        xy = xy.copy()
        xy[:, 0] += i * pitch - xy[:, 0].mean()
        xy[:, 1] -= xy[:, 1].mean()

        vis = faces[_front_facing(verts, faces, sign_x)]        # only the half facing the camera
        fv = _face_labels(vals[h], vis) if kind == "label" else _face_values(vals[h], vis)
        order = np.argsort(depth[vis].mean(axis=1))             # painter's algorithm: far → near
        polys = xy[vis[order]]
        fv = fv[order]

        missing = ~np.isfinite(fv)
        parts.setdefault("missing", []).append(polys[missing])
        if kind == "label":
            for code in np.unique(fv[~missing]):
                name = (categories or {}).get(int(code), f"category-{int(code)}")
                parts.setdefault(name, []).append(polys[fv == code])
        else:
            parts.setdefault("field", []).append(polys[~missing])
            part_vals.setdefault("field", []).append(fv[~missing])

    reg = _tagger.registry_for(ax.figure)
    drawn = []

    def _add(name, polys_list, **kw):
        polys = np.concatenate([p for p in polys_list if len(p)]) if any(
            len(p) for p in polys_list) else np.empty((0, 3, 2))
        if polys.shape[0] == 0:
            return None
        coll = PolyCollection(list(polys), edgecolor=edgecolor, linewidth=linewidth,
                              antialiased=False, **kw)
        ax.add_collection(coll)
        drawn.append(coll)
        return coll

    # ---- missing data: its own part, never a data value ------------------------------------
    coll = _add("missing", parts.pop("missing", []), facecolor=missing_color)
    if coll is not None:
        reg.add(Mark(role="surface-missing", series=series, name="missing", kind="surface",
                     artists=[coll],
                     data={"surface": {"part": "missing", "color": missing_color,
                                       "meaning": "medial wall / non-cortex / sentinel — not a value"}}))

    style_common = {
        "views": list(views), "hemispheres": list(hemispheres),
        "missingColor": missing_color,
        "missingRule": {"below": missing_below, "values": [float(m) for m in missing_values],
                        "zeroIsData": True},
    }

    if kind == "label":
        # One collection per category → each block is independently selectable and recolourable.
        resolved = {}
        for name in sorted(parts):
            colour = None
            if palette:
                colour = palette.get(name)
                if colour is None and categories:
                    for code, nm in categories.items():
                        if nm == name and code in palette:
                            colour = palette[code]
                            break
            if colour is None:
                colour = f"C{len(resolved) % 10}"
            colour = to_hex(colour)
            resolved[name] = colour
            coll = _add(name, parts[name], facecolor=colour)
            if coll is None:
                continue
            reg.add(Mark(role="surface-region", series=series, name=name, kind="surface",
                         label=label,
                         artists=[coll],
                         data={"surface": dict(style_common, kind="label", part=name,
                                               color=colour)}))
        summary = {"kind": "label", "palette": resolved, **style_common}
    else:
        import matplotlib as mpl
        vmin, vmax = _resolve_range(finite, color_range, percentile)
        norm = Normalize(vmin=vmin, vmax=vmax)
        if cmap is None:
            cmap_obj = mpl.colormaps[mpl.rcParams["image.cmap"]]
        elif isinstance(cmap, str):
            cmap_obj = mpl.colormaps[cmap]
        else:
            cmap_obj = cmap
        fv = np.concatenate(part_vals["field"])
        coll = _add("field", parts["field"], facecolor=cmap_obj(norm(fv)))
        summary = {"kind": "continuous", "cmap": getattr(cmap_obj, "name", str(cmap)),
                   "vmin": vmin, "vmax": vmax,
                   "percentile": list(percentile) if percentile else None, **style_common}
        if coll is not None:
            coll.set_array(fv)            # keeps the mapping live for a colorbar
            coll.set_cmap(cmap_obj)
            coll.set_norm(norm)
            reg.add(Mark(role="surface-field", series=series, name="field", kind="surface",
                         label=label, artists=[coll],
                         data={"surface": dict(summary, part="field")}))
        if colorbar and coll is not None:
            cb = ax.figure.colorbar(coll, ax=ax, fraction=0.03, pad=0.02)
            if cbar_label:
                cb.set_label(cbar_label)
            reg.add(Mark(role="surface-colorbar", series=series, name="colorbar", kind="surface",
                         artists=[cb.solids] if cb.solids is not None else [],
                         data={"surface": {"part": "colorbar", "vmin": vmin, "vmax": vmax,
                                           "cmap": summary["cmap"], "label": cbar_label,
                                           "editable": ["vmin", "vmax", "cmap"]}}))

    # One summary mark carrying the whole value→colour contract, so the mapping round-trips even if
    # a downstream editor only reads the series-level entry.
    reg.add(Mark(role="surface", series=series, kind="surface", label=label,
                 data={"surface": dict(summary, nVertices=int(all_vals.size),
                                       nMissing=int((~np.isfinite(all_vals)).sum()),
                                       panes=[f"{h}-{v}" for h, v in panes])}))

    ax.set_xlim(-unit_w * 0.55, (len(panes) - 1) * pitch + unit_w * 0.55)  # panes centred on i*pitch
    ax.set_ylim(-unit_h * 0.55, unit_h * 0.55)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if len(drawn) == 0:
        warnings.warn("surface(): nothing was drawn", stacklevel=2)
    return drawn
