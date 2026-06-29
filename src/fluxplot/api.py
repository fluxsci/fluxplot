"""Public API — free functions over real matplotlib artists (spec §11.1).

Free functions (not a wrapped Axes) keep the user in full matplotlib (P2): the convenience helpers
and the ``tag()`` escape hatch are the same shape, and untagged matplotlib keeps working. ``save()``
orchestrates the pipeline: assign gids → draw → auto-tag scaffold → capture coords → render
deterministically → inject ``data-*`` → emit manifest + recipe.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import matplotlib

from . import canonical_json as _cjson
from . import capture as _capture
from . import ids as _ids
from . import manifest as _manifest
from . import postprocess as _postprocess
from . import recipe as _recipe
from . import render as _render
from . import roles as _roles
from . import tagger as _tagger
from .descriptors import Mark
from .version import SPEC_VERSION, __version__


def _list(a):
    return None if a is None else list(a)


# ---------------------------------------------------------------------------
# convenience helpers (auto-tagging) — each returns the real matplotlib artist(s)
# ---------------------------------------------------------------------------
def line(ax, x, y, *, series, marker=None, label=None, **kw):
    """A named line. With ``marker=`` it also draws an addressable per-point group."""
    reg = _tagger.registry_for(ax.figure)
    (ln,) = ax.plot(x, y, label=label, **kw)
    reg.add(Mark(role="line", series=series, kind="line", x=_list(x), y=_list(y), label=label, artists=[ln]))
    if marker:
        (pts,) = ax.plot(x, y, linestyle="none", marker=marker, color=ln.get_color())
        reg.add(Mark(role="point", series=series, kind="line", x=_list(x), y=_list(y), artists=[pts], indexed=True))
        return ln, pts
    return ln


def scatter(ax, x, y, *, series, label=None, **kw):
    reg = _tagger.registry_for(ax.figure)
    coll = ax.scatter(x, y, label=label, **kw)
    reg.add(Mark(role="point", series=series, kind="scatter", x=_list(x), y=_list(y), label=label, artists=[coll], indexed=True))
    return coll


def bar(ax, x, height, *, series, label=None, **kw):
    reg = _tagger.registry_for(ax.figure)
    container = ax.bar(x, height, label=label, **kw)
    reg.add(Mark(role="bar", series=series, kind="bar", x=_list(x), y=_list(height), label=label, artists=list(container.patches), indexed=True))
    return container


def errorbar(ax, x, y, *, series, yerr=None, label=None, **kw):
    reg = _tagger.registry_for(ax.figure)
    container = ax.errorbar(x, y, yerr=yerr, label=label, **kw)
    _data_line, _caps, barlinecols = container
    reg.add(Mark(role="errorbar", series=series, kind="errorbar", x=_list(x), y=_list(y), label=label, artists=list(barlinecols), data={"yerr": _list(yerr)}))
    return container


def area(ax, x, y1, y2=0, *, series, label=None, **kw):
    reg = _tagger.registry_for(ax.figure)
    poly = ax.fill_between(x, y1, y2, label=label, **kw)
    reg.add(Mark(role="area", series=series, kind="area", x=_list(x), y=_list(y1), label=label, artists=[poly]))
    return poly


# ---------------------------------------------------------------------------
# escape hatch — tag arbitrary raw matplotlib artists
# ---------------------------------------------------------------------------
def tag(artist, *, role, series=None, index=None, name=None, **identity):
    """Tag any raw matplotlib artist with a semantic role + identity."""
    reg = _tagger.registry_for(_tagger.fig_of(artist))
    data = dict(identity)
    if index is not None:
        data["index"] = index
    reg.add(Mark(role=_roles.validate(role), series=series, name=name, artists=[artist], data=data, indexed=index is not None))
    return artist


def tag_points(points, *, series, x=None, y=None):
    """Tag an existing markers ``Line2D`` / ``PathCollection`` as an addressable point group."""
    reg = _tagger.registry_for(_tagger.fig_of(points))
    if (x is None or y is None) and hasattr(points, "get_data"):
        gx, gy = points.get_data()
        x = _list(gx) if x is None else x
        y = _list(gy) if y is None else y
    reg.add(Mark(role="point", series=series, kind="scatter", x=_list(x), y=_list(y), artists=[points], indexed=True))
    return points


# ---------------------------------------------------------------------------
# first-class overlays
# ---------------------------------------------------------------------------
def significance_bracket(ax, *, x0, x1, y, label, between=None, p=None, name=None, height=None, **kw):
    """A p-value bracket (spec §5: deliberately first-class — ubiquitous in science)."""
    reg = _tagger.registry_for(ax.figure)
    idx = reg.next_overlay_index("significance-bracket")
    if name is None:
        name = str(idx)
    if height is None:
        if ax.get_yscale() == "log":
            ytop = y * 1.06
        else:
            lo, hi = ax.get_ylim()
            ytop = y + (hi - lo) * 0.03
    else:
        ytop = y + height
    (br,) = ax.plot([x0, x0, x1, x1], [y, ytop, ytop, y], color=kw.pop("color", "black"), linewidth=kw.pop("linewidth", 1.0), **kw)
    txt = ax.text((x0 + x1) / 2.0, ytop, label, ha="center", va="bottom")
    data = {"label": label, "index": idx, "label_artist": txt}
    if between is not None:
        data["between"] = list(between)
    if p is not None:
        data["p"] = p
    reg.add(Mark(role="significance-bracket", series=None, name=name, artists=[br], data=data))
    return br


def reference_line(ax, *, y=None, x=None, name, **kw):
    reg = _tagger.registry_for(ax.figure)
    if y is not None:
        ln = ax.axhline(y, **kw)
    elif x is not None:
        ln = ax.axvline(x, **kw)
    else:
        raise ValueError("reference_line needs x= or y=")
    reg.add(Mark(role="reference-line", series=None, name=name, artists=[ln], data={"x": x, "y": y}))
    return ln


def annotation(ax, *, name, text, **kw):
    reg = _tagger.registry_for(ax.figure)
    art = ax.annotate(text, **kw)
    reg.add(Mark(role="annotation", series=None, name=name, artists=[art], data={"text": text}))
    return art


# ---------------------------------------------------------------------------
# the export
# ---------------------------------------------------------------------------
@dataclass
class SaveResult:
    svg: str
    manifest: str
    recipe: str
    warnings: list = field(default_factory=list)


def _infer_plot_type(reg) -> str:
    kinds = [m.kind for m in reg.marks if m.kind]
    for k in ("line", "scatter", "bar", "errorbar", "area"):
        if k in kinds:
            return k
    return "plot"


def _validate(manifest_obj, recipe_obj) -> None:
    try:
        import jsonschema
        from importlib.resources import files

        sch = files("fluxplot.schemas")
        mschema = json.loads((sch / "manifest.schema.json").read_text())
        rschema = json.loads((sch / "recipe.schema.json").read_text())
    except (FileNotFoundError, ModuleNotFoundError):
        return
    jsonschema.validate(manifest_obj, mschema)
    jsonschema.validate(recipe_obj, rschema)


def save(fig, path, *, recipe=None, addressable_points=None, style_classes=False, validate=True, _now=None) -> SaveResult:
    """Emit ``<path>.svg`` + ``<path>.fluxplot.json`` + ``<path>.recipe.json`` for ``fig``."""
    base, _ext = os.path.splitext(path)
    plot_name = os.path.basename(base)
    svg_path = base + ".svg"
    manifest_path = base + ".fluxplot.json"
    recipe_path = base + ".recipe.json"
    svg_filename = os.path.basename(svg_path)
    manifest_filename = os.path.basename(manifest_path)

    reg = _tagger.registry_for(fig)
    alloc = _ids.IdAllocator()

    # 1. deterministic gids on the user-tagged marks
    _tagger.resolve_gids(reg, alloc)

    # 2. finalize layout (so ticks/labels exist + transforms are final), then auto-tag scaffold
    fig.canvas.draw()
    guides = []
    for ax in fig.axes:
        guides.extend(_tagger.autotag_scaffold(ax, alloc))

    # 3. capture the data↔SVG mapping (after layout is final)
    axes_capture = []
    for ax in fig.axes:
        cap = {"id": "plot-area", "svgId": "plot-area"}
        cap.update(_capture.capture_axes(ax, fig))
        axes_capture.append(cap)

    # 4. render deterministically (hashsalt derived from the plot name)
    svg_bytes = _render.render_svg(fig, hashsalt=plot_name or "fluxplot")

    # 5. inject data-* + canonicalize
    plot_type = _infer_plot_type(reg)
    out_svg, warnings, present = _postprocess.postprocess(svg_bytes, reg, guides, plot_type)

    # 6. assemble manifest + recipe. Drop scaffold guides matplotlib culled at draw
    # (boundary ticks/gridlines, empty axis titles) so the manifest references only
    # parts that exist in the SVG — keeps the parts tree / group members honest.
    kept_guides = [g for g in guides if g.gid in present]
    man = _manifest.build_manifest(
        fig, reg, kept_guides, axes_capture, plot_type, svg_filename,
        SPEC_VERSION, __version__, matplotlib.__version__, present=present,
    )
    rec = _recipe.build_recipe(
        recipe, plot_name=plot_name, svg_filename=svg_filename,
        manifest_filename=manifest_filename, spec_version=SPEC_VERSION, now=_now,
    )
    if validate:
        _validate(man, rec)

    # 7. write all three
    out_dir = os.path.dirname(os.path.abspath(svg_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(svg_path, "wb") as f:
        f.write(out_svg)
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(_cjson.dumps(man))
    with open(recipe_path, "w", encoding="utf-8") as f:
        f.write(_cjson.dumps(rec))

    return SaveResult(svg=svg_path, manifest=manifest_path, recipe=recipe_path, warnings=warnings)
