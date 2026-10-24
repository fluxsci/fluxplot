"""Elastic-layout inputs captured with matplotlib's SVG font engine at 72 dpi.

These are constraints and measurements, never a second authoritative drawing.
The SVG remains the source of geometry. No Python executes when Flux resizes it.
"""
from __future__ import annotations

import copy
import math
import re

import matplotlib as mpl
from matplotlib import ticker
from matplotlib.collections import PathCollection
from matplotlib.legend import Legend
from matplotlib.text import Annotation, Text
import numpy as np


def _numbers(value):
    if value is np.ma.masked:return None
    if isinstance(value,np.ndarray) and value.ndim==0:return _numbers(value.item())
    if isinstance(value, dict):
        return {str(k): _numbers(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_numbers(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(format(float(value), ".15g")) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def locator_record(loc):
    if isinstance(loc, ticker.MaxNLocator):
        return dict(kind="maxn", nbins=loc._nbins, steps=loc._steps, integer=loc._integer,
                    symmetric=loc._symmetric, prune=loc._prune, minNTicks=loc._min_n_ticks)
    if isinstance(loc, ticker.MultipleLocator):
        return dict(kind="multiple", base=loc._edge.step, offset=getattr(loc, "_offset", 0))
    if isinstance(loc, ticker.LogLocator):
        return dict(kind="log", base=loc._base, subs=loc._subs, numticks=loc.numticks)
    if isinstance(loc, ticker.AutoMinorLocator):
        return dict(kind="auto-minor", n=loc.ndivs)
    if isinstance(loc, ticker.NullLocator):
        return dict(kind="null")
    module = type(loc).__module__
    if module == "matplotlib.dates":
        return dict(kind="date")
    if module == "matplotlib.category":
        return dict(kind="category")
    return dict(kind="fixed" if isinstance(loc, ticker.FixedLocator) else "custom")


def formatter_record(fmt):
    if isinstance(fmt, ticker.ScalarFormatter):
        return dict(kind="scalar", useOffset=fmt.get_useOffset(), offset=fmt.offset if not fmt.get_useOffset() else None,
                    offsetThreshold=fmt._offset_threshold, useMathText=fmt.get_useMathText(),
                    powerlimits=fmt._powerlimits, scientific=fmt._scientific,
                    useLocale=fmt.get_useLocale(), unicodeMinus=mpl.rcParams["axes.unicode_minus"])
    if isinstance(fmt, ticker.LogFormatterSciNotation):
        return dict(kind="log-sci", base=fmt._base, labelOnlyBase=fmt.labelOnlyBase,
                    minorThresholds=["inf" if v==float("inf") else "-inf" if v==float("-inf") else v for v in fmt.minor_thresholds], minExponent=mpl.rcParams["axes.formatter.min_exponent"])
    if isinstance(fmt, ticker.PercentFormatter):
        return dict(kind="percent", xmax=fmt.xmax, decimals=fmt.decimals, symbol=fmt.symbol)
    if isinstance(fmt, ticker.StrMethodFormatter) and re.fullmatch(r"\{x:(?:,?\.\d+[fe]|g)\}", fmt.fmt):
        return dict(kind="str-method", fmt=fmt.fmt)
    return dict(kind="fixed" if isinstance(fmt, (ticker.FixedFormatter, ticker.NullFormatter)) else "custom")


def capture_layout(fig, engine, panels, reg, alloc, initial_positions=None, initial_domains=None):
    from matplotlib.backends.backend_svg import RendererSVG
    from io import StringIO
    from .panels import all_axes

    w, h = fig.get_size_inches() * 72
    unsupported = ["subfigures"] if fig.subfigs else []
    if any(getattr(ax, "name", None) == "3d" for ax in fig.axes):
        unsupported.append("3d")
    if unsupported:
        return _numbers(dict(version=1, supported=False, reasons=unsupported,
            figure=dict(widthPt=w, heightPt=h, suptitle=None, supxlabel=None, supylabel=None),
            engine=dict(kind="none", subplotpars={k: getattr(fig.subplotpars, k)
                for k in ("left", "right", "bottom", "top", "wspace", "hspace")}),
            grids=[], axes=[], colorbars=[], legends=[], texts={}, frames={}, fonts={}))
    renderer = RendererSVG(w, h, StringIO())
    reasons, fonts, font_keys, texts, frames = [], {}, {}, {}, {}
    axes, grids, grid_ids, colorbars, legends = [], [], {}, [], []
    by_axes = {id(a): p.svg_id + ("." + key if a is not p.axes else "")
               for p in panels for a, key in [(p.axes, ""), *p.twins]}
    actual_axes = all_axes(fig)
    by_axes.update({id(a): a.get_gid() for a in actual_axes if getattr(a, "_colorbar", None) is not None})
    axis_artists, legend_children = {}, set()

    def refuse(code, gid=None):
        reason = code + (":" + gid if gid else "")
        if reason not in reasons:
            reasons.append(reason)

    def box(bb):
        return [bb.x0, h-bb.y1, bb.x1, h-bb.y0]

    def point(xy):
        return [xy[0], h-xy[1]]

    def font(prop):
        key = (tuple(prop.get_family()), prop.get_size_in_points(), prop.get_style(), str(prop.get_weight()))
        if key not in font_keys:
            fid = f"f{len(fonts)}"
            font_keys[key] = fid
            advances = {ch: renderer.get_text_width_height_descent(ch, prop, False)[0]
                        for ch in "0123456789.,−-+eE×% ()"}
            _, height, descent = renderer.get_text_width_height_descent("lp", prop, False)
            scale=prop.get_size_in_points()/100
            large=prop.copy();large.set_size(100)
            power=renderer._text2path.mathtext_parser.parse(r'$\mathdefault{10^{0}}$',72,large)
            coeff=renderer._text2path.mathtext_parser.parse(r'$\mathdefault{2\times10^{0}}$',72,large)
            base_glyph,sup_glyph=power.glyphs[0],power.glyphs[-1]
            from matplotlib.ft2font import LoadFlags
            parser=renderer._text2path.mathtext_parser
            fontset=parser._font_type_mapping[large.get_math_fontfamily()](large,LoadFlags.NO_HINTING)
            raw_width=parser._parser.parse(r'$\mathdefault{10^{0}}$',fontset,100,72).width
            double=parser.parse(r'$\mathdefault{10^{00}}$',72,large)
            negative=parser.parse(r'$\mathdefault{10^{-0}}$',72,large)
            fonts[fid] = dict(family=", ".join(prop.get_family()), sizePt=prop.get_size_in_points(),weight=prop.get_weight(),style=prop.get_style(),
                              ascentPt=height-descent, descentPt=descent, advances=advances,
                              math=dict(powerRawWidthPt=raw_width*scale,
                                supAdvancePt=(double.glyphs[-1][-2]-sup_glyph[-2])*scale,
                                minusAdvancePt=(negative.glyphs[-1][-2]-sup_glyph[-2])*scale,
                                quantumPt=scale,supScale=sup_glyph[1]/base_glyph[1],
                                supRisePt=(sup_glyph[-1]-base_glyph[-1])*scale,
                                baseAdvance10=(sup_glyph[-2]-base_glyph[-2])*scale,
                                powerWidthPt=power.width*scale,powerHeightPt=power.height*scale,powerDescentPt=power.depth*scale,
                                coeffPrefixPt=(coeff.glyphs[-1][-2]-sup_glyph[-2])*scale,
                                timesPadPt=(coeff.glyphs[1][-2]-coeff.glyphs[0][-2])*scale-advances['2']))
        return font_keys[key]

    def text_record(t):
        gid = t.get_gid()
        if not gid or not t.get_visible() or not t.get_text():
            return
        if t.get_usetex():
            refuse("text:usetex", gid)
        texts[gid] = dict(bbox=box(Text.get_window_extent(t,renderer) if isinstance(t,Annotation) else t.get_window_extent(renderer)),
                          anchor=point(t.get_transform().transform(t.get_position())),
                          ha=t.get_ha(), va=t.get_va(), rotation=t.get_rotation(),
                          font=font(t.get_fontproperties()), text=t.get_text())

    def slot(ss):
        gs = ss.get_gridspec()
        if id(gs) not in grid_ids:
            parent = slot(gs._subplot_spec) if hasattr(gs, "_subplot_spec") else None
            gid = f"g{len(grids)}"
            grid_ids[id(gs)] = gid
            grids.append(dict(id=gid, nrows=gs.nrows, ncols=gs.ncols,
                              widthRatios=gs.get_width_ratios(), heightRatios=gs.get_height_ratios(),
                              wspace=getattr(gs, "wspace", getattr(gs, "_wspace", None)),
                              hspace=getattr(gs, "hspace", getattr(gs, "_hspace", None)), parent=parent))
        return dict(grid=grid_ids[id(gs)], rows=[ss.rowspan.start, ss.rowspan.stop-1],
                    cols=[ss.colspan.start, ss.colspan.stop-1])

    def axis_record(axis, key, owner):
        tick = axis.majorTicks[0]
        far = axis.get_ticks_position() in ("top", "right")
        label = tick.label2 if far else tick.label1
        side = ("top" if far else "bottom") if key.startswith("x") else ("right" if far else "left")
        axis_artists[id(axis.label)] = dict(f="axis", axes=owner, axis=key, part="title")
        axis_artists[id(axis.offsetText)] = dict(f="axis", axes=owner, axis=key, part="offset")
        for minor, ticks in ((False, axis.get_major_ticks()), (True, axis.get_minor_ticks())):
            for tk in ticks:
                for art, part in ((tk.label1, "ticklabel"), (tk.label2, "ticklabel"),
                                  (tk.tick1line, "tick"), (tk.tick2line, "tick"), (tk.gridline, "gridline")):
                    axis_artists[id(art)] = dict(f="axis", axes=owner, axis=key, part=part,
                                                 value=tk.get_loc(), minor=minor, side=("top" if key=="x" else "right") if art in (tk.label2,tk.tick2line) else ("bottom" if key=="x" else "left"))
        major, minor = locator_record(axis.get_major_locator()), locator_record(axis.get_minor_locator())
        fmt = formatter_record(axis.get_major_formatter())
        style = dict(side=side, labelSide=axis.get_label_position(), tickDir=tick._tickdir,
                     tickLenPt=tick._size, tickPadPt=tick._base_pad,
                     minorTickLenPt=axis.get_minor_ticks()[0]._size if axis.get_minor_ticks() else 2,
                     labelPadPt=axis.labelpad, labelAuto=axis._autolabelpos,
                     font=font(label.get_fontproperties()), labelRotation=label.get_rotation(), labelHa=label.get_ha(),
                     offsetText=axis.offsetText.get_gid(), offsetSpec=dict(id=axis.offsetText.get_gid(),
                         anchor=point(axis.offsetText.get_transform().transform(axis.offsetText.get_position())),
                         font=font(axis.offsetText.get_fontproperties()),ha=axis.offsetText.get_ha(),va=axis.offsetText.get_va(),rotation=axis.offsetText.get_rotation()),label=axis.label.get_gid(),
                     visible=axis.get_visible(), tickTop=tick.tick2line.get_visible(), labelTop=tick.label2.get_visible(),
                     tickBottom=tick.tick1line.get_visible(), labelBottom=tick.label1.get_visible())
        ticking = dict(major=major, minor=minor, formatter=fmt,
                       retick=major["kind"] in ("maxn", "multiple", "log") and fmt["kind"] != "custom",
                       domain=list(axis.get_view_interval()), scale=axis.get_scale(),
                       values=[float(t.get_loc()) for t in axis.get_major_ticks() if np.isfinite(t.get_loc())],
                       labels=[(t.label2 if far else t.label1).get_text() for t in axis.get_major_ticks() if np.isfinite(t.get_loc())],
                       offsetText=axis.get_major_formatter().get_offset(),minorFormatter=formatter_record(axis.get_minor_formatter()),
                       minorValues=[float(t.get_loc()) for t in axis.get_minor_ticks() if np.isfinite(t.get_loc())],
                       minorLabels=[(t.label2 if far else t.label1).get_text() for t in axis.get_minor_ticks() if np.isfinite(t.get_loc())],
                       minorRetick=minor['kind'] in ('auto-minor','log') and (isinstance(axis.get_minor_formatter(),ticker.NullFormatter) or formatter_record(axis.get_minor_formatter())['kind'] not in ('custom','fixed')))
        return style, ticking

    if fig.subfigs:
        refuse("subfigures")
    kind = type(engine).__name__
    opts = engine.get() if engine is not None else {}
    if kind == "ConstrainedLayoutEngine":
        if getattr(engine, "_compress", False):
            refuse("layout:compressed")
        engine_rec = dict(kind="constrained", wPad=opts["w_pad"]*72, hPad=opts["h_pad"]*72,
                          wSpace=opts["wspace"], hSpace=opts["hspace"], rect=opts.get("rect", [0, 0, 1, 1]))
    elif kind == "TightLayoutEngine":
        engine_rec = dict(kind="tight", pad=opts["pad"], hPad=opts["h_pad"], wPad=opts["w_pad"],
                          rect=opts.get("rect") or [0, 0, 1, 1], fontPt=mpl.rcParams["font.size"])
    else:
        if engine is not None and kind != "PlaceHolderLayoutEngine":
            refuse("layout:custom")
        engine_rec = dict(kind="none", subplotpars={k: getattr(fig.subplotpars, k)
                          for k in ("left", "right", "bottom", "top", "wspace", "hspace")})

    for ax in actual_axes:
        aid = by_axes.get(id(ax))
        if not aid:
            continue
        if ax.name != "rectilinear":
            refuse("3d" if ax.name == "3d" else "projection:" + ax.name)
        ss = ax.get_subplotspec()
        locator = ax.get_axes_locator()
        locator = getattr(locator,"_orig_locator",locator)
        parent = next((a for a in actual_axes if ax in a.child_axes), None)
        inset = None
        if locator is not None:
            if type(locator).__name__ == "_TransformedBoundsLocator" and parent is not None and locator._transform == parent.transAxes:
                inset = dict(parent=by_axes.get(id(parent)), bounds=locator._bounds, frame="axes")
            elif not getattr(ax, "_functions", None):
                refuse("axes-locator", aid)
        original = ax.get_position(original=True)
        actual = ax.get_window_extent(renderer)
        if ax.patch.get_gid():frames[ax.patch.get_gid()]=dict(f="axes",axes=aid,frac=[0,0],geometry=True)
        tight = ax.get_tightbbox(renderer, bbox_extra_artists=[], for_layout_only=True) or actual
        styles, ticking = {}, {}
        for key, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            styles[key], ticking[key] = axis_record(axis, key, aid)
        for side, spine in ax.spines.items():
            if side not in ("top", "bottom", "left", "right") or not spine.get_visible() or not spine.get_gid():
                continue
            axis_artists[id(spine)] = dict(f="axis", axes=aid, axis="x" if side in ("top", "bottom") else "y", part="spine", side=side,
                                           position=spine._position if isinstance(spine._position, tuple) else ("axes", .5) if spine._position == "center" else ("data", 0) if spine._position == "zero" else ("outward", 0))
        titles = {name: getattr(ax, attr).get_gid() for name, attr in
                  (("left", "_left_title"), ("center", "title"), ("right", "_right_title"))}
        titles["padPt"] = ax.titleOffsetTrans.get_matrix()[1, 2]
        for name, attr in (("left", "_left_title"), ("center", "title"), ("right", "_right_title")):
            axis_artists[id(getattr(ax, attr))] = dict(f="title", axes=aid, slot=name)
        rec = dict(id=aid, slot=slot(ss) if ss is not None else None,
                   initialPosition=(initial_positions or {}).get(id(ax), original.bounds),
                   initialDomain=(initial_domains or {}).get(id(ax),dict(x=list(ax.get_xlim()),y=list(ax.get_ylim()))) if type(ax).__name__!="SecondaryAxis" else dict(x=list(ax.get_xlim()),y=list(ax.get_ylim())),
                   dataLim=dict(x=list(ax.dataLim.intervalx),y=list(ax.dataLim.intervaly)),
                   position=original.bounds, positionMode="inset" if inset else "grid" if ss is not None and ax.get_in_layout() else "manual",
                   inset=inset, inLayout=ax.get_in_layout(), aspect=ax.get_aspect(),dataRatio=ax.get_data_ratio(),
                   adjustable=ax.get_adjustable(), anchor=ax.get_anchor(), boxAspect=ax.get_box_aspect(),
                   twins=[], sharex=None, sharey=None, title=titles, axisStyle=styles, ticking=ticking,
                   margins=[actual.x0-tight.x0, actual.y0-tight.y0, tight.x1-actual.x1, tight.y1-actual.y1],
                   pixelBox=box(actual))
        rec['extraLayoutIds'] = [a.get_gid() for a in ax.get_default_bbox_extra_artists() if a.get_gid() and a.get_in_layout()]
        for which, grouper in (("sharex", ax.get_shared_x_axes()), ("sharey", ax.get_shared_y_axes())):
            rec[which] = next((by_axes[id(a)] for a in actual_axes if a is not ax and id(a) in by_axes and grouper.joined(ax, a)), None)
        cb = getattr(ax, "_colorbar", None)
        if cb is not None:
            info = getattr(ax, "_colorbar_info", {})
            parents = info.get("parents", [getattr(ax, "_fluxplot_owner_axes", getattr(cb.mappable, "axes", None))])
            rec.update(dict(dataAspect=rec["aspect"],managed=bool(info),parents=[by_axes[id(a)] for a in parents if id(a) in by_axes],
                            location=info.get("location", "right" if cb.orientation == "vertical" else "bottom"),
                            orientation=cb.orientation, fraction=info.get("fraction", .15), shrink=info.get("shrink", 1),
                            aspect=info.get("aspect", 20), padFrac=info.get("pad", .05),
                            panchor=info.get("panchor", [1, .5]), anchor=info.get("anchor", [0, .5]), extendFrac=cb.extendfrac))
            colorbars.append(rec)
        else:
            axes.append(rec)

    for p in panels:
        primary = next((a for a in axes if a["id"] == p.svg_id), None)
        if primary:
            primary["twins"] = [dict(id=by_axes[id(a)], axis=key) for a, key in p.twins]

    for leg in [*(a.get_legend() for a in actual_axes), *fig.legends]:
        if leg is None or not leg.get_gid():
            continue
        gid = leg.get_gid()
        legend_children.update(id(a) for a in leg.findobj() if a is not leg)
        bb = leg.get_window_extent(renderer)
        requested = next((name for name, code in Legend.codes.items() if code == leg._loc), str(leg._loc))
        resolved = requested
        if leg._loc == 0:
            candidates = [(np.linalg.norm(np.array(leg._get_anchored_bbox(i, bb, leg.get_bbox_to_anchor(), renderer)) - [bb.x0, bb.y0]), i)
                          for i in range(1, 11)]
            best = min(candidates)[1]
            resolved = next(name for name, code in Legend.codes.items() if code == best)
        owner = by_axes.get(id(leg.axes), "figure")
        anchor = None
        if leg._bbox_to_anchor is not None:
            tr = leg.axes.transAxes if leg.axes is not None else fig.transFigure
            anchor = dict(bbox=leg.get_bbox_to_anchor().transformed(tr.inverted()).bounds,
                          frame="axes" if leg.axes is not None else "figure")
        collision=None
        if requested=='best':
            collision_boxes,lines,offsets=leg._auto_legend_data(renderer)
            collision=dict(boxes=[box(b) for b in collision_boxes],lines=[[point(p) for p in line.vertices] for line in lines],points=[point(p) for p in np.ma.filled(np.ma.asarray(offsets),np.nan) if np.isfinite(p).all()])
        legends.append(dict(collision=collision,id=gid, owner=owner, locRequested=requested, locResolved=resolved,
                            bboxToAnchor=anchor, borderAxesPadPt=leg.borderaxespad*leg._fontsize,
                            inLayout=leg.get_in_layout(), sizePt=[bb.width, bb.height], bbox=box(bb)))
        frames[gid] = dict(f="legend")

    for art in fig.findobj():
        gid = art.get_gid()
        if not gid:
            continue
        if isinstance(art, Text):
            text_record(art)
        if id(art) in legend_children or gid in frames or id(art) in by_axes:
            continue
        if id(art) in axis_artists:
            frames[gid] = axis_artists[id(art)]
            continue
        ax = getattr(art, "axes", None)
        aid = by_axes.get(id(ax))
        if art is fig.patch:
            frames[gid] = dict(f="figure", frac=[0, 0], geometry=True)
        elif isinstance(art, Annotation):
            def endpoint(coords, pt):
                if coords in ("data", "axes fraction", "figure fraction"):
                    return dict(f={"data": "data", "axes fraction": "axes", "figure fraction": "figure"}[coords], axes=aid, pt=pt, nativePt=point(ax.transData.transform(pt)) if coords == "data" else None)
                refuse("transform:annotation", gid)
                return dict(f="figure", pt=[0, 0])
            base = endpoint(art.xycoords, art.xy)
            offset = art.xyann if art.anncoords == "offset points" else [0, 0]
            textbase = base if art.anncoords == "offset points" else endpoint(art.anncoords, art.xyann)
            frames[gid] = dict(f="offset", axes=aid, base=textbase, offsetPt=offset, textcoords=str(art.anncoords))
            if art.arrow_patch is not None and art.arrow_patch.get_gid():
                arrow = art.arrow_patch
                frames[arrow.get_gid()] = dict(f="arrow", text=gid, xy=base,
                      shrinkA=arrow.shrinkA, shrinkB=arrow.shrinkB,
                      style=type(arrow.get_arrowstyle()).__name__, connection=type(arrow.get_connectionstyle()).__name__)
        elif aid:
            transform = art.get_transform()
            if type(art).__name__ == "NotchedBox" and getattr(art, "axes_ref", None) is ax:
                frames[gid] = dict(f="glyph", axes=aid, box=dict(x=art.x, y0=art.y0, y1=art.y1,
                    notch=art.notch, widthPt=art.width, depthPt=art.depth, heightPt=art.height))
            elif isinstance(art, PathCollection):
                transform = art.get_offset_transform()
                frames[gid] = dict(f="glyph", axes=aid, anchor="use-xy", anchorsPt=[point(p) for p in np.ma.filled(transform.transform(art.get_offsets()),np.nan) if np.isfinite(p).all()])
            elif transform == ax.transAxes:
                frac = art.get_position() if isinstance(art, Text) else [0, 0]
                frames[gid] = dict(f="axes", axes=aid, frac=frac, geometry=not isinstance(art, Text))
            elif transform == ax.get_xaxis_transform():
                frames[gid] = dict(f="blend", axes=aid, x="data", y="axes")
            elif transform == ax.get_yaxis_transform():
                frames[gid] = dict(f="blend", axes=aid, x="axes", y="data")
            elif transform.contains_branch(ax.transData):
                frames[gid] = dict(f="image" if art.__class__.__name__ == "AxesImage" else "data", axes=aid)
            elif transform.contains_branch(ax.transAxes):
                frames[gid] = dict(f="axes", axes=aid, frac=[0, 0], geometry=True)
            elif transform.output_dims == 2 and any(transform.contains_branch_seperately(ax.transData)):
                xdata, ydata = transform.contains_branch_seperately(ax.transData)
                frames[gid] = dict(f="blend", axes=aid, x="data" if xdata else "axes", y="data" if ydata else "axes")
            elif gid not in (ax.xaxis.get_gid(), ax.yaxis.get_gid()):
                refuse("transform:custom", gid)
        elif isinstance(art, Text):
            frames[gid] = dict(f="figure", frac=art.get_position())

    return _numbers(dict(version=1, supported=not reasons, reasons=reasons,
                         figure=dict(widthPt=w, heightPt=h, suptitle=getattr(fig._suptitle, "get_gid", lambda: None)(),
                                     supxlabel=getattr(fig._supxlabel, "get_gid", lambda: None)(),
                                     supylabel=getattr(fig._supylabel, "get_gid", lambda: None)(),auto={key:bool(getattr(getattr(fig,"_"+key,None),"_autopos",False)) for key in ("suptitle","supxlabel","supylabel")}),
                         engine=engine_rec, grids=grids, axes=axes, colorbars=colorbars,
                         legends=legends, texts=texts, frames=frames, fonts=fonts))


def retain_layout_references(layout, present):
    """Matplotlib culls empty labels and out-of-view ticks; reference only emitted ids."""
    out = copy.deepcopy(layout)
    out["texts"] = {k: v for k, v in out["texts"].items() if k in present}
    out["frames"] = {k: v for k, v in out["frames"].items() if k in present}
    out["legends"] = [v for v in out["legends"] if v["id"] in present]
    for key in ("suptitle", "supxlabel", "supylabel"):
        if out["figure"][key] not in present:
            out["figure"][key] = None
    for axis in out["axes"] + out["colorbars"]:
        axis["extraLayoutIds"] = [v for v in axis.get("extraLayoutIds", []) if v in present]
        for key in ("left", "center", "right"):
            if axis["title"][key] not in present:
                axis["title"][key] = None
        for style in axis["axisStyle"].values():
            for key in ("offsetText", "label"):
                if style[key] not in present:
                    style[key] = None
    return out
