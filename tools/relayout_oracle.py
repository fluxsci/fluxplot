"""Matplotlib 3.11 oracle for elastic layout. Generated files are never edited.

Run with PYTHONPATH=src python tools/relayout_oracle.py --out <Flux fixtures>.
The spike measurements deliberately live separately from the public manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import fluxplot as fp
from fluxplot import render as _render


SPIKE_CASES = ("line", "scatter", "colorbar", "shared", "ratios", "none")
CASES = SPIKE_CASES + ("scatter-sizes", "bars", "hist", "errorbar", "fill", "blend",
    "image-equal", "image-auto", "contour", "colorbar-horizontal", "colorbar-shared",
    "datalim", "box-aspect", "span", "nested", "twin", "secondary", "inset", "log",
    "symlog", "category", "date", "offset", "rotated", "legend-corner", "legend-best",
    "legend-outside", "figure-text", "annotation", "tight", "raster", "hexmatrix", "fluxbox", "seaborn",
    "polar", "subfigures", "3d", "line-updated", "log-wide", "log-subs", "both-sides", "manual-colorbar", "annotation-symlog",
    "meanmark", "meanmark-log", "tight-rect")


def build(name, w, h):
    if name not in SPIKE_CASES:
        return build_extended(name, w, h)
    kw = dict(figsize=(w, h), layout=None if name == "none" else "constrained")
    if name == "shared":
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, **kw)
    elif name in ("colorbar", "ratios"):
        fig, axs = plt.subplots(1, 2, width_ratios=[2, 1] if name == "ratios" else None, **kw)
    else:
        fig, axs = plt.subplots(**kw)
    x = np.linspace(0, 10, 31)
    for i, ax in enumerate(np.atleast_1d(axs).flat):
        if name == "scatter":
            ax.scatter(x, 2 * x + np.sin(x), label="observations")
        elif name == "colorbar" and i == 1:
            im = ax.imshow(np.arange(100).reshape(10, 10), aspect="auto")
            fig.colorbar(im, ax=ax)
        else:
            ax.plot(x, 2 * x, label="signal")
        ax.set(xlabel="Time (s)", ylabel="Response", title=f"Panel {i + 1}")
    return fig


def build_extended(name, w, h):
    from matplotlib.ticker import FixedLocator
    if name == "subfigures":
        fig=plt.figure(figsize=(w,h));fig.subfigures(1,2)[0].subplots().plot([0,1],[0,1]);return fig
    if name=="tight-rect":
        # A live tight engine whose rect reserves the top 14 % for a figure legend.
        fig,axs=plt.subplots(1,2,figsize=(w,h));x=np.linspace(.1,10,31)
        for a in axs:a.plot(x,np.sin(x),label='Signal');a.set(xlabel='Time (s)',ylabel='Response')
        fig.legend(*axs[0].get_legend_handles_labels(),loc='upper right',bbox_to_anchor=(1,1))
        fig.set_layout_engine('tight',pad=.5,rect=(0,0,1,.86));return fig
    if name in ("polar","3d"):
        fig=plt.figure(figsize=(w,h));ax=fig.add_subplot(projection=name);ax.plot([0,1],[0,1]);return fig
    fig,ax=plt.subplots(figsize=(w,h),layout="tight" if name=="tight" else "constrained")
    x=np.linspace(.1,10,31);rng=np.random.default_rng(316)
    if name=="line-updated":ax.plot(x,np.sin(x),label="signal")
    elif name=="scatter-sizes": ax.scatter(x,np.sin(x),s=np.linspace(4,80,len(x)))
    elif name=="bars": ax.bar([0,1,2],[2,5,3])
    elif name=="hist": ax.hist(rng.normal(size=400),bins=12)
    elif name=="errorbar": ax.errorbar(x[::3],np.sin(x[::3]),yerr=.2,xerr=.1,fmt='o-',capsize=3)
    elif name=="fill": ax.fill_between(x,np.sin(x),np.sin(x)+.5)
    elif name=="blend": ax.plot(x,np.sin(x));ax.axhline(.3);ax.axvspan(2,3,alpha=.2)
    elif name in ("image-equal","image-auto","colorbar-horizontal","colorbar-shared"):
        if name=="colorbar-shared":
            fig.clear();axs=fig.subplots(1,2);ax=axs[0];im=ax.imshow(np.arange(30).reshape(5,6),aspect='auto');axs[1].imshow(np.arange(30).reshape(5,6),aspect='auto');fig.colorbar(im,ax=list(axs))
        else:
            im=ax.imshow(np.arange(30).reshape(5,6),aspect='auto' if name!='image-equal' else 'equal')
            if name=='colorbar-horizontal':fig.colorbar(im,ax=ax,orientation='horizontal',label='Intensity')
    elif name=="contour":
        xx,yy=np.meshgrid(np.linspace(-2,2,15),np.linspace(-2,2,15));im=ax.contourf(xx,yy,xx*xx+yy*yy,levels=6);fig.colorbar(im,ax=ax)
    elif name in ("datalim","box-aspect"):
        ax.plot(x,np.sin(x));ax.set_aspect(1,adjustable='datalim') if name=='datalim' else ax.set_box_aspect(1)
    elif name in ("span","nested"):
        fig.clear();gs=fig.add_gridspec(2,2)
        if name=='span': axes=[fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
        else: inner=gs[0,0].subgridspec(2,1);axes=[fig.add_subplot(inner[0]),fig.add_subplot(inner[1]),fig.add_subplot(gs[:,1])]
        for a in axes:a.plot(x,np.sin(x));a.set(xlabel='Time',ylabel='Value')
        ax=axes[0]
    elif name in ("twin","secondary","inset"):
        ax.plot(x,np.sin(x))
        if name=='twin':a=ax.twinx();a.plot(x,x*20,color='red');a.set_ylabel('Temperature')
        elif name=='secondary':ax.secondary_xaxis('top',functions=(lambda v:v*2,lambda v:v/2)).set_xlabel('Twice time')
        else:a=ax.inset_axes([.6,.55,.35,.35]);a.plot(x,np.cos(x))
    elif name=="log":ax.loglog(x,x*x)
    elif name in ("log-wide", "log-subs"):
        from matplotlib.ticker import LogLocator, LogFormatterSciNotation
        ax.plot([1e-12,1e12] if name=='log-wide' else [1e-2,1e3],[1e-9,1e9],label='signal');ax.set_xscale('log');ax.set_yscale('log')
        if name=='log-wide':ax.tick_params(labelsize=16)
        else:
            ax.xaxis.set_major_locator(LogLocator(subs=(1,2,5)))
            ax.xaxis.set_major_formatter(LogFormatterSciNotation(minor_thresholds=(float('inf'),float('inf'))))
    elif name=="both-sides":
        ax.plot(x,np.sin(x),label='signal');ax.tick_params(top=True,right=True,labeltop=True,labelright=True)
    elif name=="manual-colorbar":
        im=ax.imshow(np.arange(30).reshape(5,6),aspect='auto');cax=fig.add_axes([.85,.2,.025,.6]);fig.colorbar(im,cax=cax)
    elif name=="annotation-symlog":
        ax.plot(x-5,np.sin(x));ax.set_xscale('symlog',linthresh=1);ax.annotate('Peak',xy=(2,1),xytext=(25,18),textcoords='offset points',arrowprops={'arrowstyle':'->'})
    elif name=="symlog":ax.plot(x-5,np.sin(x));ax.set_xscale('symlog',linthresh=1)
    elif name=="category":ax.bar(['Wake','NREM','REM'],[3,5,2])
    elif name=="date":ax.plot(np.arange('2026-01-01','2026-01-11',dtype='datetime64[D]'),np.arange(10))
    elif name=="offset":ax.plot(x+1e6,np.sin(x)*1e-6)
    elif name=="rotated":ax.bar(['Control group','Intervention group','Recovery group'],[1,3,2]);ax.tick_params(axis='x',labelrotation=35)
    elif name.startswith('legend-'):
        ax.plot(x,np.sin(x),label='Signal');ax.scatter(x[::3],np.cos(x[::3]),label='Observations')
        ax.legend(loc='best' if name=='legend-best' else 'upper right' if name=='legend-corner' else 'upper left',**({'bbox_to_anchor':(1.02,1)} if name=='legend-outside' else {}),title='Series')
    elif name=="figure-text":
        ax.plot(x,np.sin(x));fig.suptitle('Experiment');fig.supxlabel('Time from stimulus');fig.supylabel('Measurements');fig.text(.5,.02,'Study A',ha='center');ax.text(.05,.95,'ROI',transform=ax.transAxes,va='top')
    elif name=="annotation":ax.plot(x,np.sin(x));ax.annotate('Peak',xy=(1.57,1),xytext=(50,20),textcoords='offset points',arrowprops={'arrowstyle':'->'})
    elif name=="raster":ax.scatter(rng.normal(size=2000),rng.normal(size=2000))
    elif name=="hexmatrix":fp.hexmatrix(matrix=rng.normal(size=(8,8)),ax=ax,series='matrix')
    elif name=="fluxbox":fp.fluxbox({'g':['a']*10+['b']*10,'v':rng.normal(size=20)},x='g',y='v',ax=ax)
    elif name=="meanmark":
        # Notched circles, arms and caps; group c's interval is shorter than the radius (cut caps).
        fp.meanmark({'g':['a']*10+['b']*10+['c']*10,'u':list(range(10))*3,
                     'v':np.r_[rng.normal(size=20),1+rng.normal(scale=.002,size=10)]},
                    x='g',y='v',units='u',ax=ax,show_notch=True)
    elif name=="meanmark-log":
        fp.meanmark({'g':['a']*12+['b']*12,'v':np.exp(rng.normal(size=24))},x='g',y='v',ax=ax,
                    orientation='horizontal',value_scale='log',interval='range')
    elif name=="seaborn":
        import seaborn as sns
        sns.barplot(x=['a']*10+['b']*10,y=rng.normal(size=20),ax=ax,seed=316);fig.tight_layout()
    else:ax.plot(x,np.sin(x))
    ax.set(xlabel='Time (s)',ylabel='Response',title='Experiment')
    return fig


def measurements(fig, engine, initial_positions=None):
    """Facts captured with the SVG renderer while figsize and gids are frozen."""
    from matplotlib.backends.backend_svg import RendererSVG
    from io import StringIO
    # Figure._get_renderer would choose Agg after draw_without_rendering's canvas
    # restoration. Use the same SVG font engine as final_layout, at 72 dpi.
    renderer = RendererSVG(fig.bbox.width, fig.bbox.height, StringIO())
    w, h = fig.get_size_inches() * 72
    grids, grid_ids = [], {}

    def slot(ss):
        gs = ss.get_gridspec()
        if id(gs) not in grid_ids:
            parent = slot(gs._subplot_spec) if hasattr(gs, "_subplot_spec") else None
            gid = f"g{len(grids)}"
            grid_ids[id(gs)] = gid
            grids.append(dict(id=gid, nrows=gs.nrows, ncols=gs.ncols,
                              widthRatios=list(gs.get_width_ratios()),
                              heightRatios=list(gs.get_height_ratios()),
                              wspace=getattr(gs,'wspace',getattr(gs,'_wspace',None)), hspace=getattr(gs,'hspace',getattr(gs,'_hspace',None)), parent=parent))
        return dict(grid=grid_ids[id(gs)], rows=[ss.rowspan.start, ss.rowspan.stop-1],
                    cols=[ss.colspan.start, ss.colspan.stop-1])

    def box(bb):
        return dict(x0=float(bb.x0), y0=float(h-bb.y1), x1=float(bb.x1), y1=float(h-bb.y0))

    axes, colorbars = [], []
    for ax in fig.axes:
        b = ax.get_window_extent(renderer)
        tight = ax.get_tightbbox(renderer, for_layout_only=True)
        ss = ax.get_subplotspec()
        cb = getattr(ax, "_colorbar_info", None)
        rec = dict(id=ax.get_gid(), box=box(b), position=list(ax.get_position(original=True).bounds),
                   slot=slot(ss) if ss is not None else None,
                   margins=[b.x0-tight.x0, b.y0-tight.y0, tight.x1-b.x1, tight.y1-b.y1],
                   aspect=ax.get_aspect(), boxAspect=np.asarray(ax.get_box_aspect()).tolist() if ax.get_box_aspect() is not None else None,
                   adjustable=ax.get_adjustable(), anchor=ax.get_anchor(),
                   domain=[list(ax.get_xlim()), list(ax.get_ylim())])
        rec["initialPosition"] = list(initial_positions[id(ax)]) if initial_positions else rec["position"]
        rec["tickMetrics"] = []
        for axis in (ax.xaxis, ax.yaxis):
            labels = axis.get_ticklabels()
            prop = labels[0].get_fontproperties() if labels else axis.label.get_fontproperties()
            widths = {ch: renderer.get_text_width_height_descent(ch, prop, False)[0]
                      for ch in "0123456789.−-+eE "}
            lo, hi = sorted(axis.get_view_interval())
            visible = [t for v, t in zip(axis.get_majorticklocs(), labels)
                       if lo <= v <= hi and t.get_visible() and t.get_text()]
            rec["tickMetrics"].append(dict(fontPt=prop.get_size_in_points(), advances=widths,
                  visible=bool(visible), width=max((t.get_window_extent(renderer).width for t in visible), default=0),
                  height=max((t.get_window_extent(renderer).height for t in visible), default=0)))
        if cb:
            rec.update({k: cb[k] for k in ("location", "fraction", "shrink", "aspect", "pad", "anchor", "panchor")})
            rec["parents"] = [a.get_gid() for a in cb["parents"]]
            colorbars.append(rec)
        else:
            axes.append(rec)
    kind = type(engine).__name__
    opts = engine.get() if engine is not None else {}
    return dict(widthPt=float(w), heightPt=float(h), engine=dict(kind="constrained" if kind == "ConstrainedLayoutEngine" else "none",
                wPad=(opts.get("w_pad") or 0)*72, hPad=(opts.get("h_pad") or 0)*72,
                wSpace=opts.get("wspace", 0), hSpace=opts.get("hspace", 0)),
                subplotpars={k: getattr(fig.subplotpars, k) for k in ("left", "right", "bottom", "top", "wspace", "hspace")},
                grids=grids, axes=axes, colorbars=colorbars)


def generate(out, selected=None, resume=False):
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for name in selected or CASES:
        native = (5., 3.2)
        sizes = [(5*a, 3.2*b) for a in (.6, .8, 1.25, 1.6, 2.) for b in (.6, .8, 1.25, 1.6, 2.)]
        sizes += [(2.4, 6.), (8., 2.), (3., 5.)]
        folder = out/name
        folder.mkdir(exist_ok=True)
        records = []
        if resume and all((folder/f"{stem}.fluxplot.json").exists() and
                          (folder/f"{stem}.oracle.json").exists() for stem in ["base", *[f"size-{i:02}" for i in range(28)]]):
            cases.append(dict(name=name, native=list(native), sizes=[dict(stem=f"size-{i:02}",wPt=w*72,hPt=h*72) for i,(w,h) in enumerate(sizes)]))
            print(f"{name}: resumed 28 completed sizes",flush=True)
            continue
        for stem, (w, h) in [("base", native)]+[(f"size-{i:02}", size) for i, size in enumerate(sizes)]:
            # V1 freezes a one-shot tight_layout and native date tick labels.
            # Resize after that one-time work; other cases rebuild at target figsize.
            if name in ('seaborn','date'):
                fig=build(name,*native)
                if name=='date':
                    fig.draw_without_rendering()
                    for a in fig.axes:
                        a.set_xticks(a.get_xticks(),[t.get_text() for t in a.get_xticklabels()])
                fig.set_size_inches(w,h)
            else:
                fig=build(name,w,h)
            if name=='nested':
                for i,a in enumerate(fig.axes):fp.panel(a,f'fixed-{i}')
            initial_positions = {id(a): a.get_position(original=True).bounds for a in fig.axes}
            engine = fig.get_layout_engine()
            captured = []
            original = _render.render_svg
            def capture_then_render(figure, *args, **kwargs):
                captured.append(measurements(figure, engine, initial_positions))
                return original(figure, *args, **kwargs)
            try:
                _render.render_svg = capture_then_render
                fp.save(fig, str(folder/f"{stem}.svg"), recipe=False)
            finally:
                _render.render_svg = original
            rec = captured[0]
            (folder/f"{stem}.oracle.json").write_text(json.dumps(rec, indent=2)+"\n")
            plt.close(fig)
            if stem != "base":
                records.append(dict(stem=stem, wPt=w*72, hPt=h*72))
        cases.append(dict(name=name, native=list(native), sizes=records))
        print(f"{name}: {len(records)} sizes", flush=True)
    (out/"cases.json").write_text(json.dumps(dict(matplotlib=matplotlib.__version__, cases=cases), indent=2)+"\n")
    fig, ax = plt.subplots(figsize=(5, 3.2), layout="constrained")
    fp.hexmatrix(matrix=np.random.default_rng(723).normal(size=(56, 56)), ax=ax, colorbar=False, series="dense")
    fp.save(fig, str(out/"hexmatrix.svg"), recipe=False, force_vectors=True)
    plt.close(fig)


def tick_tables(out, count=2400):
    from matplotlib import ticker
    from fluxplot.layout import locator_record, formatter_record
    rng=np.random.default_rng(7481);rows=[]
    for i in range(count):
        length=float(rng.uniform(30,650));font=float(rng.choice([6,8,10,12,16]));vertical=bool(i%2)
        fig=plt.figure(figsize=(2 if vertical else length/72,length/72 if vertical else 2))
        ax=fig.add_axes([0,0,1,1]);axis=ax.yaxis if vertical else ax.xaxis
        kind=i%5
        lo=float(rng.uniform(-10,10)*10**int(rng.integers(-5,7)))
        hi=lo+float(rng.uniform(.1,10)*10**int(rng.integers(-4,6)))
        if kind==0:
            lo=float(10**rng.uniform(-6,2));hi=lo*float(10**rng.uniform(.2,12))
            (ax.set_yscale if vertical else ax.set_xscale)('log')
            base=float(rng.choice([2,10]));subs=[1.] if i%3 else ([1.,2.,5.] if base==10 else [1.])
            loc=ticker.LogLocator(base=base,subs=subs)
            fmt=ticker.LogFormatterSciNotation(base=base,labelOnlyBase=i%7==0,minor_thresholds=(1,.4) if i%4 else (float('inf'),float('inf')))
        elif kind==1:
            base=float(10**int(rng.integers(-3,4)))*float(rng.choice([.5,1,2,5]));lo=0;hi=base*float(rng.uniform(2,18))
            loc=ticker.MultipleLocator(base=base,offset=base*.25);fmt=ticker.ScalarFormatter()
        else:
            loc=ticker.MaxNLocator(nbins='auto' if i%3 else int(rng.integers(2,10)),steps=[1,2,2.5,5,10],integer=i%13==0,symmetric=i%17==0,prune=rng.choice([None,'lower','upper','both']))
            if kind==2:fmt=ticker.ScalarFormatter(useOffset=i%4!=0,useMathText=i%7==0)
            elif kind==3:fmt=ticker.PercentFormatter(xmax=float(rng.choice([1,100])),decimals=None)
            else:fmt=ticker.StrMethodFormatter('{x:.2f}' if i%2 else '{x:.1e}')
        (ax.set_ylim if vertical else ax.set_xlim)(lo,hi)
        axis.set_major_locator(loc);axis.set_major_formatter(fmt)
        axis.set_tick_params(labelsize=font)
        if kind==0:minor=ticker.LogLocator(base=base,subs='auto')
        else:minor=ticker.AutoMinorLocator(None if i%3 else int(rng.choice([2,4,5])))
        axis.set_minor_locator(minor)
        values=loc();labels=fmt.format_ticks(values);minor_values=axis.get_minorticklocs()
        rows.append(dict(length=length,fontPt=font,vertical=vertical,axis=dict(major=locator_record(loc),minor=locator_record(minor),formatter=formatter_record(fmt),retick=True,domain=[lo,hi],scale='log' if kind==0 else 'linear',values=[]),values=values.tolist(),labels=labels,offset=fmt.get_offset(),minor=minor_values.tolist()))
        plt.close(fig)
    out.mkdir(parents=True,exist_ok=True)
    def native(v):return v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v)
    (out/'ticks.json').write_text(json.dumps(dict(matplotlib=matplotlib.__version__,rows=rows),default=native,indent=2)+'\n')
    print(f'ticks: {len(rows)} rows',flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", help="Comma separated case names (default: full corpus)")
    parser.add_argument("--resume", action="store_true", help="Reuse completed case families after an interrupted run")
    parser.add_argument("--ticks", action="store_true", help="Write the locator/formatter oracle table only")
    parser.add_argument("--solver",action="store_true",help="Write font-independent layout solver tables")
    args=parser.parse_args()
    if args.solver:
        from relayout_solver_oracle import generate as solver_tables
        solver_tables(args.out)
    elif args.ticks:tick_tables(args.out)
    else:generate(args.out, args.cases.split(',') if args.cases else None, args.resume)
