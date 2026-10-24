"""Font-independent tables from Matplotlib LayoutGrid and tight-layout algebra."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib._layoutgrid import LayoutGrid
from matplotlib._tight_layout import _auto_adjust_subplotpars
from matplotlib.transforms import Bbox


def generate(out,count=240):
    rng=np.random.default_rng(231023);rows=[]
    for i in range(count):
        nr,nc=int(rng.integers(1,4)),int(rng.integers(1,5));w,h=float(rng.uniform(350,950)),float(rng.uniform(250,750))
        wr=rng.integers(1,5,nc).tolist();hr=rng.integers(1,5,nr).tolist();margins=rng.uniform(0,22,(nr*nc,4));wp,hp=rng.uniform(1,9,2);ws,hs=rng.uniform(0,.2,2)
        root=LayoutGrid(parent=(0,0,1,1));grid=LayoutGrid(parent=root,parent_inner=True,nrows=nr,ncols=nc,width_ratios=wr,height_ratios=hr)
        for r in range(nr):
            for c in range(nc):
                left,bottom,right,top=margins[r*nc+c]
                for side,value,index in [('left',left/w,c),('right',right/w,c),('top',top/h,r),('bottom',bottom/h,r),('leftcb',max(wp,ws*w/2/nc) /w if c else wp/w,c),('rightcb',max(wp,ws*w/2/nc)/w if c<nc-1 else wp/w,c),('topcb',max(hp,hs*h/2/nr)/h if r else hp/h,r),('bottomcb',max(hp,hs*h/2/nr)/h if r<nr-1 else hp/h,r)]:grid.edit_margin_min(side,value,index)
        root.update_variables();expected=[]
        for r in range(nr):
            for c in range(nc):
                b=grid.get_inner_bbox(r,c);expected.append([b.x0*w,(1-b.y1)*h,b.x1*w,(1-b.y0)*h])
        rows.append(dict(engine='constrained',nr=nr,nc=nc,w=w,h=h,wr=wr,hr=hr,margins=margins.tolist(),wp=wp,hp=hp,ws=ws,hs=hs,boxes=expected))
        fig=plt.figure(figsize=(w/72,h/72),dpi=72);gs=fig.add_gridspec(nr,nc,width_ratios=wr,height_ratios=hr);axs=[];spans=[];positions=[]
        for r in range(nr):
            for c in range(nc):
                ax=fig.add_subplot(gs[r,c]);left,bottom,right,top=margins[r*nc+c];bb=ax.get_position().transformed(fig.transFigure);tight=Bbox.from_extents(bb.x0-left,bb.y0-bottom,bb.x1+right,bb.y1+top)
                ax.get_tightbbox=lambda *a,_box=tight,**kw:_box
                axs.append([ax]);spans.append((slice(r,r+1),slice(c,c+1)));positions.append(list(ax.get_position().bounds))
        pad=float(rng.uniform(.5,1.5));kwargs=_auto_adjust_subplotpars(fig,None,shape=(nr,nc),span_pairs=spans,subplot_list=axs,pad=pad,h_pad=pad,w_pad=pad)
        if kwargs:
            fig.subplots_adjust(**kwargs);expected=[]
            for group in axs:
                b=group[0].get_position();expected.append([b.x0*w,(1-b.y1)*h,b.x1*w,(1-b.y0)*h])
            rows.append(dict(engine='tight',nr=nr,nc=nc,w=w,h=h,wr=wr,hr=hr,margins=margins.tolist(),pad=pad,positions=positions,boxes=expected))
        plt.close(fig)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'solver.json').write_text(json.dumps(dict(matplotlib=matplotlib.__version__,rows=rows),indent=2)+'\n')
    print(f'solver: {len(rows)} rows')
