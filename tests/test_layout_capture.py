"""The additive 0.4 layout contract is SVG-renderer data, not runtime guesses."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pytest
import fluxplot as fp
from fluxplot.integrity import validate_references
from lxml import etree

@pytest.mark.parametrize('engine',[None,'tight','constrained'])
def test_layout_and_resave_are_stable(tmp_path,engine):
    fig,axs=plt.subplots(1,2,figsize=(5,3),layout=engine or 'none',width_ratios=[2,1])
    for ax in axs:ax.plot([1,2,3],[2,4,8]);ax.set(xlabel='Time',ylabel='Value',title='Panel')
    result=fp.save(fig,str(tmp_path/'plot.svg'),recipe=False)
    first=Path(result.manifest).read_bytes();layout=json.loads(first)['layout']
    assert layout['supported'] and layout['version']==1
    assert layout['figure']['widthPt']==360
    assert layout['engine']['kind']==(engine or 'none')
    assert layout['grids'][0]['widthRatios']==[2,1]
    assert len(layout['axes'])==2 and layout['fonts']
    ids=set(etree.parse(result.svg).xpath('//@id'))
    validate_references(json.loads(first),ids)
    fp.save(fig,str(tmp_path/'plot.svg'),recipe=False)
    assert Path(result.manifest).read_bytes()==first
    ax=axs[0];ax.set_xlim(1,10)
    result=fp.save(fig,str(tmp_path/'plot.svg'),recipe=False)
    assert json.loads(Path(result.manifest).read_text())['layout']['axes'][0]['initialDomain']['x']==[1,10]
    plt.close(fig)

@pytest.mark.parametrize('projection',['polar','3d'])
def test_refusal_is_named_without_breaking_save(tmp_path,projection):
    fig=plt.figure();ax=fig.add_subplot(projection=projection);ax.plot([0,1],[0,1])
    result=fp.save(fig,str(tmp_path/'refused.svg'),recipe=False)
    layout=json.loads(Path(result.manifest).read_text())['layout']
    assert layout['supported'] is False and any(projection in reason for reason in layout['reasons'])
    plt.close(fig)

def test_fixed_and_numeric_ticks_and_glyph_anchors(tmp_path):
    fig,ax=plt.subplots(layout='constrained');ax.scatter([1,2],[2,3],s=[4,25]);ax.set_xticks([1,2],['First','Second'])
    result=fp.save(fig,str(tmp_path/'ticks.svg'),recipe=False)
    layout=json.loads(Path(result.manifest).read_text())['layout'];axis=layout['axes'][0]
    assert axis['ticking']['x']['retick'] is False
    assert axis['ticking']['x']['labels']==['First','Second']
    assert axis['ticking']['y']['retick'] is True
    assert any(len(frame.get('anchorsPt',[]))==2 for frame in layout['frames'].values())
    plt.close(fig)


def test_tiny_log_domains_and_secondary_limits_preserve_precision(tmp_path):
    fig, ax = plt.subplots(layout='constrained')
    ax.plot([1e-12, 1e12], [0, 1]); ax.set_xscale('log')
    secondary = ax.secondary_yaxis('right', functions=(lambda v: v*2, lambda v: v/2))
    result = fp.save(fig, str(tmp_path/'tiny.svg'), recipe=False)
    axes = json.loads(Path(result.manifest).read_text())['layout']['axes']
    main = next(a for a in axes if a['ticking']['x']['scale']=='log')
    assert main['ticking']['x']['domain'][0] > 0
    child = next(a for a in axes if a['id'].endswith('.y2'))
    assert child['initialDomain']['y'] == pytest.approx(secondary.get_ylim())
    plt.close(fig)
