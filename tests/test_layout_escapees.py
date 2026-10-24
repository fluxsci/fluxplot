import json
from pathlib import Path
import matplotlib.pyplot as plt
import fluxplot as fp
from lxml import etree

def test_offset_legend_arrow_and_colorbar_are_addressable(tmp_path):
    fig,ax=plt.subplots(layout='constrained');ax.plot([1e6,1e6+1],[1,2],label='signal')
    ax.legend(title='Series');ax.annotate('Peak',xy=(1e6+1,2),xytext=(-20,10),textcoords='offset points',arrowprops={'arrowstyle':'->'})
    image=ax.imshow([[0,1],[2,3]],extent=(1e6,1e6+1,1,2),aspect='auto');fig.colorbar(image,ax=ax)
    result=fp.save(fig,str(tmp_path/'escapees.svg'),recipe=False)
    root=etree.parse(result.svg);manifest=json.loads(Path(result.manifest).read_text())
    roles=set(root.xpath('//@data-role'))
    assert {'tick-offset','legend-title','annotation-arrow','background'}<=roles
    ids=set(root.xpath('//@id'))
    assert not any(i.startswith(('text_','patch_')) for i in ids)
    for identifier in manifest['layout']['texts']:assert identifier in ids
    arrow=next(f for f in manifest['layout']['frames'].values() if f['f']=='arrow')
    assert arrow['text'] in ids
    plt.close(fig)
