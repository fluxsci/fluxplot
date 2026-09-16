"""Every rendered tick must participate in the consumer's axis/colorbar groups."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator
import pytest
from lxml import etree
import fluxplot as fp


def _save(fig, tmp_path):
    result = fp.save(fig, str(tmp_path / 'ticks.svg'), recipe=False)
    return json.loads(Path(result.manifest).read_text()), etree.parse(result.svg)


def _node(tree, gid):
    if tree.get('id') == gid:
        return tree
    for child in tree.get('children', []):
        hit = _node(child, gid)
        if hit is not None:
            return hit


def _assert_no_untagged_tick_uses(root):
    # Matplotlib encloses every rendered major/minor tick in xtick_N / ytick_N.
    for tick in root.xpath('//*[starts-with(@id,"xtick_") or starts-with(@id,"ytick_")]'):
        for mark in tick.xpath('.//*[local-name()="use"]'):
            # A tick dash must be named and inlined as a measurable path; text
            # glyphs may still use paths when a caller chooses outlined fonts.
            if not mark.xpath('ancestor::*[local-name()="text"]'):
                pytest.fail('Unclassified/unmeasurable tick use: ' + etree.tostring(mark).decode())


@pytest.mark.parametrize('secondary', [False, True])
def test_log_axis_major_minor_and_both_sides_grouped(tmp_path, secondary):
    fig, ax = plt.subplots()
    ax.set_xscale('log'); ax.set_xlim(.5, 120)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100]))
    ax.xaxis.set_minor_locator(FixedLocator([2, 5, 20, 50]))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.tick_params(axis='x', which='both', top=secondary, labeltop=secondary)
    ax.grid(True, which='both', axis='x')
    manifest, root = _save(fig, tmp_path)
    ticks = _node(manifest['parts'], 'axis.x.ticks')['members']
    labels = _node(manifest['parts'], 'axis.x.tick-labels')['members']
    grids = _node(manifest['parts'], 'axis.x.gridlines')['members']
    assert len(ticks) == len(labels) == 7 * (2 if secondary else 1)
    assert len(grids) == 7
    assert sum('.minor.' in g for g in ticks) == 4 * (2 if secondary else 1)
    if secondary:
        assert sum('.secondary.' in g for g in ticks) == 7
    assert set(ticks) == set(root.xpath('//*[@data-role="tick" and @data-axis="x"]/@id'))
    assert set(labels) == set(root.xpath('//*[@data-role="tick-label" and @data-axis="x"]/@id'))
    _assert_no_untagged_tick_uses(root)
    again, _ = _save(fig, tmp_path)
    assert manifest == again
    plt.close(fig)


@pytest.mark.parametrize('location', ['right', 'left', 'top', 'bottom'])
def test_colorbar_ticks_groups_kinds_and_range(tmp_path, location):
    fig, ax = plt.subplots()
    image = fp.heatmap(ax, [[1, 3], [10, 75]], series='counts', norm=LogNorm(1, 75))
    cb = fp.colorbar(image, name='count', ax=ax, location=location, label='Count')
    axis = cb.ax.yaxis if cb.orientation == 'vertical' else cb.ax.xaxis
    axis.set_major_locator(LogLocator(base=10, subs=(1, 3)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    axis.set_minor_locator(FixedLocator([2, 5, 20, 50]))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    manifest, root = _save(fig, tmp_path)
    guide = next(g for g in manifest['guides'] if g['role'] == 'colorbar')
    ticks = _node(manifest['parts'], 'colorbar.count.ticks')['members']
    labels = _node(manifest['parts'], 'colorbar.count.tick-labels')['members']
    assert len(ticks) == len(labels) == 8
    assert guide['ticks'] == [1, 3, 10, 30]
    assert set(ticks) == set(root.xpath('//*[@data-role="colorbar-tick"]/@id'))
    assert set(labels) == set(root.xpath('//*[@data-role="colorbar-tick-label"]/@id'))
    expected = {'colorbar-label': 'text', 'colorbar-tick-label': 'text',
                'colorbar-tick': 'line', 'colorbar-outline': 'line', 'colorbar-solids': 'shape'}
    for part in guide['parts']:
        assert part['kind'] == expected[part['role']]
    assert guide['mappable'] == 'counts.x-heatmap'
    assert len(manifest['axes']) == 1
    _assert_no_untagged_tick_uses(root)
    plt.close(fig)
