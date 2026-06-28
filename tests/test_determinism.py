"""P5: same input → byte-identical SVG + manifest (the basis for morph/diff/regeneration)."""
import matplotlib.pyplot as plt

import fluxplot as fp
from _helpers import build_growth_fig


def _save(tmp_path, name):
    fig = build_growth_fig()
    res = fp.save(fig, str(tmp_path / name), recipe=dict(script="growth.py", params={"t": 1}))
    plt.close(fig)
    svg = open(res.svg, "rb").read()
    manifest = open(res.manifest, "rb").read()
    return svg, manifest


def test_svg_and_manifest_are_byte_identical(tmp_path):
    svg1, man1 = _save(tmp_path, "a.svg")
    svg2, man2 = _save(tmp_path, "a.svg")  # same name → same hashsalt
    assert svg1 == svg2, "SVG output is not deterministic"
    assert man1 == man2, "manifest output is not deterministic"


def test_resaving_same_figure_is_stable(tmp_path):
    # saving one figure object twice must not double per-point ids (idempotent resolve)
    fig = build_growth_fig()
    r1 = fp.save(fig, str(tmp_path / "g.svg"))
    a = open(r1.manifest, "rb").read()
    r2 = fp.save(fig, str(tmp_path / "g.svg"))
    b = open(r2.manifest, "rb").read()
    plt.close(fig)
    assert a == b
