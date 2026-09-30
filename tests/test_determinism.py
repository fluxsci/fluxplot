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


def test_colour_scaled_plots_are_byte_stable(tmp_path):
    """A hexmatrix and a colour-mapped scatter (LUTs, gradients, per-element values) are as
    deterministic as a line plot."""
    import numpy as np

    def build():
        rng = np.random.default_rng(11)
        fig, (a, b) = plt.subplots(1, 2, figsize=(6, 2.6))
        fp.hexmatrix(x=rng.normal(size=300), y=rng.normal(size=300), ax=a, gridsize=8, series="h", norm="log")
        pts = fp.scatter(b, rng.normal(size=40), rng.normal(size=40), c=rng.uniform(size=40), s=rng.uniform(5, 40, 40), series="p")
        fp.colorbar(pts, extend="both")
        return fig

    outputs = []
    for _ in range(2):
        fig = build()
        res = fp.save(fig, str(tmp_path / "c.svg"), recipe=False)
        outputs.append((open(res.svg, "rb").read(), open(res.manifest, "rb").read()))
        plt.close(fig)
    assert outputs[0] == outputs[1]


def test_images_are_byte_stable(tmp_path):
    """An fp.image (per-channel LUTs, a scale bar, two colour keys) is as deterministic as a line plot."""
    import numpy as np

    def build():
        rng = np.random.default_rng(5)
        fig, ax = plt.subplots(figsize=(4, 3))
        im = fp.image(ax, rng.gamma(2.0, 30.0, (2, 24, 32)), series="cells", channels=["a", "b"], pixel_size=0.5)
        fp.scalebar(ax, 4.0)
        fp.colorbar(im.mappables["a"], ax=ax, name="a")
        return fig

    outputs = []
    for _ in range(2):
        fig = build()
        res = fp.save(fig, str(tmp_path / "i.svg"), recipe=False)
        outputs.append((open(res.svg, "rb").read(), open(res.manifest, "rb").read()))
        plt.close(fig)
    assert outputs[0] == outputs[1]
