"""Signature plots on dark grounds (B5): every mark a glowbar, fluxbox or hexmatrix draws stands off
the axes background under the light, paper and dark themes (WCAG contrast ≥ 1.5), and the
bracket / identity lines by ≥ 3; inks lift on dark grounds instead of deepening."""
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex, to_rgba  # noqa: E402

import fluxplot as fp  # noqa: E402
from fluxplot import colorcheck as cc  # noqa: E402
from fluxplot import style  # noqa: E402
from fluxplot.signature_fluxplots import _colour  # noqa: E402

THEMES = ("use_light", "use_paper", "use_dark")


@pytest.fixture(autouse=True)
def _reset():
    yield
    plt.close("all")
    mpl.rcdefaults()
    style.use_light()


def _table():
    rng = np.random.default_rng(4)
    # six units, each measured in both categories (so paired connectors exist)
    return {"type": ["SD", "S"] * 6, "APP": list(rng.normal(2, 0.4, 12)), "subject": [f"u{i // 2}" for i in range(12)]}


def _ground(ax):
    c = ax.get_facecolor()
    return c if c[3] > 0 else ax.figure.get_facecolor()


def _contrast(colour, ground):
    r, g, b, a = to_rgba(colour)
    gr = np.array(to_rgba(ground)[:3])
    blended = a * np.array([r, g, b]) + (1 - a) * gr  # what a translucent mark actually shows
    return cc.contrast(tuple(blended), to_hex(ground))


@pytest.mark.parametrize("theme", THEMES)
def test_glowbar_marks_stand_off_the_ground(theme):
    getattr(style, theme)()
    fig, ax = plt.subplots()
    gb = fp.glowbar(_table(), x="type", y="APP", units="subject", ax=ax,
                    connect_identical_points_across_x_values=True)
    ground = _ground(ax)
    for coll in gb.artists["points"]:
        # a point reads through its face or its rim: the palest shade on white is outlined by a
        # deeper rim (that is what the rim is for), the deepest on a dark ground by a lighter one
        for face, edge in zip(coll.get_facecolor(), coll.get_edgecolor()):
            assert max(_contrast(face, ground), _contrast(edge, ground)) >= 1.5, (theme, to_hex(face), to_hex(edge))
    for ln in gb.artists["mean"] + gb.artists["caps"] + gb.artists["lines"]:
        assert _contrast(ln.get_color(), ground) >= 1.5, (theme, to_hex(ln.get_color()))
    for glow in gb.artists["glow"]:
        assert _contrast(glow.get_colors()[0][:3], ground) >= 1.5
    if theme == "use_dark":
        # inks lift on a dark ground: the mean line is paler than the group colour, not deeper
        col = gb.group_colors["SD"]
        mean = next(ln for ln in gb.artists["mean"])
        assert _colour.perceptual(np.array(to_rgba(mean.get_color())[:3]))[0] > _colour.perceptual(np.array(col[:3]))[0]
        assert gb.artists["lines"][0].get_color() == style.ACTIVE["tokens"]["grid"]  # the dark grid reads
    if theme == "use_light":
        from fluxplot.signature_fluxplots.glowbar import CONNECT_GREY
        assert gb.artists["lines"][0].get_color() == CONNECT_GREY  # the light grid would not: base-300 stays


@pytest.mark.parametrize("theme", THEMES)
def test_fluxbox_marks_stand_off_the_ground(theme):
    getattr(style, theme)()
    fig, ax = plt.subplots()
    fb = fp.fluxbox(_table(), x="type", y="APP", units="subject", ax=ax, show_caps=True)
    ground = _ground(ax)
    for coll in fb.artists["points"]:
        for face, edge in zip(coll.get_facecolor(), coll.get_edgecolor()):
            assert max(_contrast(face, ground), _contrast(edge, ground)) >= 1.5, theme
    for ln in fb.artists["box"] + fb.artists["whiskers"] + fb.artists["median"] + fb.artists["caps"]:
        colour = to_rgba(ln.get_color(), alpha=ln.get_alpha())
        assert _contrast(colour, ground) >= 1.5, (theme, to_hex(colour))


@pytest.mark.parametrize("theme", THEMES)
def test_hexmatrix_ramp_and_identity_line_read_on_the_ground(theme):
    getattr(style, theme)()
    rng = np.random.default_rng(1)
    x = 10 ** rng.normal(0, 0.4, 300)
    fig, ax = plt.subplots(figsize=(3, 3))
    hm = fp.hexmatrix(x=x, y=x * 10 ** rng.normal(0, 0.2, 300), ax=ax, color="#4CB391", xscale="log",
                      yscale="log", identity_line=True, gridsize=8)
    ground = _ground(ax)
    top = hm.cmap(1.0)  # the fullest hexagon's colour is the one that must stand out
    assert _contrast(top, ground) >= 1.5, (theme, to_hex(top))
    assert _contrast(hm.artists["identity"].get_color(), ground) >= 3.0, theme
    if theme == "use_dark":
        assert hm.cmap.name.startswith("hexmatrix.mono-dark:")
        # deep near the ground, pale at the top — and the name replays the same ramp
        assert _colour.perceptual(np.array(hm.cmap(0.0)[:3]))[0] < _colour.perceptual(np.array(hm.cmap(1.0)[:3]))[0]
        from fluxplot.signature_fluxplots.hexmatrix import _resolve_cmap
        again = _resolve_cmap(hm.cmap.name)
        assert np.array_equal(again(np.arange(256)), hm.cmap(np.arange(256)))
    else:
        assert hm.cmap.name.startswith("hexmatrix.mono:")


@pytest.mark.parametrize("theme", THEMES)
def test_bracket_reads_on_the_ground(theme):
    getattr(style, theme)()
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [1, 2], series="s")
    br = fp.significance_bracket(ax, x0=0, x1=1, y=2.2, label="*")
    assert _contrast(br.get_color(), _ground(ax)) >= 3.0


def test_ground_aware_inks():
    assert _colour.is_dark("#1c1b1a") and not _colour.is_dark("#ffffff")
    assert _colour.neutral_for("#ffffff") == _colour.NEUTRAL and _colour.neutral_for("#1c1b1a") == _colour.NEUTRAL_DARK
    light_rim, dark_rim = _colour.rim("#4385be", "#ffffff"), _colour.rim("#4385be", "#1c1b1a")
    L = lambda c: _colour.perceptual(np.array(c[:3]))[0]  # noqa: E731
    assert L(light_rim) < L(to_rgba("#4385be")) < L(dark_rim)
    assert _colour.shade_bounds(88.0, 22.0, "#ffffff") == (88.0, 22.0)
    pale, dark = _colour.shade_bounds(88.0, 22.0, "#1c1b1a")
    assert pale == 88.0 and dark >= _colour.ground_lightness("#1c1b1a") + 20
