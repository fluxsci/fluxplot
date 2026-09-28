"""Colour sources shared by the signature fluxplots.

A signature plot takes *one* colour spec per group and turns it into (a) as many distinct point
colours as it needs and (b) one solid, representative group colour. The spec can be anything
fluxplot knows about:

* a colormap name from ``fp.colors.maps`` — qualified (``"cmasher.emerald"``, ``"crameri.batlow"``,
  ``"tol.sunset"``, ``"cmr.emerald"``) or bare (``"emerald"``, ``"YlGnBu"``), ``_r`` for reversed;
* a palette from ``fp.colors.palettes`` — ``"brewer.Set2"``, ``"tol.bright"``, ``"flexoki.blue"`` or a
  bare group name (``"bright"``) when no colormap has that name;
* any matplotlib-registered colormap name or a ``Colormap`` object;
* a list of colours (a hand-made palette);
* a single colour (it becomes a pale → colour → deep ramp).

**Ordered** sources (sequential maps / palettes — lightness runs one way) are oriented light → dark;
points take equal perceptual steps between two lightness bounds and the group colour sits at a fixed
position along the ramp. **Unordered** continuous sources (diverging, cyclic, rainbow) are sampled at
equal perceptual steps along the part of the map inside the lightness bounds — so a diverging map's
pale centre is skipped, never used for a dot on white. **Qualitative** palettes keep their designer's
order (their first colours are the most distinct), minus any too pale to read. For unordered sources
the group colour is the map's most chromatic colour of mid lightness — its most characteristic hue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

#: lightness window used when choosing a representative colour of an unordered source
_REP_LIGHTNESS = (35.0, 70.0)


# ---------------------------------------------------------------------------
# perceptual space
# ---------------------------------------------------------------------------
def lab(rgb) -> np.ndarray:
    """sRGB in [0, 1] (…, 3+) → CIELAB (D65)."""
    c = np.asarray(rgb, float)[..., :3]
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def perceptual(rgb) -> np.ndarray:
    """sRGB → a perceptually uniform space whose first axis is lightness on a 0–100 scale.

    CAM02-UCS (the space viridis and cmasher are designed in) when ``colorspacious`` is importable —
    it ships with cmasher, a fluxplot dependency — otherwise CIELAB.
    """
    rgb = np.asarray(rgb, float)[..., :3]
    try:
        from colorspacious import cspace_convert
    except ImportError:  # pragma: no cover - colorspacious comes with cmasher
        return lab(rgb)
    return cspace_convert(rgb, "sRGB1", "CAM02-UCS")


def darken(c, k):
    from matplotlib.colors import to_rgba
    r, g, b, a = to_rgba(c)
    return (r * (1 - k), g * (1 - k), b * (1 - k), a)


def _monotone(rgbs, tol=1.0) -> bool:
    """Lightness runs one way (within ``tol``) — i.e. the colours form an ordered ramp."""
    L = lab(np.asarray(rgbs, float))[..., 0]
    d = np.diff(L)
    return bool(len(L) >= 3 and (np.all(d <= tol) or np.all(d >= -tol)))


# ---------------------------------------------------------------------------
# resolving a spec
# ---------------------------------------------------------------------------
@dataclass
class ColourSource:
    """One group's colour source, resolved."""

    label: str
    #: "continuous" (sample a colormap) or "discrete" (pick from a list of colours)
    kind: str
    #: lightness runs one way (sequential); ordered sources are oriented light → dark
    ordered: bool
    cmap: Optional[object] = None  # continuous sources
    colours: list = field(default_factory=list)  # discrete sources, RGBA
    fixed: Optional[tuple] = None  # a single colour given as the spec: it IS the group colour
    brewer: bool = False  # a ColorBrewer ramp (its 0.75 tone is the designed group colour)

    def ramp(self):
        """A light→dark colormap for an ordered source (discrete lists are interpolated)."""
        from matplotlib.colors import LinearSegmentedColormap
        if self.cmap is not None:
            return self.cmap
        return LinearSegmentedColormap.from_list(f"glowbar-{self.label}", [c[:3] for c in self.colours])


def _from_colormap(cm, label, kind_hint=None, discrete_hint=None):
    from matplotlib.colors import ListedColormap
    discrete = discrete_hint if discrete_hint is not None else (
        isinstance(cm, ListedColormap) and cm.N <= 32)
    if discrete and hasattr(cm, "colors"):
        return _from_list(list(cm.colors), label, kind_hint)
    if kind_hint in ("sequential", "diverging", "cyclic", "qualitative"):
        ordered = kind_hint == "sequential"
    else:
        ordered = _monotone(cm(np.linspace(0, 1, 64)))
    if ordered and lab(cm(0.0))[0] < lab(cm(1.0))[0]:
        cm = cm.reversed()
    return ColourSource(label, "continuous", ordered, cmap=cm)


def _rgba(c):
    from matplotlib.colors import to_rgba
    return to_rgba(c)


def _from_list(colours, label, kind_hint=None):
    cols = [_rgba(c) for c in colours]
    if not cols:
        raise ValueError(f"colour source {label!r} is an empty palette")
    ordered = kind_hint == "sequential" if kind_hint else _monotone([c[:3] for c in cols])
    if ordered and lab(cols[0])[0] < lab(cols[-1])[0]:
        cols = cols[::-1]
    return ColourSource(label, "discrete", ordered, colours=cols)


def _palette(name):
    """``(colours, type)`` for ``"collection.group"`` or a bare group name, else ``None``."""
    from .. import colors as _colors
    pal = _colors.palettes
    if "." in name:
        cid, _, group = name.partition(".")
        candidates = [(cid, group)] if cid in pal.collections() else []
    else:
        candidates = [(cid, name) for cid in ("brewer", "tol", "flexoki")]
    for cid, group in candidates:
        for g in pal.info(cid)["groups"]:
            if g["name"] == group:
                return [s["hex"] for s in g["swatches"]], g.get("type")
    return None


def resolve(spec, label="group") -> ColourSource:
    """Resolve any colour spec fluxplot understands (see the module docstring)."""
    from matplotlib.colors import Colormap

    if isinstance(spec, Colormap):
        info = _map_info(spec.name)
        return _from_colormap(spec, label, info.get("type"), info.get("discrete"))
    if isinstance(spec, str):
        info = _map_info(spec)
        src = _resolve_str(spec, label, info)
        if src is not None:
            src.brewer = _is_brewer(spec, info)
            return src
    elif isinstance(spec, (list, tuple)) and len(spec) and not _is_single_colour(spec):
        return _from_list(list(spec), label)
    return _single(spec, label)


def _resolve_str(spec, label, info):
    """A named colormap or palette → its source, or ``None`` when the name is neither."""
    import matplotlib as mpl

    from .. import colors as _colors
    if spec in mpl.colormaps:  # matplotlib's exact map (built-ins, cmr.*, fluxplot-registered names)
        return _from_colormap(mpl.colormaps[spec], label, info.get("type"), info.get("discrete"))
    try:  # fluxplot's collections: bare names ("emerald", "batlow"), "cmasher.emerald", …
        return _from_colormap(_colors.maps.get(spec), label, info.get("type"), info.get("discrete"))
    except KeyError:
        pass
    found = _palette(spec)
    if found is not None:
        colours, kind = found
        return _from_list(colours, label, kind)
    return None


_BREWER = None


def _is_brewer(spec, info):
    global _BREWER
    if _BREWER is None:
        from .. import colors as _colors
        _BREWER = set(_colors.palettes.names("brewer"))
    base = spec[:-2] if spec.endswith("_r") else spec
    cid, _, bare = base.rpartition(".")
    return bare in _BREWER and cid in ("", "brewer", "mpl")


def _single(spec, label):
    import matplotlib as mpl
    try:
        colour = _rgba(spec)
    except (ValueError, TypeError):
        raise ValueError(
            f"{label}: {spec!r} is not a colormap (fp.colors.maps / matplotlib), a palette "
            "(fp.colors.palettes, e.g. 'brewer.Set2' or 'tol.bright'), a list of colours or a colour"
        ) from None
    rgb = np.array(colour[:3])
    ramp = [tuple(1 - 0.82 * (1 - rgb)), tuple(rgb), darken(colour, 0.6)[:3]]
    src = _from_colormap(mpl.colors.LinearSegmentedColormap.from_list(f"glowbar-{label}", ramp),
                         label, "sequential")
    src.fixed = colour
    return src


def _is_single_colour(spec) -> bool:
    try:
        _rgba(spec)
        return True
    except (ValueError, TypeError):
        return False


def _map_info(name) -> dict:
    from .. import colors as _colors
    try:
        return _colors.maps.info(name)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# choosing colours
# ---------------------------------------------------------------------------
def even_shades(cmap, n, pale=88.0, dark=22.0):
    """``n`` colours from a light→dark colormap, spaced EVENLY IN PERCEIVED COLOUR.

    The shades run from lightness ``pale`` to ``dark`` (0 = black, 100 = white) in equal steps of
    CAM02-UCS distance, i.e. equal *visible* differences. Sampling a map evenly in its parameter
    instead makes some neighbours nearly identical (ColorBrewer's YlGnBu merges its two darkest
    navies that way). A single shade is taken from the middle of the range.
    """
    if n <= 0:
        return []
    t = np.linspace(0.0, 1.0, 2048)
    u = perceptual(cmap(t))
    lightness = np.minimum.accumulate(u[:, 0])  # sequential maps darken monotonically; drop wiggles
    keep = (lightness <= pale) & (lightness >= dark)
    if keep.sum() < 2:
        raise ValueError(
            f"the colormap {getattr(cmap, 'name', cmap)!r} has no stretch between lightness "
            f"{pale} and {dark}; widen shade_range")
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(u[keep], axis=0), axis=1))]
    targets = [arc[-1] / 2] if n == 1 else np.linspace(0.0, arc[-1], n)
    return [cmap(float(ti)) for ti in np.interp(targets, arc, t[keep])]


def _even_along(cmap, n, pale, dark):
    """Equal perceptual steps along the parts of ANY colormap inside the lightness window.

    Jumps across excluded stretches (a diverging map's pale centre) add no distance, so the chosen
    colours stay evenly spread over what is actually usable.
    """
    t = np.linspace(0.0, 1.0, 2048)
    u = perceptual(cmap(t))
    keep = (u[:, 0] <= pale) & (u[:, 0] >= dark)
    if keep.sum() < 2:
        raise ValueError(f"the colormap {getattr(cmap, 'name', cmap)!r} has no stretch between "
                         f"lightness {pale} and {dark}; widen shade_range")
    tk, uk = t[keep], u[keep]
    step = np.linalg.norm(np.diff(uk, axis=0), axis=1)
    step[np.diff(np.flatnonzero(keep)) > 1] = 0.0  # no distance across a gap
    arc = np.r_[0.0, np.cumsum(step)]
    targets = [arc[-1] / 2] if n == 1 else np.linspace(0.0, arc[-1], n)
    return [cmap(float(tk[min(np.searchsorted(arc, a), len(tk) - 1)])) for a in targets]


def shades(src: ColourSource, n, pale=88.0, dark=22.0):
    """``n`` distinct point colours from a resolved source."""
    if n <= 0:
        return []
    if src.ordered:
        if src.kind == "discrete":
            usable = [c for c in src.colours if dark <= perceptual(c[:3])[0] <= pale]
            if len(usable) >= n:  # the designer's own classes, evenly spread
                idx = np.round(np.linspace(0, len(usable) - 1, n)).astype(int) if n > 1 else [len(usable) // 2]
                return [usable[i] for i in idx]
        return even_shades(src.ramp(), n, pale, dark)
    if src.kind == "continuous":
        return _even_along(src.cmap, n, pale, dark)
    usable = [c for c in src.colours if perceptual(c[:3])[0] <= pale] or src.colours
    if n > len(usable):
        import warnings
        warnings.warn(f"{src.label}: the palette has {len(usable)} usable colours for {n} points; "
                      "colours repeat — use a colormap (or a larger palette) to keep them distinct",
                      stacklevel=3)
    return [usable[i % len(usable)] for i in range(n)]


#: neutral group colour for multi-hue sources (Flexoki base-600): no single hue stands for them
NEUTRAL = (0x6F / 255, 0x6E / 255, 0x69 / 255, 1.0)


def representative(src: ColourSource, position=None):
    """One solid colour that stands for the whole source (glow, caps, the mean's base shade).

    * a single colour given as the spec → that colour;
    * ``position`` given → that point of the light → dark ramp (ordered sources);
    * a ColorBrewer ramp → 0.75 along it (Brewer places a strong, saturated tone there);
    * any other source → its most chromatic colour of mid lightness, provided its hues agree
      (one hue family, e.g. cmasher ``emerald``); a source whose hues spread around the wheel
      (diverging, rainbow) has no honest single hue, and a qualitative palette is a set of
      different categories — both get a neutral ink.
    """
    if src.fixed is not None:
        return src.fixed
    if src.ordered and (position is not None or src.brewer):
        return tuple(src.ramp()(0.75 if position is None else position))
    if src.kind == "discrete" and not src.ordered:
        return NEUTRAL  # a qualitative palette: every colour is a different category, none is 'the' hue
    cands = list(src.cmap(np.linspace(0, 1, 256))) if src.kind == "continuous" else list(src.colours)
    u = perceptual(np.array([c[:3] for c in cands]))
    chroma = np.hypot(u[:, 1], u[:, 2])
    mid = (u[:, 0] >= _REP_LIGHTNESS[0]) & (u[:, 0] <= _REP_LIGHTNESS[1]) & (chroma > 5)
    if not mid.any():
        return NEUTRAL
    hue = np.arctan2(u[mid, 2], u[mid, 1])
    w = chroma[mid]
    resultant = np.hypot((w * np.cos(hue)).sum(), (w * np.sin(hue)).sum()) / w.sum()
    if resultant < 0.6:  # hues spread around the wheel
        return NEUTRAL
    pool = np.flatnonzero(mid)
    return tuple(cands[int(pool[np.argmax(chroma[pool])])])


def interleaved_order(n):
    """Lane → shade rank such that neighbouring lanes are always ≥ 2 shades apart.

    ``0, h, 1, h+1, …`` with ``h = ceil(n/2)`` — e.g. ``[0, 3, 1, 4, 2, 5]`` for six lanes. The palest
    shade still sits in the first lane and the darkest in the last.
    """
    import itertools
    h = (n + 1) // 2
    order = []
    for a, b in itertools.zip_longest(range(h), range(h, n)):
        order.append(a)
        if b is not None:
            order.append(b)
    return order
