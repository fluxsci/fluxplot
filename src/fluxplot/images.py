"""``fp.image`` and ``fp.scalebar``: micrographs and other pixel data as first-class, addressable
plot content.

An image is one or more **channels** (a ``(H, W)`` matrix each), every channel with its own LUT
(a colormap, or a colour name meaning a black-to-colour ramp) and **display range** (the black and
white points; default the 1st–99.8th percentiles). fluxplot composites the channels into one RGB
``imshow`` in Python and records everything a consumer needs to redo it: each channel is a colour
scale in the manifest (``<series>.<channel>``, linear norm over the display range,
``recolor: "regenerate"`` — or ``"raster"`` with ``value_raster=True``, when the channel's values
travel beside the SVG) and a recipe control, so Flux's colour-scale editor edits a channel's LUT or
brightness/contrast and reruns. ``pixel_size`` puts the axes in physical units, and
:func:`scalebar` draws a vector bar of a stated length in those units.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import colorscale as _colorscale
from . import data as _data
from . import tagger as _tagger
from .descriptors import Mark

__all__ = ["image", "scalebar", "ImageResult", "DEFAULT_CHANNEL_COLOURS", "DEFAULT_DISPLAY_PERCENTILES"]

#: the LUT colours of a multi-channel image without ``luts``: the fluorescence-microscopy
#: convention (green / magenta first, since they stay distinct for every common colour deficiency)
DEFAULT_CHANNEL_COLOURS = ("#00ff00", "#ff00ff", "#00ffff", "#ffff00", "#ff0000", "#0000ff")
#: the default display range: the (black, white) percentiles of a channel's finite values
DEFAULT_DISPLAY_PERCENTILES = (1.0, 99.8)
COMPOSITES = ("add", "max")
_MONO_PREFIX = "image.mono:"


# ---------------------------------------------------------------------------------------------
# LUTs
# ---------------------------------------------------------------------------------------------
def resolve_lut(spec):
    """A channel LUT: a Colormap as is; a registered colormap name (matplotlib's or fluxplot's);
    else a colour (``"green"``, ``"#00ff00"``) meaning the black-to-colour ramp
    ``image.mono:<hex>``."""
    from matplotlib import colors as mcolors
    from ._fieldmap import resolve_colormap
    if spec is None:
        return resolve_colormap("gray")
    if isinstance(spec, mcolors.Colormap):
        return spec
    name = str(spec)
    if name.startswith(_MONO_PREFIX):
        return _mono(name[len(_MONO_PREFIX):])
    try:
        return resolve_colormap(name)
    except (KeyError, ValueError):
        pass
    if not mcolors.is_color_like(name):
        raise ValueError(f"image: LUT {spec!r} is neither a colormap name nor a colour")
    return _mono(name)


def _mono(colour):
    from matplotlib import colors as mcolors
    hex_ = mcolors.to_hex(colour, keep_alpha=False).lower()
    cm = mcolors.LinearSegmentedColormap.from_list(_MONO_PREFIX + hex_, ["#000000", hex_], N=256)
    cm.set_bad((0.0, 0.0, 0.0, 0.0))
    return cm


# ---------------------------------------------------------------------------------------------
# the image
# ---------------------------------------------------------------------------------------------
@dataclass
class ImageResult:
    """What :func:`image` drew."""

    ax: Any
    #: the composited RGB ``AxesImage``
    artist: Any
    #: channel names in stack order (``None`` names → ``ch0, ch1, …``; a 2-D image has one channel)
    channels: List[str]
    #: channel name → the values drawn (``float``, NaN for missing), ``(H, W)`` each
    data: Dict[str, np.ndarray]
    #: channel name → its LUT (a Colormap)
    luts: Dict[str, Any]
    #: channel name → ``(black, white)`` display range actually used
    display_range: Dict[str, Tuple[float, float]]
    #: channel name → the recipe control key / manifest colour-scale id
    keys: Dict[str, str]
    #: ``(row, column)`` pixel size in ``units``, or ``None`` (pixel coordinates)
    pixel_size: Optional[Tuple[float, float]]
    units: str
    #: the ``imshow`` extent ``[left, right, bottom, top]``
    extent: List[float]
    #: the composited RGB array, ``(H, W, 3)`` in ``[0, 1]``
    rgb: np.ndarray = field(repr=False, default=None)
    #: channel name → a ``ScalarMappable`` (LUT + display range) for ``fp.colorbar``
    mappables: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.rgb.shape[:2]


def _stack(data, channels):
    """``(values (C, H, W) float, names)`` from a 2-D image or a (C, H, W) / (H, W, C) stack."""
    arr = np.ma.filled(np.ma.asarray(data, dtype=float), np.nan)
    if arr.ndim == 2:
        if channels is not None and len(channels) != 1:
            raise ValueError(f"image: a 2-D image has one channel, but channels= names {len(channels)}")
        return arr[None], (None if channels is None else [str(channels[0])])
    if arr.ndim != 3:
        raise ValueError(f"image: data must be (H, W), (C, H, W) or (H, W, C); got shape {arr.shape}")
    if channels is not None:
        n = len(channels)
        if arr.shape[0] == n and arr.shape[-1] != n:
            stack = arr
        elif arr.shape[-1] == n:
            stack = np.moveaxis(arr, -1, 0)
        elif arr.shape[0] == n:
            stack = arr
        else:
            raise ValueError(f"image: channels= names {n} channels but data has shape {arr.shape}")
        return stack, [str(c) for c in channels]
    if arr.shape[0] <= 4 < arr.shape[-1]:
        stack = arr
    elif arr.shape[-1] <= 4:
        stack = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"image: cannot tell the channel axis of shape {arr.shape}; pass channels=[...]")
    return stack, [f"ch{i}" for i in range(stack.shape[0])]


def _per_channel(value, n, what):
    """Broadcast a per-channel option: ``None`` → ``[None] * n``; one value → repeated; a list of
    ``n`` → as is."""
    if value is None:
        return [None] * n
    if isinstance(value, (list, tuple)) and len(value) == n and (n != 2 or what != "display_range"
                                                                 or any(isinstance(v, (list, tuple)) or v is None for v in value)):
        return list(value)
    if what == "display_range":
        if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, (int, float, np.number)) for v in value):
            return [tuple(value)] * n
        raise ValueError(f"image: display_range must be (lo, hi) or one pair per channel ({n})")
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"image: {what} has {len(value)} entries for {n} channels")
        return list(value)
    return [value] * n


def _auto_range(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = (float(q) for q in np.percentile(finite, DEFAULT_DISPLAY_PERCENTILES))
    if lo == hi:
        lo, hi = float(finite.min()), float(finite.max())
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def image(ax, data, *, series, pixel_size=None, units="µm", origin="upper", channels=None, luts=None,
          display_range=None, composite="add", value_raster=False, key=None, **imshow_kw) -> ImageResult:
    """Draw a single- or multi-channel image with recorded LUTs and display ranges.

    Parameters
    ----------
    ax, data
        The axes and the pixels: ``(H, W)``, or a ``(C, H, W)`` / ``(H, W, C)`` stack (the channel
        axis is the one matching ``len(channels)``, else the one of length ≤ 4). Missing pixels
        (NaN / masked) are transparent.
    series
        The series name: the image is ``<series>.image`` and every channel's colour scale and
        recipe control is ``<series>.<channel>`` (a 2-D image with no ``channels``: ``<series>``).
    pixel_size, units
        The physical size of a pixel (one number, or ``(row, column)``) and its unit; the axes are
        then in those units (``extent``), and :func:`scalebar` lengths are too.
    origin
        ``"upper"`` (row 0 at the top, the imaging convention) or ``"lower"``.
    channels
        Channel names for a stack (default ``ch0, ch1, …``).
    luts
        Per-channel LUT (one for all, or a list): a colormap name / Colormap, or a colour name
        meaning a black-to-colour ramp. Default ``gray`` for one channel, else green, magenta,
        cyan, yellow, red, blue.
    display_range
        Per-channel ``(black, white)`` (one pair for all, or a list): the linear norm's limits.
        Default the 1st–99.8th percentiles of each channel's finite values.
    composite
        How channels combine: ``"add"`` (sum, clipped) or ``"max"``.
    value_raster
        Write each channel's values as ``<plot>.<key>.values.json`` beside the SVG and mark its
        scale ``recolor: "raster"``, so a consumer can repaint from the values.
    key
        Override the control-key root (default ``series``).
    **imshow_kw
        Passed to ``Axes.imshow`` (``interpolation`` defaults to ``"nearest"``).

    Returns
    -------
    ImageResult
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib import colors as mcolors
    from .fields import _options
    if composite not in COMPOSITES:
        raise ValueError(f"image: composite must be one of {COMPOSITES}, got {composite!r}")
    if origin not in ("upper", "lower"):
        raise ValueError(f"image: origin must be 'upper' or 'lower', got {origin!r}")
    stack, names = _stack(data, channels)
    n, h, w = stack.shape
    single_unnamed = names is None
    if single_unnamed:
        names = ["ch0"]
    if len(set(names)) != n:
        raise ValueError(f"image: channel names must be distinct, got {names}")
    lut_list = _per_channel(luts, n, "luts")
    ranges = _per_channel(display_range, n, "display_range")
    root = str(key) if key is not None else str(series)
    reg = _tagger.registry_for(ax.figure)

    rgb = np.zeros((h, w, 3))
    result_luts, result_ranges, keys, mappables, channel_records, scale_records, controls = {}, {}, {}, {}, [], [], {}
    for i, name in enumerate(names):
        vals = stack[i]
        lut = lut_list[i]
        if lut is None:
            lut = "gray" if n == 1 else DEFAULT_CHANNEL_COLOURS[i % len(DEFAULT_CHANNEL_COLOURS)]
        lo, hi = ranges[i] if ranges[i] is not None else _auto_range(vals)
        ckey = root if single_unnamed else f"{root}.{name}"
        opts = {"cmap": lut if isinstance(lut, mcolors.Colormap) else str(lut), "vmin": float(lo), "vmax": float(hi)}
        # the recipe's colour controls for this channel (a LUT or display range edited in Flux)
        ckey = _options(ax, series, ckey, opts, resolve=resolve_lut)
        cmap = resolve_lut(opts["cmap"])
        norm = opts.get("norm")
        if norm is None:
            norm = mcolors.Normalize(vmin=opts.get("vmin", lo), vmax=opts.get("vmax", hi))
        if norm.vmin is None or norm.vmax is None:
            norm.vmin, norm.vmax = (lo if norm.vmin is None else norm.vmin), (hi if norm.vmax is None else norm.vmax)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        rgba = sm.to_rgba(np.ma.masked_invalid(vals), bytes=False)
        layer = rgba[..., :3] * rgba[..., 3:4]  # a transparent (missing) pixel adds nothing
        rgb = np.maximum(rgb, layer) if composite == "max" else rgb + layer
        result_luts[name], result_ranges[name], keys[name], mappables[name] = cmap, (float(norm.vmin), float(norm.vmax)), ckey, sm
        field_like = {"normalization": {"kind": type(norm).__name__, "vmin": float(norm.vmin), "vmax": float(norm.vmax),
                                        "clip": bool(norm.clip)}, "cmap": cmap.name,
                      "cmapSpec": _colorscale.cmap_spec(cmap, resolve_lut)}
        controls[ckey] = _colorscale.controls_state(field_like)
        record = _colorscale.scale_record(ckey, sm, label=name if not single_unnamed else None,
                                          extend="neither", recolor="raster" if value_raster else "regenerate")
        scale_records.append(record)
        channel_records.append({"name": name, "lut": controls[ckey]["cmap"], "displayRange": [float(norm.vmin), float(norm.vmax)],
                                "scale": ckey, "controlKey": ckey})
    rgb = np.clip(rgb, 0.0, 1.0)

    if pixel_size is not None:
        py, px = (float(pixel_size), float(pixel_size)) if np.ndim(pixel_size) == 0 else (float(pixel_size[0]), float(pixel_size[1]))
        if py <= 0 or px <= 0:
            raise ValueError("image: pixel_size must be positive")
        extent = [0.0, w * px, h * py, 0.0] if origin == "upper" else [0.0, w * px, 0.0, h * py]
        size = (py, px)
    else:
        extent = [-0.5, w - 0.5, h - 0.5, -0.5] if origin == "upper" else [-0.5, w - 0.5, -0.5, h - 0.5]
        size = None
    imshow_kw.setdefault("interpolation", "nearest")
    artist = ax.imshow(rgb, origin=origin, extent=extent, **imshow_kw)

    payload = {"shape": [int(h), int(w)], "channels": channel_records, "composite": composite, "origin": origin,
               "extent": [float(v) for v in extent], "units": str(units),
               "pixelSize": None if size is None else [size[0], size[1]]}
    mark_data = {"image": payload, "color_scales": scale_records, "color_controls": controls, "color_paint": "fill"}
    if value_raster:
        mark_data["value_rasters"] = [
            {"key": keys[name], "payload": {"spec": "fluxplot/values", "scale": keys[name], "shape": [int(h), int(w)],
                                            "values": _data.values(stack[i].reshape(-1))}}
            for i, name in enumerate(names)]
    reg.add(Mark(role="image", series=series, kind="image", x=None, y=None, artists=[artist], axes=ax, data=mark_data))
    return ImageResult(ax=ax, artist=artist, channels=list(names), data={nm: stack[i] for i, nm in enumerate(names)},
                       luts=result_luts, display_range=result_ranges, keys=keys, pixel_size=size, units=str(units),
                       extent=[float(v) for v in extent], rgb=rgb, mappables=mappables)


# ---------------------------------------------------------------------------------------------
# the scale bar
# ---------------------------------------------------------------------------------------------
_LOCS = {
    "lower right": (1.0, 0.0), "lower left": (0.0, 0.0), "upper right": (1.0, 1.0), "upper left": (0.0, 1.0),
    "lower center": (0.5, 0.0), "upper center": (0.5, 1.0),
}


def scalebar(ax, length, units="µm", *, loc="lower right", label=None, color=None, thickness=2.0,
             pad=0.4, name=None, text_kw=None):
    """A vector scale bar of ``length`` data units, anchored in a corner of the axes.

    The bar is a ``Line2D`` whose x extent is exactly ``length`` in data coordinates (so it is
    true to :func:`image`'s ``pixel_size``) and whose vertical anchor is an axes fraction, with
    its label (default ``"<length> <units>"``) centred above it (below, for the upper corners).
    ``pad`` is the inset from the axes edge in units of the label's font size. ``color`` paints
    bar and label (default: the theme's ink, so it reads on a dark ground too). Registers
    ``Mark(role="scalebar")`` with ``{length, units}``; the label is ``<id>.label``.
    """
    import matplotlib
    from matplotlib import transforms as mtransforms
    from matplotlib.lines import Line2D
    if length <= 0:
        raise ValueError("scalebar: length must be positive")
    if loc not in _LOCS:
        raise ValueError(f"scalebar: loc must be one of {sorted(_LOCS)}, got {loc!r}")
    reg = _tagger.registry_for(ax.figure)
    idx = reg.next_overlay_index("scalebar")
    if name is None:
        name = str(idx)
    text_kw = dict(text_kw or {})
    fontsize = text_kw.get("fontsize", matplotlib.rcParams["font.size"])
    fontsize_pt = matplotlib.font_manager.FontProperties(size=fontsize).get_size_in_points()
    # the axes' size in points, for pad (font sizes) → axes fraction
    bbox = ax.get_position()
    fig_w, fig_h = ax.figure.get_size_inches()
    ax_w_pt, ax_h_pt = max(bbox.width * fig_w * 72, 1e-9), max(bbox.height * fig_h * 72, 1e-9)
    pad_x, pad_y = pad * fontsize_pt / ax_w_pt, pad * fontsize_pt / ax_h_pt
    hx, hy = _LOCS[loc]
    x0, x1 = ax.get_xlim()
    span = x1 - x0
    frac_len = length / abs(span) if span else 0.0
    if hx == 1.0:
        right = 1.0 - pad_x
        left = right - frac_len
    elif hx == 0.0:
        left = pad_x
        right = left + frac_len
    else:
        left, right = 0.5 - frac_len / 2, 0.5 + frac_len / 2
    y = pad_y if hy == 0.0 else 1.0 - pad_y
    # x in data units (the bar's length is the statement), y as an axes fraction (the anchor)
    xa = x0 + left * span
    xb = xa + (length if span >= 0 else -length)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    themed = color is None
    if themed:
        color = matplotlib.rcParams["text.color"]
    bar = Line2D([xa, xb], [y, y], transform=trans, color=color, linewidth=thickness, solid_capstyle="butt", zorder=10)
    ax.add_line(bar)
    if label is None:
        label = f"{length:g} {units}".rstrip()
    gap = 0.4 * fontsize_pt / ax_h_pt
    if hy == 0.0:
        ty, va = y + gap, "bottom"
    else:
        ty, va = y - gap, "top"
    text_kw.setdefault("fontsize", fontsize)
    txt = ax.text((xa + xb) / 2, ty, label, transform=trans, ha="center", va=va, color=color, zorder=10, **text_kw)
    data = {"length": float(length), "units": str(units), "label": str(label), "label_artist": txt, "index": idx, "loc": loc}
    if themed:
        data["ink"] = {"stroke": "ink"}
        data["ink_label"] = {"fill": "ink"}
    reg.add(Mark(role="scalebar", series=None, name=name, artists=[bar], axes=ax, data=data))
    bar._fluxplot_label = txt
    return bar
