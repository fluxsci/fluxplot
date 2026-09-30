"""Capture each axis's data↔SVG-pixel mapping + scale type (spec §6).

Consumers must NOT have to reconstruct matplotlib's transform. We store, per axis, explicit
``(data, svg)`` anchor pairs plus the scale type — two anchors + scale is the complete, transform-free
contract a morph / resize / agent needs. The conversion is the dpi-proof *fraction* method verified
in ``NOTES_matplotlib_svg.md`` §3 (exact round-trip against emitted marker positions).

Call :func:`capture_axes` AFTER ``fig.canvas.draw()`` (so layout/transforms are final) and BEFORE
the controlled render.
"""
from __future__ import annotations

import numpy as np


def svg_viewbox(fig) -> tuple[float, float]:
    """SVG user-space size in points — always ``figsize_inches * 72`` regardless of dpi."""
    return fig.get_figwidth() * 72.0, fig.get_figheight() * 72.0


def data_to_svg(ax, fig, x: float, y: float) -> tuple[float, float]:
    """Map a data point to SVG user-space coordinates (origin top-left, points)."""
    disp = ax.transData.transform((x, y))
    w_px, h_px = fig.bbox.width, fig.bbox.height
    vbw, vbh = svg_viewbox(fig)
    sx = disp[0] / w_px * vbw
    sy = (1.0 - disp[1] / h_px) * vbh
    return float(sx), float(sy)


def _log_base(axis) -> float:
    try:
        return float(axis.get_transform().base)
    except Exception:
        return 10.0


def tick_kinds(axis):
    """Portable names for an axis' major tick locator and formatter classes: the consumer that
    re-ticks a view (or redraws a colour key) picks the matching scheme."""
    from matplotlib import ticker
    loc, fmt = axis.get_major_locator(), axis.get_major_formatter()
    if isinstance(loc, ticker.LogLocator):
        locator = 'log'
    elif isinstance(loc, ticker.FixedLocator):
        locator = 'fixed'
    elif isinstance(loc, ticker.MultipleLocator):
        locator = 'multiple'
    elif type(loc).__module__ == 'matplotlib.dates':
        locator = 'date'
    elif type(loc).__module__ == 'matplotlib.category':
        locator = 'category'
    else:
        locator = 'auto'
    if isinstance(fmt, ticker.LogFormatter):
        formatter = 'log'
    elif isinstance(fmt, ticker.PercentFormatter):
        formatter = 'percent'
    elif isinstance(fmt, ticker.ScalarFormatter):
        formatter = 'sci' if fmt.get_offset() else 'plain'
    elif isinstance(fmt, ticker.FixedFormatter):
        formatter = 'fixed'
    elif type(fmt).__module__ == 'matplotlib.dates':
        formatter = 'date'
    elif type(fmt).__module__ == 'matplotlib.category':
        formatter = 'category'
    else:
        formatter = 'custom'
    return locator, formatter


def _axis_capture(ax, fig, which: str) -> dict:
    if which == "x":
        scale = ax.get_xscale()
        lo, hi = ax.get_xlim()
        label = ax.get_xlabel()
        y_ref = ax.get_ylim()[0]
        endpoints = [(lo, y_ref), (hi, y_ref)]
        mpl_axis = ax.xaxis
    else:
        scale = ax.get_yscale()
        lo, hi = ax.get_ylim()
        label = ax.get_ylabel()
        x_ref = ax.get_xlim()[0]
        endpoints = [(x_ref, lo), (x_ref, hi)]
        mpl_axis = ax.yaxis

    out: dict = {
        "scale": scale,
        "label": label,
        "domain": [float(lo), float(hi)],
        "anchors": [],
    }
    if scale == "log":
        out["base"] = _log_base(mpl_axis)

    supported = getattr(ax, "name", "rectilinear") == "rectilinear" and scale in ("linear", "log")
    out["supported"] = supported
    out["tickLocator"], out["tickFormatter"] = tick_kinds(mpl_axis)
    out["ticks"] = [{"value": float(v), "label": t.get_text()}
                    for v, t in zip(mpl_axis.get_ticklocs(), mpl_axis.get_ticklabels()) if np.isfinite(v)]
    converter = getattr(mpl_axis, "get_converter", lambda: getattr(mpl_axis, "converter", None))()
    module = type(converter).__module__ if converter else ""
    if module == "matplotlib.dates":
        from matplotlib.dates import get_epoch
        out["units"] = {"kind": "date", "epoch": get_epoch(), "unit": "day"}
    elif module == "matplotlib.category":
        out["units"] = {"kind": "category"}
    for (dx, dy), data_val in zip(endpoints, (lo, hi)):
        if not supported:
            break
        sx, sy = data_to_svg(ax, fig, dx, dy)
        out["anchors"].append({"data": float(data_val), "svg": sx if which == "x" else sy})
    return out


def capture_axes(ax, fig, twins=()) -> dict:
    """Return ``{"x": {...}, "y": {...}, "pixelBox": {...}}`` for one Axes — plus ``"y2"`` /
    ``"x2"`` for each twin axes (``[(axes, "y2" | "x2")]``) sharing its frame."""
    # plot-area rectangle in SVG coords (convenience; the SVG clipPath stays authoritative).
    vbw, vbh = svg_viewbox(fig)
    box = ax.get_window_extent()
    sx0, sx1 = box.x0 / fig.bbox.width * vbw, box.x1 / fig.bbox.width * vbw
    sy0, sy1 = (1 - box.y1 / fig.bbox.height) * vbh, (1 - box.y0 / fig.bbox.height) * vbh
    out = {
        "projection": getattr(ax, "name", "rectilinear"),
        "x": _axis_capture(ax, fig, "x"),
        "y": _axis_capture(ax, fig, "y"),
        "pixelBox": {
            "x0": min(sx0, sx1),
            "y0": min(sy0, sy1),
            "x1": max(sx0, sx1),
            "y1": max(sy0, sy1),
        },
    }
    for twin, which in twins:
        if which not in out:  # the first twin of each kind; a third value axis has no slot
            out[which] = _axis_capture(twin, fig, "y" if which == "y2" else "x")
            functions = getattr(twin, "_functions", None)
            if functions is not None and callable(functions[0]):
                # a secondary axis: the parent → secondary transform as samples across the parent's range
                lo, hi = (ax.get_xlim() if which == "x2" else ax.get_ylim())
                grid = np.linspace(lo, hi, 9)
                try:
                    mapped = np.asarray(functions[0](grid), dtype=float)
                    out[which]["secondary"] = {"of": "x" if which == "x2" else "y",
                                               "samples": [[float(a), float(b)] for a, b in zip(grid, mapped) if np.isfinite(b)]}
                except Exception:  # a transform that rejects the grid: the axis stays, unsampled
                    out[which]["secondary"] = {"of": "x" if which == "x2" else "y", "samples": []}
    return out
