"""Label map → one exact outline per object (SPEC 6), a port of the reference ``outlines.ts``.

The outline follows pixel edges, so it encloses exactly the labelled pixels: nothing is smoothed
or resampled. Pixel (row r, column c) covers x in [c, c+1] and y in [r, r+1]. Outer boundaries run
clockwise on screen (y down) and holes counter-clockwise, so both fill rules agree; where two parts
of an object touch only at a corner they stay separate loops. Given the same label map this
produces the same path strings as the reference, character for character.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LabelOutline", "label_outlines", "outline_area"]


@dataclass
class LabelOutline:
    label: int
    #: SVG path data: one closed subpath per boundary loop
    d: str
    #: number of pixels carrying the label
    pixels: int
    #: bounding box in pixels: x, y, width, height
    box: tuple
    #: centroid of the labelled pixels, pixel centres at +0.5
    cx: float
    cy: float


def _directed_edges(mask):
    """Boundary edges of a (bh, bw) mask between pixel corners (corner key ``y * (bw + 1) + x``),
    the object on the right-hand side of travel, in the reference's insertion order: pixels in
    row-major order, and per pixel top, right, bottom, left."""
    bh, bw = mask.shape
    stride = bw + 1
    pad = np.zeros((bh + 2, bw + 2), dtype=bool)
    pad[1:-1, 1:-1] = mask
    inner = pad[1:-1, 1:-1]
    sides = (
        inner & ~pad[:-2, 1:-1],   # top:    (x, y)     -> (x+1, y)
        inner & ~pad[1:-1, 2:],    # right:  (x+1, y)   -> (x+1, y+1)
        inner & ~pad[2:, 1:-1],    # bottom: (x+1, y+1) -> (x, y+1)
        inner & ~pad[1:-1, :-2],   # left:   (x, y+1)   -> (x, y)
    )
    starts, ends, order = [], [], []
    for kind, (sel, (ax, ay, bx, by)) in enumerate(zip(sides, ((0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 0, 0)))):
        ys, xs = np.nonzero(sel)
        starts.append((ys + ay) * stride + xs + ax)
        ends.append((ys + by) * stride + xs + bx)
        order.append((ys * bw + xs) * 4 + kind)
    order = np.concatenate(order)
    perm = np.argsort(order, kind="stable")
    return np.concatenate(starts)[perm].tolist(), np.concatenate(ends)[perm].tolist(), stride


def _loops(mask, x0, y0):
    starts, ends, stride = _directed_edges(mask)
    nxt = {}
    for a, b in zip(starts, ends):
        lst = nxt.get(a)
        if lst is None:
            nxt[a] = [b]
        else:
            lst.append(b)
    keys = list(nxt)       # insertion order; keys are never re-added, so the first live one is found by a cursor
    cursor = 0
    parts = []
    while nxt:
        while keys[cursor] not in nxt:
            cursor += 1
        start = keys[cursor]
        cur, pdx, pdy, first = start, 0, 0, True
        pts = [start]
        while True:
            cands = nxt[cur]
            cx, cy = cur % stride, cur // stride
            pick = 0
            if len(cands) > 1 and not first:
                # where two loops touch at a corner, turn right: diagonal neighbours stay separate loops
                best = None
                for i, c in enumerate(cands):
                    turn = pdx * (c // stride - cy) - pdy * (c % stride - cx)
                    if best is None or turn > best:
                        best, pick = turn, i
            n = cands.pop(pick)
            if not cands:
                del nxt[cur]
            dx, dy = n % stride - cx, n // stride - cy
            if not first and dx == pdx and dy == pdy:
                pts[-1] = n
            else:
                pts.append(n)
            pdx, pdy, first, cur = dx, dy, False, n
            if cur == start:
                break
        if len(pts) > 1 and pts[-1] == pts[0]:
            pts.pop()
        # the start corner may sit mid-edge: drop it when its neighbours are collinear through it
        if len(pts) > 2:
            a, m, c = pts[-1], pts[0], pts[1]
            ax, ay, mx, my, cx2, cy2 = a % stride, a // stride, m % stride, m // stride, c % stride, c // stride
            if (mx - ax) * (cy2 - my) == (my - ay) * (cx2 - mx):
                pts.pop(0)
        parts.append("M" + " L".join(f"{p % stride + x0},{p // stride + y0}" for p in pts) + " Z")
    return " ".join(parts)


def label_outlines(labels) -> list:
    """One :class:`LabelOutline` per positive label of a 2-D integer map, in ascending label order."""
    lab = np.asarray(labels)
    if lab.ndim != 2:
        raise ValueError("label_outlines needs a 2-D label map")
    H, W = lab.shape
    flat = lab.reshape(-1)
    where = np.flatnonzero(flat > 0)
    if where.size == 0:
        return []
    values = flat[where]
    order = np.argsort(values, kind="stable")
    values, where = values[order], where[order]
    cuts = np.flatnonzero(np.diff(values)) + 1
    out = []
    for idx, l in zip(np.split(where, cuts), values[np.r_[0, cuts]]):
        ys, xs = idx // W, idx % W
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        mask = lab[y0:y1 + 1, x0:x1 + 1] == l
        n = int(idx.size)
        out.append(LabelOutline(label=int(l), d=_loops(mask, x0, y0), pixels=n,
                                box=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                                cx=int(xs.sum()) / n + 0.5, cy=int(ys.sum()) / n + 0.5))
    return out


def outline_area(d: str) -> float:
    """Signed area of path data produced above (clockwise on screen is positive): the pixel count."""
    area = 0.0
    for sub in filter(None, d.split("M")):
        pts = [tuple(map(float, p.strip().split(","))) for p in sub.replace("Z", "").split("L")]
        for i, (x0, y0) in enumerate(pts):
            x1, y1 = pts[(i + 1) % len(pts)]
            area += x0 * y1 - x1 * y0
    return area / 2
