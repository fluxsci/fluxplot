# matplotlib → SVG mechanics (verified on matplotlib 3.11.0, Python 3.13)

Findings from the de-risk spike. These are the external matplotlib behaviours FluxPlot relies on.
Re-run `tests/test_marker_dom.py` after any matplotlib upgrade — if these change, the generator
breaks silently.

## 1. `Artist.set_gid(gid)` → `<g id="gid">`
Confirmed for every artist type we tag:
- `Line2D` (line) → `<g id="control.line"><path d="…" clip-path="url(#…)" style="fill:none;stroke:…"/></g>`
- `Line2D` markers (`ax.plot(x,y,'o')`) and `PathCollection` (`ax.scatter`) → identical shape:
  ```
  <g id="control.points">
    <defs><path id="m7782986621" d="…glyph…" style="…"/></defs>
    <g clip-path="url(#clip)">
      <use xlink:href="#m7782986621" x="46.14" y="160.56" style="fill:…;stroke:…"/>   ← one per point,
      <use xlink:href="#m7782986621" x="96.87" y="97.20" style="…"/>                   in DATA ORDER
      …
    </g>
  </g>
  ```
- `Rectangle` bar (`ax.bar`) → `<g id="counts.bar.0"><path d="M…z" style="fill:…"/></g>` (one group per
  bar — per-bar addressing is trivial; set_gid on each `Rectangle` in the `BarContainer`).

## 2. Per-point addressability → collection-split (primary strategy)
Points are drawn as ONE artist (`series.points` gid). In post-processing, enumerate the `<use>`
descendants of that group **in document order** and assign each `id="series.point.k"` + `data-index`
+ `data-x`/`data-y`. Per-point **style lives on the `<use>`** (`style="fill:…;stroke:…"`), so a GUI
restyle of point k = set style on its `<use>` — no need to touch the shared `<defs>` glyph.
- **Guard:** if `len(<use>) != len(data)` (off-axis points can be culled), skip per-point ids and keep
  only the `series.points` group addressable. (Fallback, not expected for typical in-view data.)

## 3. Coordinate capture → dpi-proof fraction method (EXACT match verified)
SVG user space is always **points** (72/inch): `viewBox="0 0 figw_in*72 figh_in*72"`, origin top-left.
matplotlib display space is pixels @dpi, origin bottom-left. Round-trip verified exactly:
```
disp = ax.transData.transform((x,y))                 # display px, origin bottom-left
fracx = disp[0] / fig.bbox.width                     # dpi-invariant
fracy = disp[1] / fig.bbox.height
svg_x = fracx * (fig.get_figwidth()  * 72)
svg_y = (1 - fracy) * (fig.get_figheight() * 72)     # y flipped
```
Capture **after** `fig.canvas.draw()` and **before** the controlled save. (Direct `disp*72/dpi` also
works but the SVG backend can report dpi=72 at draw time; the fraction form is invariant either way.)

## 4. `svg.fonttype='none'` keeps `<text>`
14 real `<text>` nodes with readable content ("0.0", "Time (h)") + `font-family:'DejaVu Sans'`.
Axis titles / tick labels / legend text stay editable, restyleable, addressable.

## 5. Determinism
`rcParams['svg.hashsalt'] = <fixed>` → render twice → **byte-identical** (verified). With
`hashsalt=None`, clip/marker/`<defs>` ids are salted by a fresh uuid4 each run → output differs.
Also set `metadata={'Date': None}` and strip `Creator`/comment in post-process.

## Internal references that the GUI must re-prefix when inlining
`<use xlink:href="#m…">` ↔ `<path id="m…">` (defs), and `clip-path="url(#p…)"` ↔ `<clipPath id="p…">`.
Per-instance id prefixing in the app must rewrite both sides to keep them paired.
