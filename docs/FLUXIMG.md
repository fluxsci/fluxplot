# Scientific images (fluximg)

A **fluximg** is one N-dimensional scientific image together with what is known about it: its
pixels, physical axes, channels, and layers drawn on the same pixel grid (other images, labelled
objects with a table, lines and circles). It ranges from a single micrograph to a multi-gigabyte
stack with channel, depth and time dimensions. Flux places it on a page and lets you browse planes,
change contrast and LUTs, and show or hide layers without rerunning Python.

```python
import fluxplot as fp

img = fp.fluximg(mean_image, dims=("c", "y", "x"), pixel_size=0.25, unit="µm",
                 channels=["structure", "iGluSnFR"], meta=dict(subject="S1", session=3))
img.add_image("activity", activity_map, lut="magenta", opacity="value")
img.add_labels("footprints", label_map, background=-1, weights=weights,
               table=sources, key="source_id")
img.add_shapes("dendrites", lines=dendrite_paths, coords="row_col", key="dendrite_id")
img.add_shapes("discs", circles=discs, diameter=1.0, key="disc_id")
img.save("plots/mean")

fp.imsave("plots/ref_stack", stack, dims=("c", "z", "y", "x"), coords={"z": z_um},
          pixel_size=0.25, nodata=-1)
fp.imsave("plots/trial1", movie, dims=("c", "t", "x", "y"), dt=1 / 80, t0=start_s,
          pixel_size=0.25, layers="plots/mean")      # the mean image's layers, copied over

img = fp.imread("plots/ref_stack")                  # .values, .dims, .axes, .layer(id), .table(id)
```

`save` prints what it did, for example:

```
fluximg  plots/mean.fluximg
  pixels   c=2 × y=213 × x=924, float64 → stored linear16 (uint16)
  why      stored at 16 bits across each channel's range (linear16): the rounding step is at most
           0.00013 of the typical difference between neighbouring pixels (limit 0.1), so nothing
           visible or measurable is lost. Pass lossless=True to keep the float64 values exactly.
  size     258.5 kB on disk: pixels 130.2 kB (4% of the 3.1 MB raw), layers 88.0 kB, manifest + preview 40.4 kB
  layers   activity (intensity), footprints_weights (intensity), footprints (labels), dendrites
           (vector), discs (vector)
  regions  ROI 1, ROI 2, ROI 3
```

and returns a report with the same facts (`.encoding`, `.reason`, `.bytes`, `.raw_bytes`, `.pixel_bytes`,
`.layer_bytes`, `.other_bytes`, `.layers`, `.regions`, `.notes`) and the manifest it wrote. The percentage
compares the stored pixels (overview levels included) with the raw pixels; layers and the manifest and
preview are listed apart, so a small image with big layers does not look badly compressed. `help(fp.fluximg)`, `help(fp.imsave)` and
`help(fp.imread)` list every parameter.

Writing needs a zstd library: `pip install "fluxplot[images]"` (installs `zstandard`). Python 3.14
has one built in, and `numcodecs` or `imagecodecs` also work when already installed.

## Dimensions and axes

`dims` names every axis of the array: `y` and `x` (required), and optionally `t` (time), `c`
(channels), `z` (depth) and any other lower-case name (`trial`, `condition`). A 2-D array defaults to
`("y", "x")`; anything else needs `dims`. Give them in the array's own order. The bundle stores
them as extra dimensions first, then `t, c, z, y, x`, so a `(c, t, x, y)` movie is stored as
`(t, c, y, x)`, one plane at a time without copying the whole array.

| Keyword | Meaning |
|---|---|
| `pixel_size`, `unit` | The physical size of a pixel (one number, or `(y, x)`) and its unit. Without it the axes are in pixels. |
| `coords={"z": z_um}` | Positions along a dimension. Evenly spaced values become a scale and an offset; anything else is kept as explicit positions. Strings name the positions of a categorical dimension. |
| `dt`, `t0` | Seconds per frame and the time of frame 0. |
| `default_index={"z": 10}` | Where the image opens. Without it Flux opens at the middle z plane and the first position along everything else; `preview.png` shows the same plane. |
| `transform` | A pixel-to-world matrix, kept verbatim as `world.matrix`: 3 x 3 for a 2-D mapping of `(x, y, 1)`, 4 x 4 for `(x, y, z, 1)`. The image coordinates are continuous pixel coordinates, as for layers: pixel (column `i`, row `j`) covers `[i, i+1) x [j, j+1)`, so the first pixel's centre is `(0.5, 0.5)`, and `z` is the plane index (0 at the first plane). The result is in `unit`. Flux keeps it but does not use it yet. |
| `meta` | A dict kept in the bundle's provenance (subject, session, acquisition, …). |

An xarray `DataArray` works as `data`: its dims, evenly spaced or explicit coordinates, the
labels of its `c` coordinate (channel names) and each coordinate's `attrs["units"]` are used, and
explicit keywords win. Datetime coordinates become seconds from their first value, which is kept in
`meta` as `<dim>_start`.

## Channels

`channels` gives one entry per channel: a name, or a dict `{"name", "color" | "lut", "range", "visible"}`.
A `lut` that is a colour (`"magenta"`, `"#ff2bd6"`) means the black-to-colour ramp; anything else must
be a colour map Flux can draw: a name from fluxplot's collections (`"gray"`, `"viridis"`, `"magma"`,
`"cmasher.ember"`, `"crameri.batlow"`, any with `_r`), or a matplotlib `Colormap` whose name is one of
those. A colour map registered only in your Python session, or a `Colormap` made on the spot, is
refused when the image (or layer) is built, with a message saying what to give instead, so nothing is
written that Flux could not show. `range` is the default display window `(black, white)`.
`"visible": False` makes the channel start hidden in Flux (it is still stored, and one click shows it);
the preview leaves it out.

Default colours: one channel is gray; two are green and magenta; three are gray, green and magenta,
the trio that stays distinguishable for deuteranopes, protanopes and tritanopes. Four or more continue
with cyan, azure and red: no set of four or more additive colours stays apart for every colour-vision
deficiency, so the first three keep their colours when a channel is added, and for four or more
channels it is worth choosing the colours yourself. The default window is the 1st to 99.8th percentile. Every channel
also carries statistics (exact min and max, percentiles and a 256-bin histogram), so Flux can draw
a histogram and set contrast without reading pixels. `photometric="rgb"` marks a true-colour
photograph: three channels shown as red, green and blue and adjusted together.

## How the pixels are stored

Pixels are stored in **chunks of one plane** (planes over 1024 px are cut into 512 px tiles and also
get 4x, 8x, … overview levels for fast zooming), compressed with zstd. Chunks without data are not
stored at all.

- **Integer data is stored exactly**, always, in its own type (`int64` and the like in the
  narrowest integer type that holds every value). `nodata=` names the stored value that means "no
  data".
- **Float data is stored at 16 bits across each channel's own range** (`linear16`) when that is
  provably harmless: one rounding step must be less than a tenth of the typical difference between
  neighbouring pixels, for every channel. That is true of any real noisy image, and it halves the
  size of float32 data (quarters float64) before compression. Otherwise the floats are kept exactly:
  a constant or piecewise-constant image, or one whose range is stretched by a few extreme values,
  stays exact, and the report says so and why.
- **`lossless=True`** keeps the floats exactly, always. `nodata=` on float data turns that value into
  missing (NaN).

`fp.imread` returns values: exact data in its stored type (NaN where a float image has no data),
`linear16` data decoded to float64. A decoded value is within half a step of what was written.

## Layers

Layers share the image's `(y, x)` pixel grid. Each has a lower-case id, an optional `name`, and
whether it starts `visible`.

**`add_image(id, data, lut=, range=, opacity=, visible=False, lossless=False)`**: another image,
such as an activity map or a mask. `opacity` is a number 0..1 or `"value"` (opacity follows the
normalized value). Stored by the same rule as the pixels.

**`add_labels(id, labels, background=0, table=, key=, weights=, color=)`**: objects as an
integer map. `background` is the value meaning "no object"; every other value is the object's
label and must be positive (objects are numbered from 1). Flux gets one exact outline per object,
following pixel edges, holes included. `table` (a polars or pandas DataFrame, or a dict of columns)
has one row per object, and `key` names the column that names an object (what Flux shows). The
label column is `label`, or else the `key` column when it holds the label numbers
(`table=sources, key="source_id"`). Without a table, one is made with `label`, `pixels`, `cx`, `cy`.
`weights` (a float map) becomes the intensity layer `<id>_weights`, stored exactly.

The join between the map and the table is checked so it cannot silently go wrong by one: a table row
for label 0 (or the background value, or below) is refused, because label 0 means "no object". That
usually means a 0-based id column was used as the labels of a map numbered from 1; the message says to
add a `label` column holding the map's values (`table["label"] = table["source_id"] + 1` when the map
is that id plus one) and to keep `key=` naming your own id column. Every object in the map must have a
row; rows whose label has no object are kept, and the report notes how many there are.

**`add_shapes(id, lines=, circles=, polygons=, coords="xy", diameter=, radius=, key=, table=)`**:
vector shapes in image pixels.

- `lines` / `polygons`: a list of `(N, 2)` arrays, a dict key → array, or a list of dicts with
  `"vertices"` plus other fields (the fields become table columns).
- `circles`: a DataFrame, dict of columns or list of dicts whose centre columns are
  `center_row`/`center_col`, `row`/`col`, `cy`/`cx` or `y`/`x`; or an `(N, 2)` array of centres.
  `diameter` / `radius` is one number, one per circle, or a column name, in the image's physical
  unit when it has a pixel size (an ellipse when pixels are not square), else in pixels.
- `coords="row_col"` means array indices: an integer vertex addresses a pixel centre, so 0.5 is
  added and the pair is swapped to x, y. `"xy"` means pixel coordinates as given. Centre columns
  named `row` / `col` are array indices whatever `coords` says.
- `key` names the field that identifies a shape. When there is exactly one text column (besides
  the centres) it is found on its own.

Label and vector layers given no `color` take distinct colours in turn: yellow, orange, white, coral,
lavender (then again), none of which is a default channel colour. An `add_image` layer defaults to
magenta; give it a colour that is not one of the channels'.

**Borrowing layers.** `save(..., layers="plots/mean")` (or `fp.imsave(..., layers=...)`) copies
every layer of another bundle into this one, recording where it came from. That is how a movie of
one trial gets the footprints and dendrites drawn on the mean image. It is refused when the two
pixel grids differ. With `regions="auto"` the other bundle's regions come along too, so the movie and
the mean image name the same rectangles.

## Regions

`img.add_region("Cell body", (x, y, width, height))` names a rectangle in pixels. By default
(`regions="auto"`), when the image has missing data, such as a scan of a few areas, each connected
area that has data becomes a region `ROI 1`, `ROI 2`, … from left to right. Areas smaller than
`region_min_pixels` (default 64 pixels, an 8 x 8 patch) are ignored as stray specks, an area whose box
is the whole image is never a region, and nothing is made when more than 64 areas remain. When
`layers=` borrows another bundle's layers, its regions are used instead (see above). `regions=None`
turns this off.

## Saving and reading

`save` appends `.fluximg` when the path lacks it. An existing bundle is replaced safely: everything
is written to a temporary sibling directory, `fluximg.json` last, and swapped in, so a failure
leaves the old bundle untouched. A directory that is not a fluximg bundle is never replaced.
`recipe=True` (or a dict, as for `fp.save`) writes `<stem>.recipe.json` beside the bundle so Flux
can regenerate it. `FLUXPLOT_ONLY` filters saves as it does for `fp.save`. `zstd_level` trades
time for size (default 5).

`fp.imread(path)` reads it back: `.values` (full resolution, or an overview with `level=1, 2, …`),
`.dims`, `.shape`, `.axes`, `.channels` (names), `.manifest`, `.layer(id)` (a raster layer's
array), `.table(id)` (a layer's table as a dict of columns; `pd.DataFrame(img.table("footprints"))`)
and `.to_xarray()`. `fp.imread(path, layer="footprints")` returns just that layer's array.

## Large stacks

Planes are streamed: one pass learns the image (range, statistics, the encoding check, profiles
along every extra dimension), a second compresses and writes it, with chunks compressed on several
threads and written straight into their shard files. A 1.28 GB float32 movie of shape
`(2, 816, 924, 213)` writes in about 2 s with well under 100 MB of memory beyond the array itself.

No file in a bundle is much over 64 MB, whatever the order of the dimensions, so a bundle can live in a
git repository (GitHub refuses files over 100 MB). The chunks of a stack are grouped into shard files
by the rule in Flux's `SPEC.md` 3.3: a file always holds whole planes, and as many of the last
dimensions (z before c before t …) as fit, judged from the compressed size of the first planes. A
channel-first `(3, 25, 2048, 2048)` uint16 confocal stack, for example, is written as runs of z planes
of one channel, not one file per channel.

## What is on disk

```
plots/mean.fluximg/
  fluximg.json          the manifest: dims, axes, channels and their statistics, layers, regions
  pixels/               a Zarr v3 group (OME-Zarr 0.5 metadata when every dimension is t, c, z, y or x)
    0/                  full resolution: sharded, zstd-compressed chunks
    1/, 2/, …           overviews of large planes
  layers/<id>/          a raster layer (Zarr), and layers/<id>.svg / <id>.table.json for outlines,
                        shapes and tables
  preview.png           a small picture of the default view (8-bit gray when that view is one gray
                        channel, else RGB; with alpha where there is no data)
plots/mean.recipe.json  optional: how to regenerate it
```

The format is **fluximg/1**, specified in Flux's `docs/fluximg/SPEC.md`. Any Zarr v3 reader that
supports sharding can read the pixels; the manifest says how stored numbers become values.
