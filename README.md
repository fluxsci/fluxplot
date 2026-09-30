# FluxPlot

*matplotlib, but every meaningful thing has a name.*

FluxPlot is a thin, additive layer over [matplotlib](https://matplotlib.org/). You keep plotting in
real matplotlib; FluxPlot records **what each mark means** as you draw it, and `fluxplot.save()` turns
your figure into a **semantic SVG** — a plot whose every part (a series' line, its 4th data point, the
x-axis title) is individually named, addressable, and restylable, with a sidecar that knows the data
behind every pixel.

This is the keystone format of the **Flux** ecosystem. The same file is, without compromise, a
publication figure, an animatable slide element, a reproducible artifact, and an object an AI agent
(or you, in a terminal) can point at precisely.

---

## The problem it solves

When matplotlib (or any plotting library) exports an SVG, it produces a flat soup of anonymous shapes:

```xml
<g id="line2d_7"><path d="M 46 160 L 96 97 …"/></g>
<g id="PathCollection_1"><use x="46" y="179"/><use x="147" y="141"/> …</g>
```

A human looking at the rendered picture knows that one path is "the control group" and that dot is
"the treatment value at hour 12." **The file does not.** That meaning existed at *plot time* — your
code had the arrays, knew the series names, knew the axis was log-scaled — and then it was thrown away
the instant the figure was flattened to pixels and anonymous geometry.

Everything you might want to do later needs that meaning back:

- **Edit one part** of a figure ("make the control line red, the axis labels 8pt") — you have to know
  which shape is which.
- **Animate it** ("draw the lines, then stagger the points in") — you have to address parts by role.
- **Morph it** ("the data moves from the t-test version to the Mann-Whitney version") — you have to
  interpolate in *data* space, which means knowing each point's value and the axis scale.
- **Ask about it** ("what's the peak treatment value?") — you need numbers, not pixel coordinates.

You cannot reliably recover any of this from a flattened SVG. Colors collide, log axes look linear,
two points at the same pixel are indistinguishable. **The only place the meaning is knowable for
certain is at the moment the plot is drawn.** So that is where FluxPlot captures it.

> **Principle 1 — meaning is captured at birth, never reverse-engineered.** FluxPlot tags marks *as
> your code draws them*, while the program still knows what they are. This single idea rules out the
> whole category of "post-process an existing SVG with heuristics" — by then it's already guesswork.

---

## Install & quickstart

```bash
pip install fluxplot          # depends on matplotlib, numpy, lxml, jsonschema
```

```python
import matplotlib.pyplot as plt
import fluxplot as fp

t          = [0, 4, 8, 12, 16, 20, 24]
control    = [0.02, 0.05, 0.13, 0.41, 0.95, 1.6, 1.9]
treatment  = [0.02, 0.07, 0.25, 0.80, 1.5, 1.95, 2.1]

fig, ax = plt.subplots(figsize=(6.4, 4.8))
fp.line(ax, t, control,   series="control",   marker="o", label="Control")
fp.line(ax, t, treatment, series="treatment", marker="s", label="Treatment")
ax.set_xlabel("Time (h)"); ax.set_ylabel("OD600"); ax.set_yscale("log"); ax.legend()

fp.significance_bracket(ax, x0=20, x1=24, y=2.0, label="**",
                        between=("control", "treatment"), p=0.003)

fp.save(fig, "plots/growth.svg")
```

You write **ordinary matplotlib** — `fp.line` is `ax.plot` with a `series=` name attached. `fp.save`
then produces three files. The recipe is rerunnable with zero ceremony: when called from a `.py`
script, `save` discovers the producing script automatically (deterministic, conservative rules —
never notebook history, never a guess) and records how it was discovered
(`provenance.scriptDiscovery`: `automatic` / `explicit` / `unavailable`). Pass
`recipe=dict(script=..., params={...}, inputs=[...])` to record parameters and input hashes —
explicit fields always win — or `recipe=False` to suppress discovery entirely (notebooks,
privacy-sensitive callers). Inputs are **never** discovered automatically.

```
plots/growth.svg            ← the semantic SVG (renders & edits like any vector, but every part is named)
plots/growth.fluxplot.json  ← the manifest (the data + coordinate mapping + build order behind the picture)
plots/growth.recipe.json    ← the recipe (how it was made — script, params, input hashes)
```

That's the whole library surface for most users: name your series, call `save`.

---

## One plot, four lives

From that **one source**, four jobs are served without compromise:

| Job | What consumes it | What it needs that a flat SVG can't give |
|-----|------------------|------------------------------------------|
| **Publication figure** | Flux Figure, Illustrator, Inkscape | edit a part by name; restyle a whole role ("all axis titles → Helvetica 8pt") |
| **Animated slide** | Flux Slide | reveal/morph parts *by role and identity* ("draw every line, stagger every point") |
| **Reproducible artifact** | the recipe + your code | rerun with a different test; regenerate; the references survive |
| **Agent-addressable object** | an AI agent, a script, the CLI | "the control series' 4th point" = a real, resolvable handle |

The design rule is strict: *if a choice helps one job but breaks another, it's wrong.* That's why the
output is built on SVG (vector **and** a structured, web-native, universally-interoperable tree) rather
than a raster, a canvas, or a private JSON scene graph.

---

## The three files (and why three)

### `growth.svg` — the renderable, addressable truth
A completely normal SVG: open it in any browser or vector editor and it just works (it never *depends*
on the manifest — degrade gracefully, always). What makes it *semantic* is that FluxPlot has added, to
each meaningful element:

- a **stable, meaningful `id`** — `control.line`, `control.point.3`, `axis.x.title`, `legend`;
- **`data-*` attributes** carrying role + identity (+ the datum value, for convenience):

```xml
<g id="control.line" data-role="line" data-series="control"><path d="…"/></g>
…
<use id="control.point.3" data-role="point" data-series="control"
     data-index="3" data-x="12" data-y="0.41" x="182.05" y="103.5"/>
```

Text stays as real, editable `<text>` (axis titles, tick labels) — not outlined to paths — so it can be
restyled and read.

### `growth.fluxplot.json` — the semantic index
A sidecar that **points into the SVG by id** and adds everything SVG can't naturally express. It never
duplicates geometry (no path data, no bounding boxes — those live authoritatively in the SVG). It
holds:

```jsonc
{
  "axes": [{
    "x": { "scale": "linear", "domain": [-1.2, 25.2],
           "anchors": [{"data": -1.2, "svg": 57.6}, {"data": 25.2, "svg": 414.72}] },
    "y": { "scale": "log", "base": 10, "domain": [0.0158, 2.6767],
           "anchors": [{"data": 0.0158, "svg": 307.6}, {"data": 2.6767, "svg": 41.5}] }
  }],
  "series": [{ "id": "control", "kind": "line", "svg": {"line": "control.line", "points": "control.points"},
               "data": { "x": [0,4,8,…], "y": [0.02,0.05,0.13,…] },
               "points": [ … {"index": 3, "svgId": "control.point.3", "x": 12, "y": 0.41} … ] }],
  "build": { "order": ["axis.x","axis.y","gridlines","control.line","treatment.line",
                       "control.points","treatment.points","legend","significance-bracket.0"],
             "presets": { "line": {"animation": "draw-on"}, "point": {"animation": "stagger-in"} } }
}
```

This is the file an agent or Flux Slide reads *first* — it's where the *meaning* lives.

### `growth.recipe.json` — the provenance
The script (auto-discovered, or recorded explicitly), the parameters, references (+ hashes) of the
input data, and a `provenance` block (script hash, interpreter, package versions, git state when
available). Enough to **re-run the plot here** — which is what makes "rerun Figure 6d with a
Mann-Whitney test" a real operation. All host-varying material lives here, never in the SVG/manifest.

A plot made in a notebook cell has no script to re-run. `fp.save(..., recipe={"notebook": path,
"cell": "fig-growth"})` records it honestly: `provenance.scriptDiscovery: "notebook"` and a
`notebook: {path, cell, sha256}` block, with no `command`. In a live kernel the notebook is
detected from what the host states outright — `$QUARTO_DOCUMENT_PATH`, or the
`__vsc_ipynb_file__` / `__session__` globals — and never guessed from the working directory.

### Consistency between the three files
The manifest records `artifact.svgSha256` — the checksum of the final SVG bytes. `save` stages all
three files and commits them with atomic per-file renames (SVG → manifest → recipe), so a watcher
never sees a partially written file, and a stale SVG/manifest pair is *detectable* via the checksum
rather than silently misinterpreted.

**Why split SVG and manifest?** Because they answer different questions in the representation each is
good at. The SVG answers *"how does it look and which part is which?"* — that belongs in a vector
tree. The manifest answers *"what does it mean and how does it move?"* — nested numeric data,
coordinate transforms, choreography — which is miserable to cram into SVG attributes and natural as
JSON. The authority rule keeps them from drifting: **geometry is authoritative in the SVG; data,
coordinate-mapping, and choreography are authoritative in the manifest; identity + role appear in both
as the join key.**

---

## How it actually works (under the hood)

FluxPlot rides matplotlib's one real hook from "an artist" to "a named SVG element":
`artist.set_gid("control.line")` makes matplotlib's SVG backend wrap that artist's output in
`<g id="control.line">…</g>`. The pipeline in `save()` is:

1. **Assign deterministic gids.** Walk the marks you tagged and give each a semantic id (`ids.py`).
2. **Auto-tag the scaffold.** Name the axes, tick labels, legend, and title so you never have to.
3. **Capture coordinates.** Read each axis's real transform and record the data↔pixel mapping + scale
   (see below).
4. **Render deterministically.** Save to SVG with the determinism knobs set (see below).
5. **Inject `data-*` + canonicalize.** matplotlib writes only `id`, never arbitrary attributes — so a
   post-render pass (`postprocess.py`, via lxml) finds each element **by the id we ourselves set** and
   adds `data-role`/`data-series`/… , splits the points group into addressable per-point elements, and
   strips volatile metadata.
6. **Emit the manifest and recipe.**

> **"Isn't step 5 the reverse-engineering you said was a sin?"** No — and the distinction is the whole
> point. Reverse-engineering means looking at an anonymous `<path>` and *guessing from its color or
> shape* that it's the control line. Here, **we set `id="control.line"` before rendering**, so the
> post-pass is an *exact structural join on an id we authored* plus annotation with values our code
> already had. Nothing is inferred from pixels. It's serialization, not divination.

### Per-point addressability
"The 4th point of the control series" needs each point to be its *own* element. matplotlib draws a
set of markers as one artist: a marker glyph defined once in `<defs>`, then one `<use>` per point in
data order. FluxPlot's post-pass enumerates those `<use>` children and stamps each with
`id="control.point.k"` + `data-index`/`data-x`/`data-y`. (If matplotlib ever culls off-axis points and
the count stops matching the data, FluxPlot detects the mismatch and keeps the group addressable
rather than mislabel indices.)

### Heavy layers are rasterized by default
An artist becomes as many SVG nodes as it draws primitives. A `LineCollection` built from per-edge
segments — the ordinary way to draw an SWC reconstruction or a graph — emits **one `<path>` per
segment**, and a `scatter` emits **one `<use>` per point**. At real data scale that is 10⁴–10⁵ nodes
in a single panel, and consumers inline that markup as live DOM, where it is ruinous. (Measured: a
14-panel figure carrying three neuron reconstructions and 8.7k-point scatters reached 260,907 nodes
and ~390 ms per pan frame — about 2.5 fps.)

So `fp.save` rasterizes any artist over `raster_threshold` primitives (default 800) into a single
embedded `<image>` at `raster_dpi` (default 600), and says so:

```
fluxplot: 'medoid' — rasterized 2 heavy layer(s) at 600 dpi: axon.x-morphology (72,586 primitives),
dendrite.x-morphology (4,199 primitives). Axes, ticks, labels and legend stay vector.
Pass force_vectors=True to keep everything as vectors.
```

**Only the heavy layer is rasterized.** Axes, spines, ticks, tick labels, the legend, annotations and
every lighter series stay fully vector and fully editable — this is matplotlib's own `set_rasterized`
applied per artist, which is what journals expect for dense scatter and line art anyway. The
rasterized layer **keeps its id, its `data-role`/`data-series` and its manifest entry**, so it stays
addressable as a whole; only *per-point* ids are unavailable, because a rasterized cloud has no
per-point nodes. `SaveResult.rasterized` lists what was rasterized and the manifest marks those
entries `"rasterized": true`.

On a real morphology panel: **13.42 MB / 76,852 nodes → 0.06 MB / 67 nodes**, rendering
pixel-indistinguishable (mean channel difference 0.11/255).

Opt out with `fp.save(..., force_vectors=True)` or `FLUXPLOT_FORCE_VECTORS=1`; the heavy layers are
then still reported, as a warning naming them and their node cost. A per-artist
`set_rasterized(False)` does **not** override the default — that is matplotlib's factory setting
rather than a considered choice, and silently emitting an unusable SVG is the failure this exists to
prevent. `force_vectors` is how you say you meant it.

### Determinism — the same plot always produces the same bytes
This is non-negotiable, because **morphing, diffing, and regeneration all break if ids or structure
wobble between runs.** FluxPlot pins the sources of nondeterminism:

- `svg.hashsalt` is fixed (otherwise matplotlib salts every clip-path / glyph id with a fresh UUID);
- `svg.fonttype='none'` keeps text as `<text>` (the default outlines glyphs into font-version-dependent
  path data, *and* makes labels un-restylable);
- the SVG date metadata is dropped, path-simplification is pinned, and the JSON is written canonically
  (sorted keys, fixed float precision).

Result: same input → byte-identical SVG and manifest. Timestamps and content hashes (which *must*
vary) live only in the recipe, so the SVG/manifest stay stable for diffs and morphs.

### Coordinate capture — why the manifest stores "anchors"
A morph that moves a point from value 2 to value 8 on a **log** axis must travel correctly, which is
impossible from pixels alone. So per axis FluxPlot stores the scale type and two `(data, svg)` anchor
pairs. A consumer interpolates between them — linearly, or in log space for a log axis — and gets the
exact pixel position for any data value, **without ever reconstructing matplotlib's transform.** Two
anchors + a scale type is the complete, portable contract. (FluxPlot computes them with a
dpi-invariant fraction method that round-trips *exactly* against the positions matplotlib actually
emits — verified in `NOTES_matplotlib_svg.md`.)

---

## Semantic IDs: the universal join key

Ids are **derived from meaning**, not random — a dotted path of `[a-z0-9-]` segments:

```
control.line          control.point.3        axis.x.title       axis.x.tick.2
legend                annotation.peak        reference-line.threshold        panel.a
```

The same id is simultaneously the SVG `id`, the manifest key, the handle a caption or animation step
refers to, and a coordinate in "meaning space" (`role=point, series=control, index=3`). Making it
**deterministic** is what lets four different operations work:

- **Durable references** — a restyle, a caption, or an animation attached to `control.line` survives the
  plot being **regenerated with new data**: the line is still "the control line" even though its shape
  changed, because it has the same id.
- **Morphing** — match parts across two versions of a plot by id, then tween.
- **Legible diffs** — a regenerated plot produces a meaningful git diff instead of noise.
- **Addressing** — humans and agents name a part the same way the file does.

(matplotlib's own ids use underscores and hex hashes and never contain dots, so FluxPlot's dotted
namespace is provably disjoint from anything it autogenerates.)

Two orthogonal axes of labeling make this work: **role** = *what kind of thing it is* (`line`,
`point`, `axis-title`) and **identity** = *which one* (`series=control`, `index=3`). Animation targets
by role ("draw every `line`"); a caption targets by identity ("the `treatment` series"); a journal
restyle targets by role across the whole figure ("every `axis-title` → 8pt").

---

## The role vocabulary

The part names aren't ad hoc — they're the **Grammar of Graphics** (the model behind ggplot2 and
Vega-Lite): a plot is *data → marks + scales + guides + annotations*. FluxPlot's v0 core roles:

- **Containers:** `figure`, `panel`, `plot-area`, `legend`, `colorbar`, `title`
- **Scaffold / guides:** `axis`, `spine`, `tick`, `tick-label`, `axis-title`, `gridline`, `background`
- **Data marks:** `series`, `line`, `point`, `bar`, `area`, `errorbar`, `box`, `violin`
- **Composite sub-parts:** `whisker`, `cap`, `flier`, `median`, `mean`, `segment`, `regions`
- **Overlays:** `annotation`, `reference-line`, `highlight-region`, `significance-bracket`, `label`

Science is unbounded (heatmaps, networks, brain maps), so the vocabulary is a **versioned core plus a
namespaced extension mechanism**: anything unrecognized becomes an `x-…` role. An unknown role still
gets a stable id and still renders and is still addressable — at worst it's a clean, tagged, editable
figure. The standard only ever *adds* power; it never makes a plot worse than a plain SVG.

---

## The API

Two styles, both pure matplotlib underneath — you can mix them freely with raw matplotlib calls.

**Convenience helpers** (auto-tagging) — each returns the real matplotlib artist(s):

```python
fp.line(ax, x, y, *, series, marker=None, label=None, **mpl_kwargs)
fp.scatter(ax, x, y, *, series, label=None, key=None, **mpl_kwargs)   # c=values → a colour scale
fp.bar(ax, x, height, *, series, **mpl_kwargs)
fp.errorbar(ax, x, y, *, series, yerr=None, **mpl_kwargs)
fp.area(ax, x, y1, y2=0, *, series, **mpl_kwargs)                    # manifest band = {x, y1, y2}
fp.image(ax, data, *, series, pixel_size=None, units="µm", channels=None, luts=None, display_range=None, ...)
fp.scalebar(ax, length, units="µm", *, loc="lower right", label=None, color=None, thickness=2.0, pad=0.4)
fp.band(ax, x, lo, hi, *, series, what="95% CI", **mpl_kwargs)        # <series>.band beside <series>.line
fp.box(ax, values, *, series, label=None, include_values=False, **mpl_kwargs)     # one box per call
fp.violin(ax, values, *, series, label=None, include_values=False, **mpl_kwargs)  # one violin per call
fp.hist(ax, values, *, series, bins=None, label=None, include_values=False, **mpl_kwargs)
```

**Images and scale bars** — a micrograph is data too:

```python
im = fp.image(ax, np.stack([dapi, gfp]), series="cells", channels=["dapi", "gfp"],
              luts=["blue", "green"], display_range=[(0, 900), None],    # None → 1st–99.8th percentiles
              pixel_size=0.325, units="µm", composite="add")           # or "max"; value_raster=True
fp.scalebar(ax, 10, "µm")                                              # a vector bar, exactly 10 data units
fp.colorbar(im.mappables["gfp"], ax=ax, name="gfp", label="GFP")
```

`fp.image` takes a `(H, W)` image or a `(C, H, W)` / `(H, W, C)` stack, gives every channel a LUT
(a colormap name, or a colour meaning a black-to-colour ramp) and a display range (the black and
white points), composites them into one RGB `imshow` in Python, and puts the axes in physical
units from `pixel_size`. Each channel is a colour scale in the manifest (`cells.dapi`,
`cells.gfp`; a 2-D image's is just `cells`) with a linear norm over its display range and
`recolor: "regenerate"` (`"raster"` with `value_raster=True`, when the channel's values travel
beside the SVG), and a recipe control of the same name — so Flux's colour-scale editor adjusts a
channel's brightness/contrast or LUT and reruns. The series' `image` payload records the channels,
LUTs, display ranges, pixel size, units, extent and composite. `fp.scalebar` is a `Line2D` whose x
extent is exactly `length` data units, anchored in a corner (`loc`), with its label
(`"<length> <units>"`) as `scalebar.<n>.label`; it takes the theme's ink. `examples/image_example.py`
draws a two-channel field.

`fp.band` is the uncertainty band of a line: registered under the **same series** (so `ctl.band`
sits beside `ctl.line`), in the line's colour at `alpha=0.25`, with `band = {x, lo, hi, what}` in
the manifest saying what it is (`"95% CI"`, `"SEM"`, …) — a consumer can re-fit or re-label it
from the inputs rather than from polygon vertices. `fp.area` records `{x, y1, y2}` likewise.

`fp.box`/`fp.violin`/`fp.hist` wrap matplotlib's documented return structures, so every statistic
is individually addressable and grouped per series (`control.whiskers`, `control.medians`, …).
`fp.hist` records the exact `binEdges`/`counts` in the manifest's `distribution` payload; raw
sample values are recorded only with `include_values=True` (samples can be large or sensitive).

**Surface maps** — a value per mesh vertex, drawn as a set of views and kept addressable:

```python
fp.surface(ax, values, *, series, surfaces, kind="categorical"|"continuous", ...)
```

`surfaces` is a `{hemisphere: (vertices, faces)}` mapping, or paths to GIFTI files (read with
`nibabel`, an optional dependency). `values` is one value per vertex. A categorical map draws **one
collection per category**, so every region becomes a separately addressable, separately recolourable
part (`blocks.regions`, or `blocks.<category>` by name) with a matching legend key; a continuous map
draws one field plus a colorbar, and the manifest records the complete value→colour contract —
`cmap`, `color_range`, the category→colour table, and which vertices were treated as missing. Views
(`lateral`, `medial`, …) and hemispheres are laid out side by side inside the one axes.

Vertices with no data are an explicit `missing` part rather than a value, so an on-mesh zero stays
distinguishable from absent data and sentinel codes grey out instead of becoming a spurious category.
Rendering is orthographic with back-face culling — a fold cannot paint over the surface in front of
it — with optional Lambertian `shading` for relief; a face straddling a boundary takes the majority
label rather than being dropped.

**Signature fluxplots** — complete, opinionated plot types that are unique to Flux. They take a
DataFrame (pandas, polars or a dict of columns) plus column names, seaborn-style, and name every
part for you. The first is the **glowbar**:

```python
gb = fp.glowbar(df, x="condition", y="APP/GAPDH", units="subject", ax=ax)
gb = fp.glowbar(df, x="condition", y="signal", units="mouse",          # paired / repeated measures
                connect_identical_points_across_x_values=True, ax=ax)
```

Every observation is a dot; beside each group sits a slim bar that *glows* — its ink densest at
the centre of the distribution and fading to hard caps at the interval ends — with the **mean** as
a heavy, haloed line across the bar and the **median** as a V-notch cut into both of its edges.
The interval is mean ± SEM by default, glowing around the mean; `interval="iqr"` spans the box of
a box plot and glows around the median, `"sd"` gives mean ± SD, and a callable
`values -> (low, high)` is accepted. Default groups use ColorBrewer's `YlGnBu`, then `YlOrRd`.

With `units=` every unit (animal, subject, cell) keeps a **fixed lane and colour derived from the
table, never from the values**, so plots of different measures made from one table agree dot for
dot — even when a unit is missing from one of them. Unit colours are equal *perceptual* steps
(CAM02-UCS) of each group's ColorBrewer map between `shade_range=(88, 22)` lightness, dealt across
lanes so neighbours always contrast (`interleave_shades=True`), and rimmed in a deeper shade of
themselves so the palest dots stay crisp (`point_edge="rim"`). Connectors are a quiet neutral grey
and break at a unit's missing category rather than bridging it.

| part | default id | role |
|---|---|---|
| interval glow | `<category>.glow` | `box` |
| interval caps | `<category>.caps` | `cap` |
| mean line | `<category>.mean` | `mean` |
| median notch | `<category>.median` | `median` |
| a unit's point(s) / connector | `<unit>.points`, `<unit>.point.<k>` / `<unit>.line` | `point` / `line` |
| points without `units` | `<category>.points`, `<category>.point.<k>` | `point` |

Each category series carries a `glowbar` manifest payload with the exact statistics drawn (`n`,
`mean`, `median`, `sd`, `sem`, `q1`, `q3`, `interval`, `low`, `high`, `center`, `x`, `groupColor`),
and each unit series its identity (`units`, `unit`, `categories`, `colors`); the manifest's
`plotType` is `"glowbar"`. Series names default to the category / unit values — `series=` and
`unit_series=` (a mapping or a callable) rename them. Every visual choice is a keyword:
`bar_width`, `bar_offset`, `bar_side` (`"outer"` — the default: the first category's bar to
the left of its points, every other to the right — or `"left"` / `"right"` for all), `glow_steps`, `glow_alpha`, `mean_line_width`, `mean_color`,
`mean_halo_width`, `median_notch_depth`, `median_notch_height`, `cap_width`, `cap_color`,
`show_mean`/`show_median`/`show_caps`/`show_individual_points`, `point_size`, `jitter`,
`point_edge`, `point_fill_alpha` (fill only — the rim stays opaque), `palette` (per category: any
`fp.colors.maps` colormap such as `"cmasher.emerald"`, any `fp.colors.palettes` palette such as
`"brewer.Set2"` / `"tol.bright"`, a matplotlib map, a list of colours or one colour — glowbar picks
as many distinct point colours as it needs plus a solid group colour), `group_color`, `group_color_position`, `point_colors`, `shade_range`,
`interleave_shades`, `connect_line_width`, `connect_color`, `connect_alpha` (see
`help(fp.glowbar)`). It returns a `GlowbarResult` (`.ax`, `.categories`, `.stats`, `.group_colors`,
`.point_colors`, `.series`, `.unit_series`, `.artists`). `examples/glowbar_example.py` draws both
designs.

The **fluxbox** is the glowbar with a box plot for its summary — the same call, the same lanes,
colours, connectors and names, so the two can be swapped for one another dot for dot:

```python
fb = fp.fluxbox(df, x="condition", y="APP/GAPDH", units="subject", ax=ax)
```

Beside each group sits a slim box (Q1–Q3), a half-strength wash of the group colour
(`box_alpha=0.5`). The **median** is a solid line across it in the group's own hue — deepened, or on
a dark background lifted, only as far as it takes to differ from the box by `median_contrast=30`
units of perceived lightness, so it reads for any palette and theme; the whiskers (and fliers) share
that colour, so the box is the only wash. The **mean** is a V-notch cut
into both edges of the box — the glowbar's median notch; a mean outside the box (a strongly skewed
group) keeps its mark as the same two V's drawn solid, pointing in at the whisker. The capless
whiskers reach the most extreme observations within `whis` × IQR of the box (`1.5` — Tukey's rule,
exactly `plt.boxplot`'s whiskers), `"range"`, or a `(low, high)` pair of percentiles. Observations
beyond the whiskers are not drawn again as fliers — the points already show them — unless the
points are hidden (`show_fliers="auto"`).

| part | default id | role |
|---|---|---|
| box (Q1–Q3) | `<category>.box` | `box` |
| whiskers | `<category>.whiskers` | `whisker` |
| whisker caps (`show_caps=True`) | `<category>.caps` | `cap` |
| median line | `<category>.median` | `median` |
| mean notch | `<category>.mean` | `mean` |
| fliers (points hidden) | `<category>.fliers` | `flier` |
| points / connectors | as for the glowbar | `point` / `line` |

Each category series carries a `fluxbox` manifest payload with the exact statistics drawn (`n`,
`mean`, `median`, `sd`, `sem`, `q1`, `q3`, `iqr`, `whis`, `whiskerLow`, `whiskerHigh`, `outliers`,
`x`, `groupColor`); the manifest's `plotType` is `"fluxbox"`. The box keywords are `whis`,
`box_width`, `box_alpha`, `box_offset`, `box_side`, `median_line_width`, `median_color` (sets the
median outright), `median_contrast`, `mean_notch_depth`, `mean_notch_height`, `whisker_width`, `whisker_color`,
`cap_size`, `cap_width`, `cap_color`, `flier_size`, `cut_color` and
`show_mean`/`show_median`/`show_whiskers`/`show_caps`/`show_fliers`; every point, colour, connector
and naming keyword is the glowbar's. It returns a `FluxboxResult` with the same fields as a
`GlowbarResult`. `examples/fluxbox_example.py` draws both designs.

The **hexmatrix** tiles the plane with regular hexagons, each one a named part. One call covers a
point cloud's density, a 2D gradient of a third variable, and a matrix drawn on a hex lattice:

```python
hm = fp.hexmatrix(df, x="x", y="y", ax=ax, color="#4CB391", marginals=True)       # jointplot-style
hm = fp.hexmatrix(df, x="wake", y="nrem", ax=ax, xscale="log", yscale="log",      # log-log density
                  norm="log", identity_line=True, colorbar_label="Synapses per hexbin")
hm = fp.hexmatrix(df, x="ccf_x", y="ccf_z", ax=ax, aspect="equal", binwidth=0.15, # spatial, 0.15 mm bins
                  norm="log", colorbar_label="somata / hexbin")
hm = fp.hexmatrix(df, x="x", y="y", C="rate", reduce="median", ax=ax,             # a gradient of C
                  cmap="RdBu_r", center=0)
hm = fp.hexmatrix(matrix=weights, ax=ax, gap=0.08)                                # a hex lattice map
```

Hexagons are addressed by their lattice position: `row` counts up the y axis and `col` along x, so
`<series>.hex.<row>.<col>` names the same hexagon in every plot with the same `extent` and grid,
whatever the data. They are regular *on the page*. `aspect="auto"` locks the axes' box aspect so a
later layout pass can't squash them, and `aspect="equal"` bins in true data units. Log axes bin in
log space.

The colour can be a count, `stat="density"`/`"probability"`/`"percent"` (optionally `weights=`), or
a `reduce` of `C` (`mean`, `median`, `sum`, `min`, `max`, `std`, `count` or a callable). The scale is
set by `cmap`/`color` (a single-hue ramp), `norm` (`linear`/`log`/`sqrt` or any `Normalize`),
`vmin`/`vmax`, `robust` and `center`. Further keywords:

* `mincnt=0` draws the empty hexagons too;
* `sparse=k` draws points instead of hexagons where fewer than `k` observations fall;
* `show_points` overlays every observation;
* `gap`, `edgecolor`, `linewidth` and `orientation="flat"` shape the hexagons;
* `marginals`, `colorbar` and `identity_line` add furniture.

The colormap and colour limits are recipe controls, exactly as for `fp.heatmap`: Flux's Color scales
editor can change them and regenerate.

| part | default id | role |
|---|---|---|
| all hexagons | `<series>.hexes` | `x-hexbin` |
| one hexagon (`data-row`, `data-column`, `data-x`, `data-y`, `data-count`, `data-value`) | `<series>.hex.<row>.<col>` | `x-hex` |
| sparse / overlaid points | `<series>-points.points`, `….point.<k>` | `point` |
| identity line | `reference-line.identity` | `reference-line` |
| marginal histograms (own panels) | `<series>-x.bar.<k>`, `<series>-y.bar.<k>` | `bar` |

The series carries a `hexmatrix` manifest payload: the lattice (orientation, radius, aspect,
scales, extent), the statistic, and every hexagon drawn (`row`, `col`, `x`, `y`, `count`, `value`).
It also carries the usual `field` colour payload; the `plotType` is `"hexmatrix"`. The call returns
a `HexMatrixResult` (`.ax`, `.hexes`, `.bins`, `.cmap`, `.norm`, `.colorbar`, `.marginal_axes`,
`.points`, `.hex_id(row, col)`, `.lookup(x, y)`). A layer of more than 800 hexagons is rasterized
like any heavy layer; its hexagons stay in the manifest. Pass `force_vectors=True` to `fp.save` to
keep every hexagon addressable. `examples/hexmatrix_example.py` draws all five.

**Statistics** — `fp.stats` holds the tests behind the plots, one per branch of the house
statistics guidance. Each two-group test takes `(a, b)`, orients signs as `a - b`, and returns one
reporting row (a dict keyed by `fp.stats.REPORT_COLUMNS`: `sig_test_used`, `test_statistic_value`,
`p-value`, `p_corrected_holm`, `dof`, `effect_size_method`, `effect_size_value`,
`effect_size_95_CI`, then the appended `effect_size_ci_low` / `effect_size_ci_high` (numbers),
`n_a`, `n_b`, `n_total`, `groups` (the names compared), `alternative`, `p_corrected_bh` and
`dof_error`), ready to save as a CSV in the plot's `_stats` dissection:

```python
row = fp.stats.welch_hedges(sd_values, sleep_values, names=("SD", "sleep"))   # alternative="two-sided"
pl.DataFrame([row]).write_csv("plots/_dissections/app_gapdh/_stats/welch_ttest.csv")
```

| function | design | test | effect size + 95% CI |
|---|---|---|---|
| `welch_hedges` | independent, means | Welch's t-test | Hedges' g, non-pooled SD `sqrt((var_a + var_b) / 2)`; Bonett (2008) CI |
| `mann_whitney_cliff` | independent, ranks | Mann–Whitney U (`U` of `a`) | Cliff's delta; Newcombe (2006) Method 5 score CI |
| `paired_t_hedges` | paired, mean difference | paired t-test | Hedges' g_z (SD of the differences); exact noncentral-t CI |
| `wilcoxon_rank_biserial` | paired, ranks | Wilcoxon signed-rank (`W+`, zeros dropped) | Kerby's matched-pairs rank-biserial r; score CI |

The CIs target the population effect size, so they are never multiplied by Hedges' `J`. Both
rank-based CIs stay non-degenerate at complete separation (`delta` or `r` = ±1), which is common at
n = 6. Rank tests report `dof = None`. `tests/test_stats.py` pins each against scipy and against
the equation that defines its interval.

*Multiple comparisons* — a row on its own has `p_corrected_holm == p-value` (a family of one).
Holm's step-down correction needs every p-value in the family, so it is a separate pass over the
rows of the comparisons that belong together (e.g. every measure tested on the same animals in one
figure); extra keys such as a measure name ride along:

```python
rows = [dict(measure=m, **fp.stats.welch_hedges(sd[m], sleep[m])) for m in measures]
rows = fp.stats.holm(rows)   # fills p_corrected_holm across the family; order is kept
```

`fp.stats.holm_adjusted(p)` does the same for a bare array of p-values; `fp.stats.bh(rows)` /
`bh_adjusted(p)` fill `p_corrected_bh` with Benjamini–Hochberg (false-discovery-rate) values for a
screen of many measures. NaN p-values pass through both untouched.

*Three or more groups* — the omnibus tests return one row (`groups` lists every group; F tests
carry `dof` / `dof_error`), the post-hoc tests one row per pair, and `pairwise` runs any two-group
test above over the pairs of a `{name: sample}` family and corrects across them:

| function | test | effect size + 95% CI |
|---|---|---|
| `anova_oneway(*groups, names=, effect="eta2"\|"omega2")` | one-way ANOVA | η² or ω²; Steiger's noncentral-F CI |
| `welch_anova(*groups, names=)` | Welch's ANOVA | ω²; noncentral-F CI on Welch's dof |
| `kruskal_epsilon(*groups, names=)` | Kruskal–Wallis | ε² = H / (N − 1); seeded bootstrap CI |
| `rm_anova(table, subject, within, dv)` | repeated-measures ANOVA, Greenhouse–Geisser dof and p | partial η²; noncentral-F CI |
| `friedman_kendall(table, subject, within, dv)` | Friedman | Kendall's W; seeded bootstrap CI |
| `tukey_hsd(*groups, names=)` | Tukey HSD (family-wise p as is) | Hedges' g, pooled SD |
| `games_howell(*groups, names=)` | Games–Howell (studentized range on Welch dof) | Hedges' g, non-pooled SD; Bonett CI |
| `dunn(*groups, names=, adjust="holm")` | Dunn's rank-sum test after Kruskal–Wallis | Cliff's delta; Newcombe CI |
| `pairwise(test, groups, pairs=None, adjust="holm"\|"bh")` | any two-group test per pair | that test's |

`tests/test_stats_multi.py` pins them against pingouin / scikit-posthocs reference values.

*From rows to brackets* — `fp.brackets(ax, rows, positions=…)` draws one significance bracket per
post-hoc row and stacks them automatically: shortest span first, each bracket one `step` above the
data it spans and above every bracket it overlaps in x (multiplicative steps on a log axis), so
nothing crosses. `positions` maps group name → x; a glowbar / fluxbox result provides it, and
`gb.brackets(rows)` is the one-liner:

```python
rows = fp.stats.pairwise(fp.stats.welch_hedges, {"ctl": ctl, "drug": drug, "sham": sham})
gb = fp.glowbar(data=df, x="group", y="value", ax=ax)
gb.brackets(rows)                                   # *** / ** / * / ns from p_corrected_holm
gb.brackets(rows, label="p", ns=False, p_column="p_corrected_bh")   # "p = 0.003", drop ns pairs
```

`label` is `"stars"`, `"p"`, `"both"` or a callable on the row; `thresholds`, `top`, `step` and
`tip` shape the stack; extra keywords reach `fp.significance_bracket`. Each bracket's manifest
overlay carries `between: [a, b]`, `p` and a `stats` block — test, statistic, raw and corrected p,
correction, effect size and its CI, sizes — so the figure states exactly which test each star came
from.

**Labels are identity** — a *conventional* labeled plot needs no helpers at all. At save time,
raw artists carrying a public label (`ax.plot(..., label="Control")`, labeled `scatter`/`bar`/
`fill_between`) are promoted to named series with their exact artist data, marked in the manifest
with `capture: {identity: artist-label, data: artist}`. Nothing is ever guessed: private/absent
labels stay addressable as `extra.*`, duplicated labels decline with one actionable warning, and
role meaning (threshold? fit? band?) is never inferred from geometry or style — use the explicit
overlays for that.

**First-class overlays** (deliberately included because they're ubiquitous in science):

```python
fp.significance_bracket(ax, *, x0, x1, y, label, between=None, p=None)
fp.reference_line(ax, *, y=None, x=None, name)
fp.annotation(ax, *, name, text, **mpl_kwargs)
```

**The escape hatch** — tag *any* raw matplotlib artist, so you never lose matplotlib's full breadth:

```python
ln, = ax.plot(x, y, "--", color="k")          # plain matplotlib
fp.tag(ln, role="reference-line", name="model")

sc = ax.scatter(x, y)
fp.tag_points(sc, series="treatment")          # make an existing collection addressable per-point
```

**Seaborn in one line** — seaborn is matplotlib underneath, so `fp.tag_seaborn` inspects what a
seaborn axes-level call drew (`lineplot`/`scatterplot`/`barplot`/`histplot`/`kdeplot`/`regplot`) and
names it — mean lines → `line`, CI bands → `area`, points → per-point `point`, bars → `bar` (+ their
`errorbar`) — using a named plotting adapter (or explicit `series=[...]` in artist draw order):

```python
sns.lineplot(data=fmri, x="timepoint", y="signal", hue="region", ax=ax)
fp.tag_seaborn(ax, plot="lineplot")             # → {"parietal": ["line","area"], "frontal": [...]}
```

**Recipes for artists without a first-class helper** — `fp.tag` covers all of them; these are the
patterns that come up constantly in practice (copy them verbatim). Unknown roles like `x-heatmap`
degrade gracefully: they still get stable ids, a `data-role`, and a manifest entry.

```python
# Stackplot — one PolyCollection per layer, tagged as areas:
polys = ax.stackplot(x, series_a, series_b, labels=["A", "B"])
for poly, name in zip(polys, ["a", "b"]):
    fp.tag(poly, role="area", series=name)

# Heatmap (imshow or pcolormesh) — the whole image is one addressable mark:
im = ax.imshow(matrix, aspect="auto", cmap=fx.SEQUENTIAL)
fp.tag(im, role="x-heatmap", series="counts-by-decade")

# Hexbin — the PolyCollection is one mark (per-hex addressing isn't meaningful):
hb = ax.hexbin(w, h, gridsize=58, xscale="log", yscale="log", bins="log", mincnt=1)
fp.tag(hb, role="x-hexbin", series="artwork-density")

# Ridgeline (a fill_between + outline per row):
for i, (name, dens) in enumerate(rows):
    band = ax.fill_between(grid, offset(i), offset(i) + dens, alpha=0.8)
    fp.tag(band, role="area", series=f"ridge-{name}")

# Horizontal bars on a LOG x-axis — never anchor at 0 (log(0) serializes as a
# huge off-canvas coordinate; fp.save warns and flux validate-plot rejects it).
# Draw from 1 so the geometry is finite and the length still encodes count:
bars = ax.barh(ypos, counts - 1, left=1)
for i, p in enumerate(bars.patches):
    fp.tag(p, role="bar", series="classification-count", index=i)
ax.set_xscale("log")
```

**Export:**

```python
fp.save(fig, path, *, recipe=None, validate=True)
```

`save` auto-tags the axes/legend/title for you, so the *only* thing you normally add to a matplotlib
script is a `series=` on your plotting calls.

A **figure-level script** that `fp.save`s several plots stays fully rerunnable per-plot: with
`FLUXPLOT_ONLY=<name[,name…]>` in the environment (fnmatch patterns work), every non-matching
`save` becomes a no-op — nothing written, sibling triplets untouched on disk. `flux rerun-plot
<recipe> --only` sets it for you, so one script per figure and per-panel regeneration coexist.

---

## What downstream tools do with it

The file *is* the API — every consumer reads the same artifacts, no private state:

- **Flux Figure** inlines the SVG (so its tagged nodes are live, clickable DOM), lets you select a part
  and restyle it, and stores the override **keyed by semantic id** so it survives regeneration.
- **Flux Slide** reads the manifest's `build.order` + per-role presets to animate by role, and morphs
  between two versions by matching ids and interpolating in data space via the anchors.
- **A "journal style" pass** restyles a whole figure by role — the figure analogue of restyling
  citations with a CSL file.
- **An agent** reads the manifest to answer questions, or edits the recipe and re-runs to regenerate a
  panel — and because ids are stable, the caption, layout, and animation re-attach automatically.

---

## What FluxPlot is *not*

- **Not a new plotting API or grammar.** It rides matplotlib's; your full matplotlib knowledge applies.
- **Not a renderer or a matplotlib replacement.** It adds names; it removes nothing.
- **Not a figure compositor.** Its unit is *one semantically-tagged plot + its manifest + recipe*.
  It supports Matplotlib subplots in one exported plot. Composing independent plot files into a
  publication figure remains Flux Figure's job.

A small, sharp contract is adoptable and composable; a sprawling one rots.

---

## Repo layout & development

```
fluxplot/
  src/fluxplot/
    api.py            # public surface + the save() pipeline
    ids.py            # the semantic ID grammar (a public, long-lived contract)
    tagger.py         # the registry (artist → meaning) + scaffold auto-tagging
    render.py         # deterministic SVG rendering (the P5 knobs)
    capture.py        # data↔SVG coordinate anchors + scale capture
    postprocess.py    # lxml: gid-join, data-* injection, per-point split, canonicalize
    manifest.py       # assemble *.fluxplot.json
    recipe.py         # assemble *.recipe.json
    roles.py          # the role vocabulary (+ x- extensions)
    signature_fluxplots/  # preset plot types unique to Flux (fp.glowbar, fp.fluxbox, fp.hexmatrix)
    stats/            # tests behind the plots, returning reporting rows (fp.stats.welch_hedges, …)
    schemas/          # JSON Schemas for the manifest and recipe (validated on every save)
  examples/growth_plot.py     # the worked example above
  examples/glowbar_example.py # the glowbar signature plot, unpaired + paired
  examples/fluxbox_example.py # the fluxbox signature plot, unpaired + paired
  examples/hexmatrix_example.py # the hexmatrix: joint, log-log, spatial, gradient, lattice map
  tests/                      # determinism, the marker-DOM probe, ids, capture, schema
  NOTES_matplotlib_svg.md     # the verified matplotlib SVG mechanics this rides on
```

```bash
python -m pytest                 # determinism is byte-checked; the marker-DOM probe guards mpl upgrades
python examples/growth_plot.py   # writes examples/out/growth.{svg,fluxplot.json,recipe.json}
```

`test_determinism.py` renders twice and asserts byte-identical output; `test_marker_dom.py` asserts
matplotlib's marker SVG structure so an upgrade that changes it fails loudly instead of silently
corrupting per-point ids.

---

## Status

**v0, in active development.** The library generates semantic SVGs + manifests + recipes today, and
Flux Figure consumes them (inline render, part selection, restyle-by-part, semantic export). The
conceptual spec is `../Flux_SemanticSVG_Spec.md`; the verified matplotlib mechanics are
`NOTES_matplotlib_svg.md`.

## Scientific export and panels (schema 0.3)

Saving captures the current artist state: edit a line with `set_data`, remove an annotation,
then save again. Scientific numbers retain their precision; NaN and masked observations become
JSON `null` at their original indices. Missing line observations remain gaps. Nonfinite recipe
parameters are rejected before any existing output file is replaced. Normal figure closing and
garbage collection release the registry.

Use ordinary Matplotlib scalar inputs, categorical bars and datetime coordinates. `barh` shares
`bar`'s orientation/baseline metadata. Histograms record bin edges, heights, count/density,
weighting and cumulative settings; `counts` remains the legacy height key. Line markers inherit
size, color, opacity, z-order and `markevery`. Numeric exported coordinates use Matplotlib's
converted units, with display tick labels and date epoch information alongside them.

```python
fig, axes = plt.subplots(1, 2, layout="constrained")
for name, ax in zip(["baseline", "followup"], axes):
    fp.panel(ax, name)               # stable even when the layout changes
    fp.line(ax, [0, 1, 2], [1, 3, 2], series="control")
fp.save(fig, "comparison.svg")
```

A legacy single axes keeps IDs such as `control.line`. Multiple axes (including twins and
insets) use panel namespaces; explicit names produce `panel.baseline.control.line`. Automatic
names follow layout order. Name panels explicitly when identities must survive rearrangement.
Colorbars belong to their source panel and do not masquerade as additional plotting axes.

Axis tick groups include visible major and minor marks on both sides, with matching
label and gridline groups. Existing major bottom/left IDs are preserved; minor and
opposite-side components use `minor` and `secondary` ID segments. Named colorbars
likewise expose `.ticks` and `.tick-labels` groups, include the visible tick side,
and report only major tick locations inside the displayed range. Colorbar parts
carry explicit text/line/shape kinds, and tick marks export as measurable paths.

The manifest's `components` inventory includes every repeated role, and both the parts tree
and build order use that inventory. Coordinates are captured after layout with the SVG renderer.
`projection`, axis `supported`, and series `capabilities.dataMorph` describe whether data-space
interpolation is exact. Flux interpolates supported line/scatter plots per panel and preserves
gaps; unsupported scales, polar/3D projections, raster layers and composite marks use a complete
transition. A 3D scatter retains group identity without claiming depth-sorted point indices.

For multi-hue Seaborn output, pass the plotting adapter: `fp.tag_seaborn(ax, plot="histplot")`,
`plot="lineplot"`, `plot="barplot"`, etc. Distribution adapters account for reversed hue draw
order. Explicit `series=[...]` means **artist draw order**. Ambiguous automatic tagging warns
and leaves addressable extras; it never guesses scientific identity from color.

## Matrices, contours and color keys

```python
fig, ax = plt.subplots(layout="constrained")
image = fp.heatmap(ax, [[1, .2], [.2, 1]], series="correlation",
                   cmap="viridis", vmin=0, vmax=1, cells=True)
fp.colorbar(image, name="correlation", label="Correlation")
fp.save(fig, "correlation.svg")

# Ordinary Matplotlib contour arguments/options and return objects:
fig, ax = plt.subplots()
bands = fp.contourf(ax, x, y, z, series="energy", levels=[0, 1, 3, 5])
fp.colorbar(bands, name="energy", label="Energy")
```

- `heatmap` uses `imshow` by default. Supplying `x` and `y` selects `pcolormesh` and supports
  irregular grids. `cells=True` uses mesh edges (default unit edges, origin at the lower left)
  and names each small-matrix cell by row/column. It does not change the raster budget.
- `contour` and `contourf` retain exact boundaries and addressable levels/bands, including the
  different ContourSet structures in Matplotlib 3.7 and later.
- Field metadata records shape, extent or grid, masks, colormap, normalization and its range.
  `include_values=True` additionally records raw matrix values; it is off by default. Regular
  meshes store compact one-dimensional edge arrays. Large layers rasterize as one named part.
- Named colorbars expose the ramp, label and ticks, and link to the mappable. Ordinary
  `fig.colorbar` calls receive the same automatic guide capture.

### Colour scales: the exact law, portably

Every colour-mapped mark — a heatmap, a hexmatrix, a filled contour, a `fp.scatter(..., c=values)`,
even a raw `ax.imshow` or `ax.pcolormesh` you never tagged — records its complete value → colour
law in the manifest's `colorScales[]`:

```jsonc
{"id": "rates",                          // == the recipe's colour-control key
 "kind": "continuous",                   // continuous | binned (a BoundaryNorm)
 "colormap": {"name": "viridis", "source": "matplotlib", "N": 256, "lut": ["#440154ff", …],
              "under": "#440154ff", "over": "#fde725ff", "bad": "#00000000", "discrete": false},
 "norm": {"kind": "log", "vmin": 1.0, "vmax": 53.0, "clip": false, "base": 10, "extend": "neither"},
 "mappables": ["rates.hexes"], "colorbars": ["colorbar.color"], "label": "Synapses per hexbin",
 "recolor": "live",                      // live | raster | regenerate (an imshow is a raster)
 "editable": {"cmap": true, "limits": true, "normKinds": ["linear", "log", "power", "symlog"], "center": false}}
```

`lut` is the full lookup table matplotlib indexes, so a consumer reproduces its colours exactly:
`lut[trunc(x · N)]` for the normalised `x`, `under` / `over` beyond the limits, `bad` for a missing
value (`fluxplot.colorscale.apply` is the reference implementation, tested hex for hex against
matplotlib for every norm kind; `tests/fixtures/colorscale_vectors.json` carries the vectors Flux
checks too). In the SVG every coloured element carries the value it was coloured by —
`data-value` on each cell, hexagon, band (with `data-level-low` / `-high`), contour line and point,
`data-missing="1"` for a gap — and the group carries `data-color-scale` and `data-paint` (`fill`,
`stroke` or `fill stroke`), so Flux can recolour or re-range a plot live, without Python.
`series[].field` and `series[].color.scale` point at the scale; `fp.heatmap(..., value_raster=True)`
additionally writes the matrix as `<plot>.<key>.values.json` beside an image layer.

Colour keys are exact vectors: the 256 solid quads become one `<rect>` filled by a hard-stepped
`<linearGradient>` (no rasterization, no warning), and the colorbar guide records `anchors`
(vmin / vcenter / vmax ↔ SVG position), `axisLength`, `orientation`, `tickLocator`,
`tickFormatter`, `extend` and the extend triangles (`colorbar.color.extend-min` / `-max`).

`fp.tag_seaborn(ax, plot="heatmap")` (and the bivariate `histplot` / `kdeplot`) turn a seaborn
mesh or contour set into the same kind of field, cells named and colour key linked.

### Colormap and palette collections

Every colormap and palette fluxplot knows is plain data in the package —
`src/fluxplot/definitions/colormaps.json` and `palettes.json` — so nothing beyond
matplotlib is imported at runtime and Flux bundles the very same files for its pickers:

| `fx.maps` collection | what | maps |
| --- | --- | --- |
| `flexoki` | fluxplot's own maps (`flexoki_diverging`, …) | 5 |
| `mpl` | matplotlib's built-ins, in its documented groups | 86 |
| `crameri` | Fabio Crameri's Scientific colour maps | 60 |
| `tol` | Paul Tol's continuous and discrete maps | 20 |
| `cmasher` | cmasher | 57 |

Every shipped map is registered with matplotlib under its qualified name and its
reverse from `import fluxplot` on, so `cmap="crameri.batlow"` (or `"tol.sunset_r"`)
works in any matplotlib call; `fx.maps.get("batlow")` resolves a bare name through the
collections in that order, `fx.maps.names("crameri")` lists a collection and
`fx.maps.info("tol.sunset")` tells you its type (`sequential` / `diverging` / `cyclic` /
`qualitative` / `misc`), family, table size `N`, whether it is `discrete` (a listed map of at
most `fx.DISCRETE_MAX` = 32 colours — a set of classes to pick from) and whether it is
perceptually `uniform` (`True` / `False` / `None` = not assessed). Each definition's `colors`
is the map's exact lookup table, so a consumer reproduces it colour for colour.

The house maps are the `flexoki` collection — `flexoki.sequential`, `.warm`, `.diverging`,
`.terrain`, `.spectrum`, linear ramps through palette anchors (`fx.FLEXOKI_MAP_ANCHORS`) and
therefore *not* uniform (`info()["uniform"] is False`); their historical bare names
(`flexoki_diverging`) are aliases of the same tables.

Two derived maps: `fx.maps.truncate("batlow", 0.2, 0.8)` is the stretch of a map as a map of its
own, named `crameri.batlow[0.2:0.8]` — a name `fx.maps.get` (and so a recipe) resolves again —
and `fx.maps.discretize("viridis", [0, 2, 5, 10], extend="both")` returns a `(ListedColormap,
BoundaryNorm)` pair, one colour per bin, that any helper takes as `cmap=` / `norm=` (the colour
scale is then recorded as `kind: "binned"`).

In a signature plot's `palette=`, a colour *name* is the colour (`"red"` → a pale → red → deep
ramp of one hue); the 13-step Flexoki ramp is `"flexoki.red"`.

Palettes live beside them under `fx.palettes`: `flexoki` (the default), `brewer`
(ColorBrewer — 9-class sequential, 11-class diverging and the qualitative sets) and
`tol` (Paul Tol's colour sets). `fx.palettes.brewer["Blues"]` and
`fx.palettes.get("tol", "bright")` are lists of hex strings.

The JSON is built by `tools/build_color_definitions.py` from the upstream packages
(matplotlib, cmcrameri, tol-colors, cmasher) — a build-time input only, re-run when a
collection should be refreshed (`uv run --with cmcrameri --with tol-colors python
tools/build_color_definitions.py`; without a package its shipped definition is kept). The
Flexoki palette itself has one canonical source, the design-token export
`src/fluxplot/definitions/flexoki.tokens.json`: `fx.flex` is built from it at import and the
`flexoki` palette collection is generated from it (fluxplot's "green" is the tokens' custom
`GRN` hue; Flexoki's original green is `olive`).

Every `fx.use_*` theme installs the house sequential map (`cmasher.rainforest`, `fx.SEQUENTIAL`)
as matplotlib's default `image.cmap`, so a heatmap without `cmap=` is in the house style;
`fx.DEFAULT_DIVERGING` names the diverging default.

### Themes a consumer can follow

The manifest records the theme in force at save — `style = {"theme": "light" | "dark" | …, "tokens":
{"ink", "label", "tick", "axis", "grid", "plot", "paper"}}`, the scaffold colours by role as
lowercase hex (`theme` is `null` once an rcParam was changed by hand) — and every scaffold element
painted with one of them carries `data-ink-fill` / `data-ink-stroke` naming the token: tick labels
and titles `ink`, axis titles `label`, ticks `tick`, spines `axis`, gridlines `grid`, the axes and
figure backgrounds (`axes.background`, `figure.background`, now parts of their own) `plot` and
`paper`. Overlays a helper drew in the theme's ink (the significance bracket) say so outright.
Data colours are never tagged, so Flux can restyle a light plot's furniture onto a dark deck and
leave the science alone. `fp.save(..., theme_vars=True)` additionally rewrites those paints to
`var(--fx-<token>, <hex>)` for a CSS-aware host (off by default until checked in Illustrator and
Inkscape; browsers and rsvg honour the fallback). A rerun with
`FLUX_PARAMS={"__fluxplot__": {"theme": "dark"}}` makes every `fx.use_*` call apply that theme,
and the recipe records the theme in force under `__fluxplot__.theme`.

### Series colours, and the same colour for the same thing everywhere

Every series records its primary paint in the manifest — `series[].color = {"hex": "#35ab49",
"alpha": 1.0, "token": "flexoki.green-400", "palette": {"name": "flexoki.light", "index": 2}}`:
the exact palette token that names it when one does, its slot in the active cycle when it came
from there (`"varies"` for a colour-mapped collection, whose `scale` names the colour scale).
The glowbar and fluxbox payloads record the `palette` spec each category's shades came from.

`fp.colors.categories` keeps categories consistent across figures: `get("SD")` returns the colour
pinned to a category, else assigns the next unused slot of the palette (default: the theme's
cycle) in first-request order and remembers it, so `"b"` is the same colour whether or not `"a"`
is on the plot — `categorical_colors` (surface maps), glowbar and fluxbox group colours and
`tag_seaborn`'s hue levels all consult it. `assign({...})` pins colours, `save()` / `load()`
write and read a project file, `fluxplot.colors.json`
(`{"spec": "fluxplot/colors", "version": 1, "categories": {"SD": "#bc5215"}, "palette": "flexoki"}`),
which the first use auto-loads from `$FLUXPLOT_COLORS` or the nearest one between the working
directory and the Git root; nothing is ever written implicitly. `categories.auto_series = True`
colours `fp.line` / `fp.scatter` series by their name when the call gives no colour (off by
default). Two more recipe controls follow: `__fluxplot__.palette = "tol.bright"` makes every
`fx.use_*` install that palette as the prop cycle (and glowbar categories draw from it), and
`__fluxplot__.series = {"<series id>": {"color": "#…"}}` recolours a series on a rerun — the
helpers and the signature plots check it before drawing.

### Signature plots on dark grounds

The glowbar, fluxbox and hexmatrix judge every ink against the ground they are drawn on. On a
dark theme the point rims lighten instead of darkening, the shades stay at least 20 lightness
units above the ground, the mean and median inks lift off their marks instead of deepening
(`_colour.median_ink`, shared by both plots), connectors take the theme's grid neutral, a
qualitative palette's group ink is the light neutral, and a hexmatrix's single-colour ramp turns
(deep near the ground → pale at the top, named `hexmatrix.mono-dark:#…` so a rerun rebuilds it).
`tests/test_dark_ground.py` pins WCAG contrast ≥ 1.5 for every mark and ≥ 3 for bracket and
identity lines under the light, paper and dark themes.

### Accessibility lint

`fp.colorcheck` asks what a plot's colours do for the reader: `simulate(colors, "deuteranomaly")`
applies the Machado (2009) colour-vision-deficiency matrices (embedded at every published
severity; protanomaly / deuteranomaly / tritanomaly), `greyscale` shows the black-and-white print,
`delta_e` is CIEDE2000 and `contrast` the WCAG 2 ratio. `check_palette(colors, bg, text=…)` finds
pairs that collapse under a deficiency (`cvd-confusable`, ΔE < 10) or in greyscale
(`greyscale-confusable`, ΔL* < 10) and marks (`low-contrast-mark`, < 3:1) or text
(`low-contrast-text`, < 4.5:1) too faint against the ground; `check_colormap` flags a map whose
ΔE steps vary too much (`non-uniform`) or a sequential map whose lightness reverses
(`non-monotone`). `fp.save(..., lint="warn")` runs `check_figure` on the tagged series and text
inks: findings go to `SaveResult.warnings` and the manifest's `quality.color`; `lint="error"`
refuses the save. Off by default.

Measured with it, the house cycles were reordered (2026-09-30) so that no two *adjacent* colours
collapse for a colour-deficient reader: `fx.CYCLE_ORDER` is now blue, orange, purple, green,
magenta, yellow, cyan, red (the smallest adjacent distance under any deficiency is ΔE 20 in the
light cycle, 15 in the dark one; cyan next to magenta used to be 5.6). Plots that let the cycle
pick their colours get different colours from the third series on. Greyscale separation cannot
be fixed by order alone at one weight — vary marker or line style for a print-safe figure.

### Colour controls in the recipe

Every colour scale is a recipe control: `fp.save` writes its complete state under
`recipe.params.__fluxplot__[<key>]`, and Flux's **Color scales** editor edits it and regenerates:

```jsonc
{"cmap": "magma" | {"lut": ["#…"], "under": "#…", "over": "#…", "bad": "#…"},
 "reversed": false,
 "vmin": 1, "vmax": 50,
 "norm": {"kind": "linear" | "log" | "symlog" | "power" | "twoslope" | "centered",
          "vcenter": 0, "gamma": 0.5, "linthresh": 1, "linscale": 1},
 "extend": "neither" | "min" | "max" | "both"}
```

The helpers read the controls from `FLUX_PARAMS` automatically. A control that merely restates
what the script asked for (the replay of a recorded save) keeps the script's own objects — a
`ListedColormap` with no resolvable name included — so regeneration never fails on its own
output; a control that differs is applied, and an impossible one (an unknown map, a log norm with
`vmin <= 0`, a twoslope centre outside the limits) raises a `ValueError` naming the key.
`extend` reaches the colour key drawn with `fp.colorbar`. The old flat `{cmap, vmin, vmax}` is
still understood.

The key names the **series** (`rates`), never the axes' position, so adding a subplot cannot
orphan an edit; a second colour-mapped series with the same name gets `panel.<name>.rates` or
`axes.<n>.rates`, and `key=` names it explicitly. Controls saved by an older Flux under the
positional key are still honoured. `recipe={"args": ..., "cwd": ..., "output": ...}` overrides
are honored; cwd/output resolve relative to the recipe directory. `FLUXPLOT_ONLY` skips
unselected saves before layout or file I/O.

#### One scale for several panels

Declare the scale once and let every panel join it:

```python
fp.color_scale("corr", cmap="RdBu_r", center=0)          # norm="linear"|"log"|"sqrt"|"symlog", vmin/vmax, robust
fp.heatmap(ax1, C1, series="ctrl", scale="corr")
fp.heatmap(ax2, C2, series="drug", scale="corr")           # also contour/contourf, scatter(c=…), hexmatrix
fp.colorbar(scale="corr", ax=[ax1, ax2], label="r")        # one key, borrowing space from both panels
```

The members share the map and the norm, and their limits default to the **union** of every
member's finite values (symmetric about `center` when one is given; `robust=True` uses the
2nd–98th percentiles), resolved at `fp.save` before layout so the figure never lays out against
stale limits. The manifest carries one `colorScales` entry (`id: "corr"`) listing every member and
the key, the recipe carries one control (`__fluxplot__.corr`), and an edit to it — a new map, a
pinned `vmax` — repaints every panel at once.

`force_vectors=True` overrides and restores caller rasterization flags, including axes z-order
rasterization. It cannot vectorize a source image created with `imshow`; use a modest
`cells=True` heatmap when individual vector cells are needed. Empty/clipped raster artists no
longer disrupt another layer's IDs. Surface lighting survives SVG rendering and colorbars use
a separate scalar mappable. Surface category rendering uses per-part painter ordering, **not a
full cross-category depth buffer**: convex projections are supported, while arbitrary folded
meshes can have cross-part occlusion differences. This limitation is recorded in surface metadata.

Old saved projects remain readable. Schema 0.3 output should be used with the accompanying Flux
reader update. New imports and reloads verify SVG checksums before legacy geometry repair;
a mismatched pair leaves the last accepted plot intact. No existing project is bulk-regenerated.

### 3D fluxplots

`fp.scene3d()` holds an actual mesh with named parts, value fields and shape states:

```python
sc = fp.scene3d(figsize=(3.5, 3), units='µm', axes='triad')
fp.mesh3d(sc, (vertices, faces), series='cell')
sc.view(azimuth=30, elevation=20)
fp.save(sc, 'plots/cell')  # GLB + semantic manifest + regeneration recipe
```

Scenes display interactively in trusted notebooks with a PNG fallback. Flux can choose
another angle, restyle parts and remap fields without rerunning Python. See the
[3D guide](docs/SCENE3D.md) for shape states, same-topology morphs, supported mesh inputs,
optional decimation and a self-contained neuron demo.
