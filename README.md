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
fp.scatter(ax, x, y, *, series, label=None, **mpl_kwargs)
fp.bar(ax, x, height, *, series, **mpl_kwargs)
fp.errorbar(ax, x, y, *, series, yerr=None, **mpl_kwargs)
fp.area(ax, x, y1, y2=0, *, series, **mpl_kwargs)
fp.box(ax, values, *, series, label=None, include_values=False, **mpl_kwargs)     # one box per call
fp.violin(ax, values, *, series, label=None, include_values=False, **mpl_kwargs)  # one violin per call
fp.hist(ax, values, *, series, bins=None, label=None, include_values=False, **mpl_kwargs)
```

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
    schemas/          # JSON Schemas for the manifest and recipe (validated on every save)
  examples/growth_plot.py     # the worked example above
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

In Flux, open X-Ray on a recipe-backed plot and expand **Color scales**. Edit the palette or
range, then choose **Apply and regenerate**. Both the field and its key are regenerated from
source data; authored Flux overrides remain keyed to existing part IDs. There is no attempt to
recolor an embedded image by changing SVG fill. Nonstandard normalizations retain their own
range rules; edit those in Python.

Field helpers read reserved `FLUX_PARAMS.__fluxplot__` overrides automatically and record their
controls in the recipe. A `key=` explicitly names the control; otherwise the key includes the
owning axes and series. Set explicit panel names or control keys before plotting for stability
across source rearrangements. `recipe={"args": ..., "cwd": ..., "output": ...}` overrides are
honored; cwd/output resolve relative to the recipe directory. `FLUXPLOT_ONLY` skips unselected
saves before layout or file I/O.

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
