# fluxplot colour system and roadmap — implementation plan

*Written 2026-09-29 against fluxplot `main` @ `648e0f9` (plus the uncommitted `fp.hexmatrix` work
described in §0.7) and Flux @ `df109a96`.*

This document is **self-contained**: an implementing agent should be able to execute any work item
from this file alone. §0 gives the context you need about both codebases. §1 lists the working
rules. §2–§8 are the work items, each with a problem statement, the exact change, tests and
acceptance criteria. §9 gives the order to work in.

Tags: **[FP]** = work in the fluxplot repo (`/home/driessen2/fluxplot`, Python). **[FLUX]** = work
in the Flux repo (`/home/driessen2/flux`, TypeScript/Svelte/Electron). **[verified]** = reproduced on
2026-09-29 with a script; everything else was established by reading code — **re-check every cited
line before editing**, because line numbers drift.

---

## 0. Context an implementer needs

### 0.1 What the two systems are

- **fluxplot** is a thin semantic layer over matplotlib. Users plot with ordinary matplotlib, or
  with `fp.*` helpers that wrap matplotlib calls and **register what each artist means** (series
  name, role). `fp.save(fig, "plots/x.svg")` writes three files:
  - `x.svg` — a normal SVG in which every meaningful element has a stable semantic `id`
    (`control.line`, `control.point.3`, `axis.x.title`) plus `data-role`, `data-series`,
    `data-kind`, `data-index`, `data-x` / `data-y`, etc.;
  - `x.fluxplot.json` — the **manifest**: axes (with data↔pixel anchors), series (data, svg ids,
    payloads), guides (legend, colorbar), overlays, a parts tree, build/animation order and presets;
  - `x.recipe.json` — how to regenerate: the script, `params`, input hashes, and
    `command` / `args` / `cwd` / `output` for a rerun.
- **Flux** is the user's scientific writing studio (Figure, Slide, Paper, Library). It inlines
  fluxplot SVGs as live DOM and uses the manifest to:
  - restyle parts (`el.overrides`, keyed by semantic id);
  - re-project axes live (`el.view`);
  - animate builds on slides;
  - regenerate by rerunning the recipe.

  Agents drive Flux through verbs (`flux-core/verbs.ts`; exposed over MCP and a CLI).
- **Principle 1 (non-negotiable): meaning is captured at birth, never reverse-engineered.** fluxplot
  records meaning while the plotting code still knows it. The SVG post-process is an *exact join on
  ids fluxplot authored*, never a pixel or colour heuristic. Any new feature must follow this rule;
  in particular, "tag by matching colours" is only acceptable when fluxplot itself set that colour
  (see B1).

### 0.2 fluxplot source map (`src/fluxplot/`)

| File | Responsibility |
|---|---|
| `api.py` | Public helpers (`line`, `scatter`, `bar`, `barh`, `errorbar`, `area`, `box`, `violin`, `hist`, `tag`, `tag_points`, `tag_seaborn`, `significance_bracket`, `reference_line`, `annotation`) and **`save` / `_save`** (the pipeline, ≈ lines 560–744). `_infer_plot_type` ≈ line 487. |
| `descriptors.py` | `Mark` dataclass: `role, series, name, kind, label, x, y, artists, indexed, live_data, axes, data{}`, plus resolved `gid, member_gids, member_indices`. `GuideTag` for scaffold. |
| `tagger.py` | `Registry` (per figure, `registry_for(fig)`), `snapshot`, `resolve_gids` (Mark → gids), `autotag_scaffold` (axes, ticks, spines, legend, titles, free-text sweep, `_sweep_extra` orphan sweep). |
| `autotag.py` | Promotes labelled raw artists to series; `is_colorbar_axes`. |
| `ids.py` | `slugify`, `series_root`, `series_id`, `axis_id`, `IdAllocator` (dedups with `-2` suffixes; relying on that suffix is considered a bug). |
| `panels.py` | Multi-axes figures: `plan(fig)` → one `Panel` per non-colorbar axes (sorted by position); ids get the `panel.<label>.` prefix when there is more than one panel or any axes is named via `fp.panel(ax, name)`. `manifest()` builds per-panel docs and merges them. |
| `raster.py` | Auto-rasterizes heavy artists (>`DEFAULT_THRESHOLD = 800` primitives) into one `<image>`; `plan()`, `rasterizing()`, `reattach()`. Colorbar solids are always rasterized (≈ line 165). |
| `render.py` | Deterministic SVG render (`svg.hashsalt`, `svg.fonttype: none`); `final_layout(fig)` freezes layout with the SVG canvas. |
| `postprocess.py` | lxml pass: rename wrappers, inject `data-*` per Mark (`_inject_points`, `_inject_indexed`, `_inject_field`, `_inject_overlay`), deref tick `<use>`, return `present` ids. |
| `manifest.py` | `build_manifest`: series entries (≈ lines 67–210; payload keys copied from `m.data` at ≈ line 158: `bar, band, uncertainty, field, glowbar, fluxbox, hexmatrix`), guides, overlays, parts tree (`_group`, `_ref` ≈ line 392), build order (`_build_order` ≈ line 486). |
| `fields.py` | `fp.heatmap`, `fp.contour`, `fp.contourf`, `fp.colorbar`; `_options()` (**the colour-control key + recipe override mechanism**), `normalization()`, `_capture()`, `colorbar_guides()`. |
| `_fieldmap.py` | `resolve_colormap(name|Colormap|None)` (matplotlib → fluxplot collections), `continuous_mapping`, `categorical_colors`, `category_name`. |
| `colors.py` | The Flexoki palette table (`colors.green400` → hex), `maps` (`_MapRegistry`: `get`, `info`, `register`, `collections`), `palettes`, loading `definitions/colormaps.json` and `palettes.json`. |
| `style.py` | House themes `use_light`, `use_lighttable`, `use_paper`, `use_dark` (≈ lines 291–340) setting rcParams; `CYCLE_LIGHT` / `CYCLE_DARK`; `SEQUENTIAL` / `DIVERGING` picks. |
| `recipe.py` | `build_recipe`, **`params(defaults)`** (merges `$FLUX_PARAMS` JSON). |
| `provenance.py` | `discover_script()` (≈ line 45), `build_provenance`. |
| `capture.py` | Axis anchors/scale/ticks/units capture. |
| `data.py` | `refresh(mark)` at save (calls `fields.capture_mark`), `bar_data`, `artist_xy`, `values`. |
| `surface.py`, `surface3d.py`, `mesh3d.py`, `scene3d*.py` | 2D surface (brain) maps and 3D scenes/GLB. `surface3d.py` ≈ line 146 already records a colormap as 256 portable stops — reuse its approach. |
| `signature_fluxplots/` | `glowbar.py`, `fluxbox.py`, `hexmatrix.py`, `_colour.py` (palette resolution, perceptual shades). |
| `stats/` | `welch_hedges`, `mann_whitney_cliff`, `paired_t_hedges`, `wilcoxon_rank_biserial`, `holm`, `holm_adjusted`; rows keyed by `stats.REPORT_COLUMNS`. |
| `schemas/` | `manifest.schema.json`, `recipe.schema.json`, `scene3d.schema.json`; every save runs `jsonschema.validate` (`api._validate`). |
| `presets.py` | `ROLE_PRESETS` animation hints → `manifest.build.presets`. |
| `version.py` | `SPEC_VERSION` (currently `0.3.0`), `__version__`. |

**The save pipeline** (`api._save`), in order:
1. `tagger.snapshot` copies the registry.
2. `final_layout` runs once with the SVG canvas.
3. For each panel: auto-promote labelled artists, `data.refresh` (which calls
   `fields.capture_mark` → `mark.data['field']`), `resolve_gids` with a panel-scoped allocator,
   `autotag_scaffold` + `colorbar_guides`, `capture_axes`.
4. `raster.plan`.
5. `render_svg` inside `raster.rasterizing`.
6. `postprocess` → `present` ids.
7. `panels.manifest` → manifest; `build_recipe` → recipe; append `__fluxplot__` colour controls to
   `recipe.params` (≈ `api.py:684-691`); validate both against the schemas.
8. Staged write, in the order SVG → manifest → recipe.

### 0.3 Today's colour-control mechanism (exact behaviour)

- `fields._options(ax, series, key, kwargs)`:
  - computes `key` = explicit `key`, else `panel.<slug>.<seriesRoot>` when the axes has an
    `fp.panel` name, else `axes.<1-based index in all_axes>.<seriesRoot>`;
  - reads `recipe.params().get('__fluxplot__', {}).get(key, {})` and copies `cmap` / `vmin` /
    `vmax` into `kwargs`;
  - if `kwargs['norm']` is a `Normalize` object, it `copy()`s it and moves `vmin` / `vmax` onto the
    copy (so LogNorm stays log);
  - returns `key`.
- `fields._capture` stores `field = {kind, shape, controlKey, normalization{kind: <class name>,
  vmin, vmax, clip, vcenter?, gamma?, linthresh?, linscale?, boundaries?}, cmap: <name>,
  missingColor, underColor, overColor, …}` on the mark. The manifest copies it to `series[].field`.
- `api._save` writes `recipe.params.__fluxplot__ = {controlKey: {cmap, vmin, vmax}}` for every
  field mark, on **every** save.
- Flux: `src/lib/plot/ColorScaleControls.svelte` builds one draft per unique
  `series.field.controlKey`. It shows a palette input with `ColormapPicker` (fed by
  `src/lib/color/colormaps.gen.ts`, generated from fluxplot's `definitions/*.json` by
  `scripts/gen-color-collections.mjs`, **32 stops per continuous map**) and min/max inputs.
  `apply()` dispatches `regenerate({...params, __fluxplot__: {...}})`. It is mounted only in the
  X-ray (`src/lib/Xray.svelte` ≈ 749) and opened from the FluxFig menu `c` action
  (`src/lib/interact/propertyMenu.ts` ≈ 632).
- Regeneration: `src/lib/plot/recipeContract.mjs` merges params, passes `__fluxplot__` **only** in
  `$FLUX_PARAMS` (never argv), runs the script, and `reimportPlot` (`src/lib/io.ts` ≈ 78) swaps
  the SVG/manifest/recipe but keeps `el.overrides` / `el.view`.

### 0.4 Flux plot surfaces you will touch

- `src/lib/types.ts` ≈ 409–472: `SemanticPlotElement` (`overrides`, `view: PlotView`),
  `PlotAxisView`, `PartOverride`.
- `src/lib/plot/types.ts`: manifest TS types (`FluxPlotManifest`, `FluxPlotSeries`,
  `FluxPlotField` ≈ 107, `FluxPlotGuide` ≈ 32, `PartNode` ≈ 73). These are hand-written and have
  drifted from the Python output (see F3).
- `src/lib/plot/project.ts` (axis fits, `viewFits`, `plotViewIssues`), `src/lib/plot/projectDom.ts`
  (`preparePlotView` / `applyPlotView` ≈ 240; called from `mount.ts`, `inlineMarkup.ts`,
  `export.ts`, `slide/player/render.ts`, `slide/player/transform.ts`).
- `src/lib/plot/parse.ts`: `applyOverrides` ≈ 361, `partDomId`, `prefixIds` (Flux prefixes every
  inlined id with the element id — use `partDomId(elId, partId)` to find nodes).
- `src/lib/plot/tree.ts` ≈ 83: `inferRole` — regex role guessing (see F2).
- `src/lib/slide/autobuild.ts` ≈ 38: `ANIM_TO_PRESET`, `TEXTISH`, `STROKABLE`, `PHASE`,
  `presetForRole`.
- `src/lib/ops.ts` ≈ 1686: `setPlotView` (the pattern to copy for new element state);
  `setPartOverride` ≈ 1708.
- `flux-core/verbs.ts`:
  - `set_plot_view` ≈ 3061 (the pattern to copy for new verbs);
  - `rerun_plot` ≈ 2479 (params schema `z.record(z.union([z.string(), z.number(), z.boolean()]))`
    ≈ 2493);
  - `restyle_part` ≈ 433;
  - `set_transform` ≈ 3118.
- Flux conventions: **read `/home/driessen2/flux/CLAUDE.md` and `AGENTS.md` before editing Flux.**
  - Checks: `npm run check` (svelte-check) and `npm test` (= check + `scripts/run-verifies.mjs
    --tier pure`).
  - New verification scripts are `scripts/verify-*.ts|mjs` and **must be registered in
    `scripts/verify-manifest.json`**.

### 0.5 fluxplot conventions

- Environment: `uv` (`uv run pytest -q -p no:warnings`, `uv run python …`). There is no `python`
  on PATH. The test suite currently has **363 passed, 5 skipped**; keep it green.
- Tests use `matplotlib.use("Agg")`. The typical pattern is to save to `tmp_path`, then load
  `*.fluxplot.json` / `*.recipe.json` and assert on ids and payloads; see `tests/test_fluxbox.py`,
  `tests/test_hexmatrix.py`, `tests/test_polish.py` (≈ 140–152 covers the colour controls).
- **Determinism:** saving the same figure twice must produce byte-identical SVG + manifest (the
  recipe may differ only in timestamps and hashes). `tests/test_determinism.py` guards this. Any
  new output must be deterministic: no dict-order dependence, no floats formatted by `str()` of
  numpy arrays, and use `repr(float)` via `postprocess._fmt`.
- **Schema:** every new manifest key must be added to `schemas/manifest.schema.json` (additive,
  optional) or the save fails validation.
- **Ids are durable API.** Never rename an existing id without a migration note. New ids must not
  depend on draw order or on the allocator's `-2` collision repair.
- Style: docstrings in the house voice (see `fluxbox.py`), `from __future__ import annotations`,
  and comment density matching the surrounding code.
- Versioning: additive manifest changes keep `SPEC_VERSION` 0.3.x (bump the patch number; update
  the schema `$id`). A breaking change needs 0.4.0 and a Flux compatibility check (F3).

### 0.6 How live axis views work in Flux (the model to copy for colour)

`el.view = {x?: {domain?, scale?}, y?: …}` is stored on the element; absent = generator default.
`setPlotView` normalises values equal to the manifest defaults back to absent.
`applyPlotView(root, manifest, view, elId)` re-projects DOM nodes through fits built from
`manifest.axes[].x|y.anchors`. A slide "Change" can animate `view` because it is an ordinary
element property. The colour-scale work (A7) mirrors this exactly with `el.colorScale`.

### 0.7 State of `fp.hexmatrix` (uncommitted as of writing)

- `signature_fluxplots/hexmatrix.py` implements hex binning / hex matrices.
- Hexagons are one `PolyCollection` registered as `Mark(role="x-hexbin", name="hexes",
  kind="hexmatrix")` with `data` keys `field_config` / `field_artist` (a colour control, kind
  `hexbin`), `field_names` / `field_member_prefix="hex"` / `field_member_role="x-hex"` /
  `field_attrs` (per-hex `data-*`), and `hexmatrix` (the payload).
- `postprocess._inject_field` was generalised to honour `field_names`. Member ids are
  `<series>.hex.<row>.<col>`.
- Tests are in `tests/test_hexmatrix.py`; the example is `examples/hexmatrix_example.py`.
- If that work is still uncommitted when you start, commit it first as its own commit.

---

## 1. Working rules for every item

1. **One item per commit** (or a small group of tightly related items). The message says what
   changed and why, and ends with the attribution line the session requires.
2. **Tests first where the item is a bug:** write the failing regression test from the item's
   "Tests" list, then fix.
3. **Keep outputs deterministic** and schema-valid (§0.5).
4. **Additive contract:** new manifest keys are optional. Consumers that ignore them must keep
   working. Old keys named "kept as alias" must keep being emitted until the item that removes
   them.
5. **Cross-repo items:** land the fluxplot side first, with its schema. Then the Flux side reads the
   new fields, with a fallback for old manifests (plots in existing projects won't be regenerated).
6. After each milestone, run both test suites (`uv run pytest -q -p no:warnings` in fluxplot;
   `npm test` in Flux when Flux was touched) and regenerate `examples/out/` with the example
   scripts.

---

## 2. Workstream A — colour scales (the full version)

**Goal.** Every colour-mapped mark carries a complete, portable colour-scale record, and every
coloured element carries its value. Flux can then recolour and re-range any colour-mapped plot
**live** (no Python), animate the change on slides, edit it from agents, redraw its colorbar, and
fall back to a recipe rerun only where the layer is a raster.

### A0 — bugs to fix first [FP] (Milestone 1)

**A0.1 Custom colormaps break every regeneration. [verified]**
- Problem: `fp.heatmap(ax, M, series='m', cmap=ListedColormap([...]))` records `cmap: "unnamed"`
  in `recipe.params.__fluxplot__`. Flux's Regenerate replays those params, and the rerun raises
  `ValueError: 'unnamed' is not a valid value for cmap`. The controls are written on every save,
  so *any* regeneration of such a plot fails.
- Change, in `fields._options`:
  - only apply an override value when it **differs** from what the script itself would have used;
    compare the override `cmap` against the script's `kwargs['cmap']` (name, or `.name` of an
    object) — equal means "not edited", so keep the script's object;
  - when an override `cmap` is not resolvable, raise
    `ValueError("colour control '<key>': unknown colormap '<name>' …")` naming the key.

  After A1 lands, `_options` also accepts `{"stops": [...]}` / `{"lut": [...]}` overrides (see A5).
- Tests (`tests/test_colorscale.py`, new): save a heatmap with a `ListedColormap`, then set
  `FLUX_PARAMS` to exactly the recorded params and re-run the plotting function →
  - no error;
  - the rendered cmap `==` the original object's colours;
  - an override `cmap="magma"` still applies.
- Acceptance: the repro above passes; `tests/test_polish.py` still passes.

**A0.2 fluxplot map names fail in `fp.heatmap` / `contour` / `contourf`. [verified]**
- Problem: `fp.heatmap(cmap='emerald')` raises, because the string goes straight to
  `ax.imshow` / `pcolormesh` / `contour` (`fields.py` ≈ 90). `fp.hexmatrix` and `fp.surface` accept
  it through `_fieldmap.resolve_colormap`.
- Change:
  - in `fields.heatmap` and `fields._contour`, after `_options`, replace a `str` `kwargs['cmap']`
    with `resolve_colormap(kwargs['cmap'])`;
  - in `colors._MapRegistry.register`, also register `cmap.reversed()` (name + `_r`) with
    matplotlib and `self._custom`;
  - have `resolve_colormap` try `maps.get()` for names ending in `_r`.
- Tests: parametrise over `['emerald', 'crameri.batlow', 'batlow', 'viridis_r',
  'flexoki_diverging', 'flexoki_diverging_r']` × `[heatmap, contourf]` → no error, and the recorded
  `field.cmap` equals the resolved map's name.

**A0.3 The house default colormap is never applied.**
- Problem: `style.py` defines `SEQUENTIAL = cmr.rainforest` (≈ 126–143), but no `use_*` sets
  `rcParams['image.cmap']`, so it stays `viridis`. The docs promise otherwise.
- Change:
  - in each `use_light` / `use_lighttable` / `use_paper` / `use_dark`, set
    `mpl.rcParams["image.cmap"] = SEQUENTIAL.name`;
  - register `SEQUENTIAL` via `colors.maps.register` first, so the name resolves;
  - add a module constant `DEFAULT_DIVERGING = DIVERGING.name`, which helpers with `center=` use
    when no cmap is given.
- Tests: after `fx.use_light()`, `mpl.rcParams['image.cmap'] == fx.SEQUENTIAL.name`, and a
  `fp.heatmap` without `cmap` records that name.
- Also fix the stale "cmasher optional" comments (`style.py` ≈ 34, 138; `colors.py` ≈ 293).
  `cmasher` is a hard dependency in `pyproject.toml`.

**A0.4 hexmatrix `center=` bugs.**
- Problem: `hexmatrix._make_norm` returns `TwoSlopeNorm` whenever `center` is set, silently
  ignoring `norm='log'`. With `center=0, vmin=0.5, vmax=3`, matplotlib's autoscale widened `vmin`
  to −3.
- Change:
  - raise `ValueError("hexmatrix: center= needs a linear norm")` when `center is not None` and
    `norm` is not `"linear"`;
  - when `center` is outside `[vmin, vmax]`, raise
    `ValueError("hexmatrix: center must lie between vmin and vmax")`.
- Tests: both raise; `center=0` with symmetric auto limits still works.

**A0.5 A `__fluxplot__` param arriving as a string crashes `_options`.**
- Problem: the Flux CLI `rerun-plot --__fluxplot__ '{…}'` passes a JSON string; `_options` calls
  `.get` on a str.
- Change: in `recipe.params()`, after merging, if `out.get('__fluxplot__')` is a `str`, `json.loads`
  it. On failure, raise `ValueError("FLUX_PARAMS __fluxplot__ is not valid JSON")`.
- Tests: `FLUX_PARAMS='{"__fluxplot__": "{\"k\": {\"cmap\": \"magma\"}}"}'` applies.

**A0.6 Colour-control keys depend on axes order.**
- Problem: the key is `axes.<n>.<series>`. Adding a subplot before the heatmap renumbers the key,
  and the saved Flux colour settings are silently orphaned.
- Change the default key rule in `_options`:
  1. the explicit `key=`;
  2. otherwise `<seriesRoot>` if no other colour-controlled series in the figure has the same root
     at call time. Track this in `registry_for(fig)._color_keys`, a set;
  3. otherwise `<panel slug>.<seriesRoot>` if the axes is named;
  4. otherwise `axes.<n>.<seriesRoot>` (today's rule).

  When looking up overrides, try the new key, then the legacy `axes.<n>.<seriesRoot>` key, so
  existing Flux projects keep their edits. Record the key actually used in `field.controlKey`.
- Tests:
  - two figures, one with an extra subplot added first → same key;
  - two heatmaps with the same series name in one figure → distinct keys;
  - a legacy `axes.1.m` override still applies.

### A1 — the `colorScales` manifest contract [FP] (Milestone 2)

**Problem.** The colour scale is recorded as a matplotlib name plus norm numbers, duplicated in
`series.field` and `guides[colorbar]`. Custom maps aren't portable, and there is nothing a
non-Python consumer can recolour with.

**Change.** Add a new module `src/fluxplot/colorscale.py`:

```python
def colormap_record(cmap) -> dict:
    """The exact lookup table matplotlib uses, portable: every Colormap → {name, source, N, lut, under, over, bad}."""
    # lut = [to_hex(c, keep_alpha=True) for c in cmap(np.arange(cmap.N))]   (exact; mpl indexes the LUT)
    # if cmap.N > 1024: resample to 1024 and set "approximate": true
    # source: "matplotlib" | "fluxplot" | "crameri" | "tol" | "cmasher" | "custom"   (via maps.info / mpl registry)
    # discrete: True for ListedColormap with N <= 32 (match _colour._from_colormap; see B4 on unifying thresholds)

NORM_KINDS = {"Normalize": "linear", "LogNorm": "log", "SymLogNorm": "symlog", "PowerNorm": "power",
              "TwoSlopeNorm": "twoslope", "CenteredNorm": "centered", "BoundaryNorm": "boundary",
              "NoNorm": "none"}

def norm_record(norm, *, extend="neither") -> dict:
    """{kind, vmin, vmax, clip, vcenter?, halfrange?, gamma?, linthresh?, linscale?, base?, boundaries?, ncolors?, extend}"""
    # unknown subclasses → {"kind": "custom", "className": type(norm).__name__, "vmin", "vmax"}

def scale_record(scale_id, mappable, *, label=None, extend="neither") -> dict:
    """One manifest colorScales[] entry for a matplotlib ScalarMappable."""
```

Manifest shape: a new optional top-level array `colorScales`.

```jsonc
{
  "id": "rates",                          // == field.controlKey (A0.6 rule); unique in the figure
  "kind": "continuous",                   // continuous | binned (BoundaryNorm) | categorical (B2)
  "colormap": {"name": "viridis", "source": "matplotlib", "N": 256, "lut": ["#440154ff", "…"],
               "under": "#440154ff", "over": "#fde725ff", "bad": "#00000000", "discrete": false},
  "norm": {"kind": "log", "vmin": 1.0, "vmax": 53.0, "clip": false, "base": 10, "extend": "neither"},
  "mappables": ["rates.hexes"],           // svg ids of every group coloured by this scale
  "colorbars": ["colorbar.color"],        // colorbar guide ids drawing it
  "label": "Synapses per hexbin",
  "recolor": "live",                      // live | regenerate  (regenerate: some mappable is a raster)
  "editable": {"cmap": true, "limits": true, "normKinds": ["linear", "log", "power", "symlog"], "center": false}
}
```

Rules:
- `lut` is the full matplotlib LUT (`cmap(np.arange(N))`). This gives exact parity: matplotlib maps
  a normalised `x` to `lut[clip(int(x*N), 0, N-1)]`, `under` for `x<0`, `over` for `x>1` (and
  `x==1 → N-1`), and `bad` for NaN or masked values.
- `editable.limits` is false for `boundary` / `custom` / `none`. `editable.center` is true only for
  `twoslope` / `centered`. `normKinds` lists the kinds a consumer may switch to: all of
  linear/log/power/symlog for continuous scales, `[]` for binned ones.
- `recolor` is `"regenerate"` if any mappable id is in the save's `rasterized` set or the mappable
  is an image (`AxesImage`), unless A2's value raster is emitted.
- Keep `series[].field` and the colorbar guide's `cmap` / `normalization` as **aliases** for one
  minor version. Add `field.colorScale: "<id>"` and colorbar guide `colorScale: "<id>"` refs.

Where it is built:
- `fields._capture` computes `scale_record` for field marks;
- `hexmatrix` gets it for free through `field_config`;
- A3 adds scatter and images.

`api._save` collects the records (dedup by id; union the `mappables`), passes them to
`panels.manifest` / `build_manifest` as a new kwarg, and `build_manifest` emits `colorScales`.
Panel namespacing: add `colorScales[].mappables` and `.colorbars` to `panels.namespace` (prefix
each id). The scale `id` itself is **not** prefixed: it is the recipe key and must equal
`controlKey`.

Schema: add `colorScales` (array of objects with required `id`, `colormap.lut`, `norm.kind`).

Tests (`tests/test_colorscale.py`):
- a LogNorm heatmap → `norm.kind == "log"`, `lut` length 256, `recolor == "live"` with
  `cells=True`, `"regenerate"` for imshow;
- **parity**: for 1,000 random values, the Python function
  `colorscale.apply(record, values)` (implement it: it is the reference implementation the Flux
  side must match) equals `mappable.to_rgba(values)` hex-for-hex, for each norm kind
  (linear/log/symlog/power/twoslope/centered/boundary) with `extend` both/neither;
- a custom `ListedColormap` → `source == "custom"`, `lut` equals its colours;
- `mono:` ramps from hexmatrix → exact LUT.

Also write `tests/fixtures/colorscale_vectors.json`, generated by a script
`tests/generate_colorscale_vectors.py`, containing `{record, values, expected_hex}` cases. The Flux
parity test (A7.1) consumes the same file; copy it into Flux as a fixture and note the source
commit.

### A2 — per-element values [FP] (Milestone 2)

**Problem.** Live recolouring needs each coloured SVG element to carry the value it was coloured
by, and a statement of which paint properties the scale drives.

**Change.** For each mappable in a colour scale, postprocess sets:
- on the mappable group: `data-color-scale="<scale id>"`;
- on each element: `data-value="<repr float>"`, or `data-missing="1"` for NaN/masked values;
- `data-paint="fill"` | `"stroke"` | `"fill stroke"`, the properties the scale colours for that
  mappable (on the group). Rules: a collection whose `get_edgecolor()` is `'face'` (or equals the
  facecolors) → `"fill stroke"`; `contour` (lines) → `"stroke"`; filled shapes → `"fill"`.

Per mark type:

| Mark | Elements | Implementation |
|---|---|---|
| hexmatrix | each `<path>` hex | already has `data-value`; add the group attrs; add `"svgId"` to each `hexmatrix.bins[]` entry (index-aligned with `field_names`; namespaced by `panels.namespace`) |
| heatmap `cells=True` (QuadMesh) | each cell `<path>` | extend `_inject_field`: `data-value` from the captured array (row-major, masked → missing) |
| heatmap default (`imshow` → one `<image>`) | none | `recolor: "regenerate"`. Optional flag `fp.heatmap(..., value_raster=True)` writes `<plot>.<scale id>.values.json` (shape + row-major float list, NaN → null) next to the SVG, referenced from the colour scale as `"valueRaster": "<file>"`; it is written with the staged-write triplet |
| contourf | each band `<path>` | `data-level-low` / `data-level-high`; the Flux recolour uses the band's matplotlib colour rule: the colour of `norm((low+high)/2)`, except extend bands (verify against `ContourSet` for 3.11 and pin it in a test) |
| contour (lines) | each level | `data-value` = level; `data-paint="stroke"` |
| scatter `c=` | each point `<use>` / `<path>` | see A3 |

Tests:
- every hex / cell element has `data-value` equal to the manifest value;
- groups carry `data-color-scale` and `data-paint`;
- the contourf band colour rule reproduces matplotlib's RGBA for every band.

### A3 — coverage: every colour-mapped call gets a scale [FP] (Milestone 2)

1. **`fp.scatter(..., c=…)`** (`api.py` ≈ 74):
   - when the collection has a non-None `get_array()`, run `fields._options` for its
     cmap/vmin/vmax (series = the scatter series), register `field_config = {"kind": "scatter"}`
     and `field_artist`;
   - add `mark.data['c'] = values(get_array())` and, when sizes vary, `mark.data['size']`;
   - in `_inject_points`, write `data-value` from `c`;
   - the series payload gains `color: {"scale": "<id>"}`.

   **Prerequisite: C1** (bubble / array-style scatters emit `<path>` not `<use>`).
2. **Auto-promotion of raw mappables.** At `_save` step 3, for every untagged `ScalarMappable` in
   `ax.images` and `ax.collections` with `get_array() is not None` (excluding colorbar solids),
   create an anonymous scale with id `<its gid>`, `recolor: "regenerate"` for images. This is not a
   series, just a scale record + `data-color-scale` on its group. It gives raw matplotlib colour
   linking with colorbars and editability via A5.
3. **seaborn.** In `tag_seaborn`, add `plot="heatmap"`: the QuadMesh becomes an `fp.heatmap`-style
   field mark with `cells` ids (if ≤ the raster threshold) and its colorbar is linked. Also
   `histplot` / `kdeplot` 2D (`bivariate`) as fields.
4. **`fp.image`** (D1) images get one scale per channel.

Tests: scatter with `c=` + colorbar → one scale with both mappable and colorbar; `data-value` on
each point; an override `vmax` applies on rerun; a raw `imshow` gets an anonymous scale.

### A4 — shared scales across panels [FP] (Milestone 4)

- API: `fp.color_scale(name, *, cmap=None, norm="linear", vmin=None, vmax=None, center=None,
  robust=False)` declares a scale on the current figure (`registry_for(fig)._scales[name]`).
- Every colour helper (`heatmap`, `contour`, `contourf`, `scatter c=`, `hexmatrix`) accepts
  `scale="name"`:
  - its `key` becomes `name`;
  - its cmap/norm come from the declaration;
  - limits default to the **union** of all members' finite values.

  Implement the union lazily: members register their value arrays; at `_save`, before render,
  compute the shared vmin/vmax and call `mappable.set_norm(copy)` / `set_clim` on every member.
  This must run before `final_layout`, so move the resolution into `api.save` before `snapshot`.
- `fp.colorbar(scale="name", ax=[...])` draws one key for the scale.
- The manifest has one `colorScales` entry with all mappables. A Flux edit then recolours all
  panels.
- Tests:
  - two heatmaps with `scale="corr"` → identical norm limits = union;
  - one scale entry, two mappables;
  - an override applies to both.

### A5 — recipe colour controls v2 [FP] (Milestone 2)

Extend `__fluxplot__[<scale id>]`. The old flat `{cmap, vmin, vmax}` stays valid.

```jsonc
{
  "cmap": "magma" | {"lut": ["#…"], "under"?: "#…", "over"?: "#…", "bad"?: "#…"},
  "reversed": false,
  "vmin": 1, "vmax": 50,
  "norm": {"kind": "linear|log|symlog|power|twoslope|centered", "vcenter"?: 0, "gamma"?: 0.5,
           "linthresh"?: 1, "linscale"?: 1},
  "extend": "neither|min|max|both"
}
```

Implement in `fields._options`, factored into `colorscale.apply_override(kwargs, override)`:
- `cmap` as a LUT → `ListedColormap(lut, name="custom:<sha1 of lut>[:10]")` + `set_under` /
  `set_over` / `set_bad`;
- `reversed` → `cmap.reversed()`;
- a norm-kind switch builds the matplotlib norm (`LogNorm`, `SymLogNorm(linthresh, linscale)`,
  `PowerNorm(gamma)`, `TwoSlopeNorm(vcenter)`, `CenteredNorm(vcenter)`) with the given or carried
  limits;
- validation: log needs `vmin > 0`; twoslope needs `vmin < vcenter < vmax` → raise a `ValueError`
  that names the key;
- `extend` is passed to the colorbar (`fp.colorbar` reads the scale's `extend`) and recorded.

`api._save` writes back the *complete* current state in this v2 shape, so the Flux editor always
starts from the real values.

Tests: each norm switch round-trips (the recipe after the rerun equals the override); invalid
combinations raise with the key in the message.

### A6 — colorbars as derived, re-drawable guides [FP] (Milestone 2)

**Problem.** Colorbar solids (256 quads) are always rasterized (`raster.py` ≈ 165); every save
prints a rasterization warning. Flux can't redraw a colorbar or its ticks.

**Change:**
1. **Vector gradient solids.**
   - Stop rasterizing colorbar solids in `raster.plan`.
   - In postprocess, replace the solids group's children with a single `<rect>` (the
     min/max bbox of the original quads in SVG user units) filled by
     `url(#<solidsId>.gradient)`: a `<linearGradient>` in `<defs>`, direction by orientation. It
     has **two stops per LUT entry** at the entry's start/end offsets, giving exact hard steps. For
     a log norm the offsets are still uniform in the colorbar's axis space (the colorbar axis is
     log-scaled and the solids are uniform in normalised space; verify with a LogNorm fixture).
   - Keep the solids id and the `colorbar-solids` role.
   - Gradient id `<solidsId>.gradient`, deterministic.
2. **Colorbar anchors.** Extend the colorbar guide in `fields.colorbar_guides`:
   - `anchors: [{value, svg}]` along the long axis, computed exactly like `capture.capture_axes`
     does for axes, from the colorbar axes' `transData` for vmin and vmax (and vcenter for
     twoslope);
   - `axisLength`, `orientation`;
   - `tickLocator: "auto" | "log" | "fixed"`, `tickFormatter: "plain" | "sci" | "log" | "percent"
     | "custom"` (inspect the colorbar axis' major locator/formatter classes);
   - `ticks` (already present);
   - `extendParts: {min?: id, max?: id}` — tag the extend triangles (colorbar `_extend_patches`)
     as `<cb>.extend-min` / `.extend-max` with role `colorbar-extend`.
3. Colorbar guide field `colorScale: "<scale id>"`.

Tests:
- no rasterization warning for a plain heatmap + colorbar;
- the gradient has 2×N stops;
- the rendered PNG (rsvg-convert, if available; otherwise skip) is visually equal within a
  tolerance to matplotlib's own PNG of the colorbar;
- the anchors map vmin/vmax to the solids rect ends ±0.01 user units.

### A7 — Flux: live colour scales [FLUX] (Milestone 3)

**A7.1 Pure colour-scale math.**
- New file `src/lib/plot/colorscale.ts`, a pure module (no DOM):
  - `normalize(norm, v): number` for every norm kind — **the formulas must match matplotlib 3.11
    exactly**; port from `matplotlib/colors.py`: `Normalize`, `LogNorm` (log10-transformed
    linear), `SymLogNorm` (`_SymmetricalLogTransform` with `linthresh`, `linscale`, `base`),
    `PowerNorm` (clip negatives to 0 before `**gamma`), `TwoSlopeNorm` (piecewise
    `[vmin, vcenter, vmax] → [0, .5, 1]`), `CenteredNorm`, `BoundaryNorm` (bin index /
    `ncolors`);
  - `lookup(colormap, x): string` (LUT indexing rule from A1);
  - `colorFor(scale, v)`.
- Test `scripts/verify-colorscale-parity.ts` reads the fixture `tests/fixtures/colorscale_vectors.json`
  (copied from fluxplot to `scripts/fixtures/`) and asserts every `expected_hex`. Register it in
  `scripts/verify-manifest.json` (tier `pure`).

**A7.2 Element state and op.**
- In `src/lib/types.ts`:

  ```ts
  export interface ColorScaleView {
    cmap?: string | { lut: string[]; under?: string; over?: string; bad?: string };
    reversed?: boolean;
    norm?: { kind?: "linear" | "log" | "symlog" | "power" | "twoslope" | "centered";
             vmin?: number; vmax?: number; vcenter?: number; gamma?: number;
             linthresh?: number; linscale?: number };
    extend?: "neither" | "min" | "max" | "both";
  }
  // on SemanticPlotElement:
  colorScale?: Record<string /* scale id */, ColorScaleView>;
  ```

- In `src/lib/ops.ts`, `setPlotColorScale(p, elementId, scaleId, patch | null, defaults?)`, modelled
  on `setPlotView`: merge the patch; normalise values equal to the manifest defaults to absent;
  `null` resets.
- Resolving a named `cmap` needs a full LUT. `colormaps.gen.ts` carries only 32 stops. Change
  `scripts/gen-color-collections.mjs` to also emit full 256-entry LUTs into a lazily imported
  `colormapLuts.gen.ts`, to keep the main bundle small. The source of truth remains fluxplot's
  `definitions/colormaps.json`; if that file only has sampled stops, fluxplot's
  `tools/build_color_definitions.py` must emit 256-entry LUTs too (B4).

**A7.3 DOM application.**
- New `src/lib/plot/colorScaleDom.ts` with `applyPlotColorScale(root, manifest, colorScale, elId)`:
  - for each manifest `colorScales[]` entry with a view override, find
    `[data-color-scale=id]` groups under `root` (ids are element-prefixed; use `partDomId`);
  - for each descendant with `data-value`, compute the colour and write `style.fill` and/or
    `style.stroke` per the group's `data-paint`; elements with `data-missing` get `bad`;
  - store the original paint in `data-flux-paint0` so reset restores it exactly (mirror how
    `restoreProjection` works);
  - skip scales with `recolor: "regenerate"` and report them through a
    `plotColorScaleIssues(manifest, colorScale)` helper (mirror `plotViewIssues`).
- Call it wherever `applyPlotView` is called (`mount.ts`, `inlineMarkup.ts`, `export.ts`,
  `slide/player/render.ts`, `slide/player/transform.ts`), **before** `applyOverrides`, so explicit
  per-part user overrides still win.
- Fallback: manifests without `colorScales` (old plots) → no-op.

**A7.4 Colorbar redraw.**
- For each `colorbars[]` of an edited scale:
  - rewrite the `<linearGradient>` stops from the new LUT;
  - recompute tick positions from the guide's `anchors` + the new norm;
  - move existing tick and tick-label nodes and rewrite the label text using `tickFormatter`
    (plain: shortest `%g`; log: `10^k` as `10<tspan baseline-shift="super">k</tspan>`);
  - hide surplus ticks; clone the last tick node for missing ones (new ids
    `<cb>.tick.<k>` / `<cb>.tick-label.<k>` continuing the numbering).
- Put this in the same module. Shared tick generation (nice numbers, log decades) goes in
  `src/lib/plot/ticks.ts`, which F4 also uses for axes.

**A7.5 Editor.**
- Rewrite `ColorScaleControls.svelte` to read `manifest.colorScales`, falling back to
  `series.field` for old manifests.
- Per scale:
  - colormap picker + reverse toggle;
  - norm-kind select (limited to `editable.normKinds`);
  - min / max, plus center (if `editable.center`) and gamma (power), with validation per kind;
  - an extend select;
  - a live preview while scrubbing (writes `el.colorScale` through the op, the same UX as
    `AxisView.svelte`);
  - "Apply to source" (writes `__fluxplot__` v2 per A5 and regenerates, then clears the live
    override);
  - "Reset".
- Mount it in the Inspector and the FluxFig property menu as well as the X-ray (`propertyMenu.ts`
  ≈ 632 currently routes to the X-ray only).

**A7.6 Verbs.**
- In `flux-core/verbs.ts`, add `set_plot_color_scale` / CLI `set-plot-color-scale`, modelled on
  `set_plot_view` (≈ 3061). Params: `target, elementId, beatId?, scaleId?` (default: the only
  scale, else error listing them), `cmap?, reversed?, norm?, vmin?, vmax?, vcenter?, gamma?,
  extend?, reset?, regenerate?`. With `regenerate: true`, it writes the recipe `__fluxplot__` and
  reruns instead of setting live state.
- Widen `rerun_plot` params (≈ 2493) to `z.record(z.unknown())`. Keep the argv encoding in
  `recipeContract.mjs`: objects are JSON-encoded, and `__fluxplot__` stays env-only.
- Add a `get_plot_color_scales(target, elementId)` read verb returning `colorScales`.

**A7.7 Slides.** `colorScale` is an ordinary element property, so a slide Change can animate it:
- interpolate `vmin` / `vmax` / `vcenter` linearly — or in log space for a log norm;
- interpolate a colormap change per LUT index in OKLab (resample both LUTs to 256 first);
- norm-kind changes switch at t = 0.5.

Implement in `slide/player/transform.ts` next to the view interpolation, and add the
`state.colorScale` key to `set_transform`.

**A7.8 Types.** `FluxPlotField.kind` gains `"hexbin"` and `"scatter"`; add `FluxPlotColorScale`,
generated per F3 when that lands.

Acceptance for A7:
- the verify parity test passes;
- a GUI verify (`scripts/verify-colorscale-gui.mjs`, tier `ui`) opens a hexmatrix plot, sets
  `vmax` live, and asserts that a known hex's fill equals the parity-computed colour and that the
  colorbar gradient changed;
- "Apply to source" regenerates and the live override clears.

---

## 3. Workstream B — the wider colour system

### B1 — theme-able plots [FP + FLUX] (Milestone 4)

**Problem.** Flux decks have seven themes (`src/lib/slide/theme.ts` ≈ 20–144), applied as `--sl-*`
CSS variables. A fluxplot SVG hard-codes every scaffold colour, so a light plot on `flux-dark` is a
cream rectangle. The manifest doesn't record which fluxplot theme was used.

**[FP] changes:**
1. **Theme record.** Each `fx.use_*` sets a module global `style.ACTIVE = {"name": "light",
   "tokens": {...}}` with tokens taken from the rcParams it set:
   - `ink` = `text.color`;
   - `label` = `axes.labelcolor`;
   - `tick` = `xtick.color`;
   - `axis` = `axes.edgecolor`;
   - `grid` = `grid.color`;
   - `plot` = `axes.facecolor`;
   - `paper` = `figure.facecolor`.

   `fp.save` emits `manifest.style = {"theme": name|null, "tokens": {...hex}}`. Tokens are
   re-read from rcParams at save time; `theme` is null if rcParams no longer match the recorded
   tokens.
2. **Semantic paint tags.** In postprocess, for guide elements only (`GuideTag` roles: axis,
   spine, tick, tick-label, axis-title, gridline, title, legend, legend-label, annotation,
   colorbar-*) and the overlay roles significance-bracket / reference-line / annotation, add
   `data-ink-fill="<token>"` and/or `data-ink-stroke="<token>"` when the element's (or its
   drawable children's) paint equals a token hex.

   This is not a heuristic: fluxplot set those colours from those rcParams itself, and
   restricting the pass to scaffold/overlay roles excludes data marks. Mark each overlay drawn by a
   helper with its token explicitly (bracket, identity line, reference line) instead of relying on
   equality. The legend frame, the axes background patch and the figure background patch get
   `role="background"` guides (`axes.background`, `figure.background`) plus `data-ink-fill`.
3. **CSS variables (opt-in).** `fp.save(..., theme_vars=True)` also rewrites those paints to
   `var(--fx-<token>, <hex>)`. It is off by default until checked in Illustrator/Inkscape
   (`rsvg-convert` handles the fallback).
4. **Theme as a recipe control.** `__fluxplot__.theme = "dark"` makes every `fx.use_*` call apply
   that theme instead. Implement `style._theme_override()`, read from `recipe.params()`, at the top
   of each `use_*`.
5. `significance_bracket` line colour: replace `"black"` with `rcParams['text.color']`
   (`api.py` ≈ 403; B5).

**[FLUX] changes:**
- A per-element `followTheme?: boolean` on `SemanticPlotElement`, default `true` on slides and
  `false` in Paper/Figure.
- When on, a new `applyPlotTheme(root, manifest, deckTheme)` in `src/lib/plot/themeDom.ts` maps
  tokens to deck variables: `ink`, `label`, `tick` → `--sl-text`; `axis` → `--sl-text-muted`;
  `grid` → a 15% mix of text into bg; `plot`, `paper` → transparent. It writes the paint on every
  `[data-ink-fill]` / `[data-ink-stroke]` node.
- Call it in the same places as `applyPlotColorScale`, before overrides.
- Data colours are never touched.
- UI: a toggle in the plot Inspector.
- Verify: place a `use_light` plot on `flux-dark` → tick-label fill equals the deck text colour;
  data line colour unchanged.

Tests [FP]: every scaffold element of a `use_light` plot has a token tag; a `use_dark` save records
`theme: "dark"`; `FLUX_PARAMS={"__fluxplot__": {"theme": "dark"}}` + `fx.use_light()` → dark
tokens; determinism holds.

### B2 — semantic series colours and cross-figure consistency [FP, small FLUX] (Milestone 4)

**Problems:**
- Series colours are recorded nowhere in the manifest (they appear only in SVG `style=`).
- glowbar/fluxbox record raw hex, never the palette spec used.
- `_fieldmap.categorical_colors` (≈ 37–45) assigns `C{len(resolved)%10}` over *sorted names*, so a
  category's colour depends on which other categories are present: `b` is blue in `['b','c']` but
  orange in `['a','b','c']` [verified].

**[FP] changes:**
1. **Colour record per series.**
   - At `data.refresh`, capture the primary paint: `Line2D.get_color()`; a collection's facecolor
     if uniform (else `"varies"`); a bar's facecolor if uniform.
   - Emit `series[].color = {"hex": "#35ab49", "alpha": 1.0, "token": "flexoki.green-400"?,
     "palette": {"name": "tol.bright", "index": 2}?, "scale": "<id>"?}`.
   - `token` comes from an exact reverse lookup over the Flexoki table in `colors.py` plus
     `palettes.json` (build the dict once).
   - glowbar/fluxbox add `palette: <the spec string or list>` to their summary payloads.
2. **Category registry** `fp.colors.categories`, a new class in `colors.py`:
   - `assign(mapping: dict[str, color])` pins colours;
   - `get(name, *, palette=None) -> hex` returns the pinned colour, else assigns the next unused
     slot of `palette` (default: the active theme's cycle) in first-request order and remembers it;
   - `load(path=None)` / `save(path=None)`;
   - auto-load on first use from the first `fluxplot.colors.json` found walking up from the cwd to
     the git root (or `$FLUXPLOT_COLORS`);
   - file format: `{"spec": "fluxplot/colors", "version": 1, "categories": {"SD": "#bc5215", …},
     "palette": "flexoki"}`;
   - it never writes a file implicitly.
   - Integrate it in `_fieldmap.categorical_colors` (replace the positional `C{n}` fallback with
     `categories.get(name)`), `glowbar._frame` group colours (use a registry colour when the
     category is pinned, overriding `palette` for that category), `tag_seaborn` hue series, and
     `fp.line` / `fp.scatter` when `color` is not given *and* `fp.colors.categories.auto_series`
     is True (default False, to avoid changing existing plots).
3. **Palette and series colour as recipe controls:**
   - `__fluxplot__.palette = "tol.bright"` → `fx.use_*` sets the prop cycle from it, and glowbar
     `palette` defaults to it;
   - `__fluxplot__.series = {"<series id>": {"color": "#…"}}` → the helpers check it before drawing.
     Implement `_series_color_override(series)` in `api.py` and use it in line/scatter/bar/hist/
     errorbar/area and the signature plots.

**[FLUX] changes:**
- A new verb/op `set_series_color(target, elementId, seriesId, color)` writes `el.overrides` for
  **every part of the series**: `series.svg.*`, `components[].svgId` and members, plus the legend
  swatch found via `guides[legend].entries[].series == seriesId → swatch`.
- The Inspector "Series colour" field uses it.
- An optional project "Colours" panel edits `fluxplot.colors.json` and offers "Regenerate all plots".

Tests [FP]:
- `categorical_colors(['b','c'])['b'] == categorical_colors(['a','b','c'])['b']`;
- `fluxplot.colors.json` pins apply to glowbar group colours;
- `series.color.token` is set for `color=colors.green400`;
- the `__fluxplot__.series` override recolours on rerun.

### B3 — accessibility lint [FP, small FLUX] (Milestone 4)

**[FP]** New module `src/fluxplot/colorcheck.py`:
- `simulate(hex_list, kind, severity=1.0)`: Machado et al. 2009 CVD matrices (deuteranomaly,
  protanomaly, tritanomaly) applied in linear sRGB — no new dependency; embed the 3×3 matrices at
  severity 1.0 and interpolate from the paper's 0.1-step tables if a severity is needed.
- `delta_e(a, b)`: CIEDE2000 or CAM02-UCS. The repo already has perceptual helpers in
  `signature_fluxplots/_colour.py` (`lab`, `perceptual`); reuse them or implement CIEDE2000
  directly.
- `contrast(fg, bg)`: WCAG 2 relative-luminance ratio.
- `check_palette(colors, bg) -> list[Finding]`, with `Finding = {kind, a, b, value, threshold,
  message}`. Rules:
  - pairwise ΔE < 10 under any simulation → `cvd-confusable`;
  - greyscale ΔL < 10 → `greyscale-confusable`;
  - mark contrast < 3:1 → `low-contrast-mark`;
  - text < 4.5:1 → `low-contrast-text`.
- `check_colormap(cmap)`: ΔE step coefficient of variation > 0.35 → `non-uniform`; non-monotone
  lightness for a sequential map → `non-monotone`.
- `check_figure(fig)`: gathers series colours (B2 capture) + text colours vs the axes background.
- `fp.save(..., lint="off"|"warn"|"error")` (default `"off"` initially; `"warn"` after the house
  palette passes). Findings go to `SaveResult.warnings` and `manifest.quality = {"color":
  [findings]}` (schema-optional).
- Reorder `style.CYCLE_LIGHT` / `CYCLE_DARK` so the measured weak pairs are not adjacent
  (deuteranopia blue/purple ΔE 3.1 and green/red 5.1; tritanopia green/cyan 3.3 — re-measure with
  the new module). Do this in a **separate commit**, because it changes default colours of existing
  plots, and note it in the README.
- Tests: known confusable pairs (pure red / pure green under deuteranopia) are flagged; black on
  white passes; the house cycle has no adjacent confusable pair after the reorder.

**[FLUX]** A "View as" toggle in Figure and Slide views: none / deuteranopia / protanopia /
tritanopia / greyscale. It is an SVG `<filter>` with a `feColorMatrix` (the same Machado matrices)
applied to the canvas root; no data changes.

### B4 — colormap toolkit and definitions hygiene [FP] (Milestone 4)

1. `colors.maps.truncate(name_or_cmap, lo, hi, n=256) -> Colormap` named
   `"<name>[lo:hi]"` (resolvable: extend `maps.get` to parse the `[a:b]` suffix).
2. `colors.maps.discretize(name_or_cmap, boundaries | n, *, extend="neither") -> (cmap,
   BoundaryNorm)`. Helpers accept `norm=BoundaryNorm`; the scale records `kind: "binned"`.
3. `register()` registers the reversed map (A0.2).
4. One "discrete" threshold: a single constant `DISCRETE_MAX = 32` in `colors.py`, used by
   `tools/build_color_definitions.py` (≈ 40, currently N ≤ 128) and `_colour._from_colormap`
   (≈ 105).
5. Ship the house maps: `build_color_definitions.py` emits a `flexoki` collection
   (`flexoki_sequential`, `_warm`, `_diverging`, `_terrain`, `_spectrum`) into
   `definitions/colormaps.json`, and emits **full LUTs** (N entries) for every continuous map in
   addition to any sampled stops (A7.2 needs this). Re-run Flux's `scripts/gen-color-collections.mjs`
   afterwards.
6. `flexoki_diverging` / `flexoki_spectrum` are not perceptually uniform (ΔE step coefficient of
   variation 0.61 vs 0.26 for batlow). Either refit their anchors in CAM02-UCS, or set
   `info()["uniform"] = False` and stop using them as the no-cmasher fallback defaults. Recommended:
   mark them non-uniform now and refit later.
7. **Single source of truth.** `Flexoki_definition.tokens.json` (repo root), `colors.py` (≈ 95–250
   hard-coded table) and `definitions/palettes.json` are three copies with identical values
   [verified]. Make the tokens file canonical: move it to `src/fluxplot/definitions/flexoki.tokens.json`,
   have `colors.py` load it at import, and have `build_color_definitions.py` generate
   `palettes.json` from it. Add `tests/test_color_definitions.py::test_single_source`. Note that
   fluxplot's "green" is a custom hue and Flexoki's original green is stored as "olive"; preserve
   that mapping.
8. `_colour._resolve_str`: `palette='red'` currently resolves to the 13-step Flexoki red *ramp*.
   Check single colours (`matplotlib.colors.is_color_like`) first and require `flexoki.red` for the
   ramp. This is a behaviour change: grep the tests and examples for bare palette names.
9. Keep alpha: `_fieldmap.categorical_colors` uses `to_hex(..., keep_alpha=True)`.

### B5 — signature plots on dark grounds [FP] (Milestone 1 for the bracket, Milestone 4 for the rest)

- `api.significance_bracket` (≈ 395–411):
  - line colour defaults to `rcParams['text.color']`;
  - accept `text_kw: dict | None` for the label (fontsize, offset);
  - apply `color` to both the line and the label;
  - on a log axis with `y <= 0`, raise `ValueError`.
- glowbar (`glowbar.py`):
  - replace the mean line's `_darken(col, 0.35)` (≈ 686) with fluxbox's `_median_ink(col, alpha,
    ground, contrast)` (`fluxbox.py` ≈ 126) — move `_median_ink` to `_colour.py`, since both use it;
  - when the axes background is dark (perceptual L < 50), swap `shade_range` to `(22, 88)`
    semantics: lift instead of deepen;
  - replace `CONNECT_GREY` with a theme neutral (`style.ACTIVE.tokens.grid`, fallback to the
    current grey);
  - for qualitative palettes, derive the group colour's ink against the ground instead of the fixed
    `NEUTRAL` base-600 (`_colour.py` ≈ 299, 318).
- Test matrix `tests/test_dark_ground.py`: for glowbar, fluxbox and hexmatrix × `use_light` /
  `use_dark` / `use_paper`, assert every mark's colour has WCAG contrast ≥ 1.5 against the axes
  background (B3's `contrast`), and bracket/identity lines ≥ 3.

### B6 — value × confidence colouring [FP] (Milestone 6)

- `alpha_by=` on `hexmatrix`, `heatmap`, `scatter`: `"count"` (hexmatrix), a column name, or an
  array; `alpha_range=(0.25, 1.0)`; `alpha_norm="linear"|"log"`.
- Implement it by setting per-element alpha in the RGBA array after colour mapping, via the
  collection's `set_facecolor(rgba)` (so matplotlib renders it).
- Record it as a second channel `colorScales[].alpha = {"source": "count", "range": [.25, 1],
  "norm": {...}}` and a per-element `data-alpha-value`. The Flux live recolour (A7.3) applies
  `fill-opacity` from it.
- Use case: a hexmatrix mean map where few observations fall, or a correlation matrix with
  non-significant cells washed out (`alpha_by=p_values` with `alpha_norm="log"`).
- Tests: alpha values monotone in the source; the recorded channel round-trips.

---

## 4. Workstream C — core correctness [FP]

Each item: the problem → the change → the test.

**C1 Bubble scatter loses per-point ids. [verified] (M1)**
- Problem: `_inject_points` (`postprocess.py` ≈ 133) requires one `<use>` per point. With `s=`
  arrays (and some other cases) matplotlib's SVG backend emits direct `<path>` children, so
  `points: 0` and a warning.
- Change: if the `<use>` count ≠ N, collect the group's direct `<path>` children (excluding
  `<defs>` / `clipPath`). If their count == N, use them as the members (same ids / attrs).
- Tests: `fp.scatter(ax, x, y, s=np.linspace(5, 50, 40), series='b')` → 40 `b.point.k` ids, no
  warning; also with `c=` + colorbar.

**C2 `tag_seaborn` mislabels KDE curves as a bar's error bars. [verified] (M1)**
- Problem: `_is_segment` returns `bar_context` for every line (`api.py` ≈ 317–320).
- Change: a line is a segment only if it has exactly 2 finite points and constant x (vertical
  bars) or constant y (horizontal bars), within 1e-12 of each other.
- Tests: `sns.histplot(x, kde=True)` → the KDE is a `line` series, not an `errorbar`;
  `barplot` error bars still join.

**C3 Notebook provenance names a helper module. [verified] (M1)**
- Problem: `provenance.discover_script` (≈ 45–72) falls back to the first `.py` stack frame when
  `__main__` has no `__file__`, so a notebook calling `mylib.plot()` records `mylib.py` as the
  rerunnable script.
- Change: return `None` unless rule 1 (a real `__main__` script) holds. Delete the stack walk, or
  keep it only when `__main__.__file__` exists but is inside site-packages (the `python -m` case),
  and document it.
- Tests: simulate `__main__` without `__file__` (monkeypatch) → `None`, and the recipe has
  `scriptDiscovery: "unavailable"`.

**C4 Twin axes become two panels. [verified] (M5)**
- Problem: `panels.plan` makes `ax.twinx()` its own panel with a duplicate x scaffold.
- Change:
  - in `panels.plan`, pair axes whose `get_position()` bounds are equal (to 1e-9) and that share
    x (`ax.get_shared_x_axes().joined(a, b)`) or y;
  - the secondary becomes part of the primary's panel;
  - `autotag_scaffold` for the secondary tags only its non-shared axis as `axis.y2.*` (or
    `axis.x2.*` for twiny) and skips the shared one;
  - `capture_axes` emits `axes[0].y2 = {scale, domain, anchors}`;
  - the secondary's series get `axis: "y2"`.
- Schema: add the optional `y2` / `x2`.
- Tests: a twinx plot → one panel, `axis.y2.title` present, no duplicate `axis.x` ids; series on
  the twin carry `axis: "y2"`.
- **[FLUX]** follow-up: `viewFits` handles `y2` (a separate view key `y2`).

**C5 Figure-scope artists. [verified] (M5)**
- Problem:
  - `fig.legend()` is untagged (`legend_1`);
  - `fig.suptitle` becomes `panel.a.figure.title`;
  - the first panel claims all `fig.texts` / `fig.lines` / `fig.patches` / `fig.images`
    (`tagger.py` ≈ 236, 271, 293–297).
- Change:
  - add `tagger.autotag_figure(fig, alloc)`, run once in `_save` with an **unprefixed**
    allocator. It tags `fig.legends[k]` → `figure.legend[.k]` (entries as for axes legends),
    `fig._suptitle` → `figure.title`, `_supxlabel` / `_supylabel` → `figure.xlabel` / `.ylabel`,
    and `fig.texts` / `lines` / `patches` / `images` → `figure.annotation.k` / `figure.extra.*`;
  - remove the `ax.figure.*` lists from `autotag_scaffold` / `_sweep_extra`;
  - the manifest gets top-level `figure: {title?, legends[], annotations[]}` and the parts tree
    puts them under `figure`.
- Tests: suptitle id is `figure.title` in a 2-panel figure; a `fig.legend` is tagged with entries.

**C6 `ax.artists` and `ax.tables` are never swept. [verified] (M5)**
- Change: add them to `_sweep_extra` (kind by `artist_kind`, tables → `container`). Recurse into
  `AnchoredOffsetbox` children to tag the text as `…label`.
- Tests: an `AnchoredSizeBar` gets `extra.artist.0` (or `scalebar.*` when made with
  `fp.scalebar`, D1).

**C7 Spine ids depend on collision repair. (M5)**
- Change: `tagger.py` ≈ 222 uses `axis_id(which, "spine", side)` → `axis.x.spine.bottom`,
  `axis.x.spine.top`, `axis.y.spine.left`, `axis.y.spine.right`.
- **This renames ids**: add a manifest `idAliases: {"axis.x.spine": "axis.x.spine.bottom", …}`
  (schema-optional) for one minor version, and have Flux resolve overrides through `idAliases`
  (**[FLUX]**: in `applyOverrides`' `resolveTargets`).
- Tests: `imshow` plots have no `-2` spine ids.

**C8 Non-ASCII series names collide. [verified] (M5)**
- Change, in `ids.slugify`:
  - transliterate Greek letters (`α→alpha`, `β→beta`, …, `µ/μ→mu`), `°→deg`, `%→pct` and
    superscript/subscript digits;
  - if the result is still empty or `"series"` for a non-empty input, append
    `-<sha1(name)[:6]>`.
- This changes ids only for names that were previously broken or colliding.
- Tests: `α`, `β` → `alpha`, `beta`; `IL-6 (pg/mL)` and `IL-6 [pg/mL]` → distinct valid ids.

**C9 The errorbar composite is one flat role. (M5)**
- Change: `fp.errorbar` registers:
  - the data line as `role="line"` (if its linestyle isn't none);
  - markers as a `point` group (if a marker is set, via `tag_points`-style split);
  - caps as `role="cap"`, bars as `role="errorbar"`;
  - `data.xerr` / `yerr` broadcast to length N (a scalar → repeated), plus `errShape: "scalar" |
    "symmetric" | "asymmetric"`.
- Keep `svg.errorbar` as today (compat).
- Tests: `fmt='o-'` → `s.line`, `s.point.k`, `s.caps`, `s.errorbar`; a scalar `yerr` is broadcast.

**C10 Legend entries join by text only. [verified] (M5)**
- Change: in `autotag_scaffold`'s legend pass, map each `legend.legend_handles[k]` back to the
  source artist. Matplotlib's `ax.get_legend_handles_labels()` order equals the entry order when
  `ax.legend()` is called without explicit handles; for explicit handles, record the mapping in a
  new `fp.legend(ax, handles=None, labels=None, **kw)` wrapper that stores `{entry k: artist id}`
  on the legend.
- Resolve the artist → Mark → series, and emit `entries[k].series`.
- Tests: `ax.legend([ln], ["Control"])` on `fp.line(series="ctl")` → entry 0 series `ctl`.

**C11 seaborn coverage. [verified] (M5)**
- Problem: box/violin/strip/swarm/point plots tag nothing; `barplot(x=g, hue=g)` tags nothing
  silently; `scatterplot(hue=)` is fused into one series.
- Change: `tag_seaborn(ax, plot=..., data=df, x=, y=, hue=)`. When `data` / `hue` are given, derive
  identity from the frame exactly:
  - hue levels in seaborn's order (`categorical_order`);
  - for scatterplot(hue), split the single PathCollection's offsets into per-hue series by row
    mask. The PathCollection stays one artist, so register one `point` mark per hue with
    `member_indices` subsets — this needs a `Mark.data['point_subset']` supported by
    `data.point_indices`;
  - box/violin via `ax.containers` / patches in category order, strip/swarm per-category
    collections, pointplot lines + markers.
- Always `warnings.warn` when nothing was tagged.
- Tests: one per plot kind, asserting the series ids and counts.

**C12 Small items. (M5)**
- `tag_seaborn` only removes empty lines seaborn created. Record `id`s of lines existing before
  the seaborn call is impossible here, so instead remove only empty lines whose label starts with
  `_` or is a legend proxy (`get_label()` in the legend texts).
- FacetGrid fallback name: use `f"{ax.get_title() or 'panel'}"` rather than `data`.
- Replace the bare `except Exception: pass` in legend swatch tagging (`tagger.py` ≈ 246) with a
  recorded warning.
- `build.order` emits `"gridlines"` only when gridlines exist (`manifest.py` ≈ 370).

---

## 5. Workstream D — missing capabilities [FP]

**D1 `fp.image` + `fp.scalebar` (M5, high).** New module `src/fluxplot/images.py`.

```python
def image(ax, data, *, series, pixel_size=None, units="µm", origin="upper",
          channels=None,                 # names for a (C,H,W)/(H,W,C) stack; None = single channel
          luts=None,                     # per-channel cmap name/Colormap; default gray / green/magenta/…
          display_range=None,            # per-channel (lo, hi); default (1st, 99.8th) percentile
          composite="add",               # add | max  (how channels combine to RGB)
          value_raster=False, **imshow_kw) -> ImageResult
def scalebar(ax, length, units="µm", *, loc="lower right", label=None, color=None, thickness=2.0,
             pad=0.4, name="scalebar") -> Artist
```

- `extent` comes from `pixel_size`, so the axes are in physical units.
- Each channel is its own colour scale (A1) with id `<series>.<channel>`, `norm = linear
  display_range`, `recolor: "regenerate"` (unless `value_raster=True`).
- Channels are composited in Python into one RGB `imshow`; the manifest records channels, LUTs,
  display ranges, pixel size, units and shape. In Flux the scale editor edits the display range
  (B&C) and regenerates; with `value_raster`, Flux could recolour on a canvas later.
- `scalebar` draws a bar + label (an `AnchoredSizeBar`-like artist built from a `Line2D` and a
  `Text` in axes coordinates, so it stays vector). It registers `Mark(role="scalebar", name=name,
  data={"length": length, "units": units})` with the label as `label_gid`. It gets the house ink
  (B1).
- Tests: an extent derived from the pixel size; a 2-channel composite equals a manual RGB
  composite; the scalebar length in data units equals `length`; a manifest payload exists.

**D2 `fp.band` (M5, high).** `fp.band(ax, x, lo, hi, *, series, what="95% CI", **fill_kw)`:
- wraps `fill_between`;
- registers `role="area"`, `name="band"` under the *same series* as its line (so `ctl.band` sits
  beside `ctl.line`);
- `data["band"] = {"x": …, "lo": …, "hi": …, "what": what}` (replacing the path-vertices capture
  for helper-made bands).
- Also give `fp.area` the `{x, y1, y2}` capture: extend the `area` helper at `api.py` ≈ 113 to
  record its inputs.
- Tests: `ctl.band` id, payload arrays equal the inputs.

**D3 `fp.regression`, `fp.kde` (M6).**
- `fp.regression(ax, x, y, *, series, kind="linear"|"poly"|"lowess", ci=0.95, degree=1)` draws
  the fit line (`series.fit`) + CI band (`series.band`) + optional points, and records
  `{coefficients, r2, p, n, ci}` using scipy (already a dependency).
- `fp.kde(ax, values, *, series, bw="scott", fill=False)` → line (+ area), recording the grid and
  density.

**D4 `fp.step`, `fp.stem` (M6).** `step` records `where=` and `drawstyle` in the series payload;
`stem` registers markers (`point`), stems (`segment`) and baseline (`reference-line`) under one
series.

**D5 Categorical and date axes (M6).** In `data.refresh`, when the axis uses a `StrCategoryConverter`,
add `data.xLabels` (the category for each x); when it uses dates, add `data.xIso` (ISO-8601
strings). The same for y.

**D6 Insets, secondary axes, broken axes (M6).**
- Panels for `ax.inset_axes` children get `insetOf: "<parent panel id>"`.
- `fp.secondary_axis(ax, "top", functions=(f, g), label=)` wraps `secondary_xaxis` and records the
  transform as samples.
- A broken-axis helper is out of scope unless requested.

**D7 Notebook recipes (M5).**
- `recipe={"notebook": path, "cell": label}` records `scriptDiscovery: "notebook"` and
  `notebook: {path, cell, sha256}`. It is not rerunnable by Flux yet: omit `command`.
- In a QMD/Jupyter kernel, auto-detect the notebook path via `$QUARTO_DOCUMENT_PATH` or the
  `__session__` / `__vsc_ipynb_file__` globals when present; never guess.

**D8 hexmatrix follow-ups (M6):**
- `svgId` per bin (A2);
- `alpha_by` (B6);
- a per-mark raster threshold: honour `Mark.data['raster_threshold']` in `raster.plan`, and
  hexmatrix passes `vector_limit=5000` by default so typical hexbins stay addressable;
- `aspect="auto"` with shared axes (skip `set_box_aspect` when the axes shares x/y, and warn).

---

## 6. Workstream E — stats → figure integration [FP]

**E1 k-group and post-hoc tests (M5).** New module `stats/multi_group.py`. Each function returns
one row (omnibus) or a list of rows (post-hoc), keyed by `REPORT_COLUMNS` plus the E2 columns:

| Function | Test | Effect size |
|---|---|---|
| `anova_oneway(*groups)` | classic one-way ANOVA | η², ω² (with CI via noncentral F) |
| `welch_anova(*groups)` | Welch's ANOVA | ω² |
| `kruskal_epsilon(*groups)` | Kruskal–Wallis | ε² |
| `rm_anova(table, subject, within, dv)` | repeated-measures ANOVA with Greenhouse–Geisser | partial η² |
| `friedman_kendall(table, subject, within, dv)` | Friedman | Kendall's W |
| `tukey_hsd(*groups, names=)` | Tukey HSD (`scipy.stats.tukey_hsd`) | Hedges' g per pair |
| `games_howell(*groups, names=)` | Games–Howell | Hedges' g |
| `dunn(*groups, names=, adjust="holm")` | Dunn's test | Cliff's δ |
| `pairwise(test, groups: dict, pairs=None, adjust="holm"\|"bh")` | runs an existing two-group test per pair and adjusts | — |

- Add `bh(rows)` / `bh_adjusted(p)` (Benjamini–Hochberg).
- `holm_adjusted` must pass NaN through instead of raising (`_common.py` ≈ 79).
- Validate against scipy / statsmodels / pingouin reference values in tests (hard-code the
  reference numbers; don't add dependencies).

**E2 Row columns (M5).** Extend `report_row` / `REPORT_COLUMNS` (append, don't reorder):
- `effect_size_ci_low` and `effect_size_ci_high` (floats; keep the string column for compat);
- `n_a`, `n_b` (or `n_groups` + `n_total`);
- `groups` (names), `alternative` (`"two-sided"` default; add `alternative=` to all four
  existing tests); `p_corrected_bh`.

**E3 `fp.brackets` (M5, the killer integration).**

```python
def brackets(ax, rows, *, positions: dict | None = None, pairs=None, label="stars",  # stars | p | both
             p_column="p_corrected_holm", thresholds=((0.001, "***"), (0.01, "**"), (0.05, "*")),
             ns=True, top=None, step=0.06, tip=0.02, **line_kw) -> list
```

- Each row's `groups` gives the pair (`a`, `b`). `positions` maps group name → x (the glowbar and
  fluxbox results provide it: add `GlowbarResult.positions` / `FluxboxResult.positions`, and
  `result.brackets(rows, **kw)` convenience methods).
- **Auto-stacking:**
  - sort pairs by span (short first);
  - the stack starts at `top`, or the max data y over the spanned categories plus `step` × the
    axis range;
  - each bracket goes on the lowest level where it doesn't overlap an existing bracket's x-span;
  - on log axes, levels are multiplicative steps.
- Each bracket is an `fp.significance_bracket` with `data` extended by `{test, statistic, p,
  p_corrected, correction, effect_size_method, effect_size, ci_low, ci_high, n}` taken from the
  row. The manifest overlay then states exactly which test each star came from.
- `label="p"` formats `p = 0.003` / `p < 0.001`.
- Tests:
  - three pairs over four categories stack without overlap (assert pairwise x-span/level
    disjointness);
  - the payload carries the row fields;
  - `ns` labels appear for p ≥ 0.05 when `ns=True`.

---

## 7. Workstream F — the Flux ↔ fluxplot contract

**F1 Build-preset names don't match (M1).** [FP + FLUX]
- Problem:
  - fluxplot `presets.py` emits `grow-from-baseline` (bar, box) and `fade-rise` (legend,
    annotation, bracket);
  - Flux `ANIM_TO_PRESET` (`src/lib/slide/autobuild.ts` ≈ 38) knows only `grow` /
    `grow-baseline` / `rise`, so both fall back to `fade`;
  - `presetForRole` forces point and bar to `stagger`, so `growBaseline` never fires;
  - `delayMs` is dropped (not in `FluxPlotBuildPreset`).
- [FP]:
  - add `PRESET_NAMES = ("draw-on", "fade-in", "stagger-in", "grow-from-baseline", "fade-rise",
    "write-on", "pop-in")` to `presets.py`, and enumerate it in the schema
    (`build.presets.*.animation` enum);
  - add presets for `x-hexbin` (`{"animation": "stagger-in", "staggerBy": "value", "staggerMs":
    4}`), `x-heatmap` (fade-in), glowbar and fluxbox parts (box: grow-from-baseline; mean/median:
    fade-in), surface (fade-in), colorbar (fade-in);
  - add a `staggerBy` hint: `"x" | "y" | "index" | "value" | "count" | "category"`.
- [FLUX]:
  - add `"grow-from-baseline": "growBaseline"` and `"fade-rise": "fadeRise"` to `ANIM_TO_PRESET`;
  - in `presetForRole`, use `stagger` for bars only when the preset says `stagger-in`, else honour
    `growBaseline`;
  - add `delayMs` / `staggerBy` to `FluxPlotBuildPreset` and pass them to the player (the
    `staggerBy` implementation is F7).
- Verify: a bar plot's autobuild uses `growBaseline`; an annotation uses `fadeRise` with its delay.

**F2 Leaf parts carry no role (M1 [FP], M2 [FLUX]).**
- Problem: `manifest._ref` (≈ 392) emits `{ref, kind}` only. Flux guesses the role with
  `inferRole` regexes (`src/lib/plot/tree.ts` ≈ 83), which fail on namespaced ids
  (`startsWith("annotation")` misses `panel.a.annotation.0`), so those parts become `"part"` in
  the wrong build phase. Hex members come out as `"part"`.
- [FP]:
  - `_ref(gid, kind, role)` always includes `role`;
  - `_group` nodes add `memberRole` (the members' role);
  - series and legend-entry nodes add `label` (the series `label`, else `name`);
  - components with `members` (field members, hexes) add `memberRole` from
    `m.data['field_member_role']` or `cell` / `contour-level`.
- [FLUX]:
  - role resolution order: manifest node `role` / parent `memberRole` → the DOM `data-role` →
    `inferRole` (legacy only);
  - `PartInfo` for group members uses the parent's `memberRole` (`parse.ts` ≈ 587–590);
  - after this, delete the role-set duplicates in `partStyle.ts` (≈ 29–39) and `autobuild.ts`
    (≈ 51–68) in favour of the manifest `kind` plus a small `ROLE_PHASE` table.
- Verify: a 2-panel plot with annotations → annotation leaves have role `annotation` and phase 3.

**F3 One schema, generated types (M2).**
- Problems:
  - fluxplot's `manifest.schema.json` is incomplete: missing `rasterized`, `field`, `surface`,
    `bar`, `band`, `uncertainty`, `axis.ticks` / `units`, `panels[].index`, typed guides and
    overlays;
  - it uses `$defs` under draft-07;
  - the recipe schema `$id` says 0.1.0 while `SPEC_VERSION` is 0.3.0;
  - Flux validates with its own looser schema (`src/lib/project/schemas.ts` ≈ 412–458);
  - Flux's hand-written `types.ts` has drifted (no `PartNode.kind`, `FluxPlotField.kind` lacks
    `hexbin`, missing payload types, colorbar guide fields, legend entry fields);
  - Flux `modernPlot()` (`contract.ts` ≈ 5–8) accepts any version ≥ 0.3 with no major ceiling.
- [FP]:
  - complete the schema for everything emitted, using `definitions` for draft-07 or move to
    2020-12 and update the `jsonschema` validator class in `api._validate`;
  - fix the recipe `$id`;
  - add `tests/test_schema_complete.py`, which saves one of each plot type and validates with
    `additionalProperties: false` injected recursively (a test-only strict copy) to prove every
    emitted key is declared.
- [FLUX]:
  - vendor `manifest.schema.json` and `recipe.schema.json` into `src/lib/plot/schemas/`, copied by
    a script `scripts/sync-fluxplot-schemas.mjs` that also records the fluxplot commit;
  - generate `src/lib/plot/types.gen.ts` with `json-schema-to-typescript` (dev dependency);
    `types.ts` re-exports and narrows it;
  - `validate_plot` uses the vendored schema;
  - `modernPlot()` rejects a major version > 0 (`specVersion` `^0\.`) with a clear "made by a
    newer fluxplot" message.

**F4 The axis view regenerates ticks and moves filled marks (M6).** [FP + FLUX]
- [FP]:
  - emit per-axis `tickLocator` / `tickFormatter` kinds (as in A6);
  - add per-mark data geometry: bar `{base, height, x, width}` in data units (the `bar` payload
    already has part of this; complete it), box quartiles (fluxbox/glowbar payloads have them),
    and `data-x0 / data-x1 / data-y0 / data-y1` on heatmap cells and hexes (hex: centre + radius
    in unit space via the `hexmatrix` payload).
- [FLUX]:
  - `ticks.ts` (shared with A7.4) generates nice ticks for the new domain, cloning tick nodes;
  - `applyPlotView` re-projects bars (rect path rewrite), boxes and cells, and drops them from the
    `plotViewIssues` "filled marks unchanged" message.

**F5 `get_plot_data` verb (M5).** [FLUX] `get_plot_data(target, elementId, seriesId?, fields?)`
returns, from the manifest:
- series `{id, name, kind, data, points?, color}`;
- payloads (glowbar/fluxbox stats, `hexmatrix.bins`, `distribution`, `field.values` when present);
- `colorScales`, axis domains and scales, overlays (brackets with their stats).

Add a size cap with pagination (`offset` / `limit`) for large arrays. It is the cheapest way to let
agents reason about figures precisely instead of from the PNG.

**F6 Hover readouts and legend coupling (M5).** [FLUX]
- In Figure/X-ray, hovering a mark shows its data from the manifest and `data-*`: point x/y, bar
  height, hex count/value, cell value, box stats, bracket test and p.
- `restyle_part` of a series fill/stroke also updates its legend swatch (via
  `guides[legend].entries[].series → swatch`). With B2's `set_series_color`, this is automatic.

**F7 Value morphs and data-order stagger (M6).** [FP + FLUX]
- [FP]:
  - stable member keys: bars keyed by category label (`data-key="SD"`), hexes by `row.col`,
    cells by `r.c`, glowbar parts by category/unit;
  - `capabilities.valueMorph: true` where two versions can be tweened by key.
- [FLUX]:
  - `stagger.by: "x" | "y" | "index" | "data" | {key: "value" | "count" | "index"}` in the slide
    types (`src/lib/slide/types.ts` ≈ 271), reading values from `data-value` / `data-count` /
    `data-index`;
  - a morph between two plot versions for bars (height tween by key), boxes (quartile tween) and
    colour-scaled elements (value tween through A7's colour math);
  - a "per-panel" build mode that phases panel by panel.

**F8 Stale example outputs (M1).** [FP] Regenerate `examples/out/` (e.g. `growth.recipe.json`
still says schema 0.2.0 and has no `command`). Add `tests/test_examples_fresh.py`, which runs each
`examples/*.py` into `tmp_path` and validates the outputs; mark it `slow` if it takes more than a
few seconds.

---

## 8. Test and parity strategy (cross-cutting)

- **Python is the reference.** `colorscale.apply(record, values)` (A1) is tested against
  matplotlib's `to_rgba`. The generated `tests/fixtures/colorscale_vectors.json` covers:
  - every norm kind;
  - edge values (vmin, vmax, vcenter, below/above range, NaN, 0 for log, negatives for power);
  - listed and continuous maps with under/over/bad colours.

  Flux's `scripts/verify-colorscale-parity.ts` asserts the same vectors. Regenerate both when
  matplotlib is upgraded.
- **Round trip.** For each colour control (A5), save → apply the override via `FLUX_PARAMS` →
  rerun → the recorded state equals the override.
- **Live vs regenerate parity (M3 acceptance).** For a hexmatrix and a `cells=True` heatmap:
  apply a scale edit live in Flux (the DOM fills after `applyPlotColorScale`), and separately
  regenerate with the same edit via the recipe. The fills of every `data-value` element must be
  identical hex. Put this in a Flux verify script with a checked-in pair of fixture plots.
- **Determinism.** `tests/test_determinism.py` gains a hexmatrix, a colour-scaled scatter and an
  `fp.image` case.
- **Schema completeness** (F3) catches any key added without a schema entry.

---

## 9. Milestones (execute in order; each ends green in both repos)

| Milestone | Items | Size | Acceptance |
|---|---|---|---|
| **M1 Correctness & contract fixes** | A0.1–A0.6, C1, C2, C3, B5 (bracket only), F1 (both sides), F2 [FP], F8 | ~2–3 days | Every A0/C1–C3 regression test passes; Flux autobuild uses `growBaseline` / `fadeRise`; `examples/out` regenerated and schema-valid. |
| **M2 Colour-scale contract** | A1, A2, A3, A5, A6, F3 | ~1 week | The parity fixture passes in Python; every colour-mapped example has a `colorScales` entry with a LUT; no colorbar rasterization warnings; the strict schema test passes. |
| **M3 Flux live colour scales** | A7.1–A7.8, F2 [FLUX] | ~1 week | The Flux parity verify passes; the GUI verify (live vmax edit + colorbar redraw) passes; the verbs work from the CLI; slide animation of vmax works. |
| **M4 Colour system** | B1, B2, B3, B4, B5 (rest), A4 | ~1–1.5 weeks | A light plot follows a dark deck; category colours are stable across figures; lint findings appear in `SaveResult.warnings`; the house maps appear in Flux's picker; dark-ground contrast tests pass. |
| **M5 Capabilities** | D1, D2, D7, E1, E2, E3, C4–C12, F5, F6 | ~2 weeks | `fp.image` + `fp.scalebar` example; `fp.brackets` stacks from stats rows with the provenance in the manifest; twin axes are one panel; `get_plot_data` returns hexmatrix bins. |
| **M6 Animation & polish** | F4, F7, B6, D3–D6, D8 | ~1.5 weeks | Axis views re-tick and move bars; hexes stagger by value; bar morphs tween by category; the `alpha_by` channel recolours live. |

**Explicitly out of scope:**
- bivariate 2D colour keys (beyond B6's alpha channel);
- pie / donut charts;
- broken axes;
- polar data capture (still a HARDENING P3 item);
- rerunning notebook recipes from Flux (D7 records them only).
