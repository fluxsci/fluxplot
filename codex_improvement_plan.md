# FluxPlot: high-confidence infrastructure and automatic-semantics plan

## Purpose and decision rule

This is an implementation plan, not a request to make FluxPlot infer meaning from
pixels. The library's central guarantee remains correct: meaning must be captured
while matplotlib artists and source data still exist. The improvements below make
that guarantee useful with less user ceremony by using only information that is
already exact at save time:

1. the Python call site and active interpreter;
2. matplotlib artist type, artist-owned data, and a public artist label;
3. an explicit fp.* wrapper's return structure; and
4. the known SVG/manifest contract.

Do not add color-, draw-order-, geometry-shape-, or legend-position-based
semantic guesses. When identity is absent or ambiguous, preserve the existing
extra/unclassified fallback rather than inventing a durable series name. A less
detailed but honest scene graph is much better than a richly named wrong one.

Complete and release each numbered item independently; do not combine schema,
panel, provenance, and wrapper work into one large compatibility break.

## Current baseline (verified 2026-07-10)

FluxPlot is already in unusually good shape for an early semantic graphics
library:

- fp.save finalizes layout, tags registered marks and scaffold, captures
  transforms, renders deterministically, postprocesses the SVG, then emits the
  manifest and recipe. The orchestration is in src/fluxplot/api.py.
- Raw text and untagged line/collection/patch artists are saved as addressable
  annotations or extra.* instead of disappearing from Flux's scene graph.
- The manifest-to-SVG integrity test catches dangling references, including the
  committed gallery. Error bars, titles/subtitles, polar spines, and per-point
  markers have recently been hardened.
- Flux consumes parts, build.order, presets, recipes, and coordinate anchors
  for X-ray editing, auto-animation, and morphing. It currently assumes a
  primarily single-panel plot in several places, particularly morphing
  (axes[0]).
- uv run python -m pytest -q passes: 77 tests. The run emits third-party
  Matplotlib deprecation warnings from cmasher/seaborn, but no FluxPlot failures.

The current weaknesses are concentrated at the convenient boundary: normal
matplotlib plus only fp.save gets an SVG and a generic extra.* tree, but not
necessarily a named data series or rerunnable recipe. A figure with multiple
non-colorbar Axes is also not represented safely: save gives every capture the
id plot-area, while global axis/legend IDs collide and acquire order-dependent
-2 suffixes.

## 1. P0 — Make fp.save automatically produce an honest rerunnable recipe

### Outcome

For a normal .py analysis script, this should be enough:

    fig, ax = plt.subplots()
    ax.plot(x, y, label="Control")
    fp.save(fig, "plots/growth.svg")

The recipe should be rerunnable by Flux without requiring
recipe=dict(script=__file__, ...). It must say that the producing script was
automatically discovered, while never claiming to know inputs or parameters it
cannot know.

### Why this is a no-brainer

Flux's strongest workflow feature is regeneration. At present its documentation
has to tell every user to remember script=__file__; omitting it silently removes
the rerun block. fp.save can determine the script, current working directory,
interpreter, output location, package versions, and source hash without asking
the user to repeat facts the runtime already knows.

### FluxPlot implementation

1. Add a focused provenance module, such as src/fluxplot/provenance.py. Keep
   stack inspection and Git probing out of api.py and recipe.py.
2. Make auto provenance the semantic default while preserving source
   compatibility:

       fp.save(fig, path)                 # auto
       fp.save(fig, path, recipe=None)    # auto
       fp.save(fig, path, recipe=False)   # explicitly suppress auto provenance
       fp.save(fig, path, recipe={...})   # explicit fields override inferred fields

   recipe=False matters for notebooks, generated figures, and privacy-sensitive
   callers.
3. Discover a script only by safe deterministic rules:

   - First prefer __main__.__file__ if it names an existing regular .py file.
   - Otherwise walk inspect.currentframe callers outward and select the first
     existing .py file outside installed FluxPlot code.
   - Reject pseudo-files such as <stdin> and <ipython-input-...>, and paths
     that do not exist. Do not scrape notebook history or create a temporary
     script.
   - Resolve it absolutely before the existing recipe-relative conversion.
4. Have build_recipe emit an additive provenance block, for example:

       "provenance": {
         "scriptDiscovery": "automatic",
         "scriptSha256": "...",
         "python": "3.13.3",
         "platform": "...",
         "packages": {"fluxplot": "0.1.0", "matplotlib": "3.x"},
         "git": {"commit": "...", "dirty": false}
       }

   Git is optional and must fail closed (omit it, never raise) outside a
   repository. Script-hash failures omit only that field. Keep this
   host-specific material in the recipe, never the deterministic SVG/manifest.
5. Merge values predictably: explicit script, command, cwd, params, and inputs
   win; inferred script only fills a missing script; no automatic input
   discovery; distinguish automatic, explicit, and unavailable discovery.
6. Call the artifact rerunnable only when a real script was found. Flux will
   re-execute the recorded command/cwd and expects it to overwrite the recorded
   output, matching the existing recipe contract.

### Flux changes

1. Extend recipe types for optional provenance.
2. In X-ray/Regenerate show a compact status: Rerunnable (script
   auto-discovered), or Not rerunnable: save was called outside a Python script.
   Never imply automatic data-input provenance.
3. If the stored script hash differs from the current script, show a
   non-blocking source-changed-since-export notice. Do not block rerun.

### Tests and acceptance criteria

- Execute a tiny real script in a temporary directory. Plain fp.save must yield
  command, args, cwd, and output resolving exactly as flux-core runRecipe does.
- Assert scriptDiscovery is automatic, a script hash exists, and inputs remains
  an empty list rather than guessed.
- Explicit values override inferred values and report explicit discovery.
- recipe=False, interactive/no-file callers, and a missing caller file write a
  valid non-rerunnable recipe with no command/args.
- Repeated saves retain byte-identical SVG and manifest; recipe volatility is
  deliberately exempt.
- Extend Flux verify-f2-recipe.ts to run an automatically discovered fixture and
  verify that the changed-source notice is non-blocking.

## 2. P0 — Correct multi-panel figures before expanding automatic tagging

### Outcome

fig, axs = plt.subplots(2, 2); ...; fp.save(fig, ...) must create one semantic
figure with four independently addressable panels. No IDs may change because
another panel was inserted earlier, and no manifest reference may rely on -2
collision repair.

This is an existing correctness gap, not a cosmetic enhancement. Multi-panel
figures are normal matplotlib output and exactly the type users will save with
only fp.save.

### Contract

Use a panel namespace for every non-colorbar axes. Retain current single-panel
IDs only when there is exactly one plot axes; use the panel contract for
multi-panel output:

    panel.a
    panel.a.plot-area
    panel.a.axis.x
    panel.a.axis.x.title
    panel.a.control.line
    panel.b. ...

Panel names must not depend on matplotlib internal axes_1 numbering. Default
names are spreadsheet labels (a, b, ..., z, aa, ...) based on final axes
position: get_subplotspec where available, otherwise visual top-to-bottom then
left-to-right sorting. Do not block automatic support on a user panel-name API.

The v1 multi-panel manifest needs a top-level panels array. Each entry has id,
svgId, index, axes capture, series, guides, overlays, parts, and build. Root
parts contains panel containers. Preserve legacy top-level axes/series/guides/
overlays/build for a single-panel artifact only; do not flatten panels into an
ambiguous global series list.

### FluxPlot implementation

1. Add PanelContext in a focused panels.py module: Axes, stable panel id, SVG
   wrapper id, and local ID prefix. Exclude colorbar Axes here, once.
2. Refactor Registry/Mark so each mark records its owning Axes/panel at
   registration. Grouping only by series today would merge control from two
   panels incorrectly.
3. Pass panel context through resolve_gids, autotag_scaffold, coordinate
   capture, postprocess, and manifest construction. Replace global axis_id use
   with a builder that takes a panel prefix.
4. Map each actual matplotlib axes wrapper to panel.<id>.plot-area in
   postprocess; never rename only axes_1. Use IDs assigned before rendering, not
   SVG geometry.
5. Scope legends, titles, free-text sweep, orphan sweep, and build order to
   panels. Deduplicate the figure suptitle and put it under figure, not under
   the first panel.
6. Keep colorbars out of panels. Promote them later as figure guides (item 6);
   do not turn them into panels merely to pass tests.
7. Version this contract deliberately (for example schemaVersion 0.3.0), update
   the JSON-schema $id, README, Flux TypeScript mirror, and SPEC_VERSION
   together. The current schema $id says 0.1.0 while emitted SPEC_VERSION says
   0.2.0; repair this in the same versioned change.

### Flux implementation

1. Extend FluxPlotManifest with optional panels. Normalize legacy manifests to
   one virtual panel internally rather than scattering conditionals.
2. Update plot/tree.ts, X-ray labels, target resolution, auto-build, and
   overrides so a panel node is navigable and a group target cannot cross panel
   boundaries.
3. Require/select a panel id in morph APIs. The current axes[0] only works for
   legacy single panels. Initial UI behavior may morph matching panel IDs and
   fade unmatched panels; never invent correspondence.
4. Extend validate-plot to understand both contracts and require unique SVG IDs
   across all panels.

### Tests and acceptance criteria

- Add a 2x2 fixture with repeated series names, distinct titles, annotations, a
  shared suptitle, legends on two panels, and one colorbar.
- Assert all SVG IDs are unique, a panel refers only to its own IDs, and no
  semantic ID has a collision-derived -2 suffix.
- Save twice and byte-compare SVG/manifest.
- Explicitly test single-panel compatibility (either byte-compatible or an
  intentional migration).
- Add Flux tests for panel X-ray paths, panel-specific overrides, build order,
  and cross-panel morph rejection/fallback.
- Run manifest-to-SVG integrity on this fixture and all gallery artifacts.

## 3. P1 — Promote safe ordinary matplotlib artists during save

### Outcome

Users should not need to rewrite a conventional labeled plot for Flux semantics:

    ax.plot(t, control, label="Control", marker="o")
    ax.scatter(t, treatment, label="Treatment")
    ax.legend()
    fp.save(fig, "plots/growth.svg")

The output should contain series control/treatment, exact data, and ordinary
line/point parts. A raw artist lacking safe identity remains extra.*; it is not
promoted to a guessed series.

### Design: conservative save-time adapter registry

Create src/fluxplot/autotag.py, separate from the orphan sweep. Run it after
final canvas draw but before gid assignment. Process only artists not already
represented in the registry.

Allowed adapters use artist class and public state only:

| Exact source | Promote when | Semantic result |
|---|---|---|
| Line2D | non-empty public label, not starting underscore | line; get_data() |
| scatter PathCollection | public label and exact Nx2 finite offsets | indexed point; get_offsets() |
| BarContainer | public label; all patches unclaimed | indexed bar; centers/heights |
| fill_between PolyCollection | public label and one unambiguous candidate | area; do not invent original vectors |
| supported artist passed to fp.tag | no x/y supplied | infer only from same exact adapter |

Rules:

- A label is identity only when non-empty and not private (_child..., _nolegend_).
  Slug it with current ID rules while retaining display text.
- Duplicate labels in a panel are ambiguous. Leave all candidates extras and
  emit one actionable warning directing users to explicit series/tagging.
- Do not use positional legend entries to identify unlabelled artists.
- Never infer reference-line, fit, errorbar, confidence band, annotation, or
  causal meaning from shape, orientation, linestyle, color, or z-order.
- Record auto promotion internally as source=auto-label and emit an additive
  manifest capture/provenance field such as identity=artist-label, data=artist.
- Explicit helpers/tags always win. Never add a second mark for a registered
  artist.

Extend fp.tag with x=None, y=None. When omitted, use the same exact extractor;
otherwise preserve absent x/y rather than making an invalid spatial claim. This
fixes the current custom-tag no-spatial-stagger limitation.

### Tests and acceptance criteria

- Raw labeled plot, scatter, and bar each produce appropriate series/data/parts
  and build-order entries.
- Raw labeled scatter has one point per exact offset. For masked/non-finite
  offsets, either preserve a documented mapping or decline promotion with a
  warning; never shift indices.
- Private/default and duplicate labels remain visible extras; only ambiguous
  public duplicates warn.
- A raw horizontal threshold becomes generic line, never reference-line; callers
  use fp.reference_line for that overlay role.
- Existing helpers/tag/seaborn behavior remains free of duplicate marks.
- Add a Flux fixture proving X-ray, style override, and auto-animation work for
  raw-only labeled input.

## 4. P1 — Add the small wrapper set that unlocks real semantic composites

### Outcome

Provide high-value APIs with line-like ergonomics and normal matplotlib return
objects:

    fp.box(ax, values, series="control", label="Control")
    fp.violin(ax, values, series="treatment", label="Treatment")
    fp.hist(ax, values, series="observations", bins=20, label="Observations")

Do not mirror every matplotlib method. Start only where its documented return
structure exposes exact semantic subparts and generic extras are inadequate.

### API and representation

1. fp.box wraps Axes.boxplot. Register documented return-dict pieces: box body,
   whisker, cap, median, and flier. One wrapper call is one semantic series;
   multiple groups use multiple calls with explicit stable series names.
2. fp.violin wraps Axes.violinplot. Register body, extrema bars, min/max, and
   mean/median when returned. Missing options create no dead parts.
3. fp.hist wraps Axes.hist. Treat returned patches as indexed bars. Record exact
   bin edges/counts in an additive distribution payload. Do not treat bar
   heights as original observations. Record source values only with
   include_values=True because samples can be large/sensitive.
4. Add presets after observing results in Flux: bars fade/stagger; composite
   statistics fade; filled bodies do not receive speculative draw-on behavior.
5. Build grouped parts analogous to shipped errorbars: control.whiskers,
   control.medians, control.fliers. Each group lists concrete SVG members and
   every member appears in build order exactly once.

This completes the high-value part of the existing composite-subparts backlog.
Contours/contourf stay separate: topology and fill-band semantics need their own
data/animation design.

### Tests and acceptance criteria

- Test each returned component against supported Matplotlib versions and assert
  every declared member resolves in SVG.
- Cover optional box/violin pieces on/off, single group, multiple wrapper calls,
  and legends.
- Compare histogram counts/edges to Matplotlib/NumPy; assert raw values are
  absent by default.
- Test X-ray grouping and group-level Flux styling for every composite.
- Extend determinism/gallery fixtures. A matplotlib structural change must fail
  a DOM contract test loudly, never silently remap subparts.

## 5. P1 — Make export/import consistency explicit and fail safely

### Outcome

Flux must never silently treat a temporarily incomplete, malformed, or stale
FluxPlot sidecar as vanilla SVG. It should retry/retain the last good semantic
state or show an actionable semantic-asset error.

This matters once fp.save is the easy workflow: it writes three sidecars
sequentially while Flux watches/imports them independently. A watcher can see a
new SVG with old manifest, or partial JSON and quietly fall back to a derived
manifest.

### FluxPlot changes

1. SHA-256 the final postprocessed SVG bytes and add artifact.svgSha256 to the
   manifest. It is deterministic and has no circular dependency because SVG
   does not contain the manifest.
2. Stage all three files in destination using unique temporary names,
   flush/close them, then os.replace final paths. Clean staging files on error.
   This prevents partial individual files but cannot make three replacements
   globally atomic; do not claim otherwise.
3. Replace in dependency order: SVG, manifest, recipe. The manifest checksum is
   the commit marker. Preserve existing destination permissions where practical
   or document behavior.
4. Add the checksum to schema and README. Do not put timestamps/recipe hashes in
   manifest.

### Flux changes

1. Centralize validation at preparePlot/import: parse JSON, verify basic
   contract/schema version, manifest.svg filename, and checksum when present.
2. On watcher parse/checksum failure, retry with bounded debounce (for example
   three attempts over 250–750 ms). If still invalid, retain prior cached
   semantic plot. On first import show warning plus explicit import-as-plain-SVG
   action; never silently downgrade.
3. Legacy manifests lacking checksum remain compatible but show legacy status.
4. Extend flux validate-plot to distinguish checksum mismatch, filename
   mismatch, malformed JSON, duplicate IDs, dangling IDs, and unsupported
   schema version.

### Tests and acceptance criteria

- Simulate old-SVG/new-manifest, new-SVG/old-manifest, missing/truncated
  manifest, and final successful triplet. Flux must not permanently cache a
  derived replacement for invalid intermediate cases.
- Check checksum success/failure through CLI and GUI/import preparation.
- Verify legacy fixtures import with explicit legacy status.
- Induce a failure after staging one file and assert no partial final artifact
  replaces an earlier successful triplet.

## 6. P2 — Finish coordinates and guides only where exact

These are valuable but follow P0/P1 because they change the public manifest.

### 6a. Categorical and datetime data

The current float conversion and two-anchor affine/log contract are adequate for
numeric axes but insufficiently explicit for datetime/category inputs and known
to be degenerate for polar transforms.

Add a versioned coordinate descriptor:

- numeric: current contract;
- datetime: canonical UTC milliseconds or ISO-8601 plus exact Matplotlib numeric
  coordinate used for projection;
- categorical: displayed text plus stable ordinal coordinate;
- unsupported: present/addressable but not data-space morphable.

Only claim conversion when Matplotlib's units converter provides it exactly.
Exercise timezone-aware, timezone-naive, NaT, duplicate category labels, and
polar axes. Flux disables data-space morph for unsupported/polar panels with an
explicit reason rather than applying a false two-anchor approximation.

### 6b. Guides and colorbars

Promote already-addressable tick/gridline/spine/title/legend information into a
complete guides index only if consumers need flat querying; do not duplicate the
parts tree for symmetry. After panel namespacing, model colorbars as figure
guides with scale, ticks, title, and SVG root. The current skip-colorbar choice
was correct temporary collision avoidance.

### 6c. Eliminate non-ID build-order tokens

Replace the sole gridlines role token in build.order with actual group IDs, such
as axis.x.gridlines and axis.y.gridlines (panel-prefixed when applicable). Flux
continues accepting old role tokens for legacy manifests. Then integrity tests
can require every build entry to resolve to a parts node or SVG ID.

### Tests and acceptance criteria

Create fixture pairs for numeric, log, date, categorical, polar, and colorbar
plots. For supported mappings, project several values from manifest in Flux and
compare emitted marker positions within documented tolerance. For unsupported
ones, assert a clear capability reason and no false morph promise.

## 7. P2 — Resolve ID stability fragilities deliberately

The allocator -2 suffix is deterministic only with fixed draw order. It is a
safety net, not durable identity.

1. For explicit helpers, detect slug collisions at registration. Names A B and
   A_B both normalize to a-b; raise an actionable error by default directing the
   caller to provide stable distinct series names. Only add an opt-in collision
   policy if a real batch workflow needs it.
2. Do not use positional legend-to-series mapping as identity. Match unique exact
   public labels only; otherwise retain swatch/text as guides but omit series
   linkage.

Test that adding unrelated series/legend entries does not rename explicit IDs.
This applies naturally to panel namespacing.

## 8. Engineering gates and release discipline

1. Make JSON schemas strict enough for parts, members, presets, panels, and new
   artifact/provenance fields. Synchronize schema $id, SPEC_VERSION, README, and
   Flux TypeScript types in the same PR.
2. Add a deliberately shared/copied FluxPlot/Flux fixture directory with real
   SVG, manifest, recipe, and contract note. Test it from both repositories.
3. Run CI across advertised Python versions and at least minimum/current
   supported Matplotlib. The project claims Python >=3.9 and Matplotlib <4;
   test that claim rather than relying only on current Python 3.13.
4. Track cmasher/seaborn deprecation warnings with upstream references; do not
   globally suppress them. Warnings-as-errors should apply only to FluxPlot's
   own namespace once dependencies permit it.
5. Every feature PR runs:

       cd /home/driessen2/fluxplot
       uv run python -m pytest -q

       cd /home/driessen2/flux
       npm test --if-present
       # plus focused feature scripts, for example:
       # npx tsx scripts/verify-f2-recipe.ts
       # npx tsx scripts/verify-xray-tree.ts
       # npx tsx scripts/verify-slide-autobuild.ts
       # npx tsx scripts/verify-slide-morph.ts

   Replace npm test with Flux's established exact project command if different;
   do not invent a second runner.

## Explicit non-goals

- No general post-hoc SVG semantic classifier. Flux derived manifests remain a
  fallback, not a reason to guess meaning after serialization.
- No automatic CSV/data-input discovery from handles, stack locals, pandas
  internals, or filenames. It is unreliable and can leak sensitive paths.
- No inference that a line is threshold, fit, confidence band, annotation, or
  causal relationship. Use explicit APIs/wrappers.
- No broad wrapper explosion. Add one only where matplotlib exposes an exact
  return contract and Flux gains materially better editing/animation.
- No silent fallback from invalid semantic sidecar to vanilla import.

## Recommended implementation sequence

1. Automatic recipe discovery plus Flux recipe status/source-change display.
2. Versioned panel model end-to-end.
3. Conservative raw Line2D/PathCollection/BarContainer promotion and fp.tag
   coordinate extraction.
4. fp.box, fp.violin, and fp.hist one at a time with DOM contracts.
5. Checksum/staged-write and Flux invalid-sidecar handling.
6. Coordinate, colorbar, guide, and ID refinements behind versioned fixtures.

At every stage preserve this invariant: every manifest reference, including every
build-order entry, names a concrete current SVG element or a documented
manifest-only group that Flux resolves without ambiguity. A feature is complete
only when this is checked from both FluxPlot and Flux.
