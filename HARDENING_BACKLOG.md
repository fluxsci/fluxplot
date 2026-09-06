# FluxPlot semantic-tagging hardening — status + backlog

A review pass (driven by gaps the Flux X-ray surfaced) hardened the tagging
pipeline. This records what shipped and what is deliberately deferred.

## Done (commit "Harden semantic tagging …")

- **Titles** — `autotag_scaffold` tags left/center/right title + the figure
  suptitle as role `title` (id `figure.title`); `figure_titles` is a list in the
  manifest (no last-wins). The house style's `loc="left"` title was previously
  missed on nearly every plot.
- **Free-text sweep** — any un-tagged `ax.texts`/`fig.texts` becomes an
  addressable `annotation.N` overlay carrying its text, so equation boxes and
  value labels stop escaping as `text_N`.
- **Subtitle** — new `subtitle` role; `style.title(sub=…)` tags it (with text).
- **Colorbar axes (P0)** — `save()` skips colorbar Axes from scaffold + capture,
  fixing the duplicate `plot-area` capture and the `axis.x-2`/`axis.y-2` id
  collisions on heatmap/contour plots.
- **Errorbar caps (P1 bug)** — `fp.errorbar` now tags the central data line + the
  caplines, not only the bar collection.
- **Role vocab (P1)** — added `subtitle, violin, contour, whisker, cap, flier,
  median, segment, caption`.
- **Overlay text (P1)** — overlay entries copy `text` from `Mark.data`.
- **Regression test** — `tests/test_semantic_integrity.py`: every id the manifest
  references must exist in the SVG, plus targeted title/subtitle/annotation/
  colorbar/errorbar cases. This is the class of test that would have caught the
  escapes originally.

## Done (figure-v1 P10 run, 2026-07-09)

- **data-kind hints** — `roles.KIND_BY_ROLE` (text|line|shape|container) injected as
  `data-kind` alongside every `data-role`, mirrored as `kind` on manifest parts nodes +
  overlay entries; x-/extra kinds inferred per-artist (`descriptors.artist_kind`).
- **Errorbar composite sub-parts (from P1 below)** — series `svg.errorbars[]` +
  per-series `<series>.errorbars` group node in `parts`; every member in `build.order`.
- **Polar spine tagging** — `polar`/`start`/`end`/`inner` spines tagged (the rectangular
  side list silently dropped them); polar example pair in the notebook.
- **Gallery integrity sweep (the P3 QA item)** — `test_semantic_integrity.py` is
  parametrized over `examples/basic_example_output/*.fluxplot.json`.
- **Dead `save()` params removed (from P3)** — `addressable_points` / `style_classes`.

## Deferred backlog (prioritized)

## Done (codex improvement-plan run, 2026-07-10, branch `codex-improvements`)

- **Automatic recipe provenance (plan §1)** — `provenance.py`: `fp.save` discovers the
  producing script (`__main__`/stack walk, deterministic safe rules), emits a
  `provenance` block (scriptDiscovery/scriptSha256/python/platform/packages/git,
  git fails closed); `recipe=False` suppresses; explicit fields win.
- **Labeled raw-artist promotion (plan §3)** — `autotag.py`: public labels become
  series identity for Line2D/scatter/BarContainer/fill_between; duplicates and
  explicit collisions decline to `extra.*` with one warning; series carry
  `capture: {identity: artist-label, data: artist}`. `fp.tag`/`fp.tag_points`
  gained exact x/y extraction (closes the P2 fp.tag item below).
- **Box/violin/hist composites (plan §4)** — `fp.box`/`fp.violin`/`fp.hist`;
  the errorbar composite pattern generalized (`manifest.COMPOSITE_ROLES`) to
  whisker/cap/median/flier/mean/segment plural member lists + per-family group
  nodes + full build-order membership; `distribution` payload for hist; new
  core role `mean`. `contourf` remains open (needs its own data/animation design).
- **Checksum + staged writes (plan §5)** — `artifact.svgSha256` in the manifest;
  save stages all three sidecars then commits via `os.replace` in dependency
  order; failures preserve the previous triplet.
- **ID stability (plan §7)** — explicit-series slug collisions raise actionably at
  registration; legend entries link by exact unique label text (never position)
  and carry their display text (closes the two ID-stability items below).

## Deferred backlog (prioritized)

### P1 — composite sub-parts (Effort M) — errorbar/box/violin SHIPPED, contour open
`contourf` bands are still tagged as a single mark: topology and fill-band
semantics need their own data/animation design before sub-part tagging helps.

### P1 — `guides[]` completeness (Effort S/M)
`manifest.guides[]` emits only axis + legend (`manifest.py` `_organize_guides`/
guide loop). Ticks/gridlines/spines/titles are present in the **parts tree** (so
they ARE addressable) but absent from the flat `guides[]` index. Low urgency
because parts covers addressing; revisit if a consumer relies on `guides[]`.

### Completed — scientific/panel/field contract 0.3 (September 2026)

See [POLISH_0.3.md](POLISH_0.3.md) and the README's schema 0.3 sections. Panel ownership,
colorbar guides, full-precision/null data, repeated components and matrix/contour helpers now
ship with synchronized Flux readers and regression fixtures. The historical deferred contour
note above is superseded by `fp.contour`/`fp.contourf` and their level metadata.

### P2
- `build.order` carries the role-ref token `"gridlines"` (not an svg id). The Flux
  consumer expands it to "all gridline groups" intentionally, and the integrity
  test whitelists it — but it is the one non-id token in the order; consider
  emitting the real `axis.{x,y}.gridlines` group ids if a consumer needs strict
  id-only orders (note: that shifts their phase derivation from role to groupRole).
  Requires Flux to accept both forms FIRST (plan §6c) — sequence behind the panel release.
- Spec/schema drift: contract doc says `specVersion`/`role`/`pixelRange`; the
  manifest emits `schemaVersion`/`roles`/`pixelBox`; schema `$id` 0.1.0 vs
  `SPEC_VERSION` 0.2.0; JSON Schema is loose. Repair with the §2 version bump.

### P3
- Categorical / time-axis capture. Polar axes capture anchors but they are degenerate
  (theta 0/2π and the r endpoints can share an svg coordinate — the 2-anchor affine
  contract cannot represent a polar transform); scaffold/parts addressing is unaffected,
  data-space morph for polar would need its own capture shape.
- Swallowed exceptions in legend-swatch tagging (`tagger.py`).
- QA: align the example venv (3.12) with the package (3.13).
