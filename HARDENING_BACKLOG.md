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

### P1 — composite sub-parts (Effort M) — errorbar SHIPPED, rest open
Box/violin internals (whisker/cap/flier/median, violin IQR/min–max) and `contourf`
bands are still tagged as a single mark. Want: helpers/patterns that emit the
sub-parts under a group node so each is addressable in the X-ray (the errorbar
grouping in `manifest.py` is the pattern to follow).

### P1 — `guides[]` completeness (Effort S/M)
`manifest.guides[]` emits only axis + legend (`manifest.py` `_organize_guides`/
guide loop). Ticks/gridlines/spines/titles are present in the **parts tree** (so
they ARE addressable) but absent from the flat `guides[]` index. Low urgency
because parts covers addressing; revisit if a consumer relies on `guides[]`.

### P2
- `fp.tag` captures no x/y, so a custom-tagged mark has no spatial-stagger anchor.
- `build.order` carries the role-ref token `"gridlines"` (not an svg id). The Flux
  consumer expands it to "all gridline groups" intentionally, and the integrity
  test whitelists it — but it is the one non-id token in the order; consider
  emitting the real `axis.{x,y}.gridlines` group ids if a consumer needs strict
  id-only orders (note: that shifts their phase derivation from role to groupRole).
- ID-stability fragilities: positional legend↔series mapping; order-dependent
  slug-collision `-2` suffixes.
- Spec/schema drift: contract doc says `specVersion`/`role`/`pixelRange`; the
  manifest emits `schemaVersion`/`roles`/`pixelBox`; schema `$id` 0.1.0 vs
  `SPEC_VERSION` 0.2.0; JSON Schema is loose.

### P3
- Categorical / time-axis capture. Polar axes capture anchors but they are degenerate
  (theta 0/2π and the r endpoints can share an svg coordinate — the 2-anchor affine
  contract cannot represent a polar transform); scaffold/parts addressing is unaffected,
  data-space morph for polar would need its own capture shape.
- Swallowed exceptions in legend-swatch tagging (`tagger.py`).
- QA: align the example venv (3.12) with the package (3.13).
