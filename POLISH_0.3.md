# Fluxplot 0.3 implementation notes

The six September 2026 review areas are implemented together with the corresponding Flux
consumer update on `codex/fluxplot-polish` in both repositories. The package version remains
0.1.0 pending a release; the artifact contract is explicitly versioned 0.3.0.

1. Lossless JSON numbers, explicit null observations, original point indices and finite recipes.
2. Save snapshots, live public artist adapters, complete repeated-component inventory,
   collectible registries, staged exports and explicit recipe invocation overrides.
3. Scalar/unit-aware wrappers, marker option parity, barh, histogram and uncertainty metadata,
   explicit Seaborn adapters for trustworthy hue order and capped/horizontal errors.
4. Panel ownership and durable explicit names, final SVG layout capture, safe transform
   capabilities, per-panel/gap-aware Flux morphs, named colorbar ownership.
5. Stable surface shading, honest occlusion metadata, scoped raster draws, force-vector state
   restoration and empty-layer handling.
6. Heatmap/contour/contourf/colorbar helpers, optional cell/value capture, bounded large fields,
   and recipe-based palette/range controls in Flux's X-Ray.

`tests/test_polish.py` covers the reported scientific/lifecycle regressions, both supported
Matplotlib structures, and safe fallbacks. `tests/generate_polish_fixtures.py` produces the
small fixture bundles also checked into Flux under `scripts/fixtures/fluxplot03`; regenerate
and copy them together when the contract changes. `verify-fluxplot03.ts` and
`verify-fluxplot03-gui.mjs` exercise the actual Flux consumer/UI path. The complete package
suite runs on Python 3.9 / Matplotlib 3.7 and Python 3.13 / Matplotlib 3.11.

The review's 52 real example plots are exercised in isolated output directories. User project
outputs and the preexisting uv.lock modification are not part of this work. Raw field samples
remain opt-in. Surface cross-category occlusion on arbitrary folded meshes and exact 3D point
projection/order are not claimed; imported appearance/group editing is preserved and unsupported
data morphs use complete transitions.

## Verification — 2026-09-06

- Python 3.13 / Matplotlib 3.11 and Python 3.9 / Matplotlib 3.7.5: **151 passed,
  1 skipped** in each environment. The skip is the old gallery test whose generated fixture
  directory is absent; the 52-project-plot sweep and committed new fixtures provide coverage.
  Warnings are third-party Matplotlib/pyparsing/cmasher deprecations.
- All **52** supplied project examples regenerated in scratch and passed Flux's enhanced validator.
- Flux: **202/202** pure gates; **42/42** Paper gates; **19/19** Figure/Slides overhaul gates;
  new field-control GUI, existing X-Ray, actual desktop recipe handler, and Figure/Slide scale gates.
- Svelte: zero errors/warnings. Production renderer, slide runtime, CLI and MCP bundles build.
  Shared fixture SVG/JSON/recipe bytes agree in both repositories.
- Bundle/startup gates: **4/4**. Production startup uses **652.9 KB** of eager JavaScript against
  the unchanged **800 KB** limit (starting revision: 647.7 KB). The startup harness now holds
  idle callbacks until its eager snapshot, then releases them to observe normal preloading;
  this fixes a measurement race without changing app scheduling or the budget. It also owns
  and cleans up the preview process directly.
- SVG/manifest identity checks and Ruff F checks pass. Failure-path tests verify caller artist,
  layout, canvas and raster state restoration; invalid recipes/duplicate IDs do not replace outputs.
