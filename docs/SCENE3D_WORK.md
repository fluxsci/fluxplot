# Track F work ledger

2026-09-28: Isolated worktree scene3d based on fluxplot main 1f2629a; owner checkout untouched.
Read Flux AGENTS, engineering-guide body and recent log, complete PLAN. P0.0 committed as
71be8e6: schema, conventions, 11 deterministic golden scenes + receipts. Parent accepted
contract. Deviation: finite `_VALUE` + optional uint8 `_VALID` replaces forbidden glTF NaN;
missing-data meaning retained. P0.0 tests 3 pass; baseline full suite 166 pass, 1 skip,
115 existing upstream warnings, 15.86 s (`/tmp/fluxplot-scene3d-baseline.log`).

F1–F3 in progress: Scene3D / views / data axes, input adapters, numpy GLB writer and named
states, shared Mark/Registry identity, manifest/furniture/save+recipe dispatch, shared
`_fieldmap` extracted from surface.py, surface3d field/categorical/missing semantics.
F5 optional collapse replay implemented using fast-simplification documented `simplify`
`return_collapses=True` and `replay_simplification`; tests pending.

No F4 runtime vendored yet. Parent owns N1 live frontend verification and renderer build.
No Flux files, animation worktrees, protected config, real projects or network Git touched.

2026-09-28 06:25 UTC checkpoint: Production F1–F3 and F5 code complete for independent
cross-review by W-core, sources temporarily frozen. Corrected glTF mask alignment in
isolated contract commit 1f9de30. Real Khronos glTF Validator reports 0 errors/0 warnings
for all 11 contract fixtures and all 5 production demo GLBs. Optional simplification
and collapse replay use fast-simplification 0.1.13; replay's actual API requires float32
points and int32 collapses (covered by regression). New production + contract tests:
20 pass. Full suite: 183 pass, 1 skip, 115 existing warnings, 16.58 s; log
/tmp/fluxplot-scene3d-full-new.log. Exact base/current 2D categorical+continuous surface
SVG, manifest and recipe bytes match (`diff -rq /tmp/fluxplot-scene3d-parity-base
/tmp/fluxplot-scene3d-parity-new`, exit 0). No SVG spec/version bump.

F4 wrapper/static preview and sync script prepared; runtime bundle not yet vendored.
Static neuron demo inspected: /tmp/fluxplot-scene3d-demo/neuron.png. Production scratch
demo script examples/scene3d_demo.py creates named neuron, two shape states, two-part
morph pair and eight-frame sequence in an explicitly selected scratch output folder.

Independent QA performed for parent P0.5: poster-worker registered gate passed,
including fresh browser/native exact-pixel parity (11 checks), 3.5 s; evidence in
model3d-poster/test-results/runs/2026-09-28T06-24-16-501Z-2. Earlier cancellation/source
await bug reproduced and fixed by owner; independent unresolved-source repro now
returns deadline31.06 ms and cancellation30.40 ms with source signal aborted.
Independent W-core review: 5/5 registered model3d pure gates passed; screenshot under
test-results/model3d/core-review/furniture-contact.png shows cardinal-axis text
collisions; notified owner along with world-bound overflow and toWorld fallback bugs.

2026-09-28 06:30 UTC: Independent W-core Python review passed the complete 183-test
suite and found/confirmed normal float32 overflow. Fixed with float64 cross products,
with 1e30-coordinate finite-normal regression; concave PolyData now delegates to
triangulate() or refuses instead of an incorrect fan; impossible named-part budgets
refuse; notebook reduction reconstructs unregistered custom colormaps from stored stops.
Added repeated-series colorbar regression (allocated suffixes remain bound to their
own field), static PNG/source-preservation regression, and 11 **production public-API**
fixtures via tests/generate_scene3d_library_fixtures.py. The independent P0.0 oracle is
unchanged. Current targeted suite: 25 pass; all 11 production fixtures validate with
Khronos glTF Validator at 0 errors/0 warnings. Production fixtures reproduce bytes.
Measured 501,264 vertices / 999,698 triangles: scene construction 262.526 ms; numpy GLB
write 176.595 ms; 24,027,764 bytes (22.9147 MiB). Measurement JSON:
/tmp/fluxplot-scene3d-writer-benchmark.json. This is CPU serialization, not GPU rendering.
Core independent review follow-up approved: registered 5/5 gates passed at
2026-09-28T06-26-38-406Z-2; rerendered contact sheet shows cardinal axes without collisions.

2026-09-28 06:34 UTC: Final independent review requested tiny finite normals too;
removed the arbitrary 1e-30 denominator floor after zero-normal replacement, so both
1e30 and 1e-20 triangle scales emit unit normals. Semantic-part decimation budgets
smaller than the number of preserved parts refuse explicitly. Final F1–F3/F5 + static
preview full suite: **190 passed, 1 skipped, 115 existing upstream warnings, 16.79 s**
(`/tmp/fluxplot-scene3d-final-f135.log`). W-core reviewed the corrective code and accepted
F1–F3/F5 pending this full pass. F4 remains pending vendored runtime/native frontends.

2026-09-28 F4 final verification in progress: vendored shared Flux viewer through
scripts/sync_flux_viewer.py, runtime m3d-r1 SHA c1ca36d01dbc859412cff5500fc3c639140d8252a5edfa29d9c14f5ca75ce86c
(765961 bytes, renderer 72f8fabd85cec94cb26a97aa1b0e63a5d3468f6d). Trusted actual
VS Code Jupyter + QMD kernels have executed self-contained HTML; earlier sphere pass
verified pointer/wheel/state/copy/axis/Home and context owners 3→3→0 on rerun/clear.
Final neuron + scale-bar + copied-view save roundtrip is running in /tmp/flux-model3d-f4-vscode;
scratch native profiles/kernels only. No real notebooks/config used. Runtime sequence
Frame controls initially failed to reflect authored2.5 and Home; renderer corrected and
added independent browser regressions before this sync.

White paper stage + opaque white notebook PNG preserves authored label contrast in dark
notebook themes (GLB/Flux export transparency unchanged). Static face-colored 0.1pt edges
close triangle raster seams. Independent renderer review found perspective fallback
image-plane mismatch; fixed using distance*tan(fov/2), analytic90° test passes. Shared
production F4 HTML browser gate:4 independent outputs, real pointer, keyboard state,
authored sequence2.5/Home, zero errors,4 PNGs remain when scripts disabled. Evidence in
test-results/model3d/notebook/. Actual untrusted VS Code is a documented limitation:
it suppresses selected mixed HTML+PNG output instead of automatically choosing PNG;
PNG-only Jupyter output displays. QMD untrusted injected-output probe inconclusive,
so no claim it displays PNG untrusted. Trusted native acceptance remains valid.

F5 root review identified min-one rounding overshoot with uneven part sizes. Added shared
integer largest-remainder face_budgets allocator: reserves1 per part then distributes
remaining capacity(count−1), bounded by source counts and global quota. Applied to mesh,
semantic surface groups, private preview. Impossible preview budget warns/retains full
scene; explicit impossible export budget refuses. Regression covers100parts with one
9900face part +99single triangles, actual simplifiable mesh/surface/preview ≤199faces,
and unchanged original preview source. Current targeted27pass114upstream warnings.
Independent root review requested. Expanded checked-in demo now defaults to
 test-results/model3d/demo/plots and creates named neuron+scale, folded cortex2states,
shared-topology morph pair, separate continuous field and8frame sequence. All6 demoGLBs
Khronos Validator0errors0warnings. Full suite/wheel rerun pending.

Final F4 acceptance (2026-09-28 06:53 UTC): real native QMD+Jupyter neuron scenario
passed against c1ca36d viewer. Both: orbit30→18/elevation20→24, wheel0.9→0.99,
shape0.01, Top90, Home restoration, Copy view selection. Live QMD scratch applied copied
view and fp.save; savedmanifest fields identical,27ms. Strengthened rerun waits for
new actual kernel execution orders and visible remount: owners3→3→0 afterclear.
Screenshots, final trusted-result.json and copied-view-roundtrip.json preserved under
 test-results/model3d/notebook/native/. Static PNG visually inspected: notriangle seams,
readable title/legend/scalebar on white background. It uses120dpi versusCSS96 furniture,
so no pixel-parity claim. Untrusted limitation remains explicit (root deviationD10).
Full final suite193pass1skip115upstreamwarnings16.10s. Wheel verified exact765961byte
c1ca36d renderer hash/stamp and MITnotice. Root independentF5quota review approved,
27targetedtests. N1 agent independentF4source/visual review pending finalacknowledgment.
Role/spec coordination explicitly documented: newscene3d0.1 contract carries newroles,
SVG0.3 unchanged; ROLE_VERSION internalvocabularyrevision only, notwireversion.
Independent F4 review APPROVED by W-render: final native screenshots/receipt inspected,
27tests, analytic perspective error4.4e-16, wheel/stamp/license,4script-disabled PNGs.
Review report in Flux model3d-render/test-results/model3d/F4_REVIEW.md. No remaining
TrackF blocker; only explicitly documented frontend/static-rendering limitations.
