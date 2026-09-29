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

2026-09-28 07:52 UTC shared viewer refresh: canonical sync from Flux model3d
`945faf2` generated dist, m3d-r1 SHA
0420ed07a6e5f667c4f80081bd05f5b699a77baea868cd70116eacd72bfec3da, 766030 bytes.
This includes approved furniture label spacing and the prior canonical-validator
refactor; no vendored code was edited manually. Focused tests27pass114warnings0.28s;
uv wheel build and exact bundled SHA/license checks passed. Generated four-output
browser regression passed real pointer orbit isolation, top view, keyboard state,
frame2.5→2.51→Home2.5 and four no-script PNGs, zero page errors. Native VS Code
evidence remains earlier c1ca36d checkpoint. Final full-plan viewer sync may follow
later approved P3 shared furniture changes.

2026-09-28 14:49 UTC combined P3 viewer refresh prepared: official atomic sync from
Flux model3d-orbit `8d039d9`, generated m3d-r1 viewer SHA
3d9f40007b0a9a17d7656a6eed80e44f026ffce53183b7ae0090a40fab64db03, 766923 bytes.
Includes shared furniture near-plane/paint-order and source-default corrections,
plus the pure colormap boundary. Generated source and stamp agree; no vendored JS
hand-edits. Tests and wheel packaging are pending release of the separate native
qualification window; previous native QMD/Jupyter evidence remains c1ca36d.

P3 viewer refresh qualification completed after native timing reservation release:
`uv run --no-sync pytest -q tests/test_scene3d.py` 27/27 passed, 114 upstream
Matplotlib warnings, 0.28s. `uv build --wheel` passed; packaged viewer/stamp/license
match the canonical generated bytes exactly (receipt
`test-results/model3d/notebook/p3-wheel-receipt.json`). Regenerated four-output HTML
and ran the existing browser acceptance: real pointer orbit isolated one viewer;
Top view, shape keyboard input, frame 2.5→2.51→Home2.5 passed; four script-disabled
PNG alternatives visible; zero page errors. Both screenshots visually inspected:
`test-results/model3d/notebook/generated-multi.png` and `scripts-disabled.png`.
This qualifies the new bundle in generated HTML; original native notebook trust
and rerun/disposal evidence is retained without claiming a new VS Code run.

2026-09-28 15:49 UTC approved furniture viewer refresh: official sync from canonical
Flux model3d-orbit `4cabc64`, m3d-r1 SHA
c740ce4ed095b0645c668e19a3d94c68bc93e838622f048604a3ad2be7c27fb9, 767716 bytes.
No hand edits to the generated viewer. Focused pytest: 27/27 passed (114 upstream
warnings, 0.27 s). Wheel built; exact JS/stamp/license compared with generated Flux bytes.
Receipt test-results/model3d/notebook/furniture-wheel-receipt.json. Four-output browser
regression rerun next; native QMD/Jupyter evidence remains the earlier c1ca36d checkpoint
until the private display is available, not represented as a fresh native run.

Four-output browser refresh PASS: real pointer orbit changes only the first output,
Top view, shape range keyboard input, authored sequence 2.5→2.51→Home 2.5, all four script-free
PNG alternatives and zero page errors. Current screenshots inspected for legible axis titles,
colorbar spacing and controls. Original native trust/disposal acceptance remains separate.

2026-09-28 16:53 UTC current-viewer native S0/N1 refresh: the exact m3d-r1
viewer c740ce4ed095b0645c668e19a3d94c68bc93e838622f048604a3ad2be7c27fb9
(767716 bytes) passed actual trusted VS Code 1.138.0, QMD Notebook 0.1.0,
Jupyter 2025.9.1 and Jupyter renderers 1.3.0. A fresh scratch profile/extensions/
workspace and isolated HOME/XDG/Jupyter state used the existing disposable
Python 3.13.11 kernel. Executed cells assert the bundle hash/size; independent
inspection found the exact bundle bytes in the saved IPYNB HTML output.

Both frontends passed real pointer orbit 30/20→18/24, wheel 0.9→0.99, keyboard
shape expanded 0→0.01, Top 90° and authored Home restoration. Copy view selects
the literal readout; applying it through live QMD scratch execution and fp.save
produced matching 18/24/0.99/expanded 0.01 manifest values in 28 ms, with matching
actual GLB hash. OS clipboard bytes were not inspected. PNG-only outputs are
visible in both trusted frontends. Owners 3→3 after actual reruns→0 after output
clear, across three notebook webview instances. Current interactive/static
screenshots were independently inspected. Native PID-owned X11 sizing used a
fresh nonzero display snapshot and no browser viewport emulation.

Author artifacts: Flux model3d-qa/test-results/model3d/notebook/current-native/
(REVIEW/result, interactive/static PNGs, native qualification/stamp receipts,
corrected harness and saved workspace files/hash receipts). Independent review:
Flux model3d-source/test-results/model3d/s0-current-review/REVIEW.md. Earlier
window-adapter failures remain preserved separately. This supersedes c1ca36d
for trusted native S0 only; untrusted mixed-MIME suppression/QMD restoration
limitations remain unchanged, and this is not a performance qualification.
This follow-up changes documentation only; runtime/generated bytes and demo
provenance at source commit 718bf3aa remain unchanged.

2026-09-28 follow-up F5 plan audit: replace the fixed 400000-face hint with a
conservative per-call cap derived from original face counts, each part's fields,
colors and morph targets, plus bounded GLB JSON overhead. Each successful helper
call's part census sets the minimum permissible cap. Proven metadata/part-floor
impossibility is distinguished from inability to recommend a conservative cap;
the message never promises the simplifier can reach a quota. Warnings for many
states suggest fewer frames without implying that the fixed cap always fits.

To make the stored-vertex bound valid even for a backend returning unused output
vertices, simplification now applies sorted surviving indices to base/states and
value/color channels together. No welding or referenced-vertex reordering. Tests
cover actual state-rich simplification, disjoint-triangle worst-case storage,
multiple helper calls, fixed overhead, and an orphan-returning backend whose
reference/follower pair remains morph-compatible. Final full suite: 197 passed,
1 skipped, 115 existing upstream warnings, 16.15 s. Independent W-render source
review APPROVED; focused size/orphan/quota tests 5 passed (26 deselected), 114
upstream warnings, 0.25 s. The full suite reproduces all fixtures; generated
fixtures and viewer are unchanged. Receipt: test-results/model3d/size-warning/
pytest-receipt.json. Notebook 300k first-frame qualification remains a separate
pending measurement; no performance claim is added by this correction.

2026-09-28 notebook300k qualification: actual trusted QMD/Jupyter clean7d2a578e
frontend checks10/10 pass (activation→next paint517.6/569.3ms), but MIME preparation
1106.58/1103.56ms leaves whole-representation lower bounds1624.18/1672.86ms.
Native probe failures retained; no complete one-second pass claimed. Independent
source/receipt/screenshot review approved that limited conclusion.

Safe preparation optimization computes GLB+manifest once per representation, passes
that immutable snapshot to PNG, and skips PolyCollection automatic bounds immediately
replaced by camera limits. No cross-call cache, triangle reduction, viewer change or
rasterization difference. Quiet CPU samples950.57/909.79/917.16ms; complete PNG and HTML
byte parity passed against the old path. Core source review approved. Tests pin one
preparation per call, mutable-scene updates, static MIME, and old-auto-bounds PNG parity
for orthographic/perspective scenes with state, alpha, continuous missing values and
colorbar. Final full suite200passed1skip115existingwarnings15.89s; independent corefocused3/3
passed and sourceAPPROVED. Quiet same-process warmed ABBA old1165.66/1137.15ms versus
optimized878.72/866.88ms; PNG+completeHTML exact on every run. Receipts and baseline
loader are under test-results/model3d/mime-preparation. Separate scratch-only batched
Agg prototype is not included in this change or its parity claim.

2026-09-29 wrapping completion on scene3d-slides from main577bd8c: Python source
layout mirrors the completed Flux title-fit draft, including hard-token/legend lines,
source-hidden row reflow, deterministic absent ticks and explicit impossible-box
warnings. No source geometry, physical fonts or painter rendering changed. Conservative
20% text-width reserve fixes observed 8–18% Arial/DejaVu Sans estimation differences;
arbitrary custom fonts remain estimates. Tests inspect actual Matplotlib glyph bounds.
Scratch artifacts: test-results/model3d/furniture-fit/static.png and parity.json.
Vendored viewer remains the prior canonical stamp pending final Flux integration/sync.
Final full suite340passed2skipped116upstreamwarnings20.74s; independent static8/8
passed0.25s. Three TS/Python source fixtures match exact viewport/guide geometry.
Core source review and n1 source/actual screenshot review APPROVED. Native notebook
acceptance awaits the final canonical viewer rebuild/sync and a working display.
