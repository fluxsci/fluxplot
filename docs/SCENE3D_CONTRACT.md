# Scene3D interchange contract, 0.1.0

The authoritative schema is `src/fluxplot/schemas/scene3d.schema.json`. Flux copies it and the
fixtures byte for byte. This is a separate discriminator from the SVG manifest: consumers
must dispatch on `spec` **before** selecting a validator. Only `spec`, `schemaVersion` and
`glb` are mandatory. Invalid or future scene3d manifests produce a warning and a plain GLB
import; they must never prevent importing otherwise valid geometry. Optional blocks, when
present, must satisfy the schema. Unknown roles remain opaque addressable parts.

The `mesh`, `pane` and `scalebar` role additions belong to the new scene3d **0.1.0**
contract. The existing SVG `SPEC_VERSION=0.3.0` stays unchanged so 2D files remain byte
compatible. `roles.ROLE_VERSION` is an internal vocabulary revision only; it is not
emitted or negotiated on the wire. Consumers use the manifest discriminator and
`schemaVersion`, and still accept unknown role strings. This is the deliberate
coordination choice for the plan’s role/spec bump requirement.

## Coordinate and identity rules

* GLB geometry is right-handed glTF Y-up. `toWorld` is a **column-major** affine 4×4 matrix
  mapping original data coordinates into stored GLB coordinates. It is a proper rotation,
  never a reflection; faces keep their original winding. `bounds` contains **world**
  coordinates; axis `lim`, `ticks`, and scale-bar `length` are in original **data** units.
* `up="y"`: identity. `up="z"`: `(x,y,z) → (x,z,-y)`. `up="-y"`:
  `(x,y,z) → (x,-y,-z)`. Other supported signed axes are also proper rotations to +Y.
* Parts use fluxplot's dotted id grammar and `IdAllocator`, with `Mark` and `Registry` as
  their in-memory semantic owners. One leaf mesh part has one GLB node, one mesh, one
  TRIANGLES primitive. Its node name is exactly `part.node`, normally `part.id`.
* `order` is part-id draw order. `parent` is an optional part-id group relation. Mesh
  `kind` is `mesh`, `field`, or `missing`; labels and guides use `furniture`.
* `glbSha256` hashes the exact **original emitted GLB bytes**, before Flux preparation.
  Flux's asset SHA hashes its own prepared bytes. These are distinct receipts: compare the
  manifest against original source bytes, never against the prepared asset hash.

## Binary contract

The file is GLB 2.0: 12-byte little-endian header, JSON then BIN chunk, each 4-byte aligned.
JSON is UTF-8 and space-padded; BIN is zero-padded. The JSON buffer byteLength excludes
terminal chunk padding. There is one embedded buffer, no URI and no extensions. All
buffer views and accessors have 4-byte-aligned starts. Attributes are tightly packed,
except uint8 SCALAR `_VALID`: each value is padded to a 4-byte stride, recorded as
`bufferView.byteStride=4`, so vertex alignment satisfies glTF.

Each primitive has little-endian float32 POSITION and NORMAL VEC3 accessors; positions
carry min/max. Normals are area-weighted vertex normals normalized to unit length
(degenerate-only vertices use `[0,1,0]`). Indices are unsigned uint16 when the maximum
index is below 65536, otherwise uint32. There is no welding or position-based sorting.
Dropping unreferenced vertices uses ascending **original index order** only.

Optional COLOR_0 is normalized unsigned-byte VEC4. Colors are linear RGB as required by
glTF; alpha is linear. Material baseColorFactor is likewise linear RGB. `color` values and
colormap stops in the manifest are ordinary sRGB hex. Materials use metallicFactor 0,
roughnessFactor 0.6, doubleSided true; alphaMode BLEND is emitted for alpha below one.

A field primitive additionally carries finite float32 SCALAR `_VALUE` in original data
units. Missing values are stored as zero with an optional unsigned-byte SCALAR `_VALID`
attribute: 1 means valid, 0 means missing. When `_VALID` is absent, every value is valid.
A real zero has `_VALID=1`; it never becomes missing. Python accepts NaN and returns masked
values as NaN, but the binary contains no non-finite floating-point values, as required by
glTF. This deliberately corrects the plan's invalid NaN-in-accessor convention. COLOR_0
remains a complete foreign-viewer fallback. Missing faces form a separately addressable part. A continuous face touching a missing vertex is missing; categories use
vertex majority with first-vertex tie-break, the same rules as `surface()`.

## Shape states and separate morph partners

States are standard glTF relative morph targets: float32 POSITION and NORMAL deltas.
POSITION delta accessors carry min/max. `mesh.extras.targetNames` matches target order;
`mesh.weights` carries the default weights. Every primitive in a mesh has the same target
count. Manifest `states` uses matching names and optional display labels; `sequence: true`
means ordered frames: frame 0 is the base, frames 1..N are these states, fractional frames
blend their neighbors. Weights are stored unclamped; ordinary UI sliders are 0–1.

Framing bounds are the union of actual base positions and each named state at weight 1,
not just base bounds. Arbitrary sums of weights may extend beyond these bounds.

Two models morph only when all paired primitives have equal modes, vertex counts and
indices. Pair by node name when both are named, otherwise primitive order. `morphGroup`
is a suggestion label, never compatibility evidence. `share_topology_with` checks original
faces and vertex counts before compacting, and replays the reference decimation rather
than independently simplifying another shape.

## Mapping and furniture

`field.range` stores the resolved limits; `rule.percentile` records how they were selected.
An explicit color_range wins. `field.cmap.stops` runs from 0 to 1 in increasing order,
with hex sRGB colors. `field.missingColor` is the missing fallback. A colorbar part uses
`field: "<series>.field"`; it never embeds a second mapping. The runtime normalizes/clips
values, interpolates sRGB stops, then converts to linear RGB for glTF shading.

Axis ticks and limits are authored in Python using matplotlib locators; computed tick
values are rounded to 12 significant digits (`0.6`, not `0.6000000000000001`). Labels,
legend, colorbar and title are SVG furniture in Flux. Text uses physical pt; resizing
changes the viewport, not fonts. `axes.kind="triad"` uses original data axes after
`toWorld`; a scale bar is shown only for orthographic projection. Layout positions are
right/top in v1. The triad sits in the viewport's bottom-left corner; a scale bar sits
bottom-left, or bottom-right (right-aligned, same inset) when a triad is shown, so the
two never overlap. The scale-bar part's `length` is in data units; its `label` is the
display text (Python simplifies metric units: `10000` nm is labelled `10 µm`).

Part opacity is carried by the alpha of `color` (8-digit hex), mirrored in the material's
baseColorFactor alpha with `alphaMode: BLEND`; a vertex-coloured field keeps a white
base colour and carries only that alpha. The optional `opacity` part field remains
available to consumers but is not emitted by Python.

## Golden fixtures

`tests/generate_scene3d_fixtures.py` generates `tests/fixtures/scene3d/`: plain mesh;
named parts and legend; continuous field and colorbar; categorical labels and missing;
box axes and ticks; scale bar; a same-topology two-part sphere/blob pair; an incompatible
partner; base plus two shape states; and an eight-frame sequence (base + seven targets).
No timestamp enters GLB or manifest. SHA receipts and exact bytes are checked in tests.
Fixture changes must be regenerated and copied to Flux together with this schema.
