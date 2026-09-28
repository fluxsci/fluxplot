# 3D fluxplots

A 3D fluxplot keeps the mesh, its values, named parts and a saved viewing angle. You can
orbit it in a notebook and later choose another angle in Flux without rerunning Python.

```python
import fluxplot as fp
fp.use_paper()
sc = fp.scene3d(figsize=(3.5, 3), units='nm', up='-y',
                title='Neuron', scalebar=10_000)
fp.mesh3d(sc, neuron, series='neuron', color=fp.colors.blue400)
sc.view(azimuth=30, elevation=15, zoom=0.9)
sc
fp.save(sc, 'plots/neuron')
```

The three outputs are `neuron.glb`, `neuron.fluxplot.json`, and `neuron.recipe.json`.
`figsize` is inches and controls the default physical placement in Flux. `up` describes
which direction is up in your data; it supports all six signed axes and converts to glTF
Y-up while keeping axis labels and scale bars in your original units.

Inputs can be `(vertices, faces)` arrays or objects exposing `vertices` and `faces`
(trimesh, meshparty, cloudvolume). PyVista PolyData uses `points` and packed triangle
`faces`. GIFTI paths use optional `nibabel`; other mesh paths use optional `trimesh`.
Install `fluxplot[mesh]` for mesh-file reading and accelerated simplification. Arrays need
no new dependency. Volumes must first be converted to triangle meshes, for example with
`skimage.measure.marching_cubes`.

## Named parts and value fields

```python
fp.mesh3d(sc, {'soma': soma, 'axon': axon, 'dendrites': dendrites},
          series='neuron', palette={'soma': '#D14D41', 'axon': '#4385BE',
                                    'dendrites': '#3AA99F'}, legend=True)

cx = fp.scene3d(figsize=(3, 3), units='mm', axes='triad')
fp.surface3d(cx, thickness, series='thickness', surfaces=(vertices, faces),
             kind='continuous', cmap='viridis', percentile=(2, 98),
             colorbar=True, cbar_label='Thickness (mm)')
```

Parts have stable names such as `neuron.axon` and `thickness.field`. Flux can recolor,
fade or hide each one. Continuous values remain in the GLB, so Flux can change their
colormap and limits directly. The used colormap stops are included in the manifest.
An explicit `color_range=(min, max)` takes precedence over percentiles.

`surface3d` uses the same value-to-color functions as `surface`. Continuous 3D colors
interpolate per vertex; the 2D surface colors each face's mean value. Categorical maps
use `kind='label'`, `categories={code: name}`, and `palette={name_or_code: color}`. Each
category gets its own part. A boundary face follows the majority of its three vertices;
a three-way tie follows the first vertex. Default category colors follow sorted category
names; pass a palette to keep colors fixed across differently populated subsets.

NaN, infinity, `missing_below=...`, and `missing_values=[...]` become missing data. A face
touching missing data goes into `<series>.missing`; **zero stays real**. The GLB encodes
missing values with a validity mask because glTF forbids non-finite binary values.

## Shape states first, separate morph partners when needed

Several shapes of one mesh can live in the same file:

```python
cx = fp.scene3d(figsize=(3, 3))
fp.mesh3d(cx, pial, series='cortex', states={'inflated': inflated})
cx.view(states={'inflated': 0.25})
fp.save(cx, 'plots/cortex')
```

Every state must have the same vertex count and exact triangle indices as its base.
Flux shows a slider per state. Notebook state sliders change the displayed preview;
Copy view includes the state weights to paste into `sc.view(...)` and save. The frame
includes the base and every state at weight 1, so inflation does not reframe the model.
Combining several weights above 1 can exceed that frame.

For a time sequence, pass `states=frames[1:], sequence=True`. Frame 0 is the base,
frames 1 onward are the targets. `sc.view(frame=2.5)` blends frames 2 and 3.

For separate files, use the same series/part names and verify correspondence:

```python
a = fp.scene3d(); fp.mesh3d(a, pial, series='cortex', max_faces=200_000)
b = fp.scene3d(); fp.mesh3d(b, inflated, series='cortex', share_topology_with=a,
                           morph_group='cortex')
assert fp.can_morph(a, b)
fp.save(a, 'plots/cortex_pial'); fp.save(b, 'plots/cortex_inflated')
```

`can_morph` returns a mapping with `ok`, paired node names, and a first-mismatch reason;
its truth value is `ok`. The reference's decimation collapses are replayed on the partner
and all shape states. Independent simplification can destroy correspondence. Named parts
are reduced separately, and values are remapped using the collapse mapping. A simplifier
may retain more faces than requested; that is reported explicitly. Simplification can
change a categorical boundary's shape even though it never merges category parts.

## Axes, scale bars and notebook behavior

Use `axes='none'`, `'box'`, or `'triad'`. Box axes use authored data limits and matplotlib
tick locators. Override them with `sc.axis('x', lim=(0, 100), ticks=[0, 50, 100], label='x (µm)')`.
Fonts come from the active house style at scene creation and stay in physical points.
Scale bars appear only in orthographic projection.

Notebook output bundles a self-contained viewer and a PNG. Drag orbits, wheel zooms,
axis keys choose views, and Home restores the saved view. Copy view provides the exact
Python view arguments. Script-free frontends can show the PNG; `sc.show(static=True)`
forces the still. Trust is controlled by the notebook frontend. Trusted VS Code Jupyter
and QMD Notebook execute this viewer. In untrusted VS Code, an output containing both
HTML and PNG can appear blank because the frontend blocks its selected HTML representation;
it does not automatically select the alternate PNG. Select the PNG representation if the frontend offers that choice, or save a PNG-only
output with `sc.show(static=True)` before sharing. PNG-only output was verified in untrusted
VS Code Jupyter; untrusted QMD output restoration remains unverified.
Frontends that render HTML but disable scripts retain the inline PNG.

`preview_max_faces` defaults to 300,000. With the optional simplifier, only the private
notebook preview is reduced; `fp.save` keeps your scene's full resolution. Without it,
previews remain full resolution and warn above 30 MiB. Notebook previews use a white paper
background so authored figure labels remain readable in dark notebook themes. This does
not change the GLB background or Flux export transparency. A static PNG uses a global face
painter with the surface shading functions; intersecting transparent geometry may look
different from the WebGL depth-buffer view.

## Troubleshooting

- An all-grey map often means all values are missing. Check sentinels and `missing_below`.
- If the notebook shows only a still, its frontend may not execute trusted HTML scripts.
  The mesh still saves normally and remains interactive in Flux.
- If a state or morph partner is refused, inspect vertex count and exact face order.
  Loading files with position-based welding or independent decimation changes topology.
- If a large model is slow, use `max_faces` and fewer states. Each state adds 24 bytes
  per vertex before GLB overhead. The notebook preview budget does not shrink saved files.

When a save exceeds Flux's 50 MiB or two-million-triangle warning limits, its warning
gives a scene-specific `max_faces=N` to apply to **each** `mesh3d`/`surface3d` call.
The cap conservatively accounts for every part, value/color channel, shape state and
GLB metadata; it preserves each call's minimum of one triangle per part. It assumes
the simplifier reaches the cap: retained-face warnings still require attention.
If fixed part/state overhead cannot fit, the warning says to remove parts/states or
split the scene. If the conservative bound cannot guarantee a cap, it says so instead
of promising one. Simplification also drops backend-produced unused vertices in
original index order, applying the same map to states, values and colors; it does not
weld vertices or disturb shared-topology correspondence.

`uv run --extra mesh python examples/scene3d_demo.py` writes scratch assets under
`test-results/model3d/demo/plots/` (override with `--out`). It creates a neuron-like mesh, two shape states,
a same-topology morph pair, a field map and an eight-frame sequence without downloaded data.
The schema and binary details are in [SCENE3D_CONTRACT.md](SCENE3D_CONTRACT.md).
