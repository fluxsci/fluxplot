"""Single source of truth for version + spec version (kept separate to avoid import cycles)."""

__version__ = "0.1.0"

# The manifest/recipe schema version this build emits.
# 0.2.0 — completed the parts model: gridlines/ticks/spines/legend-entries tagged,
#         real axis.x/axis.y group wrappers, and group nodes (members[]) in the parts tree.
# 0.3.0 — lossless observations/null gaps, panel ownership, explicit transform
#         capabilities, component inventories and scalar-field/colorbar metadata.
# 0.3.1 — additive: every parts-tree leaf carries its role, groups their memberRole, series and
#         legend entries a label; build presets use a closed animation vocabulary with stagger
#         hints; colour-control keys name the series (legacy positional keys still honoured).
# 0.3.2 — additive, with two id renames carried by manifest.idAliases for this minor version:
#         spines are axis.<x|y>.spine.<side>, and series slugs transliterate / hash-suffix names
#         that used to collide. New: colorScales (LUT law, alpha channel), shared scales, style
#         tokens and themes, series colours, quality.color lint, twin axes (y2 / x2), figure-scope
#         parts, images, bands, brackets with stats, tick schemes, member keys and valueMorph,
#         categorical / date data labels, insets, secondary axes, regression / kde / step / stem.
SPEC_VERSION = "0.3.2"
