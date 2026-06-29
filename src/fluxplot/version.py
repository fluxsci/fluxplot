"""Single source of truth for version + spec version (kept separate to avoid import cycles)."""

__version__ = "0.1.0"

# The manifest/recipe schema version this build emits.
# 0.2.0 — completed the parts model: gridlines/ticks/spines/legend-entries tagged,
#         real axis.x/axis.y group wrappers, and group nodes (members[]) in the parts tree.
SPEC_VERSION = "0.2.0"
