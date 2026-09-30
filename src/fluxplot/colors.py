"""fluxplot.colors — the canonical fluxplot-flexoki palette and colormap registry.

Almost all color "defaults" in fluxplot come from here; :mod:`fluxplot.style` (and
everything else) imports its color definitions from this module, so an edit here
percolates everywhere.

Usage::

    from fluxplot import colors as fx

    fx.green400              # the flexoki palette is available at the top level (hex string)
    fx.flex.get("green-400") # full metadata dict (h, l, hex, rgb)

    fx.maps.emerald          # the cmasher "emerald" colormap
    fx.maps.flexoki_diverging  # a fluxplot custom colormap
    fx.maps.view_map_set("cmasher")  # plot a labelled grid of every map in a set

Anything that isn't the flexoki palette needs a specifier — ``fx.maps`` for
colormaps, and eventually e.g. ``fx.tol`` if we add Paul Tol's colors.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
import functools
import json
from importlib import resources

__all__ = ["flex", "maps", "palettes", "categories", "token_of", "palette_of", "palette_colors", "DISCRETE_MAX"]

#: A ``ListedColormap`` with at most this many colours is a *discrete* map (a set of classes to
#: pick from); above it, a listed map is treated as a continuous ramp. One threshold for the
#: colour-scale records, the signature plots' palette resolution and the definitions builder.
DISCRETE_MAX = 32


class _FlexPalette:
    """Flexoki color palette with convenient attribute access.

    Attribute names use the pattern ``{color}{level}`` (no separator),
    e.g. ``green300``, ``red600``, ``base500``, ``paper``, ``black``.

    Accessing an attribute returns the **hex string** by default.
    Use :meth:`get` to retrieve the full metadata dict for a color.
    """

    def __init__(self, data: dict[str, dict]) -> None:
        self._data = data

    # -- attribute access (returns hex) --------------------------------
    def __getattr__(self, name: str) -> str:
        key = _attr_to_key(name)
        if key in self._data:
            return self._data[key]["hex"]
        raise AttributeError(f"No color {name!r} in the Flexoki palette")

    def __dir__(self) -> list[str]:
        extras = [_key_to_attr(k) for k in self._data]
        return sorted(set(super().__dir__()) | set(extras))

    # -- dict-style access (returns full metadata) ---------------------
    def get(self, name: str) -> dict:
        """Return the full metadata dict (h, l, hex, rgb) for a color.

        *name* can be either attribute-style (``green300``) or
        key-style (``green-300``).
        """
        key = _attr_to_key(name) if "-" not in name else name
        if key not in self._data:
            raise KeyError(f"No color {name!r} in the Flexoki palette")
        return self._data[key]

    def __repr__(self) -> str:
        return f"FlexPalette({len(self._data)} colors)"

    def __contains__(self, name: str) -> bool:
        key = _attr_to_key(name) if "-" not in name else name
        return key in self._data

    def keys(self) -> list[str]:
        """Return all color keys (dash-separated form)."""
        return list(self._data.keys())


def _attr_to_key(attr: str) -> str:
    """Convert attribute name to dict key: ``green300`` -> ``green-300``."""
    # Find the split point: last alpha char before first digit
    for i, ch in enumerate(attr):
        if ch.isdigit():
            return f"{attr[:i]}-{attr[i:]}"
    return attr  # no digits (e.g. "paper", "black")


def _key_to_attr(key: str) -> str:
    """Convert dict key to attribute name: ``green-300`` -> ``green300``."""
    return key.replace("-", "")


# The Flexoki palette. The canonical definition is the design-token export
# ``definitions/flexoki.tokens.json`` (the W3C design-tokens format Figma writes); this table is
# built from it at import. Two house conventions ride on top: fluxplot's "green" is the custom
# hue the tokens file stores under ``GRN`` (Flexoki's own green is kept as "olive"), and
# ``base-0`` / ``base-1000`` alias ``paper`` / ``black``. ``definitions/palettes.json``'s Flexoki
# collection is generated from this table by ``tools/build_color_definitions.py``; the three
# copies can never disagree (``tests/test_color_definitions.py::test_single_source``).
_HUE_OF_GROUP = {"base": "k", "red": "r", "orange": "o", "yellow": "y", "olive": "ol", "GRN": "g",
                 "cyan": "c", "blue": "b", "purple": "p", "magenta": "m"}


@functools.lru_cache(maxsize=None)
def _definitions(name: str) -> dict:
    with resources.files("fluxplot").joinpath(f"definitions/{name}.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _rgb(hex_: str) -> tuple:
    return tuple(int(hex_[i:i + 2], 16) for i in (1, 3, 5))


def _load_flexoki_tokens() -> dict:
    tokens = _definitions("flexoki.tokens")
    data: dict[str, dict] = {}
    for group, hue in _HUE_OF_GROUP.items():
        entries = {}
        for name, token in tokens.get(group, {}).items():
            if not isinstance(token, dict) or "$value" not in token or name.endswith("-opacity-10"):
                continue
            hex_ = str(token["$value"]["hex"]).upper()
            tail = name.rsplit("-", 1)[-1]
            level = int(tail) if tail.isdigit() else {"paper": 0, "black": 1000}.get(name, 0)
            entries[name] = {"h": hue, "l": level, "hex": hex_, "rgb": _rgb(hex_)}
        if group == "base":  # paper, base-0 (= paper), base-50 … base-950, base-1000 (= black), black
            levels = sorted((e for n, e in entries.items() if n.startswith("base-")), key=lambda e: e["l"])
            data["paper"] = entries["paper"]
            data["base-0"] = dict(entries["paper"])
            for e in levels:
                data[f"base-{e['l']}"] = e
            data["base-1000"] = dict(entries["black"])
            data["black"] = entries["black"]
        else:
            for e in sorted(entries.values(), key=lambda e: e["l"]):
                data[f"{group if group != 'GRN' else 'green'}-{e['l']}"] = e
    return data


_flex_data = _load_flexoki_tokens()

flex = _FlexPalette(_flex_data)

# -------------------------------------------
# MAPS SECTION
# -------------------------------------------


# -------------------------------------------
# DEFINITIONS (JSON, shipped with the package)
# -------------------------------------------
#
# Every colormap and palette collection fluxplot knows lives in
# ``definitions/colormaps.json`` and ``definitions/palettes.json`` as plain data —
# matplotlib's maps, Fabio Crameri's Scientific colour maps, Paul Tol's maps and
# sets, cmasher, ColorBrewer, Flexoki. The files are built once by
# ``tools/build_color_definitions.py`` from the upstream packages; at runtime
# nothing but the JSON is read, so no upstream package is a dependency, and Flux
# bundles the very same files for its pickers. A continuous map is 256 samples,
# a discrete one its exact colours.

_MAP_COLLECTION_ORDER = ("mpl", "crameri", "tol", "cmasher")


def _build_cmap(full_name: str, m: dict) -> Colormap:
    """Rebuild a shipped map from its definition. ``colors`` is the map's exact lookup table:
    a listed map's own colours (``N`` of them, ``discrete`` when few enough to be classes), or
    256 samples of a continuous map — which, resampled through ``from_list`` at ``N=256``, is
    byte for byte the table matplotlib indexes."""
    colors = list(m["colors"])
    if m.get("discrete") or m.get("N", 256) != 256 or len(colors) != 256:
        return ListedColormap(colors, name=full_name)
    return LinearSegmentedColormap.from_list(full_name, colors, N=256)


class _MapRegistry:
    """Colormap access under ``fx.maps``.

    Attribute lookup resolves fluxplot's custom maps first (``maps.flexoki_diverging``,
    and eventually e.g. ``maps.fluxglow``), then the shipped collections (``maps.emerald``
    -> ``cmasher.emerald``, ``maps.batlow`` -> ``crameri.batlow``), and finally the
    `cmasher <https://cmasher.readthedocs.io>`_ package itself (a dependency), for any map
    newer than the shipped definitions.
    """

    def __init__(self) -> None:
        self._custom: dict[str, Colormap] = {}
        self._alias_of: dict[str, str] = {}  # "flexoki_diverging" -> "flexoki.diverging"
        # collection id -> {bare name: Colormap}; built from the JSON once
        self._collections: dict[str, dict[str, Colormap]] | None = None
        self._info: dict[str, dict] = {}

    # -- the JSON collections --------------------------------------------
    def _ensure(self) -> dict[str, dict[str, Colormap]]:
        if self._collections is None:
            cols: dict[str, dict[str, Colormap]] = {}
            for c in _definitions("colormaps")["collections"]:
                cols[c["id"]] = {}
                for m in c["maps"]:
                    full = f"{c['id']}.{m['name']}"
                    cm = _build_cmap(full, m)
                    cols[c["id"]][m["name"]] = cm
                    self._info[full] = {
                        "collection": c["id"], "name": m["name"], "type": m["type"],
                        "family": m.get("family"), "discrete": bool(m.get("discrete")),
                        "uniform": m.get("uniform"), "N": int(m.get("N", len(m["colors"]))),
                    }
                    for variant in (cm, cm.reversed()):
                        try:
                            mpl.colormaps.register(variant)
                        except Exception:  # already registered — best-effort
                            pass
            self._collections = cols
        return self._collections

    def collections(self) -> list[str]:
        """The map collections: ``'flexoki'`` (fluxplot's own) then the shipped ones."""
        return list(dict.fromkeys(["flexoki", *self._ensure()]))

    def get(self, name: str) -> Colormap:
        """Resolve a colormap by name: ``'crameri.batlow'``, a bare ``'batlow'`` (the
        first collection that has it — matplotlib, Crameri, Tol, cmasher), a fluxplot
        custom map, any of those with ``_r`` for the reversed map, or a truncation
        ``'batlow[0.2:0.8]'`` (see :meth:`truncate`)."""
        if name in self._custom:
            return self._custom[name]
        if name.endswith("]") and "[" in name:
            base_name, _, span = name[:-1].rpartition("[")
            lo, _, hi = span.partition(":")
            try:
                return self.truncate(base_name, float(lo), float(hi))
            except ValueError:
                raise KeyError(f"No colormap {name!r}: a truncation is spelled 'name[lo:hi]' with 0 <= lo < hi <= 1") from None
        base, reversed_ = (name[:-2], True) if name.endswith("_r") else (name, False)
        cols = self._ensure()
        found: Colormap | None = None
        if "." in base:
            cid, _, bare = base.partition(".")
            if cid == "cmr":
                cid = "cmasher"
            found = cols.get(cid, {}).get(bare)
        else:
            if base in self._custom:
                found = self._custom[base]
            else:
                for cid in _MAP_COLLECTION_ORDER:
                    if base in cols.get(cid, {}):
                        found = cols[cid][base]
                        break
        if found is None:
            raise KeyError(f"No colormap {name!r} in fluxplot's collections ({', '.join(self.collections())})")
        return found.reversed() if reversed_ else found

    def info(self, name: str) -> dict:
        """Collection, type (sequential / diverging / cyclic / qualitative / misc), family,
        discreteness, table size ``N`` and perceptual uniformity (``True`` / ``False`` /
        ``None`` = not assessed) of a shipped map (``'crameri.batlow'`` or bare)."""
        cm = self.get(name)
        key = cm.name[:-2] if cm.name.endswith("_r") else cm.name
        key = key.split("[", 1)[0]  # a truncation reports its source map
        found = self._info.get(key) or self._info.get(self._alias_of.get(key, ""))
        return dict(found or {"collection": "flexoki", "name": key, "type": "custom", "family": None,
                              "discrete": isinstance(cm, ListedColormap) and cm.N <= DISCRETE_MAX,
                              "uniform": None, "N": int(cm.N)})

    # -- derived maps -----------------------------------------------------
    def truncate(self, cmap, lo: float, hi: float, n: int = 256) -> Colormap:
        """The stretch ``[lo, hi]`` of a map as a map of its own, named ``"<name>[lo:hi]"`` —
        which :meth:`get` resolves again, so a recipe can record it."""
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(f"truncate: need 0 <= lo < hi <= 1, got {lo!r}, {hi!r}")
        cm = cmap if isinstance(cmap, Colormap) else self._resolve_any(cmap)
        import numpy as np
        name = f"{cm.name}[{lo:g}:{hi:g}]"
        return LinearSegmentedColormap.from_list(name, cm(np.linspace(lo, hi, n)), N=n)

    def discretize(self, cmap, boundaries=None, *, n: int | None = None, vmin=None, vmax=None,
                   extend: str = "neither"):
        """A binned scale: ``(ListedColormap, BoundaryNorm)`` with one colour per bin. Give the
        ``boundaries`` (bin edges), or ``n`` bins between ``vmin`` and ``vmax``. ``extend``
        adds under / over colours from the map's ends. Helpers accept the norm as ``norm=``;
        the colour scale is then recorded as ``kind: "binned"``."""
        import numpy as np
        from matplotlib.colors import BoundaryNorm
        cm = cmap if isinstance(cmap, Colormap) else self._resolve_any(cmap)
        if boundaries is None:
            if n is None or vmin is None or vmax is None:
                raise ValueError("discretize: give boundaries, or n with vmin and vmax")
            boundaries = np.linspace(vmin, vmax, int(n) + 1)
        b = np.asarray(boundaries, dtype=float)
        if b.ndim != 1 or b.size < 2 or not np.all(np.diff(b) > 0):
            raise ValueError("discretize: boundaries must be at least two increasing numbers")
        if extend not in ("neither", "min", "max", "both"):
            raise ValueError("discretize: extend must be neither, min, max or both")
        bins = b.size - 1
        extra = (extend in ("min", "both")) + (extend in ("max", "both"))
        # with extend, matplotlib's BoundaryNorm reserves the map's first / last colour for the
        # under / over region (a Colormap's default under / over ARE its end entries), so the
        # listed map carries bins + extensions colours and the norm counts all of them
        colours = cm(np.linspace(0, 1, bins + extra))
        listed = ListedColormap(colours, name=f"{cm.name}[{bins} bins{'+' + extend if extra else ''}]")
        listed.set_bad(cm.get_bad())
        return listed, BoundaryNorm(b, bins + extra, extend=extend)

    def _resolve_any(self, name: str) -> Colormap:
        if name in mpl.colormaps:
            return mpl.colormaps[name]
        return self.get(name)

    # -- attribute access -----------------------------------------------
    def __getattr__(self, name: str) -> Colormap:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._custom:
            return self._custom[name]
        try:
            return self.get(name)
        except KeyError:
            pass
        cmr = _import_cmasher()
        if cmr is None:  # pragma: no cover - cmasher is a dependency
            raise AttributeError(f"No colormap {name!r}: not a fluxplot custom map, and cmasher is unavailable")
        cm = getattr(cmr, name, None)
        if isinstance(cm, Colormap):
            return cm
        raise AttributeError(f"No colormap {name!r} in the fluxplot custom maps or cmasher")

    def __dir__(self) -> list[str]:
        names = set(super().__dir__()) | set(self._custom)
        for maps in self._ensure().values():
            names |= set(maps)
        return sorted(names)

    def __repr__(self) -> str:
        n = sum(len(m) for m in self._ensure().values())
        return f"MapRegistry({len(self._custom)} custom maps + {n} shipped in {', '.join(_MAP_COLLECTION_ORDER)})"

    # -- registration ----------------------------------------------------
    def register(self, cmap: Colormap) -> None:
        """Add a custom colormap (also registered with matplotlib, so
        ``plt.imshow(..., cmap=cmap.name)`` works by name), together with its reversed
        twin under ``<name>_r``."""
        for variant in (cmap, cmap.reversed()):
            self._custom[variant.name] = variant
            try:
                mpl.colormaps.register(variant)
            except Exception:  # already registered / older mpl — best-effort
                pass

    # -- map sets ----------------------------------------------------------
    def names(self, set_name: str = "cmasher") -> list[str]:
        """Return the colormap names in a collection: ``'flexoki'``, ``'mpl'``,
        ``'crameri'``, ``'tol'`` or ``'cmasher'`` (reversed ``*_r`` variants omitted)."""
        return [name for name, _ in self._map_set(set_name)]

    def view_map_set(self, set_name: str = "cmasher", ncols: int = 3):
        """Plot a labelled grid of every colormap in *set_name*, for quick browsing.

        ``set_name`` is ``'cmasher'`` (reversed ``*_r`` variants omitted) or
        ``'flexoki'`` (the fluxplot custom maps). Returns the figure.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        entries = self._map_set(set_name)
        nrows = -(-len(entries) // ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 3.2, 0.62 * nrows + 0.5),
            squeeze=False,
            layout="constrained",
        )
        gradient = np.linspace(0, 1, 256)[None, :]
        for ax, (name, cm) in zip(axes.flat, entries):
            ax.imshow(gradient, aspect="auto", cmap=cm)
            ax.set_title(name, loc="left", fontsize=7, family="monospace", pad=2)
        for ax in axes.flat:
            ax.set_axis_off()
        fig.suptitle(f"{set_name} colormaps", fontsize=10)
        return fig

    def _map_set(self, set_name: str) -> list[tuple[str, Colormap]]:
        s = set_name.lower()
        cols = self._ensure()
        if s in ("flexoki", "fluxplot", "flux"):
            shipped = [(f"flexoki.{n}", cm) for n, cm in cols.get("flexoki", {}).items()]
            custom = [(n, cm) for n, cm in self._custom.items() if not n.endswith("_r") and n not in self._alias_of]
            return shipped + sorted(custom)
        if s in cols:
            return list(cols[s].items())
        raise ValueError(
            f"Unknown map set {set_name!r}; available sets: {', '.join(self.collections())}"
        )


def _import_cmasher():
    try:
        import cmasher  # importing also registers "cmr.*" names in matplotlib

        return cmasher
    except ImportError:
        return None


maps = _MapRegistry()
maps._ensure()  # every shipped map is addressable by name in matplotlib from `import fluxplot` on


# -------------------------------------------
# PALETTES SECTION
# -------------------------------------------


class _Palettes:
    """Palette collections under ``fx.palettes``: ``'flexoki'`` (the default),
    ``'brewer'`` (ColorBrewer) and ``'tol'`` (Paul Tol's colour sets), from
    ``definitions/palettes.json``. ``palettes.brewer["Blues"]`` is a list of hex
    strings; ``palettes.get("tol", "bright")`` the same; ``palettes.info("brewer")``
    the collection's metadata and typed groups."""

    def collections(self) -> list[str]:
        return [c["id"] for c in _definitions("palettes")["collections"]]

    def info(self, collection: str) -> dict:
        for c in _definitions("palettes")["collections"]:
            if c["id"] == collection:
                return c
        raise KeyError(f"No palette collection {collection!r}; available: {', '.join(self.collections())}")

    def names(self, collection: str) -> list[str]:
        return [g["name"] for g in self.info(collection)["groups"]]

    def get(self, collection: str, group: str) -> list[str]:
        for g in self.info(collection)["groups"]:
            if g["name"] == group:
                return [s["hex"] for s in g["swatches"]]
        raise KeyError(f"No group {group!r} in palette collection {collection!r}")

    def __getattr__(self, collection: str) -> dict[str, list[str]]:
        if collection.startswith("_"):
            raise AttributeError(collection)
        try:
            return {g["name"]: [s["hex"] for s in g["swatches"]] for g in self.info(collection)["groups"]}
        except KeyError as e:
            raise AttributeError(str(e)) from None

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self.collections()))

    def __repr__(self) -> str:
        return f"Palettes({', '.join(self.collections())})"


palettes = _Palettes()

# Flexoki-flavoured house maps: linear ramps through palette anchors (pleasant, NOT perceptually
# uniform — ``maps.info(...)["uniform"] is False``; prefer the cmasher / Crameri maps when
# uniformity matters). The anchors are the source ``tools/build_color_definitions.py`` samples
# into the ``flexoki`` collection of ``definitions/colormaps.json``; at runtime the shipped
# collection is used and each map is aliased under its historical bare name (``flexoki_diverging``
# == ``flexoki.diverging``), so both spellings resolve to the same table.
FLEXOKI_MAP_ANCHORS = {
    # name: (type, [token names, light → dark or end → end])
    "sequential": ("sequential", ["paper", "blue-150", "blue-400", "blue-600", "blue-800"]),
    "warm": ("sequential", ["paper", "yellow-400", "orange-400", "red-600", "red-850"]),
    "diverging": ("diverging", ["blue-600", "blue-400", "paper", "red-400", "red-600"]),
    "terrain": ("sequential", ["blue-800", "blue-600", "cyan-400", "olive-400", "yellow-400", "orange-600", "paper"]),
    "spectrum": ("cyclic", ["red-600", "orange-600", "yellow-600", "olive-600", "green-600", "cyan-600",
                            "blue-600", "purple-600", "magenta-600", "red-600"]),
}


def flexoki_map_from_anchors(name: str) -> Colormap:
    """The house map ``name`` built straight from its palette anchors (what the builder samples)."""
    kind, tokens = FLEXOKI_MAP_ANCHORS[name]
    return LinearSegmentedColormap.from_list(f"flexoki.{name}", [flex.get(t)["hex"] for t in tokens])


def _install_house_maps() -> None:
    shipped = maps._ensure().get("flexoki", {})
    for name in FLEXOKI_MAP_ANCHORS:
        source = shipped.get(name) or flexoki_map_from_anchors(name)  # the JSON, or the anchors on a fresh build
        alias = source.copy()
        alias.name = f"flexoki_{name}"
        maps.register(alias)
        maps._alias_of[alias.name] = f"flexoki.{name}"
        if f"flexoki.{name}" not in maps._info:
            maps._info[f"flexoki.{name}"] = {"collection": "flexoki", "name": name, "type": FLEXOKI_MAP_ANCHORS[name][0],
                                             "family": None, "discrete": False, "uniform": False, "N": source.N}


_install_house_maps()


# -------------------------------------------
# Naming a colour: the exact reverse lookup over every palette fluxplot ships
# -------------------------------------------
@functools.lru_cache(maxsize=None)
def _token_table() -> dict:
    """lowercase hex → the first token that names it: ``flexoki.green-400``, ``brewer.Blues-3``,
    ``tol.bright.blue`` (Flexoki first, then the shipped palette collections)."""
    table: dict[str, str] = {}
    for key, meta in _flex_data.items():
        if key in ("base-0", "base-1000"):
            continue
        table.setdefault(meta["hex"].lower(), f"flexoki.{key}")
    for c in _definitions("palettes")["collections"]:
        if c["id"] == "flexoki":
            continue
        for g in c["groups"]:
            for sw in g["swatches"]:
                name = sw["name"] if c["id"] == "brewer" else f"{g['name']}.{sw['name']}"
                table.setdefault(sw["hex"].lower(), f"{c['id']}.{name}")
    return table


def token_of(color) -> str | None:
    """The palette token an exact colour is (``colors.green400`` → ``"flexoki.green-400"``),
    or ``None`` for a colour no shipped palette has. Alpha is ignored."""
    from matplotlib.colors import to_hex
    try:
        return _token_table().get(to_hex(color).lower())
    except ValueError:
        return None


def palette_of(color, cycle=None) -> dict | None:
    """``{"name", "index"}`` when ``color`` sits in the active prop cycle (``cycle`` = the list
    of cycle colours, default matplotlib's): ``flexoki.light`` / ``flexoki.dark`` for the house
    cycles, a shipped palette's name when the cycle is one, else ``"cycle"``."""
    from matplotlib.colors import to_hex
    if cycle is None:
        cycle = [c["color"] for c in mpl.rcParams["axes.prop_cycle"]]
    try:
        hexes = [to_hex(c).lower() for c in cycle]
        target = to_hex(color).lower()
    except ValueError:
        return None
    if target not in hexes:
        return None
    from . import style as _style
    if hexes == [h.lower() for h in _style.CYCLE_LIGHT]:
        name = "flexoki.light"
    elif hexes == [h.lower() for h in _style.CYCLE_DARK]:
        name = "flexoki.dark"
    else:
        name = "cycle"
        for c in _definitions("palettes")["collections"]:
            for g in c["groups"]:
                if [sw["hex"].lower() for sw in g["swatches"]] == hexes:
                    name = f"{c['id']}.{g['name']}"
    return {"name": name, "index": hexes.index(target)}


# -------------------------------------------
# Category colours: one registry, so "SD" is the same colour in every figure
# -------------------------------------------
CATEGORIES_FILENAME = "fluxplot.colors.json"


class _Categories:
    """Stable colours for named categories (``fp.colors.categories``).

    ``get(name)`` returns the colour pinned to ``name``, else assigns the next unused slot of the
    palette (default: the active theme's cycle) in first-request order and remembers it — so a
    category keeps its colour across every figure of a session, whatever other categories each
    figure shows. ``assign`` pins colours outright. ``load`` / ``save`` read and write a project
    file (``fluxplot.colors.json``: ``{"spec": "fluxplot/colors", "version": 1, "categories":
    {"SD": "#bc5215"}, "palette": "flexoki"}``); the first use auto-loads ``$FLUXPLOT_COLORS`` or
    the nearest ``fluxplot.colors.json`` between the working directory and the Git root. Nothing
    is ever written implicitly. ``auto_series=True`` makes ``fp.line`` / ``fp.scatter`` colour a
    series by its name when no colour is given (off by default: it would change existing plots).
    """

    def __init__(self) -> None:
        self._pinned: dict[str, str] = {}
        self._assigned: dict[str, str] = {}
        self._palette = None
        self._loaded = False
        self.auto_series = False
        self.path: str | None = None

    # -- state -----------------------------------------------------------
    def reset(self) -> None:
        self.__init__()

    def assign(self, mapping: dict) -> None:
        """Pin colours: ``{"SD": "#bc5215", "Sleep": colors.blue600}``."""
        from matplotlib.colors import to_hex
        for name, colour in mapping.items():
            self._pinned[str(name)] = to_hex(colour, keep_alpha=False).lower()

    def is_pinned(self, name) -> bool:
        self._autoload()
        return str(name) in self._pinned

    def pinned(self) -> dict:
        self._autoload()
        return dict(self._pinned)

    def _cycle(self, palette) -> list:
        from matplotlib.colors import to_hex
        spec = palette if palette is not None else self._palette
        if spec is None:
            return [to_hex(c["color"]).lower() for c in mpl.rcParams["axes.prop_cycle"]]
        if isinstance(spec, (list, tuple)):
            return [to_hex(c).lower() for c in spec]
        return [h.lower() for h in palette_colors(spec)]

    def get(self, name, *, palette=None) -> str:
        """The colour for ``name``: pinned, remembered, or the next free slot of ``palette``."""
        self._autoload()
        key = str(name)
        if key in self._pinned:
            return self._pinned[key]
        if key in self._assigned:
            return self._assigned[key]
        cycle = self._cycle(palette)
        used = set(self._pinned.values()) | set(self._assigned.values())
        free = [c for c in cycle if c not in used]
        colour = free[0] if free else cycle[len(self._assigned) % len(cycle)]
        self._assigned[key] = colour
        return colour

    # -- the project file --------------------------------------------------
    def _autoload(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        import os
        path = os.environ.get("FLUXPLOT_COLORS") or _find_upwards(CATEGORIES_FILENAME)
        if path:
            self.load(path)

    def load(self, path=None) -> str | None:
        """Read a ``fluxplot.colors.json`` (default: the auto-discovered one); returns its path."""
        import os
        path = path or os.environ.get("FLUXPLOT_COLORS") or _find_upwards(CATEGORIES_FILENAME)
        if not path or not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("spec") != "fluxplot/colors":
            raise ValueError(f"{path}: not a fluxplot/colors file")
        self._loaded = True
        self.path = path
        self.assign(doc.get("categories", {}))
        self._palette = doc.get("palette") or None
        return path

    def save(self, path=None) -> str:
        """Write the pinned colours (and the palette) as a ``fluxplot.colors.json``."""
        import os
        path = path or self.path or os.path.join(os.getcwd(), CATEGORIES_FILENAME)
        doc = {"spec": "fluxplot/colors", "version": 1, "categories": dict(sorted(self._pinned.items())),
               "palette": self._palette if isinstance(self._palette, str) else "flexoki"}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
            f.write("\n")
        self.path = path
        return path

    def __repr__(self) -> str:
        return f"Categories({len(self._pinned)} pinned, {len(self._assigned)} assigned, file={self.path!r})"


def _find_upwards(filename: str) -> str | None:
    """``filename`` in the working directory or an ancestor, stopping at the Git root."""
    import os
    here = os.getcwd()
    while True:
        cand = os.path.join(here, filename)
        if os.path.exists(cand):
            return cand
        if os.path.isdir(os.path.join(here, ".git")):
            return None
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


def palette_colors(spec) -> list:
    """The colours of a palette spec: ``"tol.bright"``, ``"brewer.Set2"``, ``"flexoki"`` (the
    light house cycle), ``"flexoki.dark"``, or a bare group name searched in every collection."""
    from . import style as _style
    if spec in ("flexoki", "flexoki.light"):
        return list(_style.CYCLE_LIGHT)
    if spec == "flexoki.dark":
        return list(_style.CYCLE_DARK)
    if "." in spec:
        cid, _, group = spec.partition(".")
        return palettes.get(cid, group)
    for cid in ("tol", "brewer", "flexoki"):
        try:
            return palettes.get(cid, spec)
        except KeyError:
            continue
    raise KeyError(f"No palette {spec!r}; use 'collection.group' (fp.colors.palettes.collections())")


categories = _Categories()


# -------------------------------------------
# Module-level convenience: the flexoki palette at the top level
# -------------------------------------------
def __getattr__(name: str) -> str:
    """``colors.green400`` -> the flexoki hex, without going through ``colors.flex``."""
    try:
        return getattr(flex, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None


def __dir__() -> list[str]:
    return sorted(set(globals()) | {_key_to_attr(k) for k in _flex_data})
