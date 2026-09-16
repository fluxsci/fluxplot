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

__all__ = ["flex", "maps"]


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


_flex_data = {
    # Base (and paper/black)
    "paper": {"h": "k", "l": 0, "hex": "#FFFCF0", "rgb": (255, 252, 240)},
    "base-0": {
        "h": "k",
        "l": 0,
        "hex": "#FFFCF0",
        "rgb": (255, 252, 240),
    },  # equivalent to paper
    "base-50": {"h": "k", "l": 50, "hex": "#F2F0E5", "rgb": (242, 240, 229)},
    "base-100": {"h": "k", "l": 100, "hex": "#E6E4D9", "rgb": (230, 228, 217)},
    "base-150": {"h": "k", "l": 150, "hex": "#DAD8CE", "rgb": (218, 216, 206)},
    "base-200": {"h": "k", "l": 200, "hex": "#CECDC3", "rgb": (206, 205, 195)},
    "base-300": {"h": "k", "l": 300, "hex": "#B7B5AC", "rgb": (183, 181, 172)},
    "base-400": {"h": "k", "l": 400, "hex": "#9F9D96", "rgb": (159, 157, 150)},
    "base-500": {"h": "k", "l": 500, "hex": "#878580", "rgb": (135, 133, 128)},
    "base-600": {"h": "k", "l": 600, "hex": "#6F6E69", "rgb": (111, 110, 105)},
    "base-700": {"h": "k", "l": 700, "hex": "#575653", "rgb": (87, 86, 83)},
    "base-800": {"h": "k", "l": 800, "hex": "#403E3C", "rgb": (64, 62, 60)},
    "base-850": {"h": "k", "l": 850, "hex": "#343331", "rgb": (52, 51, 49)},
    "base-900": {"h": "k", "l": 900, "hex": "#282726", "rgb": (40, 39, 38)},
    "base-950": {"h": "k", "l": 950, "hex": "#1C1B1A", "rgb": (28, 27, 26)},
    "base-1000": {
        "h": "k",
        "l": 1000,
        "hex": "#100F0F",
        "rgb": (16, 15, 15),
    },  # equivalent to black
    "black": {"h": "k", "l": 1000, "hex": "#100F0F", "rgb": (16, 15, 15)},
    # Red
    "red-50": {"h": "r", "l": 50, "hex": "#FFE1D5", "rgb": (255, 225, 213)},
    "red-100": {"h": "r", "l": 100, "hex": "#FFCABB", "rgb": (255, 202, 187)},
    "red-150": {"h": "r", "l": 150, "hex": "#FDB2A2", "rgb": (253, 178, 162)},
    "red-200": {"h": "r", "l": 200, "hex": "#F89A8A", "rgb": (248, 154, 138)},
    "red-300": {"h": "r", "l": 300, "hex": "#E8705F", "rgb": (232, 112, 95)},
    "red-400": {"h": "r", "l": 400, "hex": "#D14D41", "rgb": (209, 77, 65)},
    "red-500": {"h": "r", "l": 500, "hex": "#C03E35", "rgb": (192, 62, 53)},
    "red-600": {"h": "r", "l": 600, "hex": "#AF3029", "rgb": (175, 48, 41)},
    "red-700": {"h": "r", "l": 700, "hex": "#942822", "rgb": (148, 40, 34)},
    "red-800": {"h": "r", "l": 800, "hex": "#6C201C", "rgb": (108, 32, 28)},
    "red-850": {"h": "r", "l": 850, "hex": "#551B18", "rgb": (85, 27, 24)},
    "red-900": {"h": "r", "l": 900, "hex": "#3E1715", "rgb": (62, 23, 21)},
    "red-950": {"h": "r", "l": 950, "hex": "#261312", "rgb": (38, 19, 18)},
    # Orange
    "orange-50": {"h": "o", "l": 50, "hex": "#FFE7CE", "rgb": (255, 231, 206)},
    "orange-100": {"h": "o", "l": 100, "hex": "#FED3AF", "rgb": (254, 211, 175)},
    "orange-150": {"h": "o", "l": 150, "hex": "#FCC192", "rgb": (252, 193, 146)},
    "orange-200": {"h": "o", "l": 200, "hex": "#F9AE77", "rgb": (249, 174, 119)},
    "orange-300": {"h": "o", "l": 300, "hex": "#EC8B49", "rgb": (236, 139, 73)},
    "orange-400": {"h": "o", "l": 400, "hex": "#DA702C", "rgb": (218, 112, 44)},
    "orange-500": {"h": "o", "l": 500, "hex": "#CB6120", "rgb": (203, 97, 32)},
    "orange-600": {"h": "o", "l": 600, "hex": "#BC5215", "rgb": (188, 82, 21)},
    "orange-700": {"h": "o", "l": 700, "hex": "#9D4310", "rgb": (157, 67, 16)},
    "orange-800": {"h": "o", "l": 800, "hex": "#71320D", "rgb": (113, 50, 13)},
    "orange-850": {"h": "o", "l": 850, "hex": "#59290D", "rgb": (89, 41, 13)},
    "orange-900": {"h": "o", "l": 900, "hex": "#40200D", "rgb": (64, 32, 13)},
    "orange-950": {"h": "o", "l": 950, "hex": "#27180E", "rgb": (39, 24, 14)},
    # Yellow
    "yellow-50": {"h": "y", "l": 50, "hex": "#FAEEC6", "rgb": (250, 238, 198)},
    "yellow-100": {"h": "y", "l": 100, "hex": "#F6E2A0", "rgb": (246, 226, 160)},
    "yellow-150": {"h": "y", "l": 150, "hex": "#F1D67E", "rgb": (241, 214, 126)},
    "yellow-200": {"h": "y", "l": 200, "hex": "#ECCB60", "rgb": (236, 203, 96)},
    "yellow-300": {"h": "y", "l": 300, "hex": "#DFB431", "rgb": (223, 180, 49)},
    "yellow-400": {"h": "y", "l": 400, "hex": "#D0A215", "rgb": (208, 162, 21)},
    "yellow-500": {"h": "y", "l": 500, "hex": "#BE9207", "rgb": (190, 146, 7)},
    "yellow-600": {"h": "y", "l": 600, "hex": "#AD8301", "rgb": (173, 131, 1)},
    "yellow-700": {"h": "y", "l": 700, "hex": "#8E6B01", "rgb": (142, 107, 1)},
    "yellow-800": {"h": "y", "l": 800, "hex": "#664D01", "rgb": (102, 77, 1)},
    "yellow-850": {"h": "y", "l": 850, "hex": "#503D02", "rgb": (80, 61, 2)},
    "yellow-900": {"h": "y", "l": 900, "hex": "#3A2D04", "rgb": (58, 45, 4)},
    "yellow-950": {"h": "y", "l": 950, "hex": "#241E08", "rgb": (36, 30, 8)},
    # Olive (original flexoki green)
    "olive-50": {"h": "ol", "l": 50, "hex": "#EDEECF", "rgb": (237, 238, 207)},
    "olive-100": {"h": "ol", "l": 100, "hex": "#DDE2B2", "rgb": (221, 226, 178)},
    "olive-150": {"h": "ol", "l": 150, "hex": "#CDD597", "rgb": (205, 213, 151)},
    "olive-200": {"h": "ol", "l": 200, "hex": "#BEC97E", "rgb": (190, 201, 126)},
    "olive-300": {"h": "ol", "l": 300, "hex": "#A0AF54", "rgb": (160, 175, 84)},
    "olive-400": {"h": "ol", "l": 400, "hex": "#879A39", "rgb": (135, 154, 57)},
    "olive-500": {"h": "ol", "l": 500, "hex": "#768D21", "rgb": (118, 141, 33)},
    "olive-600": {"h": "ol", "l": 600, "hex": "#66800B", "rgb": (102, 128, 11)},
    "olive-700": {"h": "ol", "l": 700, "hex": "#536907", "rgb": (83, 105, 7)},
    "olive-800": {"h": "ol", "l": 800, "hex": "#3D4C07", "rgb": (61, 76, 7)},
    "olive-850": {"h": "ol", "l": 850, "hex": "#313D07", "rgb": (49, 61, 7)},
    "olive-900": {"h": "ol", "l": 900, "hex": "#252D09", "rgb": (37, 45, 9)},
    "olive-950": {"h": "ol", "l": 950, "hex": "#1A1E0C", "rgb": (26, 30, 12)},
    # Green
    "green-50": {"h": "g", "l": 50, "hex": "#A9E3B2", "rgb": (169, 227, 178)},
    "green-100": {"h": "g", "l": 100, "hex": "#95DCA1", "rgb": (149, 220, 161)},
    "green-150": {"h": "g", "l": 150, "hex": "#7FD68D", "rgb": (127, 214, 141)},
    "green-200": {"h": "g", "l": 200, "hex": "#67D379", "rgb": (103, 211, 121)},
    "green-300": {"h": "g", "l": 300, "hex": "#44C55A", "rgb": (68, 197, 90)},
    "green-400": {"h": "g", "l": 400, "hex": "#35AB49", "rgb": (53, 171, 73)},
    "green-500": {"h": "g", "l": 500, "hex": "#2C973E", "rgb": (44, 151, 62)},
    "green-600": {"h": "g", "l": 600, "hex": "#228833", "rgb": (34, 136, 51)},
    "green-700": {"h": "g", "l": 700, "hex": "#1C722A", "rgb": (28, 114, 42)},
    "green-800": {"h": "g", "l": 800, "hex": "#165421", "rgb": (22, 84, 33)},
    "green-850": {"h": "g", "l": 850, "hex": "#13421B", "rgb": (19, 66, 27)},
    "green-900": {"h": "g", "l": 900, "hex": "#113016", "rgb": (17, 48, 22)},
    "green-950": {"h": "g", "l": 950, "hex": "#0F1E11", "rgb": (15, 30, 17)},
    # Cyan
    "cyan-50": {"h": "c", "l": 50, "hex": "#DDF1E4", "rgb": (221, 241, 228)},
    "cyan-100": {"h": "c", "l": 100, "hex": "#BFE8D9", "rgb": (191, 232, 217)},
    "cyan-150": {"h": "c", "l": 150, "hex": "#A2DECE", "rgb": (162, 222, 206)},
    "cyan-200": {"h": "c", "l": 200, "hex": "#87D3C3", "rgb": (135, 211, 195)},
    "cyan-300": {"h": "c", "l": 300, "hex": "#5ABDAC", "rgb": (90, 189, 172)},
    "cyan-400": {"h": "c", "l": 400, "hex": "#3AA99F", "rgb": (58, 169, 159)},
    "cyan-500": {"h": "c", "l": 500, "hex": "#2F968D", "rgb": (47, 150, 141)},
    "cyan-600": {"h": "c", "l": 600, "hex": "#24837B", "rgb": (36, 131, 123)},
    "cyan-700": {"h": "c", "l": 700, "hex": "#1C6C66", "rgb": (28, 108, 102)},
    "cyan-800": {"h": "c", "l": 800, "hex": "#164F4A", "rgb": (22, 79, 74)},
    "cyan-850": {"h": "c", "l": 850, "hex": "#143F3C", "rgb": (20, 63, 60)},
    "cyan-900": {"h": "c", "l": 900, "hex": "#122F2C", "rgb": (18, 47, 44)},
    "cyan-950": {"h": "c", "l": 950, "hex": "#101F1D", "rgb": (16, 31, 29)},
    # Blue
    "blue-50": {"h": "b", "l": 50, "hex": "#E1ECEB", "rgb": (225, 236, 235)},
    "blue-100": {"h": "b", "l": 100, "hex": "#C6DDE8", "rgb": (198, 221, 232)},
    "blue-150": {"h": "b", "l": 150, "hex": "#ABCFE2", "rgb": (171, 207, 226)},
    "blue-200": {"h": "b", "l": 200, "hex": "#92BFDB", "rgb": (146, 191, 219)},
    "blue-300": {"h": "b", "l": 300, "hex": "#66A0C8", "rgb": (102, 160, 200)},
    "blue-400": {"h": "b", "l": 400, "hex": "#4385BE", "rgb": (67, 133, 190)},
    "blue-500": {"h": "b", "l": 500, "hex": "#3171B2", "rgb": (49, 113, 178)},
    "blue-600": {"h": "b", "l": 600, "hex": "#205EA6", "rgb": (32, 94, 166)},
    "blue-700": {"h": "b", "l": 700, "hex": "#1A4F8C", "rgb": (26, 79, 140)},
    "blue-800": {"h": "b", "l": 800, "hex": "#163B66", "rgb": (22, 59, 102)},
    "blue-850": {"h": "b", "l": 850, "hex": "#133051", "rgb": (19, 48, 81)},
    "blue-900": {"h": "b", "l": 900, "hex": "#12253B", "rgb": (18, 37, 59)},
    "blue-950": {"h": "b", "l": 950, "hex": "#101A24", "rgb": (16, 26, 36)},
    # Purple
    "purple-50": {"h": "p", "l": 50, "hex": "#F0EAEC", "rgb": (240, 234, 236)},
    "purple-100": {"h": "p", "l": 100, "hex": "#E2D9E9", "rgb": (226, 217, 233)},
    "purple-150": {"h": "p", "l": 150, "hex": "#D3CAE6", "rgb": (211, 202, 230)},
    "purple-200": {"h": "p", "l": 200, "hex": "#C4B9E0", "rgb": (196, 185, 224)},
    "purple-300": {"h": "p", "l": 300, "hex": "#A699D0", "rgb": (166, 153, 208)},
    "purple-400": {"h": "p", "l": 400, "hex": "#8B7EC8", "rgb": (139, 126, 200)},
    "purple-500": {"h": "p", "l": 500, "hex": "#735EB5", "rgb": (115, 94, 181)},
    "purple-600": {"h": "p", "l": 600, "hex": "#5E409D", "rgb": (94, 64, 157)},
    "purple-700": {"h": "p", "l": 700, "hex": "#4F3685", "rgb": (79, 54, 133)},
    "purple-800": {"h": "p", "l": 800, "hex": "#3C2A62", "rgb": (60, 42, 98)},
    "purple-850": {"h": "p", "l": 850, "hex": "#31234E", "rgb": (49, 35, 78)},
    "purple-900": {"h": "p", "l": 900, "hex": "#261C39", "rgb": (38, 28, 57)},
    "purple-950": {"h": "p", "l": 950, "hex": "#1A1623", "rgb": (26, 22, 35)},
    # Magenta
    "magenta-50": {"h": "m", "l": 50, "hex": "#FEE4E5", "rgb": (254, 228, 229)},
    "magenta-100": {"h": "m", "l": 100, "hex": "#FCCFDA", "rgb": (252, 207, 218)},
    "magenta-150": {"h": "m", "l": 150, "hex": "#F9B9CF", "rgb": (249, 185, 207)},
    "magenta-200": {"h": "m", "l": 200, "hex": "#F4A4C2", "rgb": (244, 164, 194)},
    "magenta-300": {"h": "m", "l": 300, "hex": "#E47DA8", "rgb": (228, 125, 168)},
    "magenta-400": {"h": "m", "l": 400, "hex": "#CE5D97", "rgb": (206, 93, 151)},
    "magenta-500": {"h": "m", "l": 500, "hex": "#B74583", "rgb": (183, 69, 131)},
    "magenta-600": {"h": "m", "l": 600, "hex": "#A02F6F", "rgb": (160, 47, 111)},
    "magenta-700": {"h": "m", "l": 700, "hex": "#87285E", "rgb": (135, 40, 94)},
    "magenta-800": {"h": "m", "l": 800, "hex": "#641F46", "rgb": (100, 31, 70)},
    "magenta-850": {"h": "m", "l": 850, "hex": "#4F1B39", "rgb": (79, 27, 57)},
    "magenta-900": {"h": "m", "l": 900, "hex": "#39172B", "rgb": (57, 23, 43)},
    "magenta-950": {"h": "m", "l": 950, "hex": "#24131D", "rgb": (36, 19, 29)},
}

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


@functools.lru_cache(maxsize=None)
def _definitions(name: str) -> dict:
    with resources.files("fluxplot").joinpath(f"definitions/{name}.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_cmap(full_name: str, m: dict) -> Colormap:
    if m.get("discrete"):
        return ListedColormap(list(m["colors"]), name=full_name)
    return LinearSegmentedColormap.from_list(full_name, list(m["colors"]), N=256)


class _MapRegistry:
    """Colormap access under ``fx.maps``.

    Attribute lookup resolves fluxplot's custom maps first (``maps.flexoki_diverging``,
    and eventually e.g. ``maps.fluxglow``), then falls through to `cmasher
    <https://cmasher.readthedocs.io>`_ (``maps.emerald`` -> ``cmasher.emerald``).
    cmasher is an optional dependency (``pip install "fluxplot[style]"``); the custom
    maps work without it.
    """

    def __init__(self) -> None:
        self._custom: dict[str, Colormap] = {}
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
        return ["flexoki", *self._ensure()]

    def get(self, name: str) -> Colormap:
        """Resolve a colormap by name: ``'crameri.batlow'``, a bare ``'batlow'`` (the
        first collection that has it — matplotlib, Crameri, Tol, cmasher), a fluxplot
        custom map, or any of those with ``_r`` for the reversed map."""
        if name in self._custom:
            return self._custom[name]
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
        """Collection, type (sequential / diverging / cyclic / qualitative / misc),
        family and discreteness of a shipped map (``'crameri.batlow'`` or bare)."""
        cm = self.get(name)
        key = cm.name[:-2] if cm.name.endswith("_r") else cm.name
        return dict(self._info.get(key) or {"collection": "flexoki", "name": key, "type": "custom", "family": None, "discrete": False})

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
        if cmr is None:
            raise AttributeError(
                f"No colormap {name!r}: not a fluxplot custom map, and cmasher is not "
                "installed (pip install 'fluxplot[style]' for the cmasher maps)"
            )
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
        ``plt.imshow(..., cmap=cmap.name)`` works by name)."""
        self._custom[cmap.name] = cmap
        try:
            mpl.colormaps.register(cmap)
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
        if s in ("flexoki", "fluxplot", "flux"):
            return sorted(self._custom.items())
        cols = self._ensure()
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

# Flexoki-flavoured custom maps, built from the canonical palette above. The
# light-centred diverging map (blue–paper–red) is handy for correlation matrices.
# NB: linear ramps through flexoki anchors — pleasant, but not perceptually
# uniform; prefer the cmasher maps when uniformity matters.
maps.register(
    LinearSegmentedColormap.from_list(
        "flexoki_sequential",
        [flex.paper, flex.blue150, flex.blue400, flex.blue600, flex.blue800],
    )
)
maps.register(
    LinearSegmentedColormap.from_list(
        "flexoki_warm",
        [flex.paper, flex.yellow400, flex.orange400, flex.red600, flex.red850],
    )
)
maps.register(
    LinearSegmentedColormap.from_list(
        "flexoki_diverging",
        [flex.blue600, flex.blue400, flex.paper, flex.red400, flex.red600],
    )
)
maps.register(
    LinearSegmentedColormap.from_list(
        "flexoki_terrain",
        [
            flex.blue800,
            flex.blue600,
            flex.cyan400,
            flex.olive400,
            flex.yellow400,
            flex.orange600,
            flex.paper,
        ],
    )
)
maps.register(
    LinearSegmentedColormap.from_list(
        "flexoki_spectrum",
        [
            flex.red600,
            flex.orange600,
            flex.yellow600,
            flex.olive600,
            flex.green600,
            flex.cyan600,
            flex.blue600,
            flex.purple600,
            flex.magenta600,
            flex.red600,
        ],
    )
)


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
