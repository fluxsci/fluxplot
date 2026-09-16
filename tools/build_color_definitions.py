"""Build fluxplot's colour definition files from the upstream packages.

    uv run --with matplotlib --with cmasher --with cmcrameri --with tol-colors \
        python tools/build_color_definitions.py

Writes ``src/fluxplot/definitions/colormaps.json`` and ``palettes.json``: every
colormap collection (matplotlib, Crameri's Scientific colour maps, Paul Tol's maps,
cmasher) and every palette collection (Flexoki, ColorBrewer, Paul Tol's colour
sets) as plain data, so fluxplot — and Flux, which bundles a copy — can list,
draw and apply them WITHOUT importing the source packages at runtime. The
packages are a build-time input only; this script is the only place they are
imported. Continuous maps are stored as 256 samples (enough to rebuild a
LinearSegmentedColormap that is indistinguishable from the original), discrete
maps as their exact colour lists.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib._cm as _mplcm
import numpy as np
from matplotlib.colors import Colormap, ListedColormap, to_hex

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "src" / "fluxplot" / "definitions"
sys.path.insert(0, str(HERE.parent / "src"))
SAMPLES = 256


def hexes(cm: Colormap, n: int = SAMPLES) -> list[str]:
    return [to_hex(c) for c in cm(np.linspace(0.0, 1.0, n))]


def entry(name: str, cm: Colormap, kind: str, family: str | None = None) -> dict:
    discrete = isinstance(cm, ListedColormap) and cm.N <= 128
    colors = [to_hex(c) for c in cm.colors] if discrete else hexes(cm)
    e = {"name": name, "type": kind, "colors": colors}
    if family:
        e["family"] = family
    if discrete:
        e["discrete"] = True
    return e


# --- matplotlib: its documented reference groups --------------------------------
MPL_GROUPS = {
    "sequential": {
        "perceptually uniform": ["viridis", "plasma", "inferno", "magma", "cividis"],
        "brewer": ["Greys", "Purples", "Blues", "Greens", "Oranges", "Reds", "YlOrBr", "YlOrRd", "OrRd",
                    "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn"],
        "classic": ["binary", "gist_yarg", "gist_gray", "gray", "bone", "pink", "spring", "summer", "autumn",
                    "winter", "cool", "Wistia", "hot", "afmhot", "gist_heat", "copper"],
    },
    "diverging": {
        "brewer": ["PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu", "RdYlGn", "Spectral"],
        "classic": ["coolwarm", "bwr", "seismic"],
        "scientific": ["berlin", "managua", "vanimo"],
    },
    "cyclic": {"": ["twilight", "twilight_shifted", "hsv"]},
    "qualitative": {"": ["Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3",
                          "tab10", "tab20", "tab20b", "tab20c"]},
    "misc": {"": ["flag", "prism", "ocean", "gist_earth", "terrain", "gist_stern", "gnuplot", "gnuplot2",
                   "CMRmap", "cubehelix", "brg", "gist_rainbow", "rainbow", "jet", "turbo", "nipy_spectral",
                   "gist_ncar"]},
}


def build_mpl() -> dict:
    maps = []
    for kind, families in MPL_GROUPS.items():
        for family, names in families.items():
            for name in names:
                if name not in matplotlib.colormaps:
                    continue
                maps.append(entry(name, matplotlib.colormaps[name], kind, family or None))
    return {
        "id": "mpl", "name": "matplotlib",
        "description": "matplotlib's built-in colormaps, in its documented groups.",
        "url": "https://matplotlib.org/stable/users/explain/colors/colormaps.html",
        "license": "matplotlib license (BSD-compatible); viridis family CC0",
        "version": matplotlib.__version__, "maps": maps,
    }


# --- Crameri: Scientific colour maps -----------------------------------------------
CRAMERI_DIVERGING = {"broc", "cork", "vik", "lisbon", "tofino", "berlin", "bam", "roma", "vanimo", "managua"}
CRAMERI_MULTI = {"oleron", "bukavu", "fes"}


def build_crameri() -> dict:
    import cmcrameri

    maps = []
    for name in sorted(cmcrameri.cm.cmaps):
        if name.endswith("_r"):
            continue
        cm = cmcrameri.cm.cmaps[name]
        base = name[:-1] if name.endswith(("O", "S")) and name[:-1] else name
        if name.endswith("S"):
            kind, family = "qualitative", "categorical"
        elif name.endswith("O"):
            kind, family = "cyclic", None
        elif base in CRAMERI_DIVERGING:
            kind, family = "diverging", None
        elif base in CRAMERI_MULTI:
            kind, family = "sequential", "multi-sequential"
        else:
            kind, family = "sequential", None
        maps.append(entry(name, cm, kind, family))
    order = {"sequential": 0, "diverging": 1, "cyclic": 2, "qualitative": 3}
    maps.sort(key=lambda m: (order[m["type"]], m["name"].lower()))
    return {
        "id": "crameri", "name": "Crameri",
        "description": "Fabio Crameri's Scientific colour maps — perceptually uniform, colour-vision-deficiency friendly.",
        "url": "https://www.fabiocrameri.ch/colourmaps/",
        "license": "MIT (Crameri, F. (2018), Scientific colour maps, Zenodo, doi:10.5281/zenodo.1243862)",
        "version": cmcrameri.__version__, "maps": maps,
    }


# --- Paul Tol -------------------------------------------------------------------------
TOL_DIVERGING = ["sunset", "nightfall", "BuRd", "PRGn"]
TOL_SEQUENTIAL = ["YlOrBr", "WhOrBr", "iridescent", "incandescent"]
TOL_RAINBOW = ["rainbow_WhBr", "rainbow_WhRd", "rainbow_PuBr", "rainbow_PuRd", "rainbow"]


def build_tol() -> dict:
    import tol_colors as tc

    maps = []
    for name in TOL_DIVERGING:
        maps.append(entry(name, tc.colormaps[name], "diverging"))
        maps.append(entry(name + "_discrete", tc.colormaps[name + "_discrete"], "diverging", "discrete"))
    for name in TOL_SEQUENTIAL:
        maps.append(entry(name, tc.colormaps[name], "sequential"))
        if name + "_discrete" in tc.colormaps:
            maps.append(entry(name + "_discrete", tc.colormaps[name + "_discrete"], "sequential", "discrete"))
    for name in TOL_RAINBOW:
        maps.append(entry(name, tc.colormaps[name], "sequential", "rainbow"))
    maps.append(entry("rainbow_discrete", tc.rainbow_discrete(), "qualitative", "rainbow"))
    return {
        "id": "tol", "name": "Paul Tol",
        "description": "Paul Tol's colour-blind-safe colour schemes: continuous and discrete maps for ordered data.",
        "url": "https://personal.sron.nl/~pault/",
        "license": "BSD-3-Clause (tol-colors)",
        "version": getattr(tc, "__version__", "?"), "maps": maps,
    }


# --- cmasher ---------------------------------------------------------------------------
def build_cmasher() -> dict:
    import cmasher

    maps = []
    for kind in ("sequential", "diverging", "cyclic", "qualitative", "misc"):
        for name, cm in sorted(cmasher.cm.cmap_cd.get(kind, {}).items()):
            if name.endswith("_r"):
                continue
            maps.append(entry(name, cm, kind))
    return {
        "id": "cmasher", "name": "cmasher",
        "description": "cmasher — scientific colormaps for making accessible, informative and 'cmashing' plots.",
        "url": "https://cmasher.readthedocs.io/",
        "license": "BSD-3-Clause",
        "version": cmasher.__version__, "maps": maps,
    }


# --- palettes ---------------------------------------------------------------------------
BREWER_SEQ = ["Blues", "BuGn", "BuPu", "GnBu", "Greens", "Greys", "OrRd", "Oranges", "PuBu", "PuBuGn", "PuRd",
              "Purples", "RdPu", "Reds", "YlGn", "YlGnBu", "YlOrBr", "YlOrRd"]
BREWER_DIV = ["BrBG", "PiYG", "PRGn", "PuOr", "RdBu", "RdGy", "RdYlBu", "RdYlGn", "Spectral"]
BREWER_QUAL = ["Accent", "Dark2", "Paired", "Pastel1", "Pastel2", "Set1", "Set2", "Set3"]
FLEX_HUES = {"k": "base", "r": "red", "o": "orange", "y": "yellow", "g": "green", "c": "cyan", "b": "blue",
             "p": "purple", "m": "magenta"}


def build_palettes() -> dict:
    from fluxplot.colors import _flex_data  # fluxplot's own Flexoki table is the source
    import tol_colors as tc

    groups: dict[str, list[dict]] = {}
    for key, meta in _flex_data.items():
        if key in ("base-0", "base-1000"):
            continue  # aliases of paper / black
        g = FLEX_HUES.get(meta["h"], key.split("-")[0])
        groups.setdefault(g, []).append({"name": key, "hex": meta["hex"].lower()})
    flexoki = {"id": "flexoki", "name": "Flexoki", "description": "Steph Ango's Flexoki — Flux's default palette.",
               "url": "https://stephango.com/flexoki", "license": "MIT",
               "groups": [{"name": g, "type": "sequential" if g != "base" else "neutral", "swatches": sw}
                          for g, sw in groups.items()]}

    def brewer_group(name: str, kind: str) -> dict:
        cm = matplotlib.colormaps[name]
        if isinstance(cm, ListedColormap):
            cols = [to_hex(c) for c in cm.colors]
        else:
            cols = [to_hex(c) for c in getattr(_mplcm, f"_{name}_data")]
        return {"name": name, "type": kind, "classes": len(cols),
                "swatches": [{"name": f"{name}-{i + 1}", "hex": h} for i, h in enumerate(cols)]}

    brewer = {"id": "brewer", "name": "ColorBrewer",
              "description": "Cynthia Brewer's ColorBrewer schemes: sequential (9 classes), diverging (11) and qualitative sets.",
              "url": "https://colorbrewer2.org/", "license": "Apache-2.0 (Brewer, Harrower and Penn State)",
              "groups": [brewer_group(n, "sequential") for n in BREWER_SEQ]
                        + [brewer_group(n, "diverging") for n in BREWER_DIV]
                        + [brewer_group(n, "qualitative") for n in BREWER_QUAL]}

    tol = {"id": "tol", "name": "Paul Tol",
           "description": "Paul Tol's qualitative colour sets, colour-blind safe and distinct in print.",
           "url": "https://personal.sron.nl/~pault/", "license": "BSD-3-Clause (tol-colors)",
           "groups": [{"name": name, "type": "qualitative",
                       "swatches": [{"name": f, "hex": getattr(cs, f).lower()} for f in cs._fields]}
                      for name, cs in tc.colorsets.items()]}
    return {"schema": "fluxplot.palettes/1", "collections": [flexoki, brewer, tol]}


def dump(data: dict) -> str:
    # Readable structure, compact colour arrays: one map per line, never one hex per line.
    text = json.dumps(data, indent=1)
    return re.sub(r'\[\s+("#[0-9a-f]{6}"(?:,\s+"#[0-9a-f]{6}")*)\s+\]',
                  lambda m: "[" + re.sub(r",\s+", ",", m.group(1)) + "]", text) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cmaps = {"schema": "fluxplot.colormaps/1", "samples": SAMPLES,
             "collections": [build_mpl(), build_crameri(), build_tol(), build_cmasher()]}
    (OUT / "colormaps.json").write_text(dump(cmaps))
    pals = build_palettes()
    (OUT / "palettes.json").write_text(dump(pals))
    for c in cmaps["collections"]:
        print(f"colormaps: {c['id']:8} {len(c['maps']):3} maps")
    for c in pals["collections"]:
        print(f"palettes:  {c['id']:8} {len(c['groups']):3} groups")


if __name__ == "__main__":
    main()
