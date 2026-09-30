"""Generate ``tests/fixtures/colorscale_vectors.json`` — the colour-law parity vectors.

Every case is ``{name, record, values, expected}``: a manifest ``colorScales[]``-shaped record,
input values (``null`` = missing) and the ``#rrggbbaa`` matplotlib paints each one with
(``ScalarMappable.to_rgba``). Python's :func:`fluxplot.colorscale.apply` and Flux's
``colorscale.ts`` both have to reproduce ``expected`` exactly; regenerate when matplotlib is
upgraded and copy the file to Flux (``scripts/fixtures/colorscale_vectors.json``), noting the
fluxplot commit. Run: ``uv run python tests/generate_colorscale_vectors.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib import colors as mc  # noqa: E402

import fluxplot  # noqa: E402,F401  (registers the shipped maps)
from fluxplot import colorscale as cs  # noqa: E402

OUT = Path(__file__).parent / "fixtures" / "colorscale_vectors.json"


def _values(rng, lo, hi, edges):
    v = np.concatenate([rng.uniform(lo, hi, 60), np.asarray(edges, dtype=float), [np.nan]])
    return v


def cases():
    rng = np.random.default_rng(20260929)
    listed = mc.ListedColormap(["#d14d41", "#da702c", "#d0a215", "#879a39", "#3aa99f", "#4385be", "#8b7ec8", "#ce5d97"],
                               name="flexoki8")
    listed.set_under("#100f0f")
    listed.set_over("#fffcf0")
    listed.set_bad("#87858080")
    maps = {"viridis": matplotlib.colormaps["viridis"], "flexoki8": listed,
            "batlow": fluxplot.colors.maps.get("crameri.batlow")}
    norms = {
        "linear": (mc.Normalize(0, 10), (-5, 15), [0, 10, -1e-9, 10.000000001, 5]),
        "linear-clip": (mc.Normalize(0, 10, clip=True), (-5, 15), [0, 10, -3, 30]),
        "log": (mc.LogNorm(1, 1000), (-5, 2000), [1, 1000, 0, -2, 0.5, 31.62]),
        "symlog": (mc.SymLogNorm(2, linscale=1.5, vmin=-100, vmax=100), (-150, 150), [-100, 100, -2, 2, 0, 1.9, -1.9]),
        "power": (mc.PowerNorm(0.5, 0, 10), (-5, 15), [0, 10, -1, 12, 2.5]),
        "twoslope": (mc.TwoSlopeNorm(3, 0, 10), (-5, 15), [0, 3, 10, -1, 11]),
        "centered": (mc.CenteredNorm(2, 5), (-5, 10), [-3, 2, 7, -4, 8]),
        "boundary-neither": (mc.BoundaryNorm([0, 2, 5, 10], 256), (-5, 15), [0, 2, 5, 10, -1, 11, 4.99]),
        "boundary-both": (mc.BoundaryNorm([0, 2, 5, 10], 256, extend="both"), (-5, 15), [0, 2, 5, 10, -1, 11]),
        "boundary-listed": (mc.BoundaryNorm([0, 2, 5, 10], 8, extend="max"), (-5, 15), [0, 2, 5, 10, -1, 11]),
    }
    for nname, (norm, (lo, hi), edges) in norms.items():
        for mname, cmap in maps.items():
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            values = _values(rng, lo, hi, edges)
            record = cs.scale_record(f"{nname}/{mname}", mappable, extend="neither")
            expected = [mc.to_hex(c, keep_alpha=True) for c in mappable.to_rgba(np.ma.masked_invalid(values))]
            assert cs.apply(record, values) == expected, (nname, mname)
            yield {"name": f"{nname}/{mname}", "record": record,
                   "values": [None if np.isnan(v) else float(v) for v in values], "expected": expected}


def generate(path=OUT):
    doc = {"spec": "fluxplot/colorscale-vectors", "version": 1, "matplotlib": matplotlib.__version__,
           "cases": list(cases())}
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


if __name__ == "__main__":
    doc = generate(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
    print(f"{len(doc['cases'])} cases -> {OUT}")
