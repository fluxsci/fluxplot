"""The glowbar — FluxPlot's first signature plot — on two small designs.

* ``out/glowbar.svg``        two independent groups, one dot per animal (fixed lane + colour per animal)
* ``out/glowbar_paired.svg`` the same animals measured twice, joined across conditions

Each writes the usual triplet (``.svg`` + ``.fluxplot.json`` + ``.recipe.json``).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import fluxplot as fp  # noqa: E402
from fluxplot import style as fx  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "out")


def main():
    fx.use_light()
    os.makedirs(OUT, exist_ok=True)
    recipe = dict(script=os.path.basename(__file__), params={}, inputs=[])

    # two independent groups: sleep (S) vs sleep deprivation (SD), six animals each
    table = {
        "subject": [f"B6_{i}" for i in range(1, 13)],
        "condition": ["S", "SD"] * 6,
        "APP/GAPDH": [1.47, 2.77, 2.38, 2.88, 1.90, 2.08, 2.30, 3.47, 1.65, 2.80, 1.82, 3.33],
    }
    fig, ax = plt.subplots(figsize=(1.6, 1.8))
    gb = fp.glowbar(table, x="condition", y="APP/GAPDH", units="subject", ax=ax)
    fp.save(fig, os.path.join(OUT, "glowbar.svg"), recipe=recipe)
    print({c: round(s["median"], 3) for c, s in gb.stats.items()})

    # paired: the same eight mice before and after treatment, as % of control
    rng = np.random.default_rng(7)
    before = 100 + rng.normal(0, 3, 8)
    after = before - rng.normal(21, 9, 8)
    paired = {
        "mouse": [f"m{i}" for i in range(1, 9)] * 2,
        "condition": ["C"] * 8 + ["T"] * 8,
        "APP (% of C)": np.r_[before, after],
    }
    fig, ax = plt.subplots(figsize=(1.6, 1.8))
    fp.glowbar(paired, x="condition", y="APP (% of C)", units="mouse", ax=ax,
               connect_identical_points_across_x_values=True)
    fp.save(fig, os.path.join(OUT, "glowbar_paired.svg"), recipe=recipe)


if __name__ == "__main__":
    main()
