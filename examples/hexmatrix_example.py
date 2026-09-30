"""The hexmatrix — hexagonal binning with every hexagon a named part — on its five jobs.

* ``out/hexmatrix_joint.svg``    a skewed cloud with marginal histograms, one-colour ramp
* ``out/hexmatrix_rates.svg``    a log-log density with a log colour key and the identity line
* ``out/hexmatrix_spatial.svg``  a spatial map in true data units (0.15 mm hexagons)
* ``out/hexmatrix_gradient.svg`` the mean of a third variable per hexagon, on a diverging map
* ``out/hexmatrix_lattice.svg``  a 2D array drawn on a hex lattice (a SOM component plane)

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
    rng = np.random.default_rng(7)

    # a skewed cloud, seaborn's hexbin_marginals
    x = rng.gamma(2.0, 0.7, 1000)
    table = {"x": x, "y": -0.5 * x + rng.normal(0, 1, 1000)}
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    fp.hexmatrix(table, x="x", y="y", ax=ax, gridsize=22, color="#4CB391", marginals=True,
                 colorbar=False, series="cloud")
    fp.save(fig, os.path.join(OUT, "hexmatrix_joint.svg"), recipe=recipe)

    # event rates in two states, log-log, one hexagon per ~0.1 decade
    wake = 10 ** rng.normal(-0.7, 0.5, 3000)
    rates = {"Wake event rate (Hz)": wake, "NREM event rate (Hz)": wake * 10 ** rng.normal(0, 0.15, 3000)}
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    fp.hexmatrix(rates, x="Wake event rate (Hz)", y="NREM event rate (Hz)", ax=ax, xscale="log",
                 yscale="log", norm="log", gridsize=35, cmap="viridis", identity_line=True,
                 colorbar_label="Synapses per hexbin", series="rates")
    fp.save(fig, os.path.join(OUT, "hexmatrix_rates.svg"), recipe=recipe)

    # somata on a coronal section: equal aspect, 0.15 mm bins
    n = 30000
    t, r = rng.uniform(0, 2 * np.pi, n), np.sqrt(rng.uniform(0, 1, n))
    X, Z = 6 + 4 * r * np.cos(t), 6 + 4.5 * r * np.sin(t)
    keep = np.abs(Z - 5.8) > 0.15
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    fp.hexmatrix({"CCF X (mm)": X[keep], "CCF Z (mm)": Z[keep]}, x="CCF X (mm)", y="CCF Z (mm)",
                 ax=ax, aspect="equal", binwidth=0.15, norm="log", robust=True,
                 colorbar_label="somata / hexbin", series="somata")
    fp.save(fig, os.path.join(OUT, "hexmatrix_spatial.svg"), recipe=recipe)

    # the mean of a third variable per hexagon
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    fp.hexmatrix({**table, "signal": np.sin(x) + table["y"]}, x="x", y="y", C="signal", ax=ax,
                 gridsize=18, cmap="RdBu_r", center=0, series="signal")
    fp.save(fig, os.path.join(OUT, "hexmatrix_gradient.svg"), recipe=recipe)

    # a 2D array on a hex lattice
    yy, xx = np.mgrid[0:12, 0:16]
    plane = np.sin(xx / 3.0) * np.cos(yy / 4.0) + rng.normal(0, 0.1, xx.shape)
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    fp.hexmatrix(matrix=plane, ax=ax, cmap="emerald", gap=0.08, colorbar_label="weight",
                 series="som")
    fp.save(fig, os.path.join(OUT, "hexmatrix_lattice.svg"), recipe=recipe)


if __name__ == "__main__":
    main()
