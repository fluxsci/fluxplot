"""``fp.image`` + ``fp.scalebar``: a two-channel micrograph in physical units.

* ``out/image_cells.svg``  a synthetic nuclei (DAPI, blue) + reporter (GFP, green) field,
  0.325 µm pixels, a 10 µm scale bar, one colour key per channel

Each channel is its own colour scale in the manifest (``cells.dapi``, ``cells.gfp``) and its own
recipe control, so an editor can change a LUT or a display range and regenerate.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import fluxplot as fp  # noqa: E402
from fluxplot import style as fx  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "out")


def synthetic_field(rng, shape=(96, 128), n_cells=18):
    """Blurry nuclei with a reporter that only some cells express."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    dapi = rng.normal(40, 6, shape)
    gfp = rng.normal(25, 5, shape)
    for _ in range(n_cells):
        cy, cx, r = rng.uniform(8, h - 8), rng.uniform(8, w - 8), rng.uniform(4, 7)
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r**2))
        dapi += rng.uniform(600, 1000) * blob
        if rng.uniform() < 0.5:
            gfp += rng.uniform(200, 500) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (1.8 * r) ** 2))
    return np.clip(dapi, 0, None), np.clip(gfp, 0, None)


def main():
    fx.use_light()
    os.makedirs(OUT, exist_ok=True)
    recipe = dict(script=os.path.basename(__file__), params={}, inputs=[])
    rng = np.random.default_rng(11)
    dapi, gfp = synthetic_field(rng)

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    im = fp.image(ax, np.stack([dapi, gfp]), series="cells", channels=["dapi", "gfp"],
                  luts=["#3a7bd5", "#4CB391"], pixel_size=0.325, units="µm")
    fp.scalebar(ax, 10, "µm")
    fp.colorbar(im.mappables["dapi"], ax=ax, name="dapi", label="DAPI", shrink=0.6, pad=0.02)
    fp.colorbar(im.mappables["gfp"], ax=ax, name="gfp", label="GFP", shrink=0.6, pad=0.08)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    fp.save(fig, os.path.join(OUT, "image_cells.svg"), recipe=recipe)


if __name__ == "__main__":
    main()
