"""The worked example (spec §12): control vs treatment growth over 24 h.

One source → many lives. Run it to produce ``out/growth.svg`` + ``out/growth.fluxplot.json`` +
``out/growth.recipe.json``, which the Flux Figure GUI consumes.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fluxplot as fp  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "out")


def main():
    t = [0, 4, 8, 12, 16, 20, 24]
    control = [0.02, 0.05, 0.13, 0.41, 0.95, 1.6, 1.9]
    treatment = [0.02, 0.07, 0.25, 0.80, 1.5, 1.95, 2.1]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fp.line(ax, t, control, series="control", marker="o", label="Control")
    fp.line(ax, t, treatment, series="treatment", marker="s", label="Treatment")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("OD600")
    ax.set_yscale("log")
    ax.legend()

    fp.significance_bracket(
        ax, x0=20, x1=24, y=2.0, label="**", between=("control", "treatment"), p=0.003
    )
    fp.reference_line(ax, y=1.0, name="threshold", color="0.6", linestyle=":")

    os.makedirs(OUT, exist_ok=True)
    res = fp.save(
        fig,
        os.path.join(OUT, "growth.svg"),
        recipe=dict(script=os.path.basename(__file__), params={"test": "t-test", "smooth": False}, inputs=[]),
    )
    print("wrote:")
    print(" ", res.svg)
    print(" ", res.manifest)
    print(" ", res.recipe)
    if res.warnings:
        print("warnings:", res.warnings)


if __name__ == "__main__":
    main()
