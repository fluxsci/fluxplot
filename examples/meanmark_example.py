"""Meanmark: paired observations and individually editable mean/error summaries.

Run this script to write SVG + manifest + recipe triplets under out/. Regeneration
controls are exposed through fp.params and honoured through Flux's FLUX_PARAMS.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fluxplot as fp

OUT = Path(__file__).resolve().parent / "out"


def main():
    OUT.mkdir(exist_ok=True)
    fp.use_paper()
    params = fp.params({"mark_size": 4.5, "mark_edge_width": 1.5, "error_width": 1.7,
                        "interval": "ci", "show_caps": True, "show_notch": False,
                        "width_mm": 80., "height_mm": 50., "font_size": 5.5})
    controls = {key: value for key, value in params.items() if not key.startswith("__")
                and key not in ("width_mm", "height_mm", "font_size")}
    rng = np.random.default_rng(12)
    categories = ["Quiet wake", "Active wake", "SWS", "Sigma", "REM"]
    colors = dict(zip(categories, ["#86b94c", "#20953b", "#2166ac", "#7699bf", "#ad2a78"]))
    baseline = rng.normal(0, .018, 14)
    values = np.concatenate([m + baseline + rng.normal(0, .012, 14) for m in [1.07, 1.02, 1.12, 1.065, 1.03]])
    data = {"State": np.repeat(categories, 14), "Coactivity / chance": values,
            "Subject": [f"Mouse {i+1}" for i in range(14)]*len(categories)}
    plt.rcParams.update({key: params["font_size"] for key in
                         ("font.size", "axes.labelsize", "xtick.labelsize", "ytick.labelsize")})
    fig, ax = plt.subplots(figsize=(params["width_mm"]/25.4, params["height_mm"]/25.4),
                           layout="constrained")
    fp.meanmark(data, x="State", y="Coactivity / chance", units="Subject", ax=ax,
                group_color=colors, **controls)
    ax.grid(False)
    ax.axhline(1, color="#777777", linewidth=.8, linestyle="--")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    fp.save(fig, str(OUT / "meanmark.svg"), recipe={"params": params})
    plt.close(fig)

    fp.use_paper()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
    styles = [
        ("Tinted · 95% CI", {}),
        ("Solid · 95% CI + caps", {"mark_fill_color":"stroke", "interval":"ci", "show_caps":True}),
        ("Hollow · mean ± SD", {"mark_fill_color":"none", "interval":"sd"}),
        ("True cutout notches", {"show_notch":True, "mark_size":12, "notch_depth":2.6, "notch_height":5}),
        ("Horizontal", {"orientation":"horizontal", "show_caps":True}),
        ("Dark ground", {}),
    ]
    for ax, (title, options) in zip(axes.flat, styles):
        if title == "Dark ground":
            ax.set_facecolor("#20252b")
        fp.meanmark(data, x="State", y="Coactivity / chance", ax=ax,
                    show_individual_points=False, group_color=colors, **dict(controls, **options))
        ax.grid(False)
        ax.set_title(title, fontsize=11)
        if options.get("orientation") != "horizontal":
            ax.set_xticks(range(5), ["Quiet", "Active", "SWS", "Sigma", "REM"])
        if title != "Dark ground":
            ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)
    fp.save(fig, str(OUT / "meanmark_styles.svg"), recipe={"params":params})
    fig.savefig(OUT / "meanmark_styles.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
