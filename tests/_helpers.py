import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fluxplot as fp  # noqa: E402


def build_growth_fig():
    """The worked example: control vs treatment, line+points, log y, bracket, reference line."""
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
    return fig
