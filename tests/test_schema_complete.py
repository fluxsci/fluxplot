"""The manifest and recipe schemas declare EVERY key fluxplot emits (F3).

The shipped schemas are additive-permissive so old consumers never break; this test validates one
figure of every plot type against a *strict* copy (``additionalProperties: false`` injected into
every object that lists properties), which proves that nothing fluxplot writes is undeclared —
the drift that let Flux's hand-written types fall behind the Python output.
"""
import copy
import json
import os
import sys
from importlib.resources import files

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jsonschema import Draft7Validator  # noqa: E402
from matplotlib import colors as mc  # noqa: E402

import fluxplot as fp  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from test_surface import _sphere  # noqa: E402
from test_glowbar import _table  # noqa: E402


def strict(schema):
    """A copy of ``schema`` in which every object that lists properties forbids others."""
    out = copy.deepcopy(schema)

    def walk(node, conditional=False):
        if isinstance(node, dict):
            if "properties" in node and not conditional and "additionalProperties" not in node:
                node["additionalProperties"] = False
            for key, child in node.items():
                walk(child, conditional or key in ("if", "then", "else"))
        elif isinstance(node, list):
            for child in node:
                walk(child, conditional)

    walk(out)
    return out


def _schema(name):
    return json.loads((files("fluxplot") / "schemas" / name).read_text())


def _errors(validator, doc):
    return sorted(f"{'/'.join(str(p) for p in e.absolute_path)}: {e.message[:140]}" for e in validator.iter_errors(doc))


# ---------------------------------------------------------------------------------------------
# one figure per plot type
# ---------------------------------------------------------------------------------------------
def _basic():
    fig, ax = plt.subplots()
    t = [0, 4, 8, 12]
    ln, pts = fp.line(ax, t, [1, 2, 3, 5], series="control", marker="o", label="Control")
    fp.line(ax, t, [1, 3, 4, 6], series="treatment", label="Treatment", linestyle="--")
    fp.scatter(ax, [1, 5, 9], [4, 4, 4], series="dots", label="Dots")
    fp.errorbar(ax, [2, 6], [2, 3], yerr=[0.2, 0.3], series="err", fmt="s")
    fp.area(ax, t, [0.5, 1, 1.5, 2], series="band", alpha=0.3)
    fp.bar(ax, [10, 11], [1, 2], series="counts", label="Counts")
    ax.set_xlabel("Time (h)"); ax.set_ylabel("OD"); ax.set_title("Growth"); ax.grid(True)
    ax.legend()
    fp.significance_bracket(ax, x0=0, x1=4, y=6.2, label="**", between=("control", "treatment"), p=0.003)
    fp.reference_line(ax, y=1.0, name="threshold", color="0.6")
    fp.annotation(ax, name="peak", text="peak", xy=(12, 5), xytext=(10, 5.5))
    ax.text(6, 0.2, "free text")
    ax.plot([0, 12], [0.1, 0.1], color="0.8")  # a raw extra
    return fig


def _distributions():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3)
    fp.box(axes[0], rng.normal(size=30), series="b", showmeans=True)
    fp.violin(axes[1], rng.normal(size=30), series="v", showmedians=True)
    fp.hist(axes[2], rng.normal(size=100), series="h", bins=8, include_values=True)
    fp.barh(axes[2], ["x", "y"], [3, 4], series="hb", left=1)
    return fig


def _fields():
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.6))
    fp.panel(axes[0], "matrix"); fp.panel(axes[1], "contours"); fp.panel(axes[2], "image")
    m = fp.heatmap(axes[0], [[1, .2, .3], [.2, 1, np.nan], [.3, .6, 1]], series="corr", cells=True, include_values=True,
                   x=[0, 1, 2, 3], y=[0, 1, 2, 3], vmin=0, vmax=1)
    fp.colorbar(m, name="corr", label="Correlation", extend="both")
    x, y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
    c = fp.contourf(axes[1], x, y, x * x + y * y, series="energy", levels=[0, 1, 3, 5, 8], include_values=True)
    fp.contour(axes[1], x, y, x * x + y * y, series="iso", levels=[1, 3])
    fp.colorbar(c, name="energy", label="Energy")
    im = fp.heatmap(axes[2], np.arange(12.0).reshape(3, 4), series="img", norm=mc.LogNorm(1, 11), value_raster=True)
    fp.colorbar(im, name="img")
    axes[2].imshow(np.eye(3))  # a raw mappable → anonymous scale
    return fig


def _scatter_c():
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots()
    pts = fp.scatter(ax, rng.normal(size=20), rng.normal(size=20), c=rng.uniform(size=20), s=rng.uniform(5, 50, 20),
                     series="pts", label="Points")
    fp.colorbar(pts)
    ax.legend()
    return fig


def _hexmatrix():
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(figsize=(3, 3))
    fp.hexmatrix(x=rng.normal(size=300), y=rng.normal(size=300), ax=ax, gridsize=6, series="h", norm="log",
                 marginals=True, identity_line=True, sparse=2)
    return fig


def _hexmatrix_matrix():
    fig, ax = plt.subplots()
    fp.hexmatrix(matrix=np.arange(12.0).reshape(3, 4), ax=ax, series="som", cmap="RdBu_r", center=5)
    return fig


def _glowbar():
    fig, ax = plt.subplots()
    fp.glowbar(_table(), x="type", y="APP", units="subject", ax=ax, connect_identical_points_across_x_values=False)
    return fig


def _fluxbox():
    fig, ax = plt.subplots()
    fp.fluxbox(_table(), x="type", y="APP", units="subject", ax=ax, show_caps=True)
    return fig


def _surface():
    mesh = {"left": _sphere(x_offset=-1.2), "right": _sphere(x_offset=+1.2)}
    n = mesh["left"][0].shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(6, 2))
    labels = {"left": np.repeat([0, 1, 2], [n // 3, n // 3, n - 2 * (n // 3)]).astype(float),
              "right": np.repeat([0, 1, 2], [n // 3, n // 3, n - 2 * (n // 3)]).astype(float)}
    fp.surface(axes[0], labels, series="atlas", surfaces=mesh, kind="label",
               categories={0: "frontal", 1: "parietal", 2: "temporal"}, legend=True)
    vals = {"left": np.linspace(0, 100, n), "right": np.linspace(0, 100, n)}
    fp.surface(axes[1], vals, series="field", surfaces=mesh, kind="continuous", colorbar=True, cbar_label="units")
    return fig


def _seaborn():
    import pandas as pd
    import seaborn as sns
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"g": rng.choice(list("abc"), 60), "v": rng.normal(size=60), "w": rng.normal(size=60)})
    fig, axes = plt.subplots(1, 2)
    sns.barplot(data=df, x="g", y="v", errorbar="sd", capsize=0.3, ax=axes[0])
    fp.tag_seaborn(axes[0], plot="barplot")
    sns.heatmap(pd.DataFrame(np.arange(6.0).reshape(2, 3)), ax=axes[1])
    fp.tag_seaborn(axes[1], plot="heatmap")
    return fig


def _polar_and_dates():
    from datetime import datetime, timedelta
    fig = plt.figure(figsize=(6, 3))
    polar = fig.add_subplot(1, 2, 1, projection="polar")
    th = np.linspace(0, 2 * np.pi, 30)
    fp.line(polar, th, 1 + 0.3 * np.sin(3 * th), series="orbit")
    ax = fig.add_subplot(1, 2, 2)
    fp.line(ax, [datetime(2020, 1, 1) + timedelta(days=i) for i in range(3)], [1, 2, 3], series="dated")
    ax.set_yscale("log")
    return fig


BUILDERS = {
    "basic": _basic, "distributions": _distributions, "fields": _fields, "scatter_c": _scatter_c,
    "hexmatrix": _hexmatrix, "hexmatrix_matrix": _hexmatrix_matrix, "glowbar": _glowbar, "fluxbox": _fluxbox,
    "surface": _surface, "seaborn": _seaborn, "polar_and_dates": _polar_and_dates,
}


@pytest.fixture(scope="module")
def validators():
    return (Draft7Validator(strict(_schema("manifest.schema.json"))),
            Draft7Validator(strict(_schema("recipe.schema.json"))))


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_emitted_key_is_declared(tmp_path, validators, name):
    manifest_validator, recipe_validator = validators
    fig = BUILDERS[name]()
    res = fp.save(fig, str(tmp_path / f"{name}.svg"), recipe=dict(script="make.py", params={"n": 3}, inputs=[]))
    plt.close(fig)
    man = json.loads(open(res.manifest).read())
    rec = json.loads(open(res.recipe).read())
    errors = _errors(manifest_validator, man) + [f"recipe {e}" for e in _errors(recipe_validator, rec)]
    assert not errors, "\n".join(errors)


def test_schema_ids_carry_the_spec_version():
    man, rec = _schema("manifest.schema.json"), _schema("recipe.schema.json")
    assert man["$id"].endswith(f"fluxplot-manifest-{fp.SPEC_VERSION}.json")
    assert rec["$id"].endswith(f"fluxplot-recipe-{fp.SPEC_VERSION}.json")
    assert "$defs" not in man and "$defs" not in rec  # draft-07 spells it "definitions"
