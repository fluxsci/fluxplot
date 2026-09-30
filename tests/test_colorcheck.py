"""Accessibility lint (B3): CVD simulation (Machado 2009), CIEDE2000, WCAG contrast, palette and
colormap findings, and fp.save(lint=...)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import fluxplot as fp  # noqa: E402
from fluxplot import colorcheck as cc  # noqa: E402
from fluxplot import style  # noqa: E402


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_machado_matrices_match_the_reference_implementation():
    colorspacious = pytest.importorskip("colorspacious")
    rgb = np.random.default_rng(0).uniform(size=(24, 3))
    for kind in cc.CVD_KINDS:
        for severity in (1.0, 0.5, 0.73):
            ref = colorspacious.cspace_convert(rgb, {"name": "sRGB1+CVD", "cvd_type": kind, "severity": severity * 100}, "sRGB1")
            assert np.allclose(cc.simulate(rgb, kind, severity), np.clip(ref, 0, 1), atol=1e-9)
    assert np.allclose(cc.machado_matrix("deuteranopia", 0.0), np.eye(3))
    with pytest.raises(ValueError):
        cc.machado_matrix("achromatopsia")


def test_ciede2000_reference_pairs():
    # Sharma, Wu & Dalal (2005) test data
    for l1, l2, expected in [((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
                             ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
                             ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
                             ((50.0, 2.5, 0.0), (50.0, 3.1736, 0.5854), 1.0),
                             ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644)]:
        assert round(float(cc._ciede2000(np.array([l1]), np.array([l2]))[0]), 4) == expected
    assert cc.delta_e("#000000", "#ffffff") == pytest.approx(100.0, abs=0.01)
    assert cc.delta_e("#123456", "#123456") == 0.0


def test_contrast_and_known_confusions():
    assert cc.contrast("#000000", "#ffffff") == pytest.approx(21.0)
    assert cc.contrast("#ffffff", "#000000") == pytest.approx(21.0)
    findings = cc.check_palette(["#ff0000", "#00c000"], "#ffffff", text=["#000000"])
    kinds = {(f.kind, f.a, f.b) for f in findings}
    assert ("cvd-confusable", "#ff0000", "#00c000") in kinds  # red and green collapse for a deuteranope
    assert not any(f.kind == "low-contrast-text" for f in findings)  # black on white passes
    assert cc.check_palette(["#000000"], "#ffffff", text=["#000000"]) == []
    weak = cc.check_palette(["#000000"], "#ffffff", text=["#bbbbbb"])
    assert [f.kind for f in weak] == ["low-contrast-text"] and weak[0].value < 4.5
    grey = cc.check_palette(["#3a7bd5", "#d53a3a"], "#ffffff")
    assert any(f.kind == "greyscale-confusable" for f in grey)


def test_colormap_findings():
    assert cc.check_colormap("viridis") == []
    assert cc.check_colormap("crameri.batlow") == []
    jet = {f.kind for f in cc.check_colormap("jet", sequential=True)}
    assert jet == {"non-uniform", "non-monotone"}
    flexoki = cc.check_colormap("flexoki.diverging")
    assert [f.kind for f in flexoki] == ["non-uniform"] and flexoki[0].value > 0.35


def test_save_lint_records_findings(tmp_path):
    style.use_light()
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [0, 1], series="red", color="#ff0000")
    fp.line(ax, [0, 1], [1, 0], series="green", color="#00c000")
    ax.set_xlabel("x")
    with pytest.warns(UserWarning, match="colour lint"):
        res = fp.save(fig, str(tmp_path / "p.svg"), recipe=False, lint="warn")
    assert any("colour lint" in w for w in res.warnings)
    man = json.loads((tmp_path / "p.fluxplot.json").read_text())
    kinds = {f["kind"] for f in man["quality"]["color"]}
    assert "cvd-confusable" in kinds
    assert all({"kind", "a", "b", "value", "threshold", "message"} <= set(f) for f in man["quality"]["color"])
    with pytest.raises(ValueError, match="colour lint failed"):
        fp.save(fig, str(tmp_path / "e.svg"), recipe=False, lint="error")
    quiet = fp.save(fig, str(tmp_path / "q.svg"), recipe=False)
    assert not any("colour lint" in w for w in quiet.warnings)
    assert "quality" not in json.loads((tmp_path / "q.fluxplot.json").read_text())
    with pytest.raises(ValueError, match="lint must be"):
        fp.save(fig, str(tmp_path / "x.svg"), recipe=False, lint="loud")
