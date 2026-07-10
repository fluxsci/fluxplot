"""Export consistency (plan §5): svgSha256 commit marker + staged, per-file-atomic writes."""
import hashlib
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import fluxplot as fp  # noqa: E402
from _helpers import build_growth_fig  # noqa: E402


def test_manifest_records_final_svg_sha256(tmp_path, growth_fig):
    res = fp.save(growth_fig, str(tmp_path / "g.svg"))
    man = json.load(open(res.manifest))
    actual = hashlib.sha256(open(res.svg, "rb").read()).hexdigest()
    assert man["artifact"]["svgSha256"] == actual


def test_no_staging_files_left_behind(tmp_path, growth_fig):
    fp.save(growth_fig, str(tmp_path / "g.svg"))
    leftovers = [p for p in os.listdir(tmp_path) if ".staging" in p]
    assert leftovers == []
    assert sorted(os.listdir(tmp_path)) == ["g.fluxplot.json", "g.recipe.json", "g.svg"]


def test_failed_commit_preserves_previous_triplet(tmp_path, monkeypatch):
    fig = build_growth_fig()
    res = fp.save(fig, str(tmp_path / "g.svg"))
    plt.close(fig)
    before = {p: open(os.path.join(tmp_path, p), "rb").read() for p in os.listdir(tmp_path)}

    real_replace = os.replace

    def failing_replace(src, dst):
        if str(dst).startswith(str(tmp_path)):
            raise OSError("disk full (simulated)")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)
    fig = build_growth_fig()
    with pytest.raises(OSError, match="disk full"):
        fp.save(fig, str(tmp_path / "g.svg"))
    plt.close(fig)
    monkeypatch.undo()

    after = {p: open(os.path.join(tmp_path, p), "rb").read() for p in os.listdir(tmp_path)}
    assert after == before  # earlier successful triplet intact, no partial replacement, no litter


def test_existing_permissions_preserved(tmp_path, growth_fig):
    res = fp.save(growth_fig, str(tmp_path / "g.svg"))
    os.chmod(res.svg, 0o600)
    fig = build_growth_fig()
    fp.save(fig, str(tmp_path / "g.svg"))
    plt.close(fig)
    assert os.stat(res.svg).st_mode & 0o777 == 0o600
