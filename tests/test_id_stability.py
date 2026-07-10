"""ID stability (plan §7): slug collisions fail actionably; legend linkage is by exact label.

The allocator's deterministic -2 suffix stays as a safety net, but it is not durable identity:
explicit series that would collide now raise at the call site, and legend entries link to
series by exact unique label text, never by position.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import fluxplot as fp  # noqa: E402


def test_colliding_series_names_raise_at_registration(tmp_path):
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [0, 1], series="A B")
    with pytest.raises(ValueError, match="collides with 'A B'"):
        fp.line(ax, [0, 1], [1, 0], series="A_B")  # same slug a-b
    plt.close(fig)


def test_same_series_name_repeats_freely(tmp_path):
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [0, 1], series="control")
    fp.errorbar(ax, [0, 1], [0, 1], yerr=[0.1, 0.1], series="control")  # same identity: fine
    res = fp.save(fig, str(tmp_path / "s.svg"))
    plt.close(fig)
    man = json.load(open(res.manifest))
    assert len([s for s in man["series"] if s["id"] == "control"]) == 1


def test_unrelated_additions_do_not_rename_explicit_ids(tmp_path):
    def series_ids(build_extra):
        fig, ax = plt.subplots()
        fp.line(ax, [0, 1, 2], [1, 2, 3], series="alpha", marker="o", label="Alpha")
        if build_extra:
            fp.line(ax, [0, 1, 2], [2, 3, 4], series="beta", label="Beta")
            fp.reference_line(ax, y=1.0, name="threshold")
            ax.legend()
        res = fp.save(fig, str(tmp_path / f"u{int(build_extra)}.svg"))
        plt.close(fig)
        man = json.load(open(res.manifest))
        alpha = next(s for s in man["series"] if s["id"] == "alpha")
        return alpha["svg"], [p["svgId"] for p in alpha.get("points", [])]

    svg_a, pts_a = series_ids(False)
    svg_b, pts_b = series_ids(True)
    assert svg_a == svg_b and pts_a == pts_b  # adding series/legend/overlays renames nothing


def test_legend_entries_link_by_exact_label_not_position(tmp_path):
    fig, ax = plt.subplots()
    c = fp.line(ax, [0, 1], [0, 1], series="control", label="Control")
    t = fp.line(ax, [0, 1], [1, 0], series="treatment", label="Treatment")
    ax.legend(handles=[t, c])  # display order deliberately reversed vs creation order
    res = fp.save(fig, str(tmp_path / "l.svg"))
    plt.close(fig)
    man = json.load(open(res.manifest))
    legend = next(g for g in man["guides"] if g["role"] == "legend")
    assert [e["series"] for e in legend["entries"]] == ["treatment", "control"]


def test_legend_entry_without_matching_series_omits_linkage(tmp_path):
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [0, 1], series="control", label="Control")
    fp.reference_line(ax, y=0.5, name="limit", label="Limit")  # overlay, not a series
    ax.legend()
    res = fp.save(fig, str(tmp_path / "o.svg"))
    plt.close(fig)
    man = json.load(open(res.manifest))
    legend = next(g for g in man["guides"] if g["role"] == "legend")
    limit = next(e for e in legend["entries"] if e.get("text") == "Limit")
    assert "series" not in limit  # swatch/label stay addressable, no invented series claim
    assert limit.get("swatch") or limit.get("label")


def test_duplicate_labels_link_no_series(tmp_path):
    fig, ax = plt.subplots()
    fp.line(ax, [0, 1], [0, 1], series="run-1", label="Same")
    fp.line(ax, [0, 1], [1, 0], series="run-2", label="Same")
    ax.legend()
    res = fp.save(fig, str(tmp_path / "d.svg"))
    plt.close(fig)
    man = json.load(open(res.manifest))
    legend = next(g for g in man["guides"] if g["role"] == "legend")
    assert all("series" not in e for e in legend["entries"])  # ambiguous → no positional guess
