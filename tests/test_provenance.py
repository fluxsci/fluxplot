"""Automatic recipe provenance (plan §1): honest, zero-ceremony rerunnable recipes.

The end-to-end test executes a real script in a temp directory — plain ``fp.save(fig, path)``
must yield a rerun block that resolves exactly the way flux-core's runRecipe resolves it
(cwd/output against the recipe's own directory, then ``command args`` from that cwd).
"""
import json
import os
import subprocess
import sys
import textwrap

import fluxplot as fp
from fluxplot import provenance

SCRIPT = textwrap.dedent(
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fluxplot as fp

    fig, ax = plt.subplots(figsize=(4, 3))
    fp.line(ax, [0, 1, 2], [1, 2, 4], series="control", label="Control")
    fp.save(fig, "plots/growth.svg")
    """
)


def _run_script(tmp_path):
    script = tmp_path / "make_growth.py"
    script.write_text(SCRIPT)
    (tmp_path / "plots").mkdir()
    env = dict(os.environ, MPLBACKEND="Agg")
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return script, json.load(open(tmp_path / "plots" / "growth.recipe.json"))


def test_plain_save_in_real_script_is_rerunnable(tmp_path):
    script, rec = _run_script(tmp_path)
    recipe_dir = str(tmp_path / "plots")

    # resolve exactly as flux-core runRecipe does: cwd/output vs the recipe dir, args vs cwd
    run_cwd = os.path.normpath(os.path.join(recipe_dir, rec["cwd"]))
    assert run_cwd == str(tmp_path)
    assert os.path.normpath(os.path.join(run_cwd, rec["args"][0])) == str(script)
    assert os.path.normpath(os.path.join(recipe_dir, rec["output"])) == str(
        tmp_path / "plots" / "growth.svg"
    )
    assert rec["command"]  # the interpreter

    prov = rec["provenance"]
    assert prov["scriptDiscovery"] == "automatic"
    assert len(prov["scriptSha256"]) == 64
    assert rec["inputs"] == []  # never guessed
    assert prov["packages"]["fluxplot"] == fp.__version__


def test_rerun_reproduces_the_output(tmp_path):
    """The recorded command/cwd re-executes and overwrites the recorded output in place."""
    script, rec = _run_script(tmp_path)
    recipe_dir = str(tmp_path / "plots")
    out_path = os.path.join(recipe_dir, rec["output"])
    before = open(out_path, "rb").read()
    os.remove(out_path)

    run_cwd = os.path.normpath(os.path.join(recipe_dir, rec["cwd"]))
    proc = subprocess.run(
        [rec["command"], *rec["args"]], cwd=run_cwd,
        env=dict(os.environ, MPLBACKEND="Agg"), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert open(out_path, "rb").read() == before  # deterministic regeneration


def test_explicit_fields_override_inferred(tmp_path, growth_fig):
    res = fp.save(growth_fig, str(tmp_path / "g.svg"), recipe=dict(script="mine.py"))
    rec = json.load(open(res.recipe))
    assert rec["script"]["path"] == "mine.py"
    assert rec["provenance"]["scriptDiscovery"] == "explicit"


def test_interactive_caller_writes_valid_nonrerunnable_recipe(tmp_path):
    """python -c (no file anywhere on the stack) → honest 'unavailable', no rerun block."""
    code = SCRIPT.replace('fp.save(fig, "plots/growth.svg")', 'fp.save(fig, "growth.svg")')
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        env=dict(os.environ, MPLBACKEND="Agg"), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    rec = json.load(open(tmp_path / "growth.recipe.json"))
    assert rec["script"] is None
    assert "command" not in rec and "args" not in rec
    assert rec["provenance"]["scriptDiscovery"] == "unavailable"


def test_discovery_rejects_pseudo_and_installed_files(tmp_path):
    assert not provenance._is_real_script("<stdin>")
    assert not provenance._is_real_script("<ipython-input-3-abc>")
    assert not provenance._is_real_script(str(tmp_path / "missing.py"))
    cell_dir = tmp_path / "ipykernel_1234"
    cell_dir.mkdir()
    cell = cell_dir / "abcdef.py"
    cell.write_text("pass")
    assert not provenance._is_real_script(str(cell))  # notebook cell temp file
    assert provenance._in_installed_packages("/x/.venv/lib/python3.13/site-packages/pytest/__main__.py")


def test_git_probe_fails_closed_outside_repo(tmp_path):
    assert provenance._git_info(str(tmp_path)) is None


def test_provenance_git_block_in_repo(tmp_path):
    if not _git_available():
        import pytest

        pytest.skip("git not installed")
    subprocess.run(["git", "-C", tmp_path, "init", "-q"], check=True)
    script = tmp_path / "s.py"
    script.write_text("pass")
    subprocess.run(["git", "-C", tmp_path, "add", "s.py"], check=True)
    subprocess.run(
        ["git", "-C", tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"],
        check=True,
    )
    prov = provenance.build_provenance(str(script), "automatic")
    assert prov["git"]["commit"] and prov["git"]["dirty"] is False
    script.write_text("changed")
    assert provenance.build_provenance(str(script), "automatic")["git"]["dirty"] is True


def _git_available() -> bool:
    try:
        return subprocess.run(["git", "--version"], capture_output=True).returncode == 0
    except OSError:
        return False
