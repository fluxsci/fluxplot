"""fp.save emits a portable, re-runnable recipe when the producing script is recorded.

The contract is the resolution behaviour of flux-core's runRecipe: it resolves `cwd` and
`output` against the recipe's own directory, then runs `command args` from that cwd. We assert
those invariants hold (rather than exact strings) so the test is robust to where pytest runs.
"""
import json
import os

import fluxplot as fp


def test_recipe_rerun_block_resolves_correctly(tmp_path, growth_fig):
    plots = tmp_path / "plots"
    plots.mkdir()
    script = tmp_path / "make.py"
    res = fp.save(growth_fig, str(plots / "g.svg"), recipe=dict(script=str(script), inputs=[]))
    rec = json.load(open(res.recipe))

    assert rec["command"]  # the interpreter to run
    recipe_dir = os.path.dirname(os.path.abspath(res.recipe))

    # cwd is relative to the recipe dir and resolves back to the dir we ran from
    run_cwd = os.path.normpath(os.path.join(recipe_dir, rec["cwd"]))
    assert run_cwd == os.path.normpath(os.getcwd())

    # args[0] is relative to that run cwd and resolves to the script
    assert os.path.normpath(os.path.join(run_cwd, rec["args"][0])) == os.path.normpath(str(script))

    # output is relative to the recipe dir and resolves to the emitted svg
    assert os.path.normpath(os.path.join(recipe_dir, rec["output"])) == os.path.normpath(os.path.abspath(res.svg))
    assert rec["output"] == "g.svg"  # the common case is just the basename


def test_no_script_means_no_rerun_block(tmp_path, growth_fig):
    res = fp.save(growth_fig, str(tmp_path / "g.svg"))  # no recipe/script recorded
    rec = json.load(open(res.recipe))
    assert "command" not in rec
    assert "args" not in rec


def test_params_merges_flux_params(monkeypatch):
    monkeypatch.setenv("FLUX_PARAMS", '{"test": "mann-whitney"}')
    assert fp.params({"test": "t-test", "n": 3}) == {"test": "mann-whitney", "n": 3}
    monkeypatch.delenv("FLUX_PARAMS")
    assert fp.params({"test": "t-test"}) == {"test": "t-test"}
