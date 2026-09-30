"""Every example script runs from a clean directory and writes schema-valid, self-consistent
triplets — so ``examples/out`` can always be regenerated and never drifts from the code.

Each script is copied into a scratch directory and run there (they resolve ``out/`` next to
themselves), then every manifest / recipe it wrote is validated against the shipped schemas and
every SVG id the manifest references is checked to exist in the SVG. Marked ``slow``: it runs
five interpreters (~10 s); ``-m "not slow"`` skips it in a quick loop.
"""
import json
import os
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import jsonschema
import pytest
from lxml import etree

from fluxplot.integrity import validate_references

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SCRIPTS = sorted(p for p in EXAMPLES.glob("*.py"))


def _schema(name):
    return json.loads((files("fluxplot") / "schemas" / name).read_text())


@pytest.mark.slow
@pytest.mark.parametrize("script", SCRIPTS, ids=[p.stem for p in SCRIPTS])
def test_example_runs_and_validates(tmp_path, script):
    target = tmp_path / script.name
    shutil.copy(script, target)
    args = [sys.executable, str(target)]
    if script.stem == "scene3d_demo":
        args += ["--out", str(tmp_path / "out")]
    proc = subprocess.run(args, cwd=tmp_path, env=dict(os.environ, MPLBACKEND="Agg"),
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    manifests = sorted((tmp_path / "out").rglob("*.fluxplot.json"))
    assert manifests, "the example wrote nothing"
    mschema, rschema, sschema = _schema("manifest.schema.json"), _schema("recipe.schema.json"), _schema("scene3d.schema.json")
    for mpath in manifests:
        man = json.loads(mpath.read_text())
        stem = mpath.name[: -len(".fluxplot.json")]
        rec = json.loads((mpath.parent / f"{stem}.recipe.json").read_text())
        jsonschema.validate(rec, rschema)
        if man.get("spec") == "fluxplot/manifest":
            jsonschema.validate(man, mschema)
            svg = etree.parse(str(mpath.parent / man["svg"])).getroot()
            ids = set(svg.xpath("//@id"))
            assert len(ids) == len(svg.xpath("//@id")), "duplicate SVG ids"
            validate_references(man, ids)
            assert rec["schemaVersion"] == man["schemaVersion"]
        else:
            jsonschema.validate(man, sschema)
