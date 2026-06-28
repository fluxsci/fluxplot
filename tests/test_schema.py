"""Generated artifacts validate against the shipped schemas + are internally consistent."""
import json
import re
from importlib.resources import files

import jsonschema
import matplotlib.pyplot as plt

import fluxplot as fp


def _schemas():
    sch = files("fluxplot.schemas")
    return (
        json.loads((sch / "manifest.schema.json").read_text()),
        json.loads((sch / "recipe.schema.json").read_text()),
    )


def test_outputs_validate_against_schemas(tmp_path, growth_fig):
    res = fp.save(growth_fig, str(tmp_path / "g.svg"), recipe=dict(script="g.py", inputs=[]))
    man = json.load(open(res.manifest))
    rec = json.load(open(res.recipe))
    mschema, rschema = _schemas()
    jsonschema.validate(man, mschema)
    jsonschema.validate(rec, rschema)


def test_every_manifest_id_exists_in_svg(tmp_path, growth_fig):
    res = fp.save(growth_fig, str(tmp_path / "g.svg"))
    man = json.load(open(res.manifest))
    svg_ids = set(re.findall(r'id="([^"]+)"', open(res.svg).read()))
    for s in man["series"]:
        for v in s["svg"].values():
            for ref in (v if isinstance(v, list) else [v]):
                assert ref in svg_ids, f"series ref {ref} missing from svg"
        for p in s.get("points", []):
            assert p["svgId"] in svg_ids, f"point {p['svgId']} missing from svg"
    for o in man["overlays"]:
        assert o["svgId"] in svg_ids, f"overlay {o['svgId']} missing from svg"


def test_unknown_role_degrades_to_x_namespace(tmp_path):
    fig, ax = plt.subplots()
    (ln,) = ax.plot([0, 1], [0, 1])
    fp.tag(ln, role="weird-geom", series="s")
    res = fp.save(fig, str(tmp_path / "g.svg"))
    plt.close(fig)
    assert 'data-role="x-weird-geom"' in open(res.svg).read()
