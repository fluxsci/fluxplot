import hashlib
import json
from pathlib import Path
import struct
import jsonschema
import pytest
from generate_scene3d_fixtures import generate, ROOT

SCHEMA = json.loads((Path(__file__).parents[1]/'src/fluxplot/schemas/scene3d.schema.json').read_text())


def test_contract_minimal_and_strict_optional_fields():
    minimal={'spec':'fluxplot/scene3d','schemaVersion':'0.1.0','glb':'mesh.glb'}
    jsonschema.validate(minimal, SCHEMA)
    for patch in ({'view':{'zoom':0}}, {'size':{'width':0,'height':1,'unit':'in'}}, {'view':{'elevation':91}}, {'toWorld':[1]}, {'states':[{'label':'Missing name'}]}, {'parts':[{'id':'Bad ID','role':'mesh'}]}):
        with pytest.raises(jsonschema.ValidationError): jsonschema.validate({**minimal,**patch},SCHEMA)


def test_fixtures_schema_receipts_and_glb_structure():
    for p in ROOT.glob('*.fluxplot.json'):
        m=json.loads(p.read_text()); jsonschema.validate(m,SCHEMA)
        b=(ROOT/m['glb']).read_bytes()
        assert hashlib.sha256(b).hexdigest()==m['glbSha256']
        magic,version,size=struct.unpack_from('<4sII',b)
        assert (magic,version,size)==(b'glTF',2,len(b))
        n,kind=struct.unpack_from('<I4s',b,12); assert kind==b'JSON' and n%4==0
        doc=json.loads(b[20:20+n]); length,kind=struct.unpack_from('<I4s',b,20+n)
        assert kind==b'BIN\0' and length%4==0 and 28+n+length==len(b)
        for mesh in doc['meshes']:
            for primitive in mesh['primitives']:
                assert {'POSITION','NORMAL'} <= primitive['attributes'].keys()
                assert primitive['mode']==4
                assert len(primitive.get('targets',[]))==len(mesh.get('extras',{}).get('targetNames',[]))
        assert {p['node'] for p in m['parts'] if 'node' in p}=={n['name'] for n in doc['nodes']}


def test_fixtures_reproduce_exactly(tmp_path):
    assert generate(tmp_path)==json.loads((ROOT/'SHA256SUMS.json').read_text())
    for p in tmp_path.iterdir(): assert p.read_bytes()==(ROOT/p.name).read_bytes()
