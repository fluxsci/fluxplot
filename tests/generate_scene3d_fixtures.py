"""Deterministic P0.0 contract fixtures; run with uv run python tests/generate_scene3d_fixtures.py."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import struct
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent / 'fixtures' / 'scene3d'


def _linear(rgb):
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def normals(v, f):
    n = np.zeros_like(v, dtype=np.float64)
    t = v[f]
    face = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    for i in range(3):
        np.add.at(n, f[:, i], face)
    length = np.linalg.norm(n, axis=1)
    n[length == 0] = (0, 1, 0)
    return (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)).astype('<f4')


def contract_glb(parts, weights=None):
    """Fixture-only writer, independent of the later production generator."""
    binary = bytearray()
    doc = dict(asset={'version':'2.0','generator':'fluxplot scene3d contract 0.1.0'},
               buffers=[{}], bufferViews=[], accessors=[], meshes=[], nodes=[], materials=[],
               scenes=[{'nodes':list(range(len(parts)))}], scene=0)
    def accessor(a, kind, component, bounds=False, normalized=False, target=None, stride=None):
        a = np.ascontiguousarray(a)
        binary.extend(b'\0' * (-len(binary) % 4))
        bv = {'buffer':0,'byteOffset':len(binary),'byteLength':a.nbytes}
        if target is not None: bv['target'] = target
        if stride is not None: bv['byteStride'] = stride
        doc['bufferViews'].append(bv)
        binary.extend(a.tobytes())
        out = {'bufferView':len(doc['bufferViews'])-1,'componentType':component,'count':len(a),'type':kind}
        if bounds:
            out.update(min=a.min(axis=0).tolist(), max=a.max(axis=0).tolist())
        if normalized: out['normalized'] = True
        doc['accessors'].append(out)
        return len(doc['accessors'])-1
    for p in parts:
        v=np.asarray(p['v'],dtype='<f4'); f=np.asarray(p['f'],dtype='<u2' if len(v)<65536 else '<u4')
        n=normals(v,f)
        attrs={'POSITION':accessor(v,'VEC3',5126,True,target=34962),
               'NORMAL':accessor(n,'VEC3',5126,target=34962)}
        if 'values' in p:
            value=np.asarray(p['values'],dtype='<f4'); valid=np.isfinite(value)
            attrs['_VALUE']=accessor(np.where(valid,value,0).astype('<f4'),'SCALAR',5126,target=34962)
            if not valid.all(): attrs['_VALID']=accessor(np.column_stack([valid,np.zeros((len(valid),3),dtype='u1')]).astype('u1'),'SCALAR',5121,target=34962,stride=4)
        if 'colors' in p:
            rgba=np.asarray(p['colors'],dtype=float).copy(); rgba[:,:3]=_linear(rgba[:,:3])
            attrs['COLOR_0']=accessor(np.rint(rgba*255).astype('u1'),'VEC4',5121,normalized=True,target=34962)
        color=p.get('color','#4385BE'); rgb=np.array([int(color[i:i+2],16)/255 for i in (1,3,5)])
        doc['materials'].append({'name':p['id'],'pbrMetallicRoughness':{'baseColorFactor':[*_linear(rgb).tolist(),1], 'metallicFactor':0,'roughnessFactor':.6},'doubleSided':True})
        prim={'attributes':attrs,'indices':accessor(f.ravel(),'SCALAR',5123 if f.dtype.itemsize==2 else 5125,target=34963),'material':len(doc['materials'])-1,'mode':4}
        mesh={'name':p['id'],'primitives':[prim]}
        if p.get('states'):
            prim['targets']=[]
            for name, target in p['states'].items():
                target=np.asarray(target,dtype='<f4')
                prim['targets'].append({'POSITION':accessor(target-v,'VEC3',5126,True,target=34962),'NORMAL':accessor(normals(target,f)-n,'VEC3',5126,target=34962)})
            mesh['extras']={'targetNames':list(p['states'])}
            mesh['weights']=[(weights or {}).get(k,0) for k in p['states']]
        doc['meshes'].append(mesh); doc['nodes'].append({'name':p['id'],'mesh':len(doc['meshes'])-1})
    doc['buffers'][0]['byteLength']=len(binary)
    js=json.dumps(doc,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    js+=b' '*(-len(js)%4); binary.extend(b'\0'*(-len(binary)%4))
    return struct.pack('<4sII',b'glTF',2,28+len(js)+len(binary))+struct.pack('<I4s',len(js),b'JSON')+js+struct.pack('<I4s',len(binary),b'BIN\0')+binary


def sphere(n=8, m=12):
    # Poles are unique; topology is deterministic and every vertex is referenced.
    v=[[0,1,0]]
    for j in range(1,n):
        phi=np.pi*j/n
        for i in range(m):
            t=2*np.pi*i/m; v.append([np.sin(phi)*np.cos(t),np.cos(phi),np.sin(phi)*np.sin(t)])
    v.append([0,-1,0]); f=[]
    for i in range(m): f.append([0,1+(i+1)%m,1+i])
    for j in range(n-2):
        a=1+j*m; b=a+m
        for i in range(m):
            k=(i+1)%m; f.extend([[a+i,a+k,b+i],[a+k,b+k,b+i]])
    bottom=len(v)-1; a=1+(n-2)*m
    for i in range(m): f.append([a+i,a+(i+1)%m,bottom])
    return np.array(v,dtype='<f4'),np.array(f,dtype='<u2')


def cases():
    v,f=sphere(); base={'id':'sample.mesh','v':v,'f':f,'color':'#4385BE'}
    yield 'plain',[base],{}
    pair=[dict(base,id='neuron.soma',v=v*.45+[-.75,0,0],color='#D14D41'),dict(base,id='neuron.dendrites',v=v*[.3,1.1,.3]+[.7,0,0],color='#3AA99F')]
    yield 'named-parts',pair,{'parts':[{'id':'legend','role':'legend','kind':'furniture','entries':[p['id'] for p in pair]}],'layout':{'legend':'right'}}
    values=v[:,1].copy(); rgba=np.column_stack([(values+1)/2,np.full(len(v),.35),(1-values)/2,np.ones(len(v))])
    field={'cmap':{'name':'contract-blue-red','stops':[[0,'#0000FF'],[1,'#FF0000']]},'range':[-1,1],'rule':{'percentile':[0,100]},'ticks':[-1,0,1],'label':'Height','missingColor':'#D8D8D8'}
    yield 'continuous',[dict(base,id='height.field',values=values,colors=rgba)],{'parts':[{'id':'height.field','role':'surface-field','series':'height','node':'height.field','kind':'field','field':field},{'id':'height.colorbar','role':'colorbar','kind':'furniture','field':'height.field'}],'layout':{'colorbar':'right'}}
    # Missing has its own mesh; nonfinite API values serialize as finite _VALUE + _VALID.
    cats=[dict(base,id='atlas.frontal',f=f[:len(f)//3],color='#D14D41'),dict(base,id='atlas.parietal',f=f[len(f)//3:2*len(f)//3],color='#4385BE'),dict(base,id='atlas.missing',f=f[2*len(f)//3:],color='#D8D8D8',values=np.full(len(v),np.nan))]
    yield 'categorical-missing',cats,{}
    axes={'kind':'box','grid':True,**{k:{'lim':[-1,1],'ticks':[-1,0,1],'label':k+' (µm)'} for k in 'xyz'}}
    furniture=[{'id':f'axes.{k}.{s}','role':role,'kind':'furniture',**({'text':k+' (µm)'} if s=='label' else {})} for k in 'xyz' for s,role in [('label','axis-title'),('ticks','tick-label'),('grid','gridline'),('pane','pane'),('axis','axis')]]
    yield 'box-axes',[base],{'axes':axes,'parts':furniture}
    yield 'scalebar',[base],{'units':'µm','parts':[{'id':'scalebar','role':'scalebar','kind':'furniture','length':.5,'label':'0.5 µm'}]}
    morph=[dict(base,id='cortex.left',v=v+[-1.2,0,0]),dict(base,id='cortex.right',v=v+[1.2,0,0],color='#3AA99F')]
    yield 'morph-a',morph,{'morphGroup':'cortex'}
    yield 'morph-b',[dict(p,v=p['v']*[1,1.35,.65]+[0,.1,0]) for p in morph],{'morphGroup':'cortex'}
    vv,ff=sphere(6,10)
    yield 'morph-incompatible',[dict(morph[0],v=vv+[-1.2,0,0],f=ff),morph[1]],{'morphGroup':'cortex'}
    yield 'states',[dict(base,states={'inflated':v*[1.35,1.2,1.1],'bent':v+np.column_stack([.3*v[:,1]**2,np.zeros(len(v)),np.zeros(len(v))])})],{'states':[{'name':'inflated','label':'Inflated'},{'name':'bent','label':'Bent'}],'view':{'states':{'inflated':.25,'bent':0}}}
    yield 'sequence',[dict(base,states={f'frame-{i}':v*[1+i*.06,1,1-i*.04] for i in range(1,8)})],{'states':[{'name':f'frame-{i}','label':f'Frame {i}'} for i in range(1,8)],'sequence':True}


def generate(root=ROOT):
    root=Path(root); root.mkdir(parents=True,exist_ok=True); receipts={}
    for name,parts,extra in cases():
        glb=contract_glb(parts,extra.get('view',{}).get('states'))
        allv=np.concatenate([a for p in parts for a in [p['v'],*p.get('states',{}).values()]])
        manifest={'spec':'fluxplot/scene3d','schemaVersion':'0.1.0','plotType':'scene3d','glb':name+'.glb','glbSha256':hashlib.sha256(glb).hexdigest(),'size':{'width':3.5,'height':3,'unit':'in'},'units':'µm','toWorld':np.eye(4).ravel(order='F').tolist(),'bounds':{'min':allv.min(axis=0).tolist(),'max':allv.max(axis=0).tolist()},'view':{'azimuth':30,'elevation':20,'zoom':.9,'projection':'orthographic'},'lighting':'studio','style':{'font':'Inter','fontSizePt':7,'titleSizePt':8,'ink':'#100F0F','muted':'#6F6E69','lineWidthPt':.6},'axes':{'kind':'none'},'parts':[],'build':{'generator':'fluxplot scene3d contract 0.1.0'}}
        authored={p['id']:p for p in extra.get('parts',[])}
        for p in parts:
            manifest['parts'].append(authored.pop(p['id'],{'id':p['id'],'role':'surface-region' if name=='categorical-missing' else 'mesh','kind':'missing' if p['id'].endswith('.missing') else 'mesh','series':p['id'].split('.')[0],'node':p['id'],'label':p['id'].split('.')[-1],'color':p['color']}))
        manifest['parts'].extend(authored.values())
        for k,val in extra.items():
            if k=='view': manifest['view'].update(val)
            elif k!='parts': manifest[k]=val
        manifest['order']=[p['id'] for p in manifest['parts']]
        (root/(name+'.glb')).write_bytes(glb)
        (root/(name+'.fluxplot.json')).write_text(json.dumps(manifest,ensure_ascii=False,allow_nan=False,sort_keys=True,indent=2)+'\n')
    for p in sorted(root.glob('*')):
        if p.suffix in ('.glb','.json') and p.name!='SHA256SUMS.json': receipts[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'SHA256SUMS.json').write_text(json.dumps(receipts,indent=2,sort_keys=True)+'\n')
    return receipts

if __name__=='__main__':
    generate(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT)
