"""Optional collapse replay; named-part boundaries never share a simplification pass."""
from __future__ import annotations
import warnings
import numpy as np


def face_budgets(counts, total):
    """Apportion one triangle per part, then remaining capacity by largest remainder."""
    counts=[int(n) for n in counts]
    if total is None: return [None]*len(counts)
    if total<len(counts): raise ValueError('max_faces must allow at least one triangle per part')
    if any(n<1 for n in counts): raise ValueError('mesh parts must contain triangles')
    target=min(int(total),sum(counts)); remaining=target-len(counts)
    capacity=sum(n-1 for n in counts)
    if not capacity: return counts
    portions=[divmod(remaining*(n-1),capacity) for n in counts]
    result=[1+q for q,r in portions]
    for i in sorted(range(len(counts)),key=lambda i:(-portions[i][1],i))[:target-sum(result)]: result[i]+=1
    return result


def reduce_part(part, *, max_faces=None, collapses=None):
    try: import fast_simplification as fs
    except ImportError as exc:
        raise ImportError('max_faces needs the optional fluxplot[mesh] extra (fast-simplification); omit max_faces to save full resolution') from exc
    vertices=part.vertices.copy(); faces=part.faces.copy()
    if collapses is None:
        _,_,collapses=fs.simplify(vertices,faces,target_count=int(max_faces),return_collapses=True)
    collapses=np.asarray(collapses,dtype=np.int32)
    out, outfaces, mapping=fs.replay_simplification(vertices.astype(np.float32),faces,collapses)
    mapping=np.asarray(mapping,dtype=np.int64)
    if len(outfaces)==0: raise ValueError(f'{part.id}: max_faces removed every triangle; choose a larger budget')
    if max_faces is not None and len(outfaces)>max_faces:
        warnings.warn(f'{part.id}: simplifier retained {len(outfaces)} faces above requested max_faces={max_faces}',stacklevel=3)
    # Some backends leave unused output vertices. Compact by index only, in the
    # same order for all states and fields; this bounds stored vertices by 3F.
    used=np.unique(outfaces)
    for name,target in list(part.states.items()):
        sv,sf,_=fs.replay_simplification(target.astype(np.float32),faces,collapses)
        if not np.array_equal(outfaces,sf): raise ValueError(f'state {name}: collapse replay changed topology')
        part.states[name]=np.asarray(sv,dtype=float)[used]
    def remap(array):
        a=np.asarray(array,dtype=float); scalar=a.ndim==1
        if scalar: a=a[:,None]
        valid=np.isfinite(a); good=(mapping>=0)&(mapping<len(out))
        sums=np.zeros((len(out),a.shape[1])); count=np.zeros_like(sums)
        np.add.at(sums,mapping[good],np.where(valid[good],a[good],0))
        np.add.at(count,mapping[good],valid[good].astype(float))
        result=np.divide(sums,count,out=np.full_like(sums,np.nan),where=count>0)
        return result[:,0] if scalar else result
    if part.values is not None: part.values=remap(part.values)[used]
    if part.colors is not None: part.colors=remap(part.colors)[used]
    part.vertices=np.asarray(out,dtype=float)[used]; part.faces=np.searchsorted(used,outfaces).astype(np.int64)
    part.collapses=collapses.copy()
