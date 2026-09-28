"""Conservative, scene-specific face caps without running a simplifier."""
from __future__ import annotations

from copy import deepcopy
import json
import struct

WARN_BYTES = 50 * 1024**2
WARN_TRIANGLES = 2_000_000


def _json_bytes(value, *, upper):
    # Counts/offsets are bounded by GLB's uint32 size; finite float32 bounds and
    # Python float weights need at most 25 characters. Reserve 34, including
    # quotes, so simple original bounds cannot underestimate collapsed bounds.
    def numbers(item):
        if isinstance(item, dict): return {k: numbers(v) for k, v in item.items()}
        if isinstance(item, list): return [numbers(v) for v in item]
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            return '0' * 32 if upper else 0
        return item
    size = len(json.dumps(numbers(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    return (size + 3) // 4 * 4


def face_cap_recommendation(scene, glb, *, max_bytes=None, max_triangles=None):
    """Return an actionable per-helper-call cap, or explain why none is safe.

    Each part is bounded independently by min(original faces, cap), even when
    a call apportions its cap among several parts. The byte bound uses at most
    three referenced vertices per face, every attribute/state, uint32 indices,
    and the actual JSON structure/strings with worst-case numeric widths.
    This is a sufficient bound, not a prediction of backend simplification.
    """
    max_bytes = WARN_BYTES if max_bytes is None else max_bytes
    max_triangles = WARN_TRIANGLES if max_triangles is None else max_triangles
    # A scene may already have been simplified. The hint applies to rerunning
    # the authored calls, so use their original face counts, not a prior quota.
    counts = [len(part.source_faces) if part.source_faces is not None else len(part.faces)
              for part in scene.parts]
    total = sum(counts)
    limit = f'{max_bytes / 1024**2:g} MiB / {max_triangles:,} triangles'
    if len(counts) > max_triangles:
        return f'No max_faces cap can fit {limit} while preserving all {len(counts):,} parts; split the scene or remove parts'
    if total == len(counts) and len(glb) > max_bytes:
        return f'No max_faces cap can fit {limit}: all {len(counts):,} parts already contain one triangle; use fewer parts or shape states, or split the scene'

    size = struct.unpack_from('<I', glb, 12)[0]
    document = json.loads(glb[20:20 + size])
    # Even ignoring every buffer and shortening every number, these names,
    # target lists and structural keys cannot be removed by a face cap.
    required = deepcopy(document)
    optional_accessors, optional_views = set(), set()
    for mesh in required['meshes']:
        for primitive in mesh['primitives']:
            index = primitive['attributes'].pop('_VALID', None)
            if index is not None:
                optional_accessors.add(index)
                optional_views.add(required['accessors'][index]['bufferView'])
    required['accessors'] = [a for i, a in enumerate(required['accessors']) if i not in optional_accessors]
    required['bufferViews'] = [v for i, v in enumerate(required['bufferViews']) if i not in optional_views]
    for material in required['materials']: material.pop('alphaMode', None)
    if 28 + _json_bytes(required, upper=False) > max_bytes:
        return f'No max_faces cap can fit {limit}: fixed GLB part/state metadata alone exceeds the byte limit; use fewer parts or shape states, or split the scene'

    # Averaging valid values cannot add a channel normally, but reserve _VALID
    # even for presently all-valid data so the bound does not rely on that.
    for part, mesh in zip(scene.parts, document['meshes']):
        attrs = mesh['primitives'][0]['attributes']
        if part.values is not None and '_VALID' not in attrs:
            attrs['_VALID'] = len(document['accessors'])
            document['accessors'].append({'bufferView': len(document['bufferViews']), 'componentType': 5121, 'count': 1, 'type': 'SCALAR'})
            document['bufferViews'].append({'buffer': 0, 'byteOffset': 0, 'byteLength': 4, 'target': 34962, 'byteStride': 4})
    for material in document['materials']: material['alphaMode'] = 'BLEND'
    overhead = 28 + _json_bytes(document, upper=True)
    costs = [24 + 24 * len(p.states) + (8 if p.values is not None else 0) + (4 if p.colors is not None else 0) for p in scene.parts]

    def fits(cap):
        faces = [min(count, cap) for count in counts]
        # Compaction after reduction guarantees at most three vertices per face.
        # Do not cap this by today's vertex count: a user may rerun an already
        # simplified scene from its original meshes with the recommended cap.
        binary = sum(3 * count * cost + 12 * count for count, cost in zip(faces, costs))
        return sum(faces) <= max_triangles and overhead + binary <= max_bytes

    low = max(1, scene._max_call_parts)
    if not fits(low):
        return f'No conservative max_faces cap can be recommended for {limit} while preserving every call\'s parts and shape states; use fewer parts or states, or split the scene'
    high = max(counts)
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle): low = middle
        else: high = middle - 1
    return (f'use max_faces={low} on each mesh3d/surface3d call to fit {limit} '
            '(conservative bound if the simplifier reaches each requested face cap; retained-face warnings still apply)')
