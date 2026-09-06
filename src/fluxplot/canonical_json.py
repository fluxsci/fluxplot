"""Deterministic standards-compliant JSON, preserving scientific float precision.

Missing observations are encoded by the data adapters, never by the serializer.
"""
from __future__ import annotations
import json


def _native(obj):
    if isinstance(obj, dict):
        return {str(k): _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    if hasattr(obj, 'tolist'):
        return _native(obj.tolist())
    if isinstance(obj, float) and obj == 0:
        return 0.0
    return obj


def dumps(obj) -> str:
    return json.dumps(_native(obj), ensure_ascii=False, allow_nan=False,
                      sort_keys=True, indent=2, separators=(',', ': ')) + '\n'
