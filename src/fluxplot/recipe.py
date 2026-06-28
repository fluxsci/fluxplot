"""Assemble the ``*.recipe.json`` provenance sidecar (spec §11.3, v0 "enough to re-run here").

The recipe is the **provenance surface**, deliberately separate from the deterministic manifest:
timestamps and content hashes (which vary run-to-run) live here, so the SVG + manifest stay
byte-stable for morph/diff while provenance stays honest.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _script_block(script) -> dict | None:
    if script is None:
        return None
    if isinstance(script, dict):
        return script
    return {"path": str(script)}


def _hash_input(inp, base_dir):
    path = inp if isinstance(inp, str) else inp.get("path")
    role = "data" if isinstance(inp, str) else inp.get("role", "data")
    entry = {"role": role, "path": path}
    full = path if base_dir is None else os.path.join(base_dir, path)
    try:
        with open(full, "rb") as f:
            data = f.read()
        entry["sha256"] = hashlib.sha256(data).hexdigest()
        entry["bytes"] = len(data)
    except OSError:
        entry["sha256"] = None
    return entry


def build_recipe(
    recipe: dict | None,
    *,
    plot_name: str,
    svg_filename: str,
    manifest_filename: str,
    spec_version: str,
    base_dir: str | None = None,
    now: str | None = None,
) -> dict:
    recipe = recipe or {}
    inputs = recipe.get("inputs", []) or []
    out = {
        "spec": "fluxplot/recipe",
        "schemaVersion": spec_version,
        "plot": plot_name,
        "outputs": {"svg": svg_filename, "manifest": manifest_filename},
        "generatedAt": now or _now_iso(),
        "script": _script_block(recipe.get("script")),
        "params": recipe.get("params", {}),
        "inputs": [_hash_input(i, base_dir) for i in inputs],
        "env": None,  # v0: full environment capture deferred (spec §11.3)
    }
    if recipe.get("command"):
        out["command"] = recipe["command"]
    return out
