"""Assemble the ``*.recipe.json`` provenance sidecar (spec §11.3, v0 "enough to re-run here").

The recipe is the **provenance surface**, deliberately separate from the deterministic manifest:
timestamps and content hashes (which vary run-to-run) live here, so the SVG + manifest stay
byte-stable for morph/diff while provenance stays honest.

When the producing ``script`` is known, the recipe also carries a small **re-run block**
(``command``/``args``/``cwd``/``output``) so Flux's ``rerun-plot`` (and the in-app *Regenerate*
button) can reproduce the plot. Those paths are written **relative** so the recipe survives the
project being moved or copied as a whole; only the interpreter (``command``) is absolute, and it is
overridable via ``recipe["command"]``.

The script no longer has to be recorded by hand: when the caller does not pass one,
:mod:`fluxplot.provenance` discovers it from the running interpreter (safe deterministic rules
only — see plan §1). Explicit fields always win; ``recipe=False`` suppresses discovery entirely
for notebooks, generated figures and privacy-sensitive callers. The recipe then says *how* the
script was determined (``provenance.scriptDiscovery``: automatic / explicit / unavailable) and
never claims to know inputs or parameters it cannot know — there is no automatic input discovery.
"""
from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone

from . import provenance as _provenance


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
    recipe: dict | bool | None,
    *,
    plot_name: str,
    svg_filename: str,
    manifest_filename: str,
    spec_version: str,
    base_dir: str | None = None,
    recipe_dir: str | None = None,
    now: str | None = None,
) -> dict:
    # recipe semantics (plan §1): None → automatic provenance; False → explicitly suppress it
    # (still writes a valid, non-rerunnable recipe); a dict → explicit fields win, an inferred
    # script only fills a *missing* script. Inputs are never discovered automatically.
    suppress = recipe is False
    recipe = {} if recipe in (None, False) else dict(recipe)

    script = _script_block(recipe.get("script"))
    if script is not None:
        discovery = "explicit"
    elif not suppress:
        found = _provenance.discover_script()
        if found:
            script = {"path": found}
            discovery = "automatic"
        else:
            discovery = "unavailable"
    else:
        discovery = "suppressed"

    inputs = recipe.get("inputs", []) or []
    out = {
        "spec": "fluxplot/recipe",
        "schemaVersion": spec_version,
        "plot": plot_name,
        "outputs": {"svg": svg_filename, "manifest": manifest_filename},
        "generatedAt": now or _now_iso(),
        "script": script,
        "params": recipe.get("params", {}),
        "inputs": [_hash_input(i, base_dir) for i in inputs],
        "env": None,  # v0: full environment capture deferred (spec §11.3)
    }
    if not suppress:
        out["provenance"] = _provenance.build_provenance(
            script.get("path") if script else None, discovery
        )

    # Re-run block — what flux-core's runRecipe needs to reproduce the plot. It resolves
    # `cwd` and `output` against the recipe's OWN directory, runs `command args` (appending
    # params as `--key value` and exporting them as $FLUX_PARAMS), so the script re-emits the
    # SVG in place. Emitted only when we know which script produced the plot. Paths are written
    # relative (to the recipe dir / the run cwd) for portability; the interpreter is absolute
    # and overridable via recipe["command"].
    script = out["script"]
    if script and script.get("path") and recipe_dir is not None:
        cwd_now = os.getcwd()
        script_abs = os.path.abspath(script["path"])
        svg_abs = os.path.join(recipe_dir, svg_filename)
        out["command"] = recipe.get("command") or sys.executable or "python"
        out["args"] = [os.path.relpath(script_abs, cwd_now)]
        out["cwd"] = os.path.relpath(cwd_now, recipe_dir)
        out["output"] = os.path.relpath(svg_abs, recipe_dir)
    elif recipe.get("command"):  # explicit command without a script — pass through (back-compat)
        out["command"] = recipe["command"]
    return out


def params(defaults: dict | None = None) -> dict:
    """Merge ``defaults`` with any ``FLUX_PARAMS`` (JSON) provided in the environment.

    Read tunables through this so ``flux rerun-plot <recipe> --key value`` (and the in-app
    *Regenerate* button, which set ``$FLUX_PARAMS``) can re-run the script with overrides::

        import fluxplot as fp
        p = fp.params({"test": "t-test", "smooth": False})
        if p["test"] == "mann-whitney":
            ...
    """
    import json as _json

    out = dict(defaults or {})
    raw = os.environ.get("FLUX_PARAMS")
    if raw:
        try:
            out.update(_json.loads(raw))
        except Exception:
            pass
    return out
