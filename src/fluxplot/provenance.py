"""Save-time provenance — automatic, honest discovery of the producing script (plan §1).

``fp.save`` can determine the producing script, interpreter, package versions, source hash and
Git state without asking the user to repeat facts the runtime already knows. Discovery is
**deterministic and conservative**: identity comes only from ``__main__.__file__`` or the call
stack, never from notebook history or heuristics, and anything ambiguous yields ``None`` (the
recipe then says so via ``scriptDiscovery: "unavailable"`` rather than guessing).

All of this is host-specific, run-varying material — it belongs in the recipe (the provenance
surface), never in the deterministic SVG/manifest.
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys

_GIT_TIMEOUT_S = 2.0


def _is_real_script(path) -> bool:
    """True for an existing regular ``.py`` file that plausibly *is* the producing script."""
    if not path or not isinstance(path, str):
        return False
    if path.startswith("<"):  # <stdin>, <string>, <ipython-input-...>
        return False
    if not path.endswith(".py"):
        return False
    if not os.path.isfile(path):
        return False
    # ipykernel writes each cell to a real temp file (tmp/ipykernel_<pid>/<hash>.py) for
    # debugger support — a cell fragment is not a rerunnable script. Reject that exact
    # pattern; legitimate scripts in temp directories are unaffected.
    if os.path.basename(os.path.dirname(path)).startswith("ipykernel_"):
        return False
    return True


def _in_installed_packages(path: str) -> bool:
    parts = os.path.abspath(path).split(os.sep)
    return "site-packages" in parts or "dist-packages" in parts


def discover_script() -> str | None:
    """The absolute path of the producing ``.py`` script, or ``None`` when there isn't one.

    Rules (deterministic, in order):

    1. ``__main__.__file__`` if it names an existing regular ``.py`` file that is not part of
       an installed package (``python -m pytest`` must not claim pytest produced the plot).
    2. Otherwise the first caller frame whose file is an existing ``.py`` outside FluxPlot's
       own code.
    3. Otherwise ``None`` — interactive sessions, notebooks, ``python -c`` and frozen apps
       are honestly not rerunnable scripts.
    """
    import __main__

    main_file = getattr(__main__, "__file__", None)
    if _is_real_script(main_file) and not _in_installed_packages(main_file):
        return os.path.abspath(main_file)

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    frame = sys._getframe()
    while frame is not None:
        fn = frame.f_code.co_filename
        if fn and not fn.startswith("<"):
            abs_fn = os.path.abspath(fn)
            if not abs_fn.startswith(pkg_dir + os.sep) and _is_real_script(abs_fn):
                return abs_fn
        frame = frame.f_back
    return None


def _git_info(directory: str) -> dict | None:
    """``{"commit": ..., "dirty": ...}`` for the repo containing ``directory``.

    Fails closed: outside a repository, without git installed, or on any error/timeout this
    returns ``None`` and the caller omits the block — provenance must never break a save.
    """
    def run(*args):
        return subprocess.run(
            ["git", "-C", directory, *args],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_S,
        )

    try:
        head = run("rev-parse", "HEAD")
        commit = head.stdout.strip()
        if head.returncode != 0 or not commit:
            return None
        out = {"commit": commit}
        status = run("status", "--porcelain")
        if status.returncode == 0:
            out["dirty"] = bool(status.stdout.strip())
        return out
    except Exception:
        return None


def build_provenance(script_path: str | None, discovery: str) -> dict:
    """The recipe's additive ``provenance`` block.

    ``discovery`` distinguishes ``automatic`` (we found the script), ``explicit`` (the caller
    recorded it) and ``unavailable`` (no script — the artifact is not rerunnable). Script-hash
    failures omit only that field; Git is optional and omitted outside a repository.
    """
    import matplotlib

    from .version import __version__

    prov = {
        "scriptDiscovery": discovery,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {"fluxplot": __version__, "matplotlib": matplotlib.__version__},
    }
    if script_path:
        try:
            with open(script_path, "rb") as f:
                prov["scriptSha256"] = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            pass
        git = _git_info(os.path.dirname(os.path.abspath(script_path)) or ".")
        if git is not None:
            prov["git"] = git
    return prov
