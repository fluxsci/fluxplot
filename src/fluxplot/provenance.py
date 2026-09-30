"""Save-time provenance — automatic, honest discovery of the producing script (plan §1).

``fp.save`` can determine the producing script, interpreter, package versions, source hash and
Git state without asking the user to repeat facts the runtime already knows. Discovery is
**deterministic and conservative**: identity comes only from ``__main__.__file__`` (or, when a
runner such as ``pytest`` is the entry point, the caller's own file), never from notebook history
or heuristics, and anything ambiguous yields ``None`` (the recipe then says so via
``scriptDiscovery: "unavailable"`` rather than guessing).

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
    2. When the interpreter was started through an installed entry point — ``python -m pkg``
       (``__main__.__file__`` inside site-packages) or a console script such as ``pytest``
       (``__main__.__file__`` that is not a ``.py`` file) — the first caller frame whose file is
       an existing ``.py`` outside FluxPlot's own code: the user's script, run by a runner.
    3. Otherwise ``None`` — interactive sessions, notebooks (``__main__`` has no ``__file__``
       at all under ipykernel), ``python -c`` and frozen apps are honestly not rerunnable
       scripts. In particular a notebook cell calling ``mylib.plot()`` must never record
       ``mylib.py`` as the recipe: rerunning it would not reproduce the figure.
    """
    import __main__

    main_file = getattr(__main__, "__file__", None)
    if _is_real_script(main_file) and not _in_installed_packages(main_file):
        return os.path.abspath(main_file)
    if not isinstance(main_file, str) or not main_file or main_file.startswith("<"):
        return None  # no entry-point file: interactive / notebook / -c

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    frame = sys._getframe()
    while frame is not None:
        fn = frame.f_code.co_filename
        if fn and not fn.startswith("<"):
            abs_fn = os.path.abspath(fn)
            if not abs_fn.startswith(pkg_dir + os.sep) and _is_real_script(abs_fn) \
                    and not _in_installed_packages(abs_fn):
                return abs_fn
        frame = frame.f_back
    return None


#: where a running kernel says which notebook it serves — each set by exactly one host, none guessed
NOTEBOOK_ENV = "QUARTO_DOCUMENT_PATH"
NOTEBOOK_GLOBALS = ("__vsc_ipynb_file__", "__session__")


def discover_notebook() -> str | None:
    """The absolute path of the notebook the current kernel serves, or ``None``.

    Only what the host states outright counts: ``$QUARTO_DOCUMENT_PATH`` (Quarto rendering or
    a live QMD kernel), then the ``__vsc_ipynb_file__`` / ``__session__`` globals VS Code and
    Jupyter put in ``__main__``. Nothing is inferred from the working directory or the stack: a
    wrong notebook would be worse than none.
    """
    import __main__

    candidates = [os.environ.get(NOTEBOOK_ENV)]
    candidates += [getattr(__main__, name, None) for name in NOTEBOOK_GLOBALS]
    for cand in candidates:
        if isinstance(cand, str) and cand and not cand.startswith("<"):
            if cand.startswith("file://"):
                from urllib.parse import unquote, urlparse
                cand = unquote(urlparse(cand).path)
            if os.path.isfile(cand):
                return os.path.abspath(cand)
    return None


def notebook_block(path: str, cell) -> dict:
    """The recipe's ``notebook`` block: the path, the cell label and the file's SHA-256 (``None``
    when it cannot be read)."""
    out = {"path": str(path), "cell": None if cell is None else str(cell), "sha256": None}
    try:
        with open(path, "rb") as f:
            out["sha256"] = hashlib.sha256(f.read()).hexdigest()
    except OSError:
        pass
    return out


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
    recorded it), ``notebook`` (a notebook cell produced it — recorded, not rerunnable) and
    ``unavailable`` (no script — the artifact is not rerunnable). Script-hash failures omit only
    that field; Git is optional and omitted outside a repository. For a notebook, ``script_path``
    is the notebook file: its hash and Git state are recorded the same way.
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
