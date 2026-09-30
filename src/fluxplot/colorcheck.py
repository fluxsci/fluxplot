"""Accessibility lint for colours — colour-vision deficiency, greyscale, contrast, colormap uniformity.

A plot's colours are chosen for the sighted author's screen. This module asks what happens to them
for the reader: **which pairs collapse** under the three common colour-vision deficiencies (Machado,
Oliveira & Fernandes 2009: 3×3 matrices in linear sRGB, embedded here at every published severity —
no extra dependency), **which vanish in greyscale** (a photocopied paper, a black-and-white print),
whether marks and text **contrast** enough with the ground (WCAG 2), and whether a colormap is
**perceptually uniform** (even ΔE steps, lightness running one way for a sequential map). Distances
are CIEDE2000 (``delta_e``) — the perceptual metric the CAM02-UCS helpers of the signature plots
approximate for their shade spacing.

:func:`check_palette` and :func:`check_colormap` return :class:`Finding` records; :func:`check_figure`
gathers a figure's series colours and text inks (the B2 colour records) and checks them against
the axes background. ``fp.save(..., lint="warn"|"error")`` runs it on the way out, puts the
findings in ``SaveResult.warnings`` and in the manifest's ``quality.color``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------------------------
# Machado, Oliveira & Fernandes (2009), "A Physiologically-based Model for Simulation of Color
# Vision Deficiency": the linear-sRGB matrices at severity 0.0 … 1.0 in steps of 0.1. Between
# published steps the matrices are interpolated linearly, as the paper's own tables are used.
# ---------------------------------------------------------------------------------------------
CVD_KINDS = ("protanomaly", "deuteranomaly", "tritanomaly")
_MACHADO = {
    "protanomaly": [[[1.0, 0.0, -0.0], [0.0, 1.0, 0.0], [-0.0, -0.0, 1.0]], [[0.856167, 0.182038, -0.038205], [0.029342, 0.955115, 0.015544], [-0.00288, -0.001563, 1.004443]], [[0.734766, 0.334872, -0.069637], [0.05184, 0.919198, 0.028963], [-0.004928, -0.004209, 1.009137]], [[0.630323, 0.465641, -0.095964], [0.069181, 0.890046, 0.040773], [-0.006308, -0.007724, 1.014032]], [[0.539009, 0.579343, -0.118352], [0.082546, 0.866121, 0.051332], [-0.007136, -0.011959, 1.019095]], [[0.458064, 0.679578, -0.137642], [0.092785, 0.846313, 0.060902], [-0.007494, -0.016807, 1.024301]], [[0.38545, 0.769005, -0.154455], [0.100526, 0.829802, 0.069673], [-0.007442, -0.02219, 1.029632]], [[0.319627, 0.849633, -0.169261], [0.106241, 0.815969, 0.07779], [-0.007025, -0.028051, 1.035076]], [[0.259411, 0.923008, -0.18242], [0.110296, 0.80434, 0.085364], [-0.006276, -0.034346, 1.040622]], [[0.203876, 0.990338, -0.194214], [0.112975, 0.794542, 0.092483], [-0.005222, -0.041043, 1.046265]], [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]],
    "deuteranomaly": [[[1.0, 0.0, -0.0], [0.0, 1.0, 0.0], [-0.0, 0.0, 1.0]], [[0.866435, 0.177704, -0.044139], [0.049567, 0.939063, 0.01137], [-0.003453, 0.007233, 0.99622]], [[0.760729, 0.319078, -0.079807], [0.090568, 0.889315, 0.020117], [-0.006027, 0.013325, 0.992702]], [[0.675425, 0.43385, -0.109275], [0.125303, 0.847755, 0.026942], [-0.00795, 0.018572, 0.989378]], [[0.605511, 0.52856, -0.134071], [0.155318, 0.812366, 0.032316], [-0.009376, 0.023176, 0.9862]], [[0.547494, 0.607765, -0.155259], [0.181692, 0.781742, 0.036566], [-0.01041, 0.027275, 0.983136]], [[0.498864, 0.674741, -0.173604], [0.205199, 0.754872, 0.039929], [-0.011131, 0.030969, 0.980162]], [[0.457771, 0.731899, -0.18967], [0.226409, 0.731012, 0.042579], [-0.011595, 0.034333, 0.977261]], [[0.422823, 0.781057, -0.203881], [0.245752, 0.709602, 0.044646], [-0.011843, 0.037423, 0.974421]], [[0.392952, 0.82361, -0.216562], [0.263559, 0.69021, 0.046232], [-0.01191, 0.040281, 0.97163]], [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.01182, 0.04294, 0.968881]]],
    "tritanomaly": [[[1.0, 0.0, -0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[0.92667, 0.092514, -0.019184], [0.021191, 0.964503, 0.014306], [0.008437, 0.054813, 0.93675]], [[0.89572, 0.13333, -0.02905], [0.029997, 0.9454, 0.024603], [0.013027, 0.104707, 0.882266]], [[0.905871, 0.127791, -0.033662], [0.026856, 0.941251, 0.031893], [0.01341, 0.148296, 0.838294]], [[0.948035, 0.08949, -0.037526], [0.014364, 0.946792, 0.038844], [0.010853, 0.193991, 0.795156]], [[1.017277, 0.027029, -0.044306], [-0.006113, 0.958479, 0.047634], [0.006379, 0.248708, 0.744913]], [[1.104996, -0.046633, -0.058363], [-0.032137, 0.971635, 0.060503], [0.001336, 0.317922, 0.680742]], [[1.193214, -0.109812, -0.083402], [-0.058496, 0.97941, 0.079086], [-0.002346, 0.403492, 0.598854]], [[1.257728, -0.139648, -0.118081], [-0.078003, 0.975409, 0.102594], [-0.003316, 0.501214, 0.502102]], [[1.278864, -0.125333, -0.153531], [-0.084748, 0.957674, 0.127074], [-0.000989, 0.601151, 0.399838]], [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.3039]]],
}
_ALIASES = {"protanopia": "protanomaly", "deuteranopia": "deuteranomaly", "tritanopia": "tritanomaly",
            "protan": "protanomaly", "deutan": "deuteranomaly", "tritan": "tritanomaly"}

#: Default thresholds of :func:`check_palette` / :func:`check_colormap`.
THRESHOLDS = {
    "cvd-confusable": 10.0,        # ΔE2000 between two palette colours under a simulation
    "greyscale-confusable": 10.0,  # ΔL* between two palette colours in greyscale
    "low-contrast-mark": 3.0,      # WCAG ratio of a mark against the ground
    "low-contrast-text": 4.5,      # WCAG ratio of text against the ground
    "non-uniform": 0.35,           # coefficient of variation of a colormap's ΔE steps
}


def machado_matrix(kind: str, severity: float = 1.0) -> np.ndarray:
    """The 3×3 linear-sRGB matrix for ``kind`` at ``severity`` (0 = normal vision, 1 = dichromacy)."""
    kind = _ALIASES.get(kind, kind)
    if kind not in _MACHADO:
        raise ValueError(f"unknown colour-vision deficiency {kind!r}; use one of {CVD_KINDS}")
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must lie in [0, 1]")
    table = np.asarray(_MACHADO[kind])
    pos = severity * 10.0
    lo = int(np.floor(pos))
    hi = min(lo + 1, 10)
    t = pos - lo
    return (1 - t) * table[lo] + t * table[hi]


# ---------------------------------------------------------------------------------------------
# colour spaces
# ---------------------------------------------------------------------------------------------
def _rgb(colors) -> np.ndarray:
    from matplotlib.colors import to_rgba_array
    return to_rgba_array(list(colors) if not isinstance(colors, str) else [colors])[:, :3]


def _linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _encode(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(np.asarray(lin, dtype=float), 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def lab(colors) -> np.ndarray:
    """sRGB colours (anything matplotlib accepts, or an (N, 3) array in [0, 1]) → CIELAB (D65)."""
    from .signature_fluxplots._colour import lab as _lab
    return _lab(_rgb(colors) if not isinstance(colors, np.ndarray) or colors.ndim != 2 else colors)


def simulate(colors, kind: str, severity: float = 1.0) -> np.ndarray:
    """What ``colors`` look like to a viewer with ``kind`` colour-vision deficiency, as sRGB in
    [0, 1] (Machado et al. 2009, applied in linear sRGB). ``kind`` is ``protanomaly`` /
    ``deuteranomaly`` / ``tritanomaly`` (or the ``-opia`` names); ``severity`` 1.0 is dichromacy."""
    rgb = _rgb(colors) if not (isinstance(colors, np.ndarray) and colors.ndim == 2) else np.asarray(colors)[:, :3]
    lin = _linear(rgb) @ machado_matrix(kind, severity).T
    return _encode(lin)


def greyscale(colors) -> np.ndarray:
    """The colours' lightness only (L* as an sRGB grey), the way a black-and-white print shows them."""
    L = lab(colors)[:, 0]
    fy = (L + 16) / 116
    y = np.where(fy ** 3 > 216 / 24389, fy ** 3, (116 * fy - 16) * 27 / 24389)
    grey = _encode(y)
    return np.stack([grey, grey, grey], axis=-1)


def delta_e(a, b) -> float:
    """CIEDE2000 colour difference between two colours (Sharma, Wu & Dalal 2005 formulation)."""
    L1, a1, b1 = lab(a)[0]
    L2, a2, b2 = lab(b)[0]
    return float(_ciede2000(np.array([[L1, a1, b1]]), np.array([[L2, a2, b2]]))[0])


def _ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]
    kL = kC = kH = 1.0
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cbar = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.where((a1p == 0) & (b1 == 0), 0.0, np.degrees(np.arctan2(b1, a1p)) % 360)
    h2p = np.where((a2p == 0) & (b2 == 0), 0.0, np.degrees(np.arctan2(b2, a2p)) % 360)
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dhp = np.where(C1p * C2p == 0, 0.0, np.where(np.abs(dh) <= 180, dh, np.where(dh > 180, dh - 360, dh + 360)))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hsum = h1p + h2p
    hbp = np.where(C1p * C2p == 0, hsum,
                   np.where(np.abs(h1p - h2p) <= 180, hsum / 2,
                            np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    RC = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T
    RT = -np.sin(np.radians(2 * dtheta)) * RC
    return np.sqrt((dLp / (kL * SL)) ** 2 + (dCp / (kC * SC)) ** 2 + (dHp / (kH * SH)) ** 2
                   + RT * (dCp / (kC * SC)) * (dHp / (kH * SH)))


def _luminance(colors) -> np.ndarray:
    lin = _linear(_rgb(colors) if not (isinstance(colors, np.ndarray) and colors.ndim == 2) else colors)
    return lin @ np.array([0.2126, 0.7152, 0.0722])


def contrast(fg, bg) -> float:
    """WCAG 2 contrast ratio between a foreground and a background colour (1 … 21)."""
    lf, lb = float(_luminance(fg)[0]), float(_luminance(bg)[0])
    hi, lo = max(lf, lb), min(lf, lb)
    return (hi + 0.05) / (lo + 0.05)


# ---------------------------------------------------------------------------------------------
# findings
# ---------------------------------------------------------------------------------------------
@dataclass
class Finding:
    """One accessibility finding: what ``kind`` of problem, between which colours (``a``, ``b``;
    ``b`` is the ground for a contrast finding), the measured ``value`` against the ``threshold``
    it failed, and a sentence a reader can act on."""

    kind: str
    a: str
    b: Optional[str]
    value: float
    threshold: float
    message: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["value"] = round(float(d["value"]), 3)
        return d


def _hex(c) -> str:
    from matplotlib.colors import to_hex
    return to_hex(c)


def check_palette(colors, bg="#ffffff", *, text=None, names=None, thresholds=None) -> List[Finding]:
    """Lint a set of mark colours against one another and their ground.

    - every pair closer than ΔE 10 under any of the three deficiencies → ``cvd-confusable``;
    - every pair whose lightness differs by less than 10 L* → ``greyscale-confusable``;
    - a mark colour with WCAG contrast < 3:1 against ``bg`` → ``low-contrast-mark``;
    - a ``text`` colour with contrast < 4.5:1 against ``bg`` → ``low-contrast-text``.
    ``names`` label the colours in the messages (default: their hex).
    """
    th = {**THRESHOLDS, **(thresholds or {})}
    colors = list(colors)
    names = list(names) if names is not None else [_hex(c) for c in colors]
    findings: List[Finding] = []
    rgb = _rgb(colors)
    n = len(colors)
    for kind in CVD_KINDS:
        sim = simulate(rgb, kind, 1.0)
        labs = lab(sim)
        for i in range(n):
            for j in range(i + 1, n):
                d = float(_ciede2000(labs[i:i + 1], labs[j:j + 1])[0])
                if d < th["cvd-confusable"]:
                    findings.append(Finding("cvd-confusable", _hex(colors[i]), _hex(colors[j]), d, th["cvd-confusable"],
                                            f"{names[i]} and {names[j]} are near-identical to a {kind[:-3]}ope (ΔE {d:.1f} < {th['cvd-confusable']:g})"))
    L = lab(rgb)[:, 0]
    for i in range(n):
        for j in range(i + 1, n):
            d = float(abs(L[i] - L[j]))
            if d < th["greyscale-confusable"]:
                findings.append(Finding("greyscale-confusable", _hex(colors[i]), _hex(colors[j]), d, th["greyscale-confusable"],
                                        f"{names[i]} and {names[j]} merge in greyscale (ΔL* {d:.1f} < {th['greyscale-confusable']:g})"))
    for c, name in zip(colors, names):
        ratio = contrast(c, bg)
        if ratio < th["low-contrast-mark"]:
            findings.append(Finding("low-contrast-mark", _hex(c), _hex(bg), ratio, th["low-contrast-mark"],
                                    f"{name} has {ratio:.2f}:1 contrast against the ground (needs {th['low-contrast-mark']:g}:1)"))
    for c in (text or []):
        ratio = contrast(c, bg)
        if ratio < th["low-contrast-text"]:
            findings.append(Finding("low-contrast-text", _hex(c), _hex(bg), ratio, th["low-contrast-text"],
                                    f"text in {_hex(c)} has {ratio:.2f}:1 contrast against the ground (needs {th['low-contrast-text']:g}:1)"))
    return findings


def check_colormap(cmap, *, sequential=None, n=256, thresholds=None) -> List[Finding]:
    """Lint a colormap: ``non-uniform`` when the ΔE2000 steps between neighbouring samples vary
    by more than a 0.35 coefficient of variation, ``non-monotone`` when a sequential map's
    lightness does not run one way (``sequential`` defaults to the map's shipped type)."""
    from matplotlib.colors import Colormap
    from .colors import maps
    th = {**THRESHOLDS, **(thresholds or {})}
    cm = cmap if isinstance(cmap, Colormap) else maps._resolve_any(cmap)
    if sequential is None:
        try:
            sequential = maps.info(cm.name).get("type") == "sequential"
        except Exception:
            sequential = False
    rgba = cm(np.linspace(0, 1, n))
    labs = lab(rgba[:, :3])
    steps = _ciede2000(labs[:-1], labs[1:])
    findings: List[Finding] = []
    cv = float(np.std(steps) / np.mean(steps)) if np.mean(steps) > 0 else 0.0
    if cv > th["non-uniform"]:
        findings.append(Finding("non-uniform", cm.name, None, cv, th["non-uniform"],
                                f"{cm.name} is not perceptually uniform (ΔE step variation {cv:.2f} > {th['non-uniform']:g})"))
    if sequential:
        dL = np.diff(labs[:, 0])
        tol = 0.5
        if not (np.all(dL >= -tol) or np.all(dL <= tol)):
            reversals = int(np.sum(np.sign(dL[np.abs(dL) > tol])[1:] != np.sign(dL[np.abs(dL) > tol])[:-1]))
            findings.append(Finding("non-monotone", cm.name, None, float(reversals), 0.0,
                                    f"{cm.name} is sequential but its lightness reverses direction ({reversals}×)"))
    return findings


def check_figure(fig, *, thresholds=None) -> List[Finding]:
    """Lint a figure: its tagged series' primary colours (the B2 colour records) against one another
    and the axes background, and its text inks against the same ground."""
    from matplotlib.colors import to_hex
    from . import tagger
    from .data import primary_paint
    reg = tagger.registry_for(fig)
    colours, names, seen = [], [], set()
    for m in reg.marks:
        if m.series is None or m.role not in ("line", "point", "bar", "area", "box"):
            continue
        paint = primary_paint(m)
        if not paint or paint.get("hex") in (None, "varies") or m.series in seen:
            continue
        seen.add(m.series)
        colours.append(paint["hex"])
        names.append(f"series {m.series!r}")
    if not fig.axes:
        return []
    ax = next((a for a in fig.axes if a.get_label() != "<colorbar>"), fig.axes[0])
    bg = ax.get_facecolor()
    if bg[3] == 0:
        bg = fig.get_facecolor()
    texts = {to_hex(t.get_color()) for t in list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label] if t.get_text()}
    texts |= {to_hex(t.get_color()) for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_text()}
    return check_palette(colours, to_hex(bg), text=sorted(texts), names=names, thresholds=thresholds)
