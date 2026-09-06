"""Generate small shared Python/Flux 0.3 contract fixtures. No user data required.

Run: python tests/generate_polish_fixtures.py [output-directory]
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import fluxplot as fp

out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / 'fixtures' / 'polish')
out.mkdir(parents=True, exist_ok=True)
for state in ('a', 'b'):
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.4), layout='constrained')
    for panel, ax in zip(('small', 'large'), axes):
        fp.panel(ax, panel)
        factor = 1e-6 if panel == 'small' else 100
        y = np.array([1, 3, np.nan, 2, 5]) * factor
        if state == 'b': y *= 1.4
        fp.line(ax, np.arange(5), y, series='control', marker='o', markersize=3, markevery=2, label='Control')
        ax.set_title(panel.capitalize())
        ax.set_xlabel('Time'); ax.set_ylabel('Response')
    fp.save(fig, str(out / f'panels-{state}'), recipe=False, _now='2026-09-06T00:00:00Z')
    plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(5, 2.4), layout='constrained')
fp.panel(axes[0], 'matrix'); fp.panel(axes[1], 'contours')
m = fp.heatmap(axes[0], [[1, .2, .3], [.2, 1, np.nan], [.3, .6, 1]], series='correlation', cells=True, include_values=True, cmap='viridis', vmin=0, vmax=1)
fp.colorbar(m, name='correlation', label='Correlation')
axes[0].set_title('Masked matrix')
x, y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
c = fp.contourf(axes[1], x, y, x*x+y*y, series='energy', levels=[0, 1, 3, 5, 8], cmap='plasma')
fp.colorbar(c, name='energy', label='Energy')
axes[1].set_title('Contour bands')
fp.save(fig, str(out / 'fields'), recipe=False, _now='2026-09-06T00:00:00Z')
fig.savefig(out / 'fields.png', dpi=150)
plt.close(fig)
