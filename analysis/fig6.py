#!/usr/bin/env python3
"""Figure 6: Ablation Study + Transfer — publication quality.

Reads ablation results from outputs/ JSON files instead of hardcoding.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json
from pathlib import Path

FIGDIR = Path(__file__).parent.parent / 'figures'
OUTDIR = Path(__file__).parent.parent / 'outputs'
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.labelcolor': 'black',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.color': 'black',
})

C_GVAE = '#0072B2'
C_GOOD = '#009E73'
C_BAD = '#CC6677'


def load_ablation_results():
    """Load ablation AUROC values from output JSON files."""
    ablations = []

    # Baseline: read from CV results
    cv_path = OUTDIR / 'melanoma_cv' / 'cv_results.json'
    if cv_path.exists():
        with open(cv_path) as f:
            r = json.load(f)
        full_auroc = r['pooled_metrics']['auroc']
    else:
        print(f'WARNING: {cv_path} not found, using N/A for FULL baseline')
        full_auroc = None
    if full_auroc is not None:
        ablations.append(('FULL', full_auroc))

    # Each ablation: read from single-run metrics.json
    abl_keys = [
        ('logreg_baseline', 'logreg baseline'),
        ('mol_only',        'mol only'),
        ('spatial_only',    'spatial only'),
        ('static_0.3',      'static 0.3'),
        ('static_0.5',      'static 0.5'),
        ('static_0.7',      'static 0.7'),
        ('no_expr',         'no expr'),
        ('gaussian',        'gaussian'),
        ('no_contrastive',  'no contrastive'),
        ('frozen_encoder',  'frozen encoder'),
        ('gcn_encoder',     'GCN encoder*'),
        ('rare_leiden',     'rare leiden'),
    ]
    for key, display_name in abl_keys:
        fp = OUTDIR / f'melanoma_{key}' / 'metrics.json'
        if fp.exists():
            with open(fp) as f:
                r = json.load(f)
            auroc = r.get('prediction', {}).get('auroc')
            if auroc is not None:
                ablations.append((display_name, float(auroc)))
            else:
                print(f'WARNING: no prediction.auroc in {fp}')
        else:
            print(f'WARNING: {fp} not found, skipping {display_name}')

    if not ablations:
        raise FileNotFoundError('No ablation results found in outputs/. Run ablations first.')
    return ablations


ABLATIONS = load_ablation_results()


def panel_label(ax, label, x=-0.14, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left', color='black')


# ── Figure ──
fig = plt.figure(figsize=(8.0, 4.2))
gs = gridspec.GridSpec(1, 2, wspace=0.45,
                       left=0.07, right=0.97, top=0.93, bottom=0.20,
                       width_ratios=[1.5, 1])

names = [a[0] for a in ABLATIONS]
aurocs = [a[1] for a in ABLATIONS]
full_auroc = ABLATIONS[0][1]
deltas = [a - full_auroc for a in aurocs]

sorted_idx = np.argsort(aurocs)[::-1]
names_s = [names[i] for i in sorted_idx]
aurocs_s = [aurocs[i] for i in sorted_idx]

# ── (a) AUROC by ablation condition ──
ax_a = fig.add_subplot(gs[0, 0])
colors_a = []
for n, a in zip(names_s, aurocs_s):
    if n == 'FULL':
        colors_a.append(C_GVAE)
    elif a >= full_auroc:
        colors_a.append(C_GOOD)
    else:
        colors_a.append('#AAC4D7')

ax_a.bar(range(len(names_s)), aurocs_s, color=colors_a, edgecolor='white',
         lw=0.4, width=0.7)
ax_a.axhline(0.5, color='black', ls='--', lw=0.8, zorder=5)
ax_a.axhline(full_auroc, color=C_GVAE, ls=':', lw=0.6, alpha=0.5)
ax_a.set_xticks(range(len(names_s)))
ax_a.set_xticklabels(names_s, rotation=45, ha='right', fontsize=7)
ax_a.set_ylabel('AUROC')
ax_a.set_ylim(0, 0.88)

# ── (b) Delta from full model ──
ax_b = fig.add_subplot(gs[0, 1])
abl_no_full = [(n, d) for n, d in zip(names, deltas) if n != 'FULL']
abl_sorted = sorted(abl_no_full, key=lambda x: x[1])
abl_names_b = [a[0] for a in abl_sorted]
abl_deltas_b = [a[1] for a in abl_sorted]
colors_b = [C_GOOD if d >= 0 else C_BAD for d in abl_deltas_b]

ax_b.barh(range(len(abl_names_b)), abl_deltas_b, color=colors_b,
          edgecolor='white', lw=0.3, height=0.6)
ax_b.axvline(0, color='black', lw=0.5)
ax_b.set_yticks(range(len(abl_names_b)))
ax_b.set_yticklabels(abl_names_b, fontsize=7)
ax_b.set_xlabel('$\\Delta$AUROC from full model')
ax_b.set_xlim(-0.65, 0.12)
ax_b.spines['left'].set_visible(False)
ax_b.tick_params(axis='y', length=0)

# ── Panel labels ──
panel_label(ax_a, 'a')
panel_label(ax_b, 'b')

fig.text(0.07, 0.02, '*GCNConv normalize=True silently discards gate edge weights ($r$=0.026)',
         fontsize=6, fontstyle='italic', color='#666666')

for ext in ['pdf', 'png']:
    fig.savefig(FIGDIR / f'fig6_ablation.{ext}',
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig)
print(f'Saved fig6_ablation to {FIGDIR}/')
