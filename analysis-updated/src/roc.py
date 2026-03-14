"""
Test ROC curve: does KL-based rare cell burden predict immunotherapy response?
Aggregates rare cell scores to patient level, then evaluates against real response labels.
Usage: python roc.py [--data path/to/adata_analysis.h5ad]
"""
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import scanpy as sc
from sklearn.metrics import roc_curve, auc
from pathlib import Path

parser = argparse.ArgumentParser(description='ROC: rare cell burden vs response')
parser.add_argument('--data', type=str, default=None,
                    help='Path to adata_analysis.h5ad')
args = parser.parse_args()

DATA_DIR  = Path.cwd().parent / 'data'
FILE_PATH = Path(args.data).resolve() if args.data else (DATA_DIR / 'adata_analysis.h5ad').resolve()
OUT_DIR   = Path.cwd()

# style matching the other figures
mpl.rcParams.update({
    'font.family':      'serif',
    'mathtext.fontset': 'dejavuserif',
    'font.size':        11,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.color':       '#EEEEEE',
    'grid.linewidth':   0.6,
    'axes.axisbelow':   True,
    'axes.facecolor':   'white',
    'figure.facecolor': 'white',
})

if not FILE_PATH.exists():
    print(f"Error: data file not found: {FILE_PATH}")
    print("Run the pipeline first, e.g.:")
    print("  python train.py --data melanoma --config local --max-cells 2000")
    print("Then: python roc.py --data ../outputs/melanoma/adata_analysis.h5ad")
    sys.exit(1)

print(f"Loading: {FILE_PATH}")
adata  = sc.read_h5ad(FILE_PATH)
scores = adata.obs['rare_score'].to_numpy()
is_rare = scores > (scores.mean() + 2 * scores.std())

if 'patient_id' not in adata.obs.columns or 'response' not in adata.obs.columns:
    print("[warn] patient_id or response missing — skipping ROC.")
    sys.exit(0)

patients = adata.obs['patient_id'].to_numpy()
resp_raw = adata.obs['response'].to_numpy()
response = ((resp_raw == 1) | (resp_raw == '1')).astype(int)

patient_df = (
    pd.DataFrame({
        'patient_id': patients,
        'score':      scores,
        'is_rare':    is_rare,
        'response':   response,
    })
    .groupby('patient_id')
    .agg(
        mean_kl  = ('score',   'mean'),
        pct_rare = ('is_rare', 'mean'),
        p90_kl   = ('score',   lambda x: np.percentile(x, 90)),
        response = ('response','first'),
        n_cells  = ('score',   'count'),
    )
    .reset_index()
)
# NOTE: higher KL means more MDSC-like suppression, which predicts non-response (not response)
# roc_curve measures prediction of response=1, so flip the sign
patient_df['mean_kl']  = -patient_df['mean_kl']
patient_df['pct_rare'] = -patient_df['pct_rare']
patient_df['p90_kl']   = -patient_df['p90_kl']

print("\n── Patient-level summary ───────────────────────────────")
print(patient_df.to_string(index=False))
print(f"\nResponders: {patient_df['response'].sum()} / {len(patient_df)}")

y_true = patient_df['response'].to_numpy()
if y_true.sum() == 0 or y_true.sum() == len(y_true):
    print("[warn] All patients have the same response label — ROC undefined.")
    sys.exit(0)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

metrics = [
    ('mean_kl', 'Mean KL','#e63946'),
    ('pct_rare', '% rare cells', '#457b9d'),
    ('p90_kl', '90th pct KL', '#2a9d8f'),
]
for col, label, color in metrics:
    fpr, tpr, _ = roc_curve(y_true, patient_df[col].to_numpy())
    auroc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f'{label}  (AUROC={auroc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUROC=0.500)')

ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title(
    f'ROC: KL Rare Cell Burden vs Immunotherapy Response\n(n={len(patient_df)} patients)',
    fontsize=11, fontweight='bold'
)
ax.legend(fontsize=9, loc='lower right')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

ax.text(0.6, 0.95,
    'Higher score = more MDSC burden = predicts non-response',
    transform=ax.transAxes, ha='right', va='bottom',
    fontsize=7.5, color='#666666', style='italic'
)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig5_roc.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved fig5_roc.png")