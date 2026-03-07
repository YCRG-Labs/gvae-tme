"""
Test ROC curve: does KL-based rare cell burden predict immunotherapy response?
Aggregates rare cell scores to patient level, then evaluates against real response labels.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.metrics import roc_curve, auc
from pathlib import Path
from plots import is_rare

DATA_DIR = Path.cwd().parent / 'data'
OUT_DIR  = Path.cwd()

adata = sc.read_h5ad(DATA_DIR / 'adata.h5ad')
scores   = adata.obs['rare_score'].to_numpy()
patients = adata.obs['patient_id'].to_numpy()
response = adata.obs['response'].to_numpy().astype(int)

# Patient-level aggregation ---------------------------------------------------
# Three ways to summarize rare cell burden per patient:
# 1. mean KL score
# 2. % cells above mean+2std threshold
# 3. 90th percentile KL score, capturing extremes

patient_df = (
    pd.DataFrame({'patient_id': patients, 'score': scores, 'is_rare': is_rare, 'response': response}).groupby('patient_id').agg(
        mean_kl = ('score', 'mean'),
        pct_rare = ('is_rare', 'mean'),
        p90_kl = ('score', lambda x: np.percentile(x, 90)),
        response = ('response','first'),
        n_cells = ('score', 'count'),
    ).reset_index()
)

print("\n---------------- Patient-level summary: ----------------")
print(patient_df.to_string(index=False))
print(f"\nResponders: {patient_df['response'].sum()} / {len(patient_df)}")

# ROC curves ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))

metrics = [
    ('mean_kl', 'Mean KL score', '#e63946'),
    ('pct_rare', '% rare cells', '#457b9d'),
    ('p90_kl', '90th percentile KL', '#2a9d8f'),
]

y_true = patient_df['response'].to_numpy()

for col, label, color in metrics:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        print(f"[warn] All patients have the same response label — ROC undefined.")
        break
    fpr, tpr, _ = roc_curve(y_true, patient_df[col].to_numpy())
    auroc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{label}  (AUROC={auroc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUROC=0.500)')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('Fig 4. ROC: KL Rare Cell Burden vs Immunotherapy Response\n (n={len(patient_df)} patients)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig4_roc_test.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved fig4_roc_test.png")