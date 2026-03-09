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
import scanpy as sc
from sklearn.metrics import roc_curve, auc
from pathlib import Path

parser = argparse.ArgumentParser(description='ROC: rare cell burden vs response')
parser.add_argument('--data', type=str, default=None, help='Path to adata (obs: rare_score, patient_id, response)')
args = parser.parse_args()
DATA_DIR = Path.cwd().parent / 'data'
OUT_DIR = Path.cwd()
FILE_PATH = (Path(args.data).resolve() if args.data else (DATA_DIR / 'adata.h5ad')).resolve()

if not FILE_PATH.exists():
    print(f"Error: data file not found: {FILE_PATH}")
    print("Run the pipeline first, e.g.: python train.py --data melanoma --config local --max-cells 2000")
    print("Then: python roc.py --data ../outputs/melanoma/adata_analysis.h5ad")
    sys.exit(1)
print(f"Loading: {FILE_PATH}")
adata = sc.read_h5ad(FILE_PATH)
scores = adata.obs['rare_score'].to_numpy()
is_rare = scores > (scores.mean() + 2 * scores.std())

if 'patient_id' not in adata.obs.columns or 'response' not in adata.obs.columns:
    print("[warn] patient_id or response missing — skipping ROC. Run on adata with clinical labels (e.g. outputs/melanoma/adata_analysis.h5ad).")
else:
    patients = adata.obs['patient_id'].to_numpy()
    response = adata.obs['response'].to_numpy()
    response = (response == 1) | (response == '1')
    response = response.astype(int)

    patient_df = (
        pd.DataFrame({'patient_id': patients, 'score': scores, 'is_rare': is_rare, 'response': response})
        .groupby('patient_id')
        .agg(
            mean_kl=('score', 'mean'),
            pct_rare=('is_rare', 'mean'),
            p90_kl=('score', lambda x: np.percentile(x, 90)),
            response=('response', 'first'),
            n_cells=('score', 'count'),
        )
        .reset_index()
    )

    print("\n---------------- Patient-level summary: ----------------")
    print(patient_df.to_string(index=False))
    print(f"\nResponders: {patient_df['response'].sum()} / {len(patient_df)}")

    y_true = patient_df['response'].to_numpy()
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        print("[warn] All patients have the same response label — ROC undefined.")
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        metrics = [
            ('mean_kl', 'Mean KL score', '#e63946'),
            ('pct_rare', '% rare cells', '#457b9d'),
            ('p90_kl', '90th percentile KL', '#2a9d8f'),
        ]
        for col, label, color in metrics:
            fpr, tpr, _ = roc_curve(y_true, patient_df[col].to_numpy())
            auroc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{label}  (AUROC={auroc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUROC=0.500)')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'Fig 4. ROC: KL Rare Cell Burden vs Immunotherapy Response\n(n={len(patient_df)} patients)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'fig4_roc_test.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nSaved fig4_roc_test.png")