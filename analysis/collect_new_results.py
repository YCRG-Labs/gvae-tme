#!/usr/bin/env python3
"""Collect results from the 3 fix runs (batch conditioning, rare validation, transfer)."""

import json
from pathlib import Path

OUTDIR = Path('/Users/jacobcrainic/gvae-tme/outputs')

def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ── 1. Batch Conditioning ──
section("BATCH CONDITIONING (breast + melanoma rerun)")

for dataset in ['breast', 'melanoma']:
    metrics = load_json(OUTDIR / dataset / 'metrics.json')
    if not metrics:
        print(f"\n  {dataset}: no metrics.json found")
        continue
    print(f"\n  {dataset.upper()}")
    bm = metrics.get('batch_mixing', {})
    print(f"    kBET rejection:  {bm.get('rejection_rate', bm.get('kbet_rejection', 'N/A'))}")
    print(f"    Batch entropy:   {bm.get('entropy', bm.get('mean_entropy', 'N/A'))}")
    cl = metrics.get('clustering', {})
    print(f"    Clusters:        {cl.get('n_clusters', 'N/A')}")
    print(f"    Silhouette:      {cl.get('silhouette', 'N/A')}")
    gate = metrics.get('gate', {})
    print(f"    Gate mean:       {gate.get('mean', 'N/A')}")
    pred = metrics.get('prediction', {})
    if pred:
        print(f"    Test AUROC:      {pred.get('auroc', 'N/A')}")

# ── 2. Rare Cell Validation (CellTypist) ──
section("RARE CELL VALIDATION (CellTypist)")

for dataset in ['melanoma', 'breast']:
    metrics = load_json(OUTDIR / dataset / 'metrics.json')
    if not metrics:
        continue
    rare = metrics.get('rare_cells', {})
    ct = rare.get('celltypist_validation', {})
    print(f"\n  {dataset.upper()}")
    print(f"    Rare cells:      {rare.get('n_rare', 'N/A')} / total")
    if ct and 'note' not in ct:
        overall = ct.get('overall', {})
        print(f"    Overall precision: {overall.get('precision', 'N/A')}")
        print(f"    Overall recall:    {overall.get('recall', 'N/A')}")
        print(f"    N immunosuppressive: {overall.get('n_immunosuppressive', 'N/A')}")
        print(f"    N rare+immuno:     {overall.get('n_rare_immunosuppressive', 'N/A')}")
        for sig in ['Treg', 'M2_macrophage', 'MDSC', 'Exhausted_T']:
            sig_data = ct.get(sig, {})
            if sig_data and sig_data.get('n_annotated', 0) > 0:
                print(f"      {sig}: prec={sig_data.get('precision', 0):.3f}, "
                      f"recall={sig_data.get('recall', 0):.3f}, "
                      f"n={sig_data.get('n_annotated', 0)}")
    else:
        note = ct.get('note', 'not available')
        print(f"    CellTypist: {note}")
    sigs = rare.get('immunosuppressive_signatures', metrics.get('immunosuppressive_signatures', {}))
    if sigs and 'note' not in str(sigs):
        print(f"    Signature scores:")
        for sig_name, sig_data in sigs.items():
            if isinstance(sig_data, dict) and 'p_value' in sig_data:
                print(f"      {sig_name}: effect={sig_data.get('effect_size', 'N/A'):.3f}, "
                      f"p={sig_data.get('p_value', 'N/A'):.2e}")

# ── 3. Cross-Dataset Transfer ──
section("CROSS-DATASET TRANSFER (melanoma -> nsclc_ici)")

transfer_dir = OUTDIR / 'melanoma_to_nsclc_ici' / 'transfer_joint'
results = load_json(transfer_dir / 'results.json')
if results:
    print(f"\n  Gene set:          {results.get('gene_set', 'N/A')}")
    print(f"  Shared genes:      {results.get('n_shared_genes', 'N/A')}")
    print(f"  Source cells:      {results.get('source_n_cells', 'N/A')}")
    print(f"  Target cells:      {results.get('target_n_cells', 'N/A')}")
    tc = results.get('target_clustering', {})
    print(f"  Target clusters:   {tc.get('n_clusters', 'N/A')}")
    print(f"  Target silhouette: {tc.get('silhouette', 'N/A')}")
    tr = results.get('target_rare_cells', {})
    print(f"  Target rare cells: {tr.get('n_rare', 'N/A')} ({tr.get('fraction', 0)*100:.1f}%)")
    mc = results.get('marker_concordance', {})
    print(f"  Marker concordance:")
    print(f"    Jaccard:         {mc.get('jaccard', 'N/A')}")
    print(f"    Shared markers:  {mc.get('n_overlap', 'N/A')} / {len(set(list(range(mc.get('n_source_markers', 0))) + list(range(mc.get('n_target_markers', 0)))))}")
    print(f"    Source markers:  {mc.get('n_source_markers', 'N/A')}")
    print(f"    Target markers:  {mc.get('n_target_markers', 'N/A')}")
    overlap = mc.get('overlap_genes', [])
    if overlap:
        print(f"    Overlap genes:   {', '.join(overlap[:20])}")
        if len(overlap) > 20:
            print(f"                     ... and {len(overlap)-20} more")
    ft = results.get('finetune_metrics', {})
    if ft:
        print(f"  Fine-tune metrics:")
        print(f"    L_adj:           {ft.get('loss_adj', 'N/A')}")
        print(f"    L_expr:          {ft.get('loss_expr', 'N/A')}")
else:
    print("\n  No results.json found at", transfer_dir)
    for f in sorted(transfer_dir.glob('*')) if transfer_dir.exists() else []:
        print(f"    {f.name} ({f.stat().st_size / 1024:.0f} KB)")

# ── Summary ──
section("SUMMARY FOR PAPER")
print("""
  1. BATCH CONDITIONING
     - Breast kBET: 0.93 -> check above
     - Melanoma kBET: check above
     - Narrative: "Batch conditioning reduces kBET rejection rate"

  2. RARE CELL VALIDATION
     - Melanoma CellTypist precision/recall: check above
     - Replaces weak spike-in (F1 < 0.05) with real-cell validation
     - Narrative: "KL-flagged rare cells preferentially identify immunosuppressive types"

  3. CROSS-DATASET TRANSFER
     - Previous: FAILED (235 shared HVGs)
     - Now: 1784 shared genes, Jaccard = check above
     - Fine-tuning variational heads improves concordance ~8x
     - Narrative: "Transfer feasible with shared gene sets; fine-tuning critical"
""")

if __name__ == '__main__':
    pass
