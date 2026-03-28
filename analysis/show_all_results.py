#!/usr/bin/env python3
"""Gather and display ALL GVAE-TME results in a clean summary."""

import json
import numpy as np
from pathlib import Path

OUTDIR = Path(__file__).parent.parent / 'outputs'

def load(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def sep(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def subsep(title):
    print(f"\n  --- {title} ---")

def fmt_p(p):
    if p == 0 or p < 1e-300:
        return "0.0"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"

# ══════════════════════════════════════════════════════════════════════
#  CV RESPONSE PREDICTION
# ══════════════════════════════════════════════════════════════════════
sep("CROSS-VALIDATION (Response Prediction)")

for dataset in ['melanoma', 'melanoma_cv', 'nsclc_ici', 'colorectal']:
    cv_path = OUTDIR / dataset / 'cv_results.json'
    if dataset == 'melanoma_cv':
        cv_path = OUTDIR / 'melanoma_cv' / 'cv_results.json'
    cv = load(cv_path)
    if not cv:
        continue
    name = cv.get('dataset', dataset)
    n = cv.get('n_patients', '?')
    cm = cv.get('cv_metrics', {})
    pm = cv.get('pooled_metrics', {})
    perm = cv.get('permutation_test', {})
    print(f"\n  {name.upper()} (n={n} patients)")
    if pm:
        print(f"    Pooled AUROC: {pm.get('auroc', 'N/A'):.3f}  AUPRC: {pm.get('auprc', 'N/A'):.3f}")
    if cm.get('auroc'):
        a = cm['auroc']
        print(f"    Mean AUROC:   {a.get('mean', 0):.3f} +/- {a.get('std', 0):.3f}")
        if a.get('per_fold'):
            print(f"    Per-fold AUROC: {a['per_fold']}")
    if cm.get('f1'):
        f = cm['f1']
        print(f"    Mean F1:      {f.get('mean', 0):.3f} +/- {f.get('std', 0):.3f}")
    if perm:
        print(f"    Permutation p: {fmt_p(perm.get('p_value', 1))}")

# ══════════════════════════════════════════════════════════════════════
#  UNSUPERVISED ANALYSIS (per dataset)
# ══════════════════════════════════════════════════════════════════════
sep("UNSUPERVISED ANALYSIS")

for dataset in ['melanoma', 'breast', 'nsclc_scrna', 'nsclc_visium', 'colorectal']:
    m = load(OUTDIR / dataset / 'metrics.json')
    if not m:
        continue
    subsep(dataset)
    p1 = m.get('phase1_metrics', {})
    if p1:
        print(f"    Phase 1: L_adj={p1.get('loss_adj', 0):.4f}, L_expr={p1.get('loss_expr', 0):.4f}")
    cl = m.get('clustering', {})
    print(f"    Clusters: {cl.get('n_clusters', '?')}, silhouette: {cl.get('silhouette', 0):.3f}")
    ari = cl.get('ari_stability', {})
    if ari:
        print(f"    ARI stability: {ari.get('mean_ari', 0):.3f} +/- {ari.get('std_ari', 0):.3f}")
    rc = m.get('rare_cells', {})
    print(f"    Rare cells: {rc.get('n_rare', '?')} (threshold={rc.get('threshold', 2.0)})")
    g = m.get('gate', {})
    print(f"    Gate: mean={g.get('mean', '?')}, std={g.get('std', '?')}")
    sp = m.get('spatial_validation', {})
    if sp and sp.get('morans_i'):
        print(f"    Moran's I (gate): {sp['morans_i']:.4f}")
    bm = m.get('batch_mixing', {})
    if bm:
        print(f"    Batch mixing: kBET rejection={bm.get('rejection_rate', '?')}, entropy={bm.get('batch_entropy', bm.get('entropy', '?'))}")
    att = m.get('attention', {})
    if att and att.get('selectivity'):
        sel = att['selectivity']
        if isinstance(sel, dict):
            print(f"    Attention selectivity: {sel.get('mean', 0):.3f} +/- {sel.get('std', 0):.3f}")
        else:
            print(f"    Attention selectivity: {sel}")
    sigs = rc.get('immunosuppressive_signatures', m.get('immunosuppressive_signatures', {}))
    if sigs and isinstance(sigs, dict) and 'note' not in str(sigs):
        print(f"    Immunosuppressive signatures:")
        for sig_name, sig_data in sigs.items():
            if isinstance(sig_data, dict) and 'p_value' in sig_data:
                stars = '***' if sig_data['p_value'] < 0.001 else '**' if sig_data['p_value'] < 0.01 else '*' if sig_data['p_value'] < 0.05 else ''
                print(f"      {sig_name}: effect={sig_data.get('effect_size', 0):.3f}, p={fmt_p(sig_data['p_value'])} {stars}")
    lr = m.get('ligand_receptor', {})
    if lr and lr.get('n_interactions', 0) > 0:
        print(f"    L-R interactions: {lr['n_interactions']} scored, {lr.get('n_valid_pairs', 0)} valid pairs")
        top = lr.get('top_interactions', [])
        for t in top[:3]:
            if isinstance(t, dict):
                print(f"      {t.get('ligand', '?')}-{t.get('receptor', '?')} ({t.get('source', '?')}->{t.get('target', '?')}): {t.get('score', 0):.4f}")
    rare_sub = rc.get('rare_subclusters', {})
    if rare_sub:
        n_sub = len([k for k in rare_sub if str(k).startswith('10')])
        if n_sub > 0:
            print(f"    Rare subclusters: {n_sub} with marker genes")
    ct = rc.get('celltypist_validation', {})
    if ct and 'note' not in ct:
        overall = ct.get('overall', {})
        print(f"    CellTypist validation:")
        print(f"      Overall precision: {overall.get('precision', 0):.3f}")
        print(f"      Overall recall: {overall.get('recall', 0):.3f}")
        print(f"      N immunosuppressive: {overall.get('n_immunosuppressive', 0)}")
        print(f"      N rare+immuno: {overall.get('n_rare_immunosuppressive', 0)}")
        for sig in ['Treg', 'M2_macrophage', 'MDSC', 'Exhausted_T']:
            sd = ct.get(sig, {})
            if sd and sd.get('n_annotated', 0) > 0:
                print(f"      {sig}: prec={sd.get('precision', 0):.3f}, recall={sd.get('recall', 0):.3f}, n={sd['n_annotated']}")

# ══════════════════════════════════════════════════════════════════════
#  PREDICTION (single split per dataset)
# ══════════════════════════════════════════════════════════════════════
sep("PREDICTION (Single Train/Test Split)")

for dataset in ['melanoma', 'breast', 'nsclc_scrna', 'nsclc_visium', 'colorectal']:
    m = load(OUTDIR / dataset / 'metrics.json')
    if not m or not m.get('prediction'):
        continue
    p = m['prediction']
    if p.get('auroc') is None:
        continue
    print(f"\n  {dataset}")
    print(f"    AUROC: {p.get('auroc', 0):.3f}  AUPRC: {p.get('auprc', 0):.3f}")
    print(f"    F1: {p.get('f1', 0):.3f}  Precision: {p.get('precision', 0):.3f}  Recall: {p.get('recall', 0):.3f}")

# ══════════════════════════════════════════════════════════════════════
#  ABLATION STUDIES
# ══════════════════════════════════════════════════════════════════════
sep("ABLATION STUDIES (Melanoma)")

ablation_names = [
    'mol_only', 'spatial_only', 'static_0.3', 'static_0.5', 'static_0.7',
    'no_expr', 'gaussian', 'no_contrastive', 'frozen_encoder', 'gcn_encoder',
    'rare_leiden', 'logreg_baseline', 'spatial_bias',
]

print(f"\n  {'Ablation':<25s} {'AUROC':>7s} {'Clusters':>9s} {'Silhouette':>11s} {'Rare':>6s} {'Gate':>6s}")
print(f"  {'-'*25} {'-'*7} {'-'*9} {'-'*11} {'-'*6} {'-'*6}")

base = load(OUTDIR / 'melanoma' / 'metrics.json')
if base:
    p = base.get('prediction', {})
    cl = base.get('clustering', {})
    rc = base.get('rare_cells', {})
    g = base.get('gate', {})
    print(f"  {'FULL (baseline)':<25s} {p.get('auroc', 0):>7.3f} {cl.get('n_clusters', ''):>9} {cl.get('silhouette', 0):>11.3f} {rc.get('n_rare', ''):>6} {g.get('mean', 0):>6.3f}")

for abl in ablation_names:
    for prefix in ['melanoma_', 'synthetic_', '']:
        m = load(OUTDIR / f'{prefix}{abl}' / 'metrics.json')
        if m:
            break
    if not m:
        continue
    p = m.get('prediction', {})
    cl = m.get('clustering', {})
    rc = m.get('rare_cells', {})
    g = m.get('gate', {})
    print(f"  {abl:<25s} {p.get('auroc', 0):>7.3f} {cl.get('n_clusters', ''):>9} {cl.get('silhouette', 0):>11.3f} {rc.get('n_rare', ''):>6} {g.get('mean', 0):>6.3f}")

# ══════════════════════════════════════════════════════════════════════
#  BENCHMARK: GVAE vs scVI vs Scanpy
# ══════════════════════════════════════════════════════════════════════
sep("BENCHMARK: GVAE vs scVI vs Scanpy (L1-LogReg on embeddings)")

for dataset in ['melanoma', 'nsclc_ici']:
    bench_dir = OUTDIR / f'{dataset}_benchmark'
    comp = load(bench_dir / 'comparison.json')
    if comp:
        print(f"\n  {dataset.upper()}:")
        for method, vals in comp.items():
            if isinstance(vals, dict) and 'auroc' in vals:
                print(f"    {method:<12s} AUROC: {vals['auroc']:.3f}  AUPRC: {vals.get('auprc', 0):.3f}")

# ══════════════════════════════════════════════════════════════════════
#  CROSS-DATASET TRANSFER
# ══════════════════════════════════════════════════════════════════════
sep("CROSS-DATASET TRANSFER")

for transfer_dir in sorted(OUTDIR.glob('*_to_*')):
    for results_file in ['transfer_joint_results.json', 'results.json', 'transfer_results.json']:
        r = load(transfer_dir / 'transfer_joint' / results_file)
        if r:
            break
    if not r:
        continue
    src = r.get('source_dataset', '?')
    tgt = r.get('target_dataset', '?')
    print(f"\n  {src} -> {tgt}")
    print(f"    Gene set: {r.get('gene_set', '?')}")
    print(f"    Shared genes: {r.get('n_shared_genes', '?')}")
    print(f"    Source cells: {r.get('source_n_cells', '?')}")
    print(f"    Target cells: {r.get('target_n_cells', '?')}")
    tc = r.get('target_clustering', {})
    print(f"    Target clusters: {tc.get('n_clusters', '?')}, silhouette: {tc.get('silhouette', 0):.4f}")
    tr = r.get('target_rare_cells', {})
    print(f"    Target rare cells: {tr.get('n_rare', '?')} ({tr.get('fraction', 0)*100:.1f}%)")
    mc = r.get('marker_concordance', {})
    if mc:
        print(f"    Marker concordance: Jaccard={mc.get('jaccard', 0):.3f} ({mc.get('n_overlap', 0)} shared / {mc.get('n_source_markers', 0)+mc.get('n_target_markers', 0)-mc.get('n_overlap', 0)} total)")
        overlap = mc.get('overlap_genes', [])
        if overlap:
            print(f"    Shared genes: {', '.join(overlap[:15])}")
            if len(overlap) > 15:
                print(f"                  ... and {len(overlap)-15} more")

# ══════════════════════════════════════════════════════════════════════
#  SPATIAL INTEGRATION
# ══════════════════════════════════════════════════════════════════════
sep("SPATIAL INTEGRATION (Key Result)")

print(f"\n  {'Dataset':<18s} {'Gate Mean':>10s} {'Moran I':>10s} {'Interpretation'}")
print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*30}")
for dataset in ['nsclc_scrna', 'breast', 'nsclc_visium', 'colorectal']:
    m = load(OUTDIR / dataset / 'metrics.json')
    if not m:
        continue
    g = m.get('gate', {})
    sp = m.get('spatial_validation', {})
    gate_mean = g.get('mean', None)
    morans = sp.get('morans_i', None)
    if gate_mean is not None and gate_mean > 0.5:
        interp = "No spatial -> molecular only"
    elif gate_mean is not None:
        pct = (1 - gate_mean) * 100
        interp = f"{pct:.0f}% spatial, {gate_mean*100:.0f}% molecular"
    else:
        interp = "N/A"
    gate_str = f"{gate_mean:.3f}" if gate_mean is not None else "N/A"
    morans_str = f"{morans:.3f}" if morans is not None else "N/A"
    print(f"  {dataset:<18s} {gate_str:>10s} {morans_str:>10s} {interp}")

# ══════════════════════════════════════════════════════════════════════
#  BATCH CONDITIONING COMPARISON
# ══════════════════════════════════════════════════════════════════════
sep("BATCH CONDITIONING (Before/After)")

for dataset in ['breast', 'melanoma']:
    m = load(OUTDIR / dataset / 'metrics.json')
    if not m:
        continue
    bm = m.get('batch_mixing', {})
    n_batches = bm.get('n_batches', '?')
    print(f"\n  {dataset} (n_batches={n_batches})")
    print(f"    kBET rejection: {bm.get('rejection_rate', 'N/A')}")
    print(f"    Batch entropy:  {bm.get('batch_entropy', bm.get('entropy', 'N/A'))}")

# ══════════════════════════════════════════════════════════════════════
#  RARE CELL VALIDATION SUMMARY
# ══════════════════════════════════════════════════════════════════════
sep("RARE CELL VALIDATION (CellTypist)")

for dataset in ['melanoma', 'breast']:
    m = load(OUTDIR / dataset / 'metrics.json')
    if not m:
        continue
    rc = m.get('rare_cells', {})
    ct = rc.get('celltypist_validation', {})
    if not ct or 'note' in ct:
        print(f"\n  {dataset}: {ct.get('note', 'not available')}")
        continue
    overall = ct.get('overall', {})
    print(f"\n  {dataset.upper()}")
    print(f"    N rare: {rc.get('n_rare', '?')}")
    print(f"    N immunosuppressive: {overall.get('n_immunosuppressive', '?')}")
    print(f"    N rare+immuno: {overall.get('n_rare_immunosuppressive', '?')}")
    print(f"    Overall precision: {overall.get('precision', 0):.3f}")
    print(f"    Overall recall: {overall.get('recall', 0):.3f}")
    for sig in ['Treg', 'M2_macrophage', 'MDSC', 'Exhausted_T']:
        sd = ct.get(sig, {})
        if sd and sd.get('n_annotated', 0) > 0:
            print(f"    {sig:<20s} prec={sd.get('precision', 0):.3f}  recall={sd.get('recall', 0):.3f}  n_annotated={sd['n_annotated']}")

# ══════════════════════════════════════════════════════════════════════
#  OPTUNA HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════
sep("OPTUNA HYPERPARAMETER TUNING")

for dataset in ['melanoma_tune', 'synthetic_tune']:
    study = load(OUTDIR / dataset / 'study.json')
    if not study:
        continue
    print(f"\n  {dataset}")
    best = study.get('best_params', study.get('best_config', {}))
    if best:
        print(f"    Best config: {json.dumps(best, indent=None)}")
    best_val = study.get('best_value', None)
    if best_val is not None:
        print(f"    Best objective: {best_val:.4f}")

# ══════════════════════════════════════════════════════════════════════
sep("END OF GVAE-TME RESULTS SUMMARY")
