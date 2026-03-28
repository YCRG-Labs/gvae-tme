#!/usr/bin/env python3
"""Gather and display ALL GVAE-TME results — every metric from every output directory."""

import json, os, glob
import numpy as np
from pathlib import Path

OUTDIR = Path(__file__).parent.parent / 'outputs'

def load(path):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

def sep(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def fmt_p(p):
    if p is None: return "N/A"
    if p == 0 or p < 1e-300: return "0.0"
    if p < 0.001: return f"{p:.2e}"
    return f"{p:.4f}"

def stars(p):
    if p is None: return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""

def safe(d, *keys, default="N/A"):
    v = d
    for k in keys:
        if isinstance(v, dict):
            v = v.get(k)
        else:
            return default
    if v is None: return default
    return v

# ══════════════════════════════════════════════════════════════════════
sep("CROSS-VALIDATION (Response Prediction)")
# ══════════════════════════════════════════════════════════════════════

for name, path in [("Melanoma", "melanoma_cv"), ("NSCLC ICI", "nsclc_ici_cv"), ("Colorectal", "colorectal_cv")]:
    r = load(OUTDIR / path / "cv_results.json")
    if not r: continue
    pm = r.get("pooled_metrics", {})
    pt = r.get("permutation_test", {})
    cm = r.get("cv_metrics", {})
    bs = r.get("bootstrap_ci", {})
    print(f"\n  {name} (n={r.get('n_patients','?')} patients)")
    print(f"    Pooled AUROC: {pm.get('auroc',0):.3f}  AUPRC: {pm.get('auprc',0):.3f}")
    a = cm.get('auroc', {})
    print(f"    Mean AUROC:   {a.get('mean',0):.3f} +/- {a.get('std',0):.3f}")
    f1 = cm.get('f1', {})
    print(f"    Mean F1:      {f1.get('mean',0):.3f} +/- {f1.get('std',0):.3f}")
    ci = bs.get('auroc_ci', [0,0])
    if ci and ci != [0,0]:
        print(f"    95% CI:       [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"    Permutation p: {fmt_p(pt.get('p_value'))}")
    if a.get('per_fold'):
        print(f"    Per-fold AUROC: {[round(x,3) for x in a['per_fold']]}")

# ══════════════════════════════════════════════════════════════════════
sep("UNSUPERVISED ANALYSIS")
# ══════════════════════════════════════════════════════════════════════

for ds in ["melanoma", "breast", "nsclc_scrna", "nsclc_visium", "colorectal", "nsclc_ici"]:
    r = load(OUTDIR / ds / "metrics.json")
    if not r: continue
    print(f"\n  {ds}")
    p1 = r.get("phase1_metrics", {})
    if p1:
        print(f"    Phase 1: L_adj={p1.get('loss_adj',0):.4f}, L_expr={p1.get('loss_expr',0):.4f}")
    cl = r.get("clustering", {})
    print(f"    Clusters: {cl.get('n_clusters','?')}, silhouette: {cl.get('silhouette',0):.3f}")
    ari = cl.get("ari_stability", {})
    if ari:
        print(f"    ARI stability: {ari.get('mean_ari',0):.3f} +/- {ari.get('std_ari',0):.3f}")
    rc = r.get("rare_cells", {})
    print(f"    Rare cells: {rc.get('n_rare','?')} (threshold={rc.get('threshold',2.0)})")
    g = r.get("gate", {})
    print(f"    Gate: mean={g.get('mean','?')}, std={g.get('std','?')}")

    sp = r.get("spatial_validation", {})
    if sp:
        mi = sp.get("morans_i", sp.get("permutation_test", {}).get("observed_I"))
        if mi: print(f"    Moran's I (gate): {mi}")
        perm = sp.get("permutation_test", {})
        if perm and perm.get("p_value") is not None:
            print(f"    Spatial perm test: p={fmt_p(perm['p_value'])}")

    bm = r.get("batch_mixing", {})
    if bm:
        print(f"    Batch mixing: kBET rejection={bm.get('rejection_rate','N/A')}, "
              f"entropy={bm.get('batch_entropy', bm.get('entropy','N/A'))}, "
              f"n_batches={bm.get('n_batches','?')}")

    att = r.get("attention", {})
    if att:
        sel = att.get("selectivity", att.get("mean_selectivity"))
        if isinstance(sel, dict):
            print(f"    Attention selectivity: {sel.get('mean',0):.3f} +/- {sel.get('std',0):.3f}")
        elif sel is not None:
            std_sel = att.get("std_selectivity", 0)
            print(f"    Attention selectivity: {sel:.3f} +/- {std_sel:.3f}")

    sigs = rc.get("immunosuppressive_signatures", r.get("immunosuppressive_signatures", {}))
    if sigs and isinstance(sigs, dict) and "note" not in str(sigs):
        print(f"    Immunosuppressive signatures:")
        for sig_name, sig_data in sigs.items():
            if isinstance(sig_data, dict) and "p_value" in sig_data:
                print(f"      {sig_name}: effect={sig_data.get('effect_size',0):.3f}, "
                      f"p={fmt_p(sig_data['p_value'])} {stars(sig_data['p_value'])}")

    lr = r.get("ligand_receptor", {})
    if lr and lr.get("n_interactions", 0) > 0:
        print(f"    L-R interactions: {lr['n_interactions']} scored, {lr.get('n_valid_pairs',0)} valid pairs")
        top = lr.get("interactions", lr.get("top_interactions", []))[:5]
        for t in top:
            if isinstance(t, dict):
                print(f"      {t.get('ligand','?')}-{t.get('receptor','?')} "
                      f"({t.get('source','?')}->{t.get('target','?')}): {t.get('score',0):.4f}")

    rare_markers = rc.get("rare_markers", {})
    if rare_markers:
        n_sub = len(rare_markers)
        print(f"    Rare subclusters: {n_sub} with marker genes")

    ct = rc.get("celltypist_validation", {})
    if ct and "note" not in ct:
        overall = ct.get("overall", {})
        print(f"    CellTypist validation:")
        print(f"      Overall precision: {overall.get('precision',0):.3f}")
        print(f"      Overall recall: {overall.get('recall',0):.3f}")
        print(f"      N immunosuppressive: {overall.get('n_immunosuppressive',0)}")
        print(f"      N rare+immuno: {overall.get('n_rare_immunosuppressive',0)}")
        for sig in ['Treg', 'M2_macrophage', 'MDSC', 'Exhausted_T']:
            sd = ct.get(sig, {})
            if sd and sd.get('n_annotated', 0) > 0:
                print(f"      {sig}: prec={sd.get('precision',0):.3f}, "
                      f"recall={sd.get('recall',0):.3f}, n={sd['n_annotated']}")

    pred = r.get("prediction", {})
    if pred and pred.get("auroc") is not None:
        print(f"    Prediction (single split): AUROC={pred['auroc']:.3f}, "
              f"AUPRC={pred.get('auprc',0):.3f}, F1={pred.get('f1',0):.3f}")

    ct_ann = r.get("cell_types", {})
    if ct_ann and isinstance(ct_ann, dict):
        n_types = ct_ann.get("n_types", len(ct_ann.get("cell_types", {})))
        if n_types:
            print(f"    CellTypist annotation: {n_types} cell types")

    logreg = r.get("logreg_baseline", {})
    if logreg and logreg.get("auroc") is not None:
        print(f"    LogReg baseline: AUROC={logreg['auroc']:.3f}, AUPRC={logreg.get('auprc',0):.3f}")

# ══════════════════════════════════════════════════════════════════════
sep("ABLATION STUDIES (Melanoma)")
# ══════════════════════════════════════════════════════════════════════

ablations = ["mol_only", "spatial_only", "static_0.3", "static_0.5", "static_0.7",
             "no_expr", "gaussian", "no_contrastive", "frozen_encoder", "gcn_encoder",
             "rare_leiden", "logreg_baseline", "spatial_bias"]

print(f"\n  {'Ablation':<25s} {'AUROC':>7s} {'Clusters':>9s} {'Silhouette':>11s} {'Rare':>6s} {'Gate':>6s}")
print(f"  {'-'*25} {'-'*7} {'-'*9} {'-'*11} {'-'*6} {'-'*6}")

base_cv = load(OUTDIR / "melanoma_cv" / "cv_results.json")
if base_cv:
    print(f"  {'FULL (baseline)':<25s} {base_cv['pooled_metrics']['auroc']:>7.3f} {'--':>9s} {'--':>11s} {'--':>6s} {'--':>6s}")

for abl in ablations:
    for prefix in ["melanoma_", ""]:
        r = load(OUTDIR / f"{prefix}{abl}" / "metrics.json")
        if r: break
    if not r: continue
    p = r.get("prediction", {})
    cl = r.get("clustering", {})
    rc = r.get("rare_cells", {})
    g = r.get("gate", {})
    auroc = p.get("auroc", 0)
    auroc_s = f"{auroc:.3f}" if isinstance(auroc, (int, float)) else str(auroc)
    sil = cl.get("silhouette", 0)
    gate = g.get("mean", 0)
    gate_s = f"{gate:.3f}" if isinstance(gate, (int, float)) else str(gate)
    print(f"  {abl:<25s} {auroc_s:>7s} {str(cl.get('n_clusters','')):>9s} {sil:>11.3f} {str(rc.get('n_rare','')):>6s} {gate_s:>6s}")

# ══════════════════════════════════════════════════════════════════════
sep("BENCHMARK: GVAE vs scVI vs Scanpy (L1-LogReg on embeddings)")
# ══════════════════════════════════════════════════════════════════════

for ds_name, ds_key in [("Melanoma", "melanoma"), ("NSCLC ICI", "nsclc_ici")]:
    for suffix in ["_benchmark_cv", "_benchmark"]:
        r = load(OUTDIR / f"{ds_key}{suffix}" / "benchmark_cv_results.json")
        if not r:
            r = load(OUTDIR / f"{ds_key}{suffix}" / "comparison.json")
        if r: break
    if not r:
        log_path = OUTDIR / f"benchmark_{ds_key}.log"
        if log_path.exists():
            print(f"\n  {ds_name} (from log):")
            with open(log_path) as f:
                for line in f:
                    if "AUROC=" in line and "+/-" in line:
                        print(f"    {line.strip()}")
        continue
    cv = r.get("cv_results", r)
    print(f"\n  {ds_name} (n={r.get('n_patients','?')} patients, {r.get('n_folds','?')}-fold CV):")
    print(f"    {'Method':<12s} {'AUROC':>14s} {'AUPRC':>14s}")
    print(f"    {'-'*12} {'-'*14} {'-'*14}")
    for method in ["gvae", "scvi", "scanpy"]:
        data = cv.get(method, {})
        if data:
            am = data.get("auroc_mean", 0)
            astd = data.get("auroc_std", 0)
            pm = data.get("auprc_mean", 0)
            pstd = data.get("auprc_std", 0)
            print(f"    {method:<12s} {am:.3f}+/-{astd:.3f}  {pm:.3f}+/-{pstd:.3f}")

print("\n  Note: All methods use L1-logistic regression on extracted features.")
print("  GVAE end-to-end (attention pooling) results are in the CV section above.")

# ══════════════════════════════════════════════════════════════════════
sep("CROSS-DATASET TRANSFER")
# ══════════════════════════════════════════════════════════════════════

for transfer_dir in sorted(OUTDIR.glob("*_to_*")):
    for sub in ["transfer_joint", ""]:
        for fname in ["transfer_joint_results.json", "results.json", "transfer_results.json"]:
            r = load(transfer_dir / sub / fname) if sub else load(transfer_dir / fname)
            if r: break
        if r: break
    if not r:
        te_r = load(transfer_dir / "transfer_results.json")
        if te_r:
            r = te_r
    if not r:
        print(f"\n  {transfer_dir.name}: no results found")
        continue
    src = r.get("source_dataset", r.get("source", "?"))
    tgt = r.get("target_dataset", r.get("target", "?"))
    print(f"\n  {src} -> {tgt}")
    print(f"    Gene set: {r.get('gene_set', 'N/A')}")
    print(f"    Shared genes: {r.get('n_shared_genes', '?')}")
    print(f"    Source cells: {r.get('source_n_cells', '?')}")
    print(f"    Target cells: {r.get('target_n_cells', '?')}")
    tc = r.get("target_clustering", r.get("transfer_evaluation", {}).get("clustering", {}))
    if tc:
        print(f"    Target clusters: {tc.get('n_clusters','?')}, silhouette: {tc.get('silhouette',0):.4f}")
    tr = r.get("target_rare_cells", r.get("transfer_evaluation", {}).get("rare_cells", {}))
    if tr:
        frac = tr.get("fraction", 0)
        print(f"    Target rare cells: {tr.get('n_rare','?')} ({frac*100:.1f}%)")
    mc = r.get("marker_concordance", {})
    if mc:
        print(f"    Marker concordance: Jaccard={mc.get('jaccard',0):.3f} "
              f"({mc.get('n_overlap',0)} shared / "
              f"{mc.get('n_source_markers',0)+mc.get('n_target_markers',0)-mc.get('n_overlap',0)} total)")
        overlap = mc.get("overlap_genes", [])
        if overlap:
            for i in range(0, len(overlap), 10):
                prefix = "    Shared marker genes: " if i == 0 else "                        "
                print(f"{prefix}{', '.join(overlap[i:i+10])}")

# ══════════════════════════════════════════════════════════════════════
sep("SPIKE-IN VALIDATION (Rare Cell Recovery)")
# ══════════════════════════════════════════════════════════════════════

spikein = load(OUTDIR / "melanoma_spike_in" / "spike_in_results.json")
if spikein:
    trials = spikein.get("results", [])
    print(f"\n  {'Fraction':>8s} {'Effect':>7s} {'GVAE P':>8s} {'GVAE R':>8s} {'GVAE F1':>8s} {'Scanpy P':>9s} {'Scanpy R':>9s} {'Scanpy F1':>10s}")
    print(f"  {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*10}")
    for t in trials:
        gvae = t.get("gvae", {})
        scanpy = t.get("scanpy", {})
        print(f"  {t.get('fraction',0):>8.2f} {t.get('effect_size',0):>7.1f} "
              f"{gvae.get('precision',0):>8.3f} {gvae.get('recall',0):>8.3f} {gvae.get('f1',0):>8.3f} "
              f"{scanpy.get('precision',0):>9.3f} {scanpy.get('recall',0):>9.3f} {scanpy.get('f1',0):>10.3f}")
else:
    log_path = OUTDIR / "spikein.log"
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if any(k in line for k in ["GVAE:", "Scanpy:", "scVI:", "fraction", "effect"]):
                    print(f"  {line}")
    else:
        print("  Not available")

# ══════════════════════════════════════════════════════════════════════
sep("SPATIAL INTEGRATION (Key Result)")
# ══════════════════════════════════════════════════════════════════════

print(f"\n  {'Dataset':<18s} {'Gate Mean':>10s} {'Moran I':>10s} {'Interpretation'}")
print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*30}")
for ds in ['nsclc_scrna', 'breast', 'nsclc_visium', 'colorectal']:
    r = load(OUTDIR / ds / "metrics.json")
    if not r: continue
    g = r.get("gate", {})
    sp = r.get("spatial_validation", {})
    gate_mean = g.get("mean")
    morans = sp.get("morans_i", sp.get("permutation_test", {}).get("observed_I"))
    if gate_mean is not None and gate_mean > 0.5:
        interp = "No spatial -> molecular only"
    elif gate_mean is not None:
        pct = (1 - gate_mean) * 100
        interp = f"{pct:.0f}% spatial, {gate_mean*100:.0f}% molecular"
    else:
        interp = "N/A"
    gate_s = f"{gate_mean:.3f}" if gate_mean is not None else "N/A"
    morans_s = f"{morans:.3f}" if morans is not None else "N/A"
    print(f"  {ds:<18s} {gate_s:>10s} {morans_s:>10s} {interp}")

# ══════════════════════════════════════════════════════════════════════
sep("BATCH CONDITIONING")
# ══════════════════════════════════════════════════════════════════════

for ds in ['breast', 'melanoma']:
    r = load(OUTDIR / ds / "metrics.json")
    if not r: continue
    bm = r.get("batch_mixing", {})
    print(f"\n  {ds} (n_batches={bm.get('n_batches','?')})")
    print(f"    kBET rejection: {bm.get('rejection_rate', 'N/A')}")
    print(f"    Batch entropy:  {bm.get('batch_entropy', bm.get('entropy', 'N/A'))}")

# ══════════════════════════════════════════════════════════════════════
sep("RARE CELL VALIDATION (CellTypist)")
# ══════════════════════════════════════════════════════════════════════

for ds in ['melanoma', 'breast']:
    r = load(OUTDIR / ds / "metrics.json")
    if not r: continue
    rc = r.get("rare_cells", {})
    ct = rc.get("celltypist_validation", {})
    if not ct or "note" in ct:
        print(f"\n  {ds}: {ct.get('note', 'not available') if ct else 'not available'}")
        continue
    overall = ct.get("overall", {})
    print(f"\n  {ds.upper()}")
    print(f"    N rare: {rc.get('n_rare', '?')}")
    print(f"    N immunosuppressive: {overall.get('n_immunosuppressive', '?')}")
    print(f"    N rare+immuno: {overall.get('n_rare_immunosuppressive', '?')}")
    print(f"    Overall precision: {overall.get('precision', 0):.3f}")
    print(f"    Overall recall: {overall.get('recall', 0):.3f}")
    for sig in ['Treg', 'M2_macrophage', 'MDSC', 'Exhausted_T']:
        sd = ct.get(sig, {})
        if sd and sd.get('n_annotated', 0) > 0:
            print(f"    {sig:<20s} prec={sd.get('precision',0):.3f}  recall={sd.get('recall',0):.3f}  n={sd['n_annotated']}")

# ══════════════════════════════════════════════════════════════════════
sep("OPTUNA HYPERPARAMETER TUNING")
# ══════════════════════════════════════════════════════════════════════

for ds in ["melanoma_tune", "synthetic_tune"]:
    for fname in ["tune_results.json", "study.json"]:
        r = load(OUTDIR / ds / fname)
        if r: break
    if not r: continue
    best = r.get("best_config", r.get("best_params", {}))
    best_val = r.get("best_value", r.get("best_val_loss"))
    print(f"\n  {ds}")
    if best_val is not None:
        print(f"    Best objective: {best_val}")
    if best:
        print(f"    Best config: {json.dumps(best)}")

# ══════════════════════════════════════════════════════════════════════
sep("OUTPUT FILE INVENTORY")
# ══════════════════════════════════════════════════════════════════════

total_json = 0
total_npy = 0
total_h5ad = 0
total_pt = 0
for d in sorted(OUTDIR.iterdir()):
    if not d.is_dir(): continue
    jsons = list(d.rglob("*.json"))
    npys = list(d.rglob("*.npy"))
    h5ads = list(d.rglob("*.h5ad"))
    pts = list(d.rglob("*.pt"))
    if not (jsons or npys or h5ads or pts): continue
    total_json += len(jsons)
    total_npy += len(npys)
    total_h5ad += len(h5ads)
    total_pt += len(pts)
    size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
    print(f"  {d.name:<35s} {len(jsons):>3d} json  {len(npys):>3d} npy  {len(h5ads):>2d} h5ad  {len(pts):>2d} pt  ({size_mb:.0f} MB)")

print(f"\n  TOTAL: {total_json} json, {total_npy} npy, {total_h5ad} h5ad, {total_pt} pt")

# ══════════════════════════════════════════════════════════════════════
sep("END OF GVAE-TME RESULTS SUMMARY")
# ══════════════════════════════════════════════════════════════════════
