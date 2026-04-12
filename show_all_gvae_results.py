#!/usr/bin/env python3
import json, os, glob
from pathlib import Path

HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, "gvae-tme")
RESULTS = os.path.join(BASE, "outputs")

def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

section("CROSS-VALIDATION (Response Prediction)")
for name, path in [("Melanoma", "melanoma_cv"), ("NSCLC ICI", "nsclc_ici_cv"), ("Colorectal", "colorectal_cv")]:
    fp = os.path.join(RESULTS, path, "cv_results.json")
    try:
        r = json.load(open(fp))
        pm = r["pooled_metrics"]
        pt = r["permutation_test"]
        cm = r["cv_metrics"]
        bs = r.get("bootstrap_ci", {})
        print(f"\n  {name} (n={r['n_patients']} patients)")
        print(f"    Pooled AUROC: {pm['auroc']:.3f}  AUPRC: {pm['auprc']:.3f}")
        print(f"    Mean AUROC:   {cm['auroc']['mean']:.3f} +/- {cm['auroc']['std']:.3f}")
        print(f"    Mean F1:      {cm['f1']['mean']:.3f} +/- {cm['f1']['std']:.3f}")
        print(f"    95% CI:       [{bs.get('auroc_ci', [0,0])[0]:.3f}, {bs.get('auroc_ci', [0,0])[1]:.3f}]")
        print(f"    Permutation p: {pt['p_value']}")
        print(f"    Per-fold AUROC: {[round(x,3) for x in cm['auroc']['per_fold']]}")
    except:
        print(f"\n  {name}: not available")

section("UNSUPERVISED ANALYSIS")
for ds in ["nsclc_scrna", "breast", "nsclc_visium"]:
    fp = os.path.join(RESULTS, ds, "metrics.json")
    try:
        r = json.load(open(fp))
        c = r["clustering"]
        g = r["gate"]
        rare = r["rare_cells"]
        bm = r.get("batch_mixing", {})
        sv = r.get("spatial_validation", {})
        attn = r.get("attention", {})
        p1 = r.get("phase1_metrics", {})
        print(f"\n  {ds}")
        if p1:
            print(f"    Phase 1: L_adj={p1.get('loss_adj',0):.4f}, L_expr={p1.get('loss_expr',0):.4f}")
        print(f"    Clusters: {c['n_clusters']}, silhouette: {c['silhouette']:.3f}")
        print(f"    ARI stability: {c['ari_stability']['mean_ari']:.3f} +/- {c['ari_stability']['std_ari']:.3f}")
        print(f"    Rare cells: {rare['n_rare']} (threshold={rare['threshold']})")
        print(f"    Gate: mean={g['mean']:.3f}, std={g['std']:.3f}")
        if sv:
            mi = sv.get("morans_i", sv.get("permutation_test", {}).get("observed_I", "N/A"))
            print(f"    Moran's I (gate): {mi}")
        if bm:
            print(f"    Batch mixing: kBET rejection={bm.get('rejection_rate','N/A')}, entropy={bm.get('batch_entropy','N/A'):.3f}")
        if attn:
            print(f"    Attention selectivity: {attn.get('mean_selectivity', 'N/A'):.3f} +/- {attn.get('std_selectivity', 'N/A'):.3f}")
        sigs = r.get("rare_cells", {}).get("immunosuppressive_signatures", {})
        if sigs:
            print(f"    Immunosuppressive signatures:")
            for sig_name, sig_data in sigs.items():
                if isinstance(sig_data, dict) and "p_value" in sig_data:
                    p = sig_data["p_value"]
                    eff = sig_data.get("effect_size", 0)
                    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    print(f"      {sig_name}: effect={eff:.3f}, p={p:.2e} {stars}")
        lr = r.get("ligand_receptor", {})
        if lr:
            n_int = lr.get("n_interactions", 0)
            n_valid = lr.get("n_valid_pairs", 0)
            print(f"    L-R interactions: {n_int} scored, {n_valid} valid pairs")
            top3 = lr.get("interactions", [])[:3]
            for i in top3:
                print(f"      {i.get('ligand','?')}-{i.get('receptor','?')} ({i.get('source','?')}->{i.get('target','?')}): {i.get('score',0):.4f}")
        rare_markers = r.get("rare_cells", {}).get("rare_markers", {})
        if rare_markers:
            n_subclusters = len(rare_markers)
            print(f"    Rare subclusters: {n_subclusters} with marker genes")
    except Exception as e:
        print(f"\n  {ds}: not available ({e})")

section("ABLATION STUDIES (Melanoma)")
ablations = ["mol_only", "spatial_only", "static_0.3", "static_0.5", "static_0.7",
             "no_expr", "gaussian", "no_contrastive", "frozen_encoder", "gcn_encoder",
             "rare_leiden", "logreg_baseline"]
print(f"\n  {'Ablation':<25s} {'AUROC':>7s} {'Clusters':>9s} {'Silhouette':>11s} {'Rare':>6s} {'Gate':>6s}")
print(f"  {'-'*25} {'-'*7} {'-'*9} {'-'*11} {'-'*6} {'-'*6}")

base_fp = os.path.join(RESULTS, "melanoma_cv", "cv_results.json")
try:
    r = json.load(open(base_fp))
    print(f"  {'FULL (baseline)':<25s} {r['pooled_metrics']['auroc']:>7.3f} {'--':>9s} {'--':>11s} {'--':>6s} {'--':>6s}")
except:
    pass

for abl in ablations:
    fp = os.path.join(RESULTS, f"melanoma_{abl}", "metrics.json")
    try:
        r = json.load(open(fp))
        auroc = r.get("prediction", {}).get("auroc", "N/A")
        if isinstance(auroc, (int, float)):
            auroc_str = f"{auroc:.3f}"
        else:
            auroc_str = str(auroc)
        n_clust = r.get("clustering", {}).get("n_clusters", "N/A")
        sil = r.get("clustering", {}).get("silhouette", 0)
        n_rare = r.get("rare_cells", {}).get("n_rare", "N/A")
        gate = r.get("gate", {}).get("mean", "N/A")
        gate_str = f"{gate:.3f}" if isinstance(gate, (int, float)) else str(gate)
        print(f"  {abl:<25s} {auroc_str:>7s} {str(n_clust):>9s} {sil:>11.3f} {str(n_rare):>6s} {gate_str:>6s}")
    except:
        print(f"  {abl:<25s} {'N/A':>7s}")

section("BENCHMARK: GVAE vs scVI vs Scanpy (L1-LogReg on embeddings)")
for ds_name, ds_key in [("Melanoma", "melanoma"), ("NSCLC ICI", "nsclc_ici")]:
    bench_dir = os.path.join(RESULTS, f"{ds_key}_benchmark_cv")
    results_file = os.path.join(bench_dir, "benchmark_cv_results.json")
    if os.path.exists(results_file):
        r = json.load(open(results_file))
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
    else:
        log_file = os.path.join(RESULTS, f"benchmark_{ds_key}.log")
        if os.path.exists(log_file):
            print(f"\n  {ds_name} (from log):")
            with open(log_file) as f:
                for line in f:
                    line = line.strip()
                    if "AUROC=" in line and "+/-" in line:
                        print(f"    {line}")

print("\n  Note: All methods use L1-logistic regression on extracted features.")
print("  GVAE end-to-end (attention pooling) results are in the CV section above.")

section("CROSS-DATASET TRANSFER (Melanoma -> NSCLC)")
transfer_dir = os.path.join(RESULTS, "melanoma_to_nsclc_ici_transfer")
transfer_json = os.path.join(transfer_dir, "transfer_results.json")
if os.path.exists(transfer_json):
    r = json.load(open(transfer_json))
    te = r.get("transfer_evaluation", {})
    print(f"\n  Source: {r.get('source','?')} -> Target: {r.get('target','?')}")
    print(f"  Shared genes: {r.get('n_shared_genes','?')}")
    tc = te.get("clustering", {})
    tr = te.get("rare_cells", {})
    if tc:
        print(f"  Target clustering: {tc.get('n_clusters','?')} clusters, silhouette={tc.get('silhouette',0):.4f}")
    if tr:
        print(f"  Target rare cells: {tr.get('n_rare','?')}")
else:
    log_path = os.path.join(RESULTS, "transfer.log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            content = f.read()
        if "Transfer failed" in content:
            for line in content.strip().split('\n'):
                if "Transfer" in line or "shared genes" in line:
                    print(f"  {line.strip()}")
        elif "ambiguous" in content:
            print("  Transfer failed (flag error)")
        else:
            for line in content.strip().split('\n')[-5:]:
                if line.strip():
                    print(f"  {line.strip()}")
    else:
        print("  Not available")

section("SPIKE-IN VALIDATION (Rare Cell Recovery)")
spikein_json = os.path.join(RESULTS, "melanoma_spike_in", "spike_in_results.json")
if os.path.exists(spikein_json):
    r = json.load(open(spikein_json))
    trials = r.get("results", [])
    print(f"\n  {'Fraction':>8s} {'Effect':>7s} {'GVAE P':>8s} {'GVAE R':>8s} {'GVAE F1':>8s} {'Scanpy P':>9s} {'Scanpy R':>9s} {'Scanpy F1':>10s}")
    print(f"  {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*10}")
    for t in trials:
        gvae = t.get("gvae", {})
        scanpy = t.get("scanpy", {})
        gp = gvae.get("precision", 0)
        gr = gvae.get("recall", 0)
        gf = gvae.get("f1", 0)
        sp = scanpy.get("precision", 0)
        sr = scanpy.get("recall", 0)
        sf = scanpy.get("f1", 0)
        print(f"  {t['fraction']:>8.2f} {t['effect_size']:>7.1f} {gp:>8.3f} {gr:>8.3f} {gf:>8.3f} {sp:>9.3f} {sr:>9.3f} {sf:>10.3f}")
else:
    spikein_log = os.path.join(RESULTS, "spikein.log")
    if os.path.exists(spikein_log):
        with open(spikein_log) as f:
            for line in f:
                line = line.strip()
                if "GVAE:" in line or "Scanpy:" in line or "scVI:" in line:
                    print(f"  {line}")
    else:
        print("  Not available")

section("OPTUNA HYPERPARAMETER TUNING")
tune_file = os.path.join(RESULTS, "melanoma_tune", "tune_results.json")
if os.path.exists(tune_file):
    r = json.load(open(tune_file))
    best = r.get("best_config", r.get("best_params", {}))
    best_val = r.get("best_value", r.get("best_val_loss", "N/A"))
    print(f"\n  Best validation loss: {best_val}")
    print(f"  Best config:")
    for k, v in best.items():
        print(f"    {k}: {v}")
else:
    for f in glob.glob(os.path.join(RESULTS, "melanoma_tune", "*.json")):
        try:
            r = json.load(open(f))
            if "best_config" in r or "best_params" in r:
                best = r.get("best_config", r.get("best_params", {}))
                print(f"\n  Best config: {best}")
                break
        except:
            pass

section("SPATIAL INTEGRATION (Key Result)")
print(f"\n  {'Dataset':<18s} {'Gate Mean':>10s} {'Morans I':>10s} {'Interpretation':<30s}")
print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*30}")
for ds in ["nsclc_scrna", "breast", "nsclc_visium"]:
    fp = os.path.join(RESULTS, ds, "metrics.json")
    try:
        r = json.load(open(fp))
        gate_mean = r.get("gate", {}).get("mean", None)
        sv = r.get("spatial_validation", {})
        morans = sv.get("morans_i", sv.get("permutation_test", {}).get("observed_I", None))
        gate_str = f"{gate_mean:.3f}" if gate_mean is not None else "N/A"
        morans_str = f"{morans:.3f}" if morans is not None else "N/A"
        if gate_mean is not None and gate_mean > 0.9:
            interp = "No spatial -> molecular only"
        elif gate_mean is not None:
            spatial_pct = int((1 - gate_mean) * 100)
            interp = f"{spatial_pct}% spatial, {100-spatial_pct}% molecular"
        else:
            interp = ""
        print(f"  {ds:<18s} {gate_str:>10s} {morans_str:>10s} {interp:<30s}")
    except Exception:
        print(f"  {ds:<18s} {'N/A':>10s} {'N/A':>10s}")

section("PAPER FRAMING (data-driven)")
# Read actual values for the summary
cv_data = {}
for name, path in [("nsclc", "nsclc_ici_cv"), ("melanoma", "melanoma_cv"), ("colorectal", "colorectal_cv")]:
    fp = os.path.join(RESULTS, path, "cv_results.json")
    try:
        cv_data[name] = json.load(open(fp))
    except Exception:
        pass

print("\n  KEY RESULTS (from output files):\n")
print("  1. RESPONSE PREDICTION")
for name, label in [("nsclc", "NSCLC ICI"), ("melanoma", "Melanoma"), ("colorectal", "Colorectal")]:
    if name in cv_data:
        r = cv_data[name]
        pm = r["pooled_metrics"]
        pt = r.get("permutation_test", {})
        print(f"     - {label} (n={r.get('n_patients','?')}): "
              f"Pooled AUROC {pm['auroc']:.3f}, p={pt.get('p_value', 'N/A')}")

n_ablations = 0
for abl in ["mol_only", "spatial_only", "static_0.3", "static_0.5", "static_0.7",
            "no_expr", "gaussian", "no_contrastive", "frozen_encoder", "gcn_encoder",
            "rare_leiden", "logreg_baseline"]:
    fp = os.path.join(RESULTS, f"melanoma_{abl}", "metrics.json")
    if os.path.exists(fp):
        n_ablations += 1
print(f"\n  2. ABLATIONS: {n_ablations}/12 completed")
if n_ablations < 12:
    print("     WARNING: Not all ablations have been run yet. Re-run with --config full.")

print("=" * 70)
print("  END OF GVAE-TME RESULTS SUMMARY")
print("=" * 70)
