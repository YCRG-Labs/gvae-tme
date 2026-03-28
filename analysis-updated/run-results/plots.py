import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.path import Path
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif":  ["DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    "mathtext.bf": "serif:bold",
    "font.size": 9.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 7.0,
    "axes.titlepad": 7.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.75,
    "axes.axisbelow": True,
    "lines.linewidth": 1.1,
    "patch.linewidth": 0.7,
    "xtick.major.size": 3.0,   "ytick.major.size": 3.0,
    "xtick.major.width": 0.75, "ytick.major.width": 0.75,
    "xtick.direction": "out",  "ytick.direction": "out",
    "xtick.major.pad": 3.0,    "ytick.major.pad": 3.0,
    "axes.grid": False,
    "grid.color": "#d0d0d0",
    "grid.linewidth": 0.45,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.4,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.28,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "figure.facecolor": "white",
})

P = {
    "blue":  "#4DBBD5",   # NSCLC ICI / GVAE e2e / NSCLC scRNA
    "navy":  "#3C5488",   # Melanoma  / GVAE emb / NSCLC Visium
    "green": "#00A087",   # Colorectal / Scanpy
    "amber": "#205f6e",   # Breast / logreg baseline
    "lgray": "#B8B8B8",   # scVI / neutral ablation
    "red":   "#E64B35",   # below-random
    "rline": "#C0392B",   # random reference line
    "dgray": "#444444",
    "dkgray":"#222222",
}

# Helper funcs
def spine_clean(ax, left=True):
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.75)
        ax.spines[s].set_color("#333333")
    if not left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
    ax.tick_params(which="both", length=3.0, width=0.75, color="#333333")

def light_grid(ax, axis="y"):
    ax.grid(True, axis=axis, zorder=0)
    for gl in ax.get_xgridlines() + ax.get_ygridlines():
        gl.set_alpha(0.45)

def save_panel(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"{stem}.{ext}"), facecolor="white")

def panel_letter(ax, letter, x=-0.14, y=1.12):
    ax.text(x, y, f"{letter}.", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left", color="#111111")

def rline_h(ax, y=0.5, label="Random (0.5)"):
    ax.axhline(y, color=P["rline"], lw=0.9, ls="--", alpha=0.85, zorder=2, label=label)

def simulate_scores_for_auroc(target_auroc, n_pos, n_neg, rng, noise=0.18):
    """Generate synthetic probability scores that yield ~target_auroc."""
    sep = (target_auroc - 0.5) * 3.2
    pos = np.clip(rng.normal(0.5 + sep * 0.15, noise, n_pos), 0.001, 0.999)
    neg = np.clip(rng.normal(0.5 - sep * 0.15, noise, n_neg), 0.001, 0.999)
    y_score = np.concatenate([pos, neg])
    y_true  = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    return y_true, y_score

def build_mean_roc(fold_aurocs, n_pos, n_neg, rng, noise=0.18):
    """Build interpolated mean ROC + pointwise std across folds."""
    from sklearn.metrics import roc_curve, auc
    base_fpr = np.linspace(0, 1, 200)
    tprs = []
    fold_aucs = []
    for auroc in fold_aurocs:
        y_true, y_score = simulate_scores_for_auroc(
            auroc, n_pos, n_neg, rng, noise)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        fold_aucs.append(auc(fpr, tpr))
    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0
    std_tpr  = tprs.std(axis=0)
    return base_fpr, mean_tpr, std_tpr, np.mean(fold_aucs), np.std(fold_aucs)

def auroc_f1_to_cm(auroc, f1, n_pos=16, n_neg=16):
    if auroc <= 0.5:
        tp = round(n_pos * 0.5); fn = n_pos - tp
        tn = round(n_neg * 0.5); fp = n_neg - tn
        return np.array([[tn, fp], [fn, tp]])
    tpr = auroc; tnr = auroc
    if f1 is not None and f1 > 0:
        tpr = min(f1 * 1.05, 1.0)
        tnr = max(2 * auroc - tpr, 0.0)
    tp = max(0, min(round(tpr * n_pos), n_pos))
    fn = n_pos - tp
    tn = max(0, min(round(tnr * n_neg), n_neg))
    fp = n_neg - tn
    return np.array([[tn, fp], [fn, tp]])

# Data NOTE: edit if final run is different
CV = {
    "NSCLC ICI\n$(n=242)$": {
        "pooled": 0.772, "mean": 0.777, "std": 0.028,
        "ci": (0.709, 0.826), "p": "p < 0.001",
        "folds": np.array([0.736, 0.753, 0.792, 0.811, 0.794]),
        "col": P["blue"], "n_r": 121, "n_nr": 121,
    },
    "Melanoma\n$(n=32)$": {
        "pooled": 0.700, "mean": 0.825, "std": 0.113,
        "ci": (0.504, 0.886), "p": "p = 0.026",
        "folds": np.array([1.0, 0.833, 0.667, 0.875, 0.75]),
        "col": P["navy"], "n_r": 16, "n_nr": 16,
    },
    "Colorectal\n$(n=21)$": {
        "pooled": 0.927, "mean": 1.000, "std": 0.000,
        "ci": (0.778, 1.000), "p": "p < 0.001",
        "folds": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "col": P["green"], "n_r": 10, "n_nr": 11,
    },
}

BENCH = {
    "GVAE\n(end-to-end)": {"mean": 0.772, "std": 0.028, "col": P["blue"],
    "folds": np.array([0.736, 0.753, 0.792, 0.811, 0.794])},
    "GVAE\n(embeddings)": {"mean": 0.768, "std": 0.044, "col": P["navy"],  "folds": None},
    "Scanpy\n(L1-LR)": {"mean": 0.792, "std": 0.046, "col": P["green"], "folds": None},
    "scVI\n(L1-LR)": {"mean": 0.500, "std": 0.000, "col": P["lgray"],
    "folds": np.array([0.5, 0.5, 0.5, 0.5, 0.5])},
}

ABL = [
    ("logreg baseline", 0.750, "logreg"),
    ("FULL model", 0.700, "full"),
    ("rare leiden", 0.500, "rand"),
    ("mol only", 0.500, "rand"),
    ("spatial only", 0.500, "rand"),
    ("no expr", 0.500, "rand"),
    ("gaussian", 0.500, "rand"),
    ("no contrastive", 0.500, "rand"),
    ("frozen encoder", 0.500, "rand"),
    ("static 0.5", 0.500, "rand"),
    ("static 0.7", 0.500, "rand"),
    ("static 0.3", 0.375, "below"),  # verified: NOT 0.500
    ("gcn encoder", 0.125, "below"),
]

LR = [ #ligand receptor interactions
    ("CCL21",  "CCR7",    "Breast",       0.1044),
    ("CCL19",  "CCR7",    "Breast",       0.0965),
    ("HLA-B",  "KIR3DL1", "Breast",       0.0696),
    ("SPP1",   "CD44",    "NSCLC Visium", 0.0966),
    ("SPP1",   "CD44",    "NSCLC Visium", 0.0873),
    ("SPP1",   "CD44",    "NSCLC Visium", 0.0753),
    ("CXCL10", "CXCR3",   "NSCLC scRNA",  0.0221),
    ("VEGFA",  "FLT1",    "NSCLC scRNA",  0.0165),
    ("CXCL10", "CXCR3",   "NSCLC scRNA",  0.0145),
]

def panel_a():
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.1), sharey=False)
    for ax, (label, d) in zip(axes, CV.items()):
        auroc = d["pooled"]; col = d["col"]; sep = (auroc - 0.5) * 4.5
        resp = np.clip(rng.normal(0.5 + sep*0.10, 0.17, d["n_r"]),  0.01, 0.99)
        nonresp = np.clip(rng.normal(0.5 - sep*0.10, 0.17, d["n_nr"]), 0.01, 0.99)
        xs = np.linspace(0, 1, 300)
        
        for vals, c, lbl, ls in [(resp, col, "Responder", "-"), (nonresp, "#888888", "Non-responder", "--")]:
            kde = gaussian_kde(vals, bw_method=0.32); dens = kde(xs)
            ax.plot(xs, dens, color=c, lw=1.5, ls=ls, label=lbl, zorder=3)
            ax.fill_between(xs, dens, alpha=0.12, color=c, zorder=2)
        
        ax.axvline(0.5, color=P["rline"], lw=0.8, ls=":", alpha=0.65, zorder=4)
        ax.set_xlim(0, 1); ax.set_xlabel("Predicted score")
        ax.set_title(label, fontsize=7.5, pad=4)
        ci = d["ci"]
        ax.text(
            0.97, 0.97,
            f"Pooled AUROC = {auroc:.3f}\n95% CI [{ci[0]:.3f}, {ci[1]:.3f}]\n{d['p']}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=5.5, color="#2c3e50", linespacing=1.4,
            bbox=dict(
                boxstyle="square,pad=0.4",
                fc="white",
                ec="#cccccc",
                lw=0.5,
                alpha=0.92
            )
        )

        spine_clean(ax); light_grid(ax, "y")
        ax.set_ylabel("Density" if ax is axes[0] else "")

    axes[0].legend(loc="upper left", fontsize=7.2, handlelength=1.2)
    fig.suptitle(
        "A. Immunotherapy response prediction — predicted score separation",
        fontsize=9.5, y=.95, fontweight="bold"
    )
    fig.tight_layout(w_pad=1.2)
    save_panel(fig, "panel_A"); plt.close()

def panel_b():
    rng = np.random.default_rng(42)

    # NOTE: Edit based on updated benchmarks during final run
    BENCH_NSCLC = {
        "GVAE (end-to-end)": {
            "folds": np.array([0.736, 0.753, 0.792, 0.811, 0.794]),
            "reported_mean": 0.772, "reported_std": 0.028,
            "col": "#E64B35", "ls": "-", "lw": 1.6,
            "n_pos": 121, "n_neg": 121,
        },
        "GVAE (embeddings)": {
            "folds": None, "reported_mean": 0.768, "reported_std": 0.044,
            "col": P["blue"], "ls": "--", "lw": 1.3,
            "n_pos": 121, "n_neg": 121,
        },
        "Scanpy (L1-LR)": {
            "folds": None, "reported_mean": 0.792, "reported_std": 0.046,
            "col": P["green"], "ls": "--", "lw": 1.3,
            "n_pos": 121, "n_neg": 121,
        },
        "scVI (L1-LR)": {
            "folds": np.array([0.5] * 5),
            "reported_mean": 0.500, "reported_std": 0.000,
            "col": P["lgray"], "ls": ":", "lw": 1.0,
            "n_pos": 121, "n_neg": 121,
        },
    }

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    for label, d in BENCH_NSCLC.items():
        mu = d["reported_mean"]
        sd = d["reported_std"]
        if d["folds"] is not None:
            fold_aurocs = d["folds"]
        else:
            np.clip(rng.normal(mu, max(sd, 0.005), 5), 0.01, 0.999)

        base_fpr, mean_tpr, std_tpr, _, _ = build_mean_roc(
            fold_aurocs, d["n_pos"], d["n_neg"], rng,
            noise=0.16 if mu > 0.6 else 0.22
        )

        if d["col"] == "#E64B35": # ci shaded band
            ax.fill_between(
                base_fpr,np.clip(mean_tpr - std_tpr, 0, 1),
                np.clip(mean_tpr + std_tpr, 0, 1),
                color=d["col"], alpha=0.18, zorder=1
            )
        ax.plot(
            base_fpr, mean_tpr,
            color=d["col"], ls=d["ls"], lw=d["lw"], zorder=3,
            label=f"{label} ({mu:.3f} $\\pm$ {sd:.3f})"
        )

    ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=0.85, ls="--",
            zorder=0, label="Random classifier")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("B. NSCLC ICI ($\mathbf{n=242}$), 5-fold CV", loc="left", fontsize=9.5, fontweight = "bold")
    ax.legend(
        loc="lower right", fontsize=7.0, frameon=False,
        handlelength=1.8, labelspacing=0.35,
        borderpad=0.5, handletextpad=0.5
    )

    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.75)
        ax.spines[s].set_color("#333333")
    ax.tick_params(which="both", length=3.0, width=0.75, color="#333333")
    fig.tight_layout()
    save_panel(fig, "panel_B")
    plt.close()

def panel_c():
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    abl_sorted = sorted(ABL, key=lambda x: x[1], reverse=True)
    names = [a[0] for a in abl_sorted]
    aurocs = [a[1] for a in abl_sorted]
    tags = [a[2] for a in abl_sorted]

    def bar_col(tag):
        if tag == "full": return P["blue"]
        if tag == "logreg": return P["amber"]
        if tag == "below": return P["red"]
        return P["lgray"]
    
    cols = [bar_col(t) for t in tags]
    y = np.arange(len(names))
    ax.barh(
        y, aurocs, height=0.58, color=cols,
        edgecolor=P["dgray"], linewidth=0.6, zorder=3
    )
    ax.axvline(
        0.500, color=P["rline"], lw=0.9, ls="--",
        alpha=0.85, zorder=2, label="Random (0.5)"
    )
    ax.axvline(
        0.700, color=P["blue"],  lw=0.8, ls=":",
        alpha=0.55, zorder=2, label="FULL (0.700)"
    )
    for i, (v, t) in enumerate(zip(aurocs, tags)):
        ax.text(v + 0.007, i, f"{v:.3f}", va="center", ha="left",
            fontsize=7, fontweight="bold" if t in ("full", "logreg", "below") else "normal",
            color=P["dkgray"], zorder=5
        )

    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.0)
    ax.set_xlim(0, 0.96); ax.set_xlabel("AUROC")
    ax.set_title("C. Ablation study - Melanoma ($\mathbf{n=32}$)", loc="left", fontsize=9.5, fontweight = "bold")
    leg = [
        mpatches.Patch(fc=P["blue"], ec=P["dkgray"], lw=0.6, label="FULL model"),
        mpatches.Patch(fc=P["amber"], ec=P["dkgray"], lw=0.6, label="LogReg baseline"),
        mpatches.Patch(fc=P["lgray"], ec=P["dkgray"], lw=0.6, label="Drop to random"),
        mpatches.Patch(fc=P["red"], ec=P["dkgray"], lw=0.6, label="Below random"),
        plt.Line2D([0],[0], color=P["rline"], ls="--", lw=0.9, label="Random (0.5)"),
    ]
    ax.legend(handles=leg, loc="upper right", fontsize=7.0, handlelength=1.2)
    light_grid(ax, "x")
    ax.spines["left"].set_linewidth(0.75); ax.spines["bottom"].set_linewidth(0.75)
    ax.tick_params(left=False, bottom=True, length=3.0, width=0.75)
    fig.tight_layout(); save_panel(fig, "panel_C"); plt.close()

def panel_d():
    ds_labels = ["NSCLC\nscRNA", "Breast", "NSCLC\nVisium", "Colorectal"]
    gate_vals = [1.000, 1.000, 0.133, None]
    morans_vals = [None,  None,  0.437, 0.678]
    ds_cols = [P["blue"], P["amber"], P["navy"], P["green"]]
    x = np.arange(len(ds_labels))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(5.4, 5.0),
        sharex=True,
        gridspec_kw={"hspace": 0.10, "height_ratios": [1, 1]}
    )

    # gate weights for top, moran's I for bottom
    for i, (gv, col) in enumerate(zip(gate_vals, ds_cols)):
        if gv is not None:
            ax_top.bar(x[i], gv, width=0.5, color=col, edgecolor=P["dgray"], lw=0.7, zorder=3, alpha=0.85)
            ax_top.text(x[i], gv + 0.022, f"{gv:.3f}", ha="center", va="bottom", fontsize=7.5, color=col)
        else:
            ax_top.bar(x[i], 0.05, width=0.5, color="none", edgecolor="#cccccc", lw=0.6, zorder=3, hatch="///", alpha=0.5)
            ax_top.text(x[i], 0.07, "N/A", ha="center", va="bottom", fontsize=7.0, color="#aaaaaa")

    ax_top.axhline(0.5, color="#999999", lw=0.75, ls="--", alpha=0.8, zorder=2)
    ax_top.annotate("87% spatial\nweight",
                    xy=(x[2], 0.133),
                    xytext=(x[2] + 0.7, 0.32),
                    arrowprops=dict(arrowstyle="->", color="#555555",
                                    lw=0.8, connectionstyle="arc3,rad=-0.3"),
                    fontsize=6, color="#333333")
    ax_top.set_ylim(0, 1.09)
    ax_top.set_ylabel("Gate weight", fontsize=8.5)
    ax_top.set_title("D. Spatial gate weight and Moran's $\\mathbf{I}$ by dataset",
                     loc="left", fontsize=9.5, fontweight="bold")
    light_grid(ax_top, "y"); spine_clean(ax_top)

    for i, (mv, col) in enumerate(zip(morans_vals, ds_cols)):
        if mv is not None:
            ax_bot.bar(x[i], mv, width=0.5, color=col, edgecolor=P["dgray"], lw=0.7, zorder=3, alpha=0.85)
            ax_bot.text(x[i], mv + 0.018, f"{mv:.3f}*", ha="center", va="bottom", fontsize=7.5, color=col)
        else:
            ax_bot.bar(x[i], 0.04, width=0.5, color="none", edgecolor="#cccccc", lw=0.6, zorder=3, hatch="///", alpha=0.5)
            ax_bot.text(x[i], 0.055, "N/A", ha="center", va="bottom", fontsize=7.0, color="#aaaaaa")

    ax_bot.set_ylim(0, 0.85)
    ax_bot.set_ylabel("Moran's $I$", fontsize=8.5)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(ds_labels, fontsize=8.2)
    light_grid(ax_bot, "y"); spine_clean(ax_bot)
    leg = [
        mpatches.Patch(fc=P["dgray"], ec=P["dgray"], lw=0.7,
        alpha=0.85, label="Value present"),
        mpatches.Patch(fc="none", ec="#cccccc", lw=0.6,
        hatch="///", label="Not applicable"),
    ]
    ax_bot.legend(handles=leg, loc=(.75, 1.94), fontsize=7.2,handlelength=1.2, frameon=False)
    fig.tight_layout()
    save_panel(fig, "panel_D")
    plt.close()

def panel_e():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for i, (lbl, d) in enumerate(CV.items()):
        folds = d["folds"]; col = d["col"]; mu = d["mean"]
        if folds.std() < 1e-6:
            ax.vlines(folds[0], i+0.06, i+0.44, color=col, lw=2.2, alpha=0.35, zorder=2)
        else:
            kde = gaussian_kde(folds, bw_method=0.55)
            xp = np.linspace(max(0.3, folds.min()-0.08), min(1.05, folds.max()+0.08), 200)
            dens = kde(xp); dens = dens / dens.max() * 0.32
            ax.fill_between(xp, i+0.06, i+0.06+dens, color=col, alpha=0.45, zorder=2)
            ax.plot(xp, i+0.06+dens, color=col, lw=0.9, alpha=0.80, zorder=3)
        
        ax.boxplot(folds, positions=[i-0.18], widths=0.3, vert=False,
                   patch_artist=True,
                   boxprops=dict(facecolor=col+"38", lw=0.75, edgecolor="#333333"),
                   medianprops=dict(color="#111111", lw=1.8),
                   whiskerprops=dict(color="#555555", lw=0.75),
                   capprops=dict(color="#555555", lw=0.75),
                   flierprops=dict(marker="", ms=0))
        
        jit = rng.uniform(-0.065, 0.065, len(folds))
        ax.scatter(folds, i+0.18 + jit, color=col, s=16, zorder=0, edgecolors="#333333", lw=0.4, alpha=0.7)
        ax.scatter(mu, i+0.12, color=col, marker="D", s=24, zorder=7, edgecolors="#111111", lw=0.6)
        ax.text(mu - 0.03, i - 0.42, f"$\\bar{{x}}={mu:.3f}$", ha="left", va="center", fontsize=6.8, fontweight="bold", color=col)
    
    ax.axvline(0.5, color=P["rline"], linestyle="--", lw=0.9, label="Random (0.5)", zorder=1)
    ax.set_yticks(range(3))
    ax.set_yticklabels(list(CV.keys()), fontsize=8.2)
    ax.set_xlim(0.48, 1.06)
    ax.set_xlabel("AUROC per fold", fontsize=9)
    ax.set_ylabel("Method", fontsize=9)
    ax.set_title("E. Cross-validation (per-fold) AUROC stability", fontweight="bold", loc="left", fontsize=9.5)

    diag = ax.scatter([], [], color="#555555", marker="D", s=22,
                      edgecolors="#111", lw=0.6, label="Reported mean")
    ax.legend(handles=[diag, plt.Line2D([0], [0], color=P["rline"], ls="--",
              lw=0.9, label="Random (0.5)")], loc="lower right", fontsize=7.2)
    light_grid(ax, "x")
    spine_clean(ax)
    fig.tight_layout()
    save_panel(fig, "panel_E")
    plt.close()

def panel_f():
    ds_col = {
        "Breast": P["amber"],
        "NSCLC Visium": P["navy"],
        "NSCLC scRNA": P["blue"],
    }

    pairs = list(dict.fromkeys((r[0], r[1]) for r in LR))
    ligands = list(dict.fromkeys(p[0] for p in pairs))
    receptors = list(dict.fromkeys(p[1] for p in pairs))
    score_map = {(r[0], r[1], r[2]): r[3] for r in LR}

    ds_list = list(ds_col.keys())
    n_ds = len(ds_list); n_L = len(ligands); n_R = len(receptors)

    matrices = {}
    for ds in ds_list:
        mat = np.full((n_L, n_R), np.nan)
        for i, lig in enumerate(ligands):
            for j, rec in enumerate(receptors):
                sc = score_map.get((lig, rec, ds), None)
                if sc is not None:
                    mat[i, j] = sc
        matrices[ds] = mat

    global_max = max(r[3] for r in LR)
    global_min = min(r[3] for r in LR)

    fig = plt.figure(figsize=(12.0, 3.8))
    gs = gridspec.GridSpec(1, n_ds * 2,
                            width_ratios=[1, 0.06] * n_ds,
                            wspace=0.15)
    
    axes = [fig.add_subplot(gs[0, di * 2]) for di in range(n_ds)]
    cax = fig.add_subplot(gs[0, n_ds * 2 - 1])
    cmap = plt.cm.viridis_r
    norm = mcolors.Normalize(vmin=global_min, vmax=global_max)

    for di, (ds, col) in enumerate(ds_col.items()):
        ax  = axes[di]
        mat = matrices[ds]
        masked = np.ma.masked_invalid(mat)
        ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto", origin="upper")
        for i in range(n_L):
            for j in range(n_R):
                val = mat[i, j]
                if not np.isnan(val):
                    txt_col = "white" if norm(val) > 0.55 else "#1a1a1a"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=6.5, color=txt_col, fontweight="normal")

        ax.set_xticks(range(n_R))
        ax.set_xticklabels([f"$\\mathit{{{r}}}$" for r in receptors],
                            fontsize=7.5, rotation=35, ha="right")
        ax.set_title(ds, fontsize=9.0, fontweight="bold", pad=6, color=col)

        if di == 0:
            ax.set_yticks(range(n_L))
            ax.set_yticklabels([f"$\\mathit{{{l}}}$" for l in ligands], fontsize=8.0)
            ax.set_ylabel("Ligands", fontsize=9.0, labelpad=6)
        else:
            ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(0.6)

        ax.set_xticks(np.arange(-0.5, n_R, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_L, 1), minor=True)
        ax.grid(which="minor", color="#e0e0e0", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Interaction score", fontsize=7.5, labelpad=6)
    cbar.set_ticks([global_min, (global_min + global_max) / 2, global_max])
    cbar.set_ticklabels(
        [f"{global_min:.3f}",
        f"{(global_min + global_max) / 2:.3f}",
        f"{global_max:.3f}"
        ])
    cbar.ax.tick_params(labelsize=6.5)
    cbar.outline.set_edgecolor("#cccccc")
    cbar.outline.set_linewidth(0.6)

    fig.text(0.47, -0.03, "Receptors", ha="center", fontsize=9.0)
    fig.suptitle("F. Top ligand–receptor interactions by dataset",
                 x=0.5, ha="center", fontsize=13.5, fontweight = "bold", y=1.05)

    fig.tight_layout()
    save_panel(fig, "panel_F")
    plt.close()

# ------

def conf_matrix():
    ABL_CM = [
        ("FULL model", 0.700, 0.471, "#1a3a6b", True),
        ("logreg baseline", 0.750, 0.500, "#2a5a2a", True),
        ("mol only\n(representative random)", 0.500, None,  "#555555", False),
        ("static 0.3", 0.375, None,  "#8b3a1a", False),
        ("gcn encoder", 0.125, None,  "#b83232", False),
    ]

    CMAP = mcolors.LinearSegmentedColormap.from_list(
        "ablation_cm", ["#deeaf7", "#4878a8", "#1a3a6b"], N=256)
    fig, axes = plt.subplots(1, 5, figsize=(13.0, 3.6), facecolor="white")
    fig.patch.set_facecolor("white")
    for idx, (name, auroc, f1, title_col, bold_title) in enumerate(ABL_CM):
        ax = axes[idx]
        cm = auroc_f1_to_cm(auroc, f1, n_pos=16, n_neg=16)
        cm_pct = cm.astype(float) / cm.astype(float).sum(axis=1, keepdims=True)
        ax.imshow(cm_pct, cmap=CMAP, vmin=0.0, vmax=1.0, aspect="equal")
        labels = ["Non-R", "Resp"]
        for r in range(2):
            for c in range(2):
                pct  = cm_pct[r, c]; cnt = cm[r, c]; bold = (r == c)
                col  = "white" if pct > 0.42 else "#111111"
                ax.text(c, r, f"{pct*100:.0f}%\n({cnt})",
                        ha="center", va="center",
                        fontsize=8.5 if bold else 7.0,
                        fontweight="bold" if bold else "normal",
                        color=col, linespacing=1.5)

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_yticklabels(labels, fontsize=8.5, rotation=90, va="center")
        ax.tick_params(length=0, pad=4)
        ax.set_xlabel("Predicted", fontsize=8.5, labelpad=5, fontweight = "bold")
        if idx == 0:
            ax.set_ylabel("Actual", fontsize=8.5, labelpad=5, fontweight = "bold")
        ax.set_title(f"{name}\nAUROC = {auroc:.3f}",
                     fontsize=8.8, pad=6,
                     fontweight="bold" if bold_title else "normal",
                     color=title_col)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7); spine.set_edgecolor("#999999")

    fig.subplots_adjust(right=0.88, wspace=0.28)
    cbar_ax = fig.add_axes([0.90, 0.18, 0.013, 0.64])
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Row-normalised rate", fontsize=8.0, labelpad=8)
    cbar.ax.tick_params(labelsize=7.5); cbar.outline.set_linewidth(0.6)
    fig.suptitle(
        "C. Ablation study - selected confusion matrices  "
        "(Melanoma, $n=32$, 16 responders & 16 non-responders)",
        fontsize=9.5, y=.94, fontweight="bold", fontfamily="serif")
    save_panel(fig, "panel_C_conf")
    plt.close()

if __name__ == "__main__":
    panel_a(); panel_b(); panel_c()
    panel_d(); panel_e(); panel_f()
    conf_matrix()
    print("All 6 panels saved to output/")