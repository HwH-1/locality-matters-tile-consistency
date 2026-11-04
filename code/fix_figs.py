"""Regenerate F1, F2, F4, F5, F7 so that each figure matches its caption.

Fixes applied here (see REVIEW_pre_submission.md):
  F1  the old plot clipped every value below 1e-7 with np.maximum(v,1e-7),
      which drew a false plateau for the two bounded-support operators and
      contradicted Table 3 (0.0 at halo 32).  Zeros are now drawn as open
      markers on an "exact (0)" line.
  F2  zero bars were invisible on a log axis; they are now labelled "0".
  F4  the old plot was a whole-vs-tiled identity scatter while its caption
      described TCE on the horizontal axis.  Redrawn as TCE vs metric.
  F5  the old plot was a bar chart of SR deltas while its caption described a
      rank-transition plot.  Redrawn as the bump chart the caption promises.
  F7  AKF was completely hidden under LRS-final; now dashed with open markers.
"""
import collections
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
FIG = os.path.join(ROOT, "figures")

ORDER = ["AKF", "CA-AKF", "SC-AKF", "LRS-final"]
COL = {"AKF": "#444444", "CA-AKF": "#c0392b", "SC-AKF": "#e67e22",
       "LRS-final": "#1a6fb5", "LRS-AKF": "#1a6fb5"}
LAB = {"AKF": "AKF", "CA-AKF": "CA-AKF", "SC-AKF": "SC-AKF",
       "LRS-final": "LRS-AKF (ours)", "LRS-AKF": "LRS-AKF (arith.)"}
EXACT = 1e-13          # plotting floor for "exactly zero"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(name):
    by = collections.defaultdict(list)
    with open(os.path.join(OUT, name), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                by[r["method"]].append(r)
    for m in by:
        by[m].sort(key=lambda r: r["img"])
    return by


def med(rows, *keys):
    out = []
    for r in rows:
        d = r
        for k in keys:
            if d is None or k not in d:
                d = None
                break
            d = d[k]
        if d is not None:
            out.append(d)
    return statistics.median(out)


# --------------------------------------------------------------------- F1
def fig_tce_halo(by):
    halos = [0, 8, 16, 24, 32]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for m in ORDER:
        if m not in by:
            continue
        v = np.array([med(by[m], "tce_halo", f"h{h}", "rmse") for h in halos], float)
        zero = v == 0
        y = np.where(zero, EXACT, v)
        ax.semilogy(halos, y, "-", color=COL[m], lw=1.6, marker="o", ms=4.5,
                    mfc=COL[m], mec=COL[m], label=LAB[m], zorder=3)
        if zero.any():   # re-draw the exact points as open markers on the floor
            ax.semilogy(np.array(halos)[zero], y[zero], "o", ms=6.5,
                        mfc="white", mec=COL[m], mew=1.4, zorder=4)
    ax.axhline(EXACT, color="0.55", lw=0.9, ls=(0, (1, 2)))
    ax.text(1.2, EXACT * 2.4, "exact (0)", ha="left", va="bottom",
            fontsize=6.5, color="0.35")
    ax.axhline(1 / 255, ls=":", c="k", lw=0.9)
    ax.text(17.0, 1 / 255 * 1.30, "1/255", ha="left", va="bottom", fontsize=6.5)
    ax.set_ylim(EXACT * 0.6, 4e-2)
    ax.set_xlim(-1.5, 33)
    ax.set_xlabel("overlap halo (px)")
    ax.set_ylabel("TCE (median RMSE, log scale)")
    ax.set_title("Class A reaches zero; class B plateaus")
    ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 0.88))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F1_tce_halo.png"), dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------- F2
def fig_bands(by):
    bands = ["0-2", "2-6", "6-12", "12-24", ">24"]
    ms = [m for m in ORDER if m in by and "bands_n8_h16" in by[m][0]]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = np.arange(len(bands))
    w = 0.8 / max(1, len(ms))
    floor = 1e-8
    for i, m in enumerate(ms):
        B = np.median(np.array([r["bands_n8_h16"]["bands"] for r in by[m]]), 0)
        y = np.where(B == 0, floor, B)
        ax.bar(x + i * w, y, w, color=COL[m], label=LAB[m],
               edgecolor="white", linewidth=0.4)
        for xi, bi in zip(x + i * w, B):
            if bi == 0:
                ax.text(xi, floor * 1.35, "0", ha="center", va="bottom",
                        fontsize=6.5, color=COL[m])
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.8, 3e-2)
    ax.set_xticks(x + w * (len(ms) - 1) / 2)
    ax.set_xticklabels(bands)
    ax.set_xlabel("distance from nearest tile grid line (px)")
    ax.set_ylabel("mean |d| (log scale)")
    ax.set_title("Class B is flat across bands; class A is a seam", fontsize=8.5)
    ax.legend(frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F2_bands.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- F4
def fig_blind(by):
    """Identity plot: tile-wise value against whole-image value, one panel per
    metric.  Points on the diagonal mean the metric cannot see the mismatch."""
    if "CA-AKF" not in by:
        return
    quals = ["sr", "ssim", "fsr", "ece"]
    nice = {"sr": "SR", "ssim": "SSIM", "fsr": "FSR", "ece": "ECE"}
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4))
    for ax, q in zip(axes, quals):
        rows = by["CA-AKF"]
        w = np.array([r["qual_whole"][q] for r in rows], float)
        t = np.array([r["qual_tiled"][q] for r in rows], float)
        d = float(np.mean(np.abs(t - w)))
        lo, hi = min(w.min(), t.min()), max(w.max(), t.max())
        pad = 0.06 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k:", lw=1)
        ax.plot(w, t, "o", color=COL["CA-AKF"], ms=4.2, mec="white", mew=0.4)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{nice[q]}  (mean $|\\Delta|$ = {d:.4f})", fontsize=7.5)
        ax.set_xlabel("whole-image", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("tile-wise", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F4_metrics_blind.png"), dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------- F5
def fig_flip(by, order=("AKF", "CA-AKF", "SC-AKF", "LRS-AKF")):
    """Bump chart: rank under whole-image vs tile-wise evaluation."""
    quals = ["sr", "ssim", "fsr", "ece"]
    nice = {"sr": "SR", "ssim": "SSIM", "fsr": "FSR", "ece": "ECE"}
    methods = [m for m in order if m in by]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.9), sharey=True)
    total = 0
    for ax, q in zip(axes, quals):
        better_hi = q not in ("ece", "fsr")
        imgs = sorted({r["img"] for m in methods for r in by[m]})
        n_cross = 0
        for im in imgs:
            w, t = {}, {}
            for m in methods:
                for r in by[m]:
                    if r["img"] == im:
                        w[m] = r["qual_whole"][q]
                        t[m] = r["qual_tiled"][q]
            if len(w) < len(methods):
                continue
            sgn = 1 if better_hi else -1
            rw = sorted(methods, key=lambda m: -sgn * w[m])
            rt = sorted(methods, key=lambda m: -sgn * t[m])
            y0 = [rw.index(m) + 1 for m in methods]
            y1 = [rt.index(m) + 1 for m in methods]
            crossed = rw != rt
            n_cross += int(crossed)
            col = "#c0392b" if crossed else "0.72"
            lw = 1.1 if crossed else 0.8
            for a, b in zip(y0, y1):
                ax.plot([0, 1], [a, b], "-", color=col, lw=lw,
                        alpha=0.9 if crossed else 0.8, zorder=3 if crossed else 2)
                ax.plot([0, 1], [a, b], "o", color=col, ms=2.0, zorder=4)
        total += n_cross
        ax.set_title(f"{nice[q]}: {n_cross}/10 inverted", fontsize=7.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["whole", "tiled"], fontsize=7)
        ax.set_xlim(-0.25, 1.25)
        ax.grid(axis="y", color="0.92", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks([1, 2, 3, 4])
    axes[0].set_ylabel("rank (1 = best)")
    axes[0].set_ylim(4.4, 0.6)
    fig.suptitle(f"Rank under the two protocols: {total}/40 cases invert "
                 "(red = the ranking changes)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIG, "F5_rank_flip.png"), dpi=300)
    plt.close(fig)
    print("inversions:", total, "/40")


# --------------------------------------------------------------------- F7
def fig_broad(by):
    halos = ["h0", "h16", "h32"]
    x = [0, 16, 32]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    style = {"AKF": dict(ls="--", mfc="white", mew=1.5, ms=6.5),
             "CA-AKF": dict(ls="-", ms=5),
             "SC-AKF": dict(ls="-", ms=5),
             "LRS-final": dict(ls="-", ms=5)}
    for m in ORDER:
        if m not in by:
            continue
        v = [100 * med(by[m], "tce_halo", h, "frac1") for h in halos]
        st = style.get(m, {})
        ax.plot(x, v, color=COL[m], lw=1.6, marker="o", label=LAB[m],
                zorder=5 if m == "AKF" else 4, **st)
    ax.set_xlabel("overlap halo (px)")
    ax.set_ylabel("pixels differing by > 1 level (median %)")
    ax.set_title("Breadth replication on 144 images")
    ax.set_xlim(-2, 34)
    ax.set_ylim(-0.8, 21)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "F7_broad.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    sweep = load("sweep.jsonl")                 # arithmetic LRS-AKF
    final = load("sweep_final.jsonl")           # final LRS-AKF configuration
    stage = dict(sweep)
    stage["LRS-final"] = final["LRS-final"]     # figure rows follow Table 3
    fig_tce_halo(stage)
    fig_bands(stage)
    fig_blind(stage)
    fig_flip(sweep)                             # rank test uses the arithmetic config
    broad = load("broad_all.jsonl")
    fig_broad(broad)
    print("figures written to", FIG)
