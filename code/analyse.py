"""Aggregate the tile-consistency sweep: tables, statistics, and figures.

Reads out/sweep.jsonl (one record per method x image) and produces:
  out/tables.md              -- paste-ready numbers
  out/stats.json             -- paired tests + rank inversions
  figures/F1_tce_halo.png    -- TCE vs overlap halo (class-A dies, class-B plateaus)
  figures/F2_bands.png       -- border-band decomposition
  figures/F3_tce_n.png       -- TCE vs tile count
  figures/F4_metrics_blind.png -- quality metrics fail to see TCE
  figures/F5_rank_flip.png   -- method ranking, whole vs tiled evaluation
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/data/tileconsistency/src")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, kendalltau

OUT = "/data/tileconsistency/out"
FIG = "/data/tileconsistency/figures"
NS = (2, 4, 8)
HALOS_H = (0, 8, 16, 24, 32)
BANDS = ["0-2", "2-6", "6-12", "12-24", ">24"]
QUAL = ("sr", "ssim", "fsr", "ece")
ORDER = ["AKF", "CA-AKF", "SC-AKF", "LRS-AKF", "LRS-geo"]
COL = {"AKF": "#444", "CA-AKF": "#c0392b", "SC-AKF": "#e67e22",
       "LRS-AKF": "#1a6fb5", "LRS-geo": "#16a085"}


def load(path=f"{OUT}/sweep.jsonl"):
    by = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            by.setdefault(r["method"], []).append(r)
    for m in by:
        by[m].sort(key=lambda r: r["img"])
    return by


def g(r, *keys):
    d = r
    for k in keys:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


def vals(rows, *keys):
    return np.array([v for v in (g(r, *keys) for r in rows) if v is not None], float)


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.allclose(a, b):
        return dict(p=1.0, r=0.0, n=int(len(a)))
    res = wilcoxon(a, b)
    d = b - a
    nz = d[d != 0]
    r = float((np.sum(nz > 0) - np.sum(nz < 0)) / len(nz)) if len(nz) else 0.0
    return dict(p=float(res.pvalue), r=r, n=int(len(a)))


def tables(by):
    L = ["# Tile-consistency sweep (TCE)", ""]
    L.append("images per method: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(by.items())))
    L += ["", "## 1. TCE (RMSE, whole vs tiled n=4) vs overlap halo", "",
          "| method | " + " | ".join(f"halo {h}" for h in HALOS_H) + " |",
          "|---" * (len(HALOS_H) + 1) + "|"]
    for m in ORDER:
        if m not in by:
            continue
        row = [m]
        for h in HALOS_H:
            v = vals(by[m], "tce_halo", f"h{h}", "rmse")
            row.append(f"{np.median(v):.6f}" if len(v) else "-")
        L.append("| " + " | ".join(row) + " |")

    L += ["", "## 2. Pixels differing by more than 1/255 (n=8, halo=16)", "",
          "| method | median | mean | max |", "|---|---|---|---|"]
    for m in ORDER:
        if m not in by:
            continue
        v = vals(by[m], "tce_n", "n8_h16", "frac1")
        L.append(f"| {m} | {np.median(v):.4f} | {v.mean():.4f} | {v.max():.4f} |")

    L += ["", "## 3. Max |d| on the same config (worst single pixel)", "",
          "| method | median |", "|---|---|"]
    for m in ORDER:
        if m not in by:
            continue
        v = vals(by[m], "tce_n", "n8_h16", "maxd")
        L.append(f"| {m} | {np.median(v):.4f} |")

    L += ["", "## 4. Deviation vs distance from the tile grid (n=8, halo=16)", "",
          "| method | " + " | ".join(BANDS) + " |", "|---" * 6 + "|"]
    for m in ORDER:
        if m not in by:
            continue
        B = np.array([r["bands_n8_h16"]["bands"] for r in by[m]
                      if "bands_n8_h16" in r])
        L.append("| " + m + " | " + " | ".join(f"{x:.5f}" for x in np.median(B, 0)) + " |")

    L += ["", "## 5. Grid-line visibility (share of deviation energy ON the grid, n=4 halo=16)", "",
          "| method | GV | concentration ratio |", "|---|---|---|"]
    for m in ORDER:
        if m not in by:
            continue
        gv = vals(by[m], "gv_n4_h16", "gv")
        ra = vals(by[m], "gv_n4_h16", "ratio")
        L.append(f"| {m} | {np.median(gv):.3f} | {np.median(ra):.1f}x |")

    L += ["", "## 6. Contrast-equivariance error (lower = more equivariant)", "",
          "| method | gain 0.5 | gain 2.0 |", "|---|---|---|"]
    for m in ORDER:
        if m not in by:
            continue
        a, b = vals(by[m], "cee", "0.5"), vals(by[m], "cee", "2.0")
        L.append(f"| {m} | {np.median(a):.6f} | {np.median(b):.6f} |"
                 if len(a) else f"| {m} | - | - |")

    L += ["", "## 7. Quality metrics are blind to the mismatch (median over images)", "",
          "| method | " + " | ".join(f"{q} whole/tiled" for q in QUAL) + " |",
          "|---" * (len(QUAL) + 1) + "|"]
    for m in ORDER:
        if m not in by:
            continue
        row = [m]
        for q in QUAL:
            w, t = vals(by[m], "qual_whole", q), vals(by[m], "qual_tiled", q)
            row.append(f"{np.median(w):.3f}/{np.median(t):.3f}")
        L.append("| " + " | ".join(row) + " |")

    L += ["", "## 8. Paired tests vs CA-AKF (published) on rmse at n=8, halo=16", "",
          "| comparison | p | rank-biserial | n |", "|---|---|---|---|"]
    base = {r["img"]: r for r in by.get("CA-AKF", [])}
    for m in ORDER:
        if m not in by or m == "CA-AKF" or not base:
            continue
        pairs = [(r, base[r["img"]]) for r in by[m] if r["img"] in base]
        if len(pairs) < 3:
            continue
        a = [g(x, "tce_n", "n8_h16", "rmse") for x, _ in pairs]
        b = [g(y, "tce_n", "n8_h16", "rmse") for _, y in pairs]
        st = paired(a, b)
        L.append(f"| {m} vs CA-AKF | {st['p']:.4g} | {st['r']:+.2f} | {st['n']} |")
    txt = "\n".join(L)
    open(f"{OUT}/tables.md", "w").write(txt + "\n")
    return txt


def rank_flips(by):
    """Does the method ranking change when evaluation is done tile-by-tile?"""
    imgs = sorted({r["img"] for m in by for r in by[m]})
    methods = [m for m in ORDER if m in by]
    out = {"per_image": [], "kendall": {}}
    for q in QUAL:
        better_hi = q not in ("ece", "fsr")          # sr/ssim higher is better
        W, T = [], []
        for im in imgs:
            w = {m: g(r, "qual_whole", q) for m in methods
                 for r in by[m] if r["img"] == im}
            t = {m: g(r, "qual_tiled", q) for m in methods
                 for r in by[m] if r["img"] == im}
            if len(w) < 2 or len(t) < 2:
                continue
            rw = sorted(methods, key=lambda m: -w[m] * (1 if better_hi else -1))
            rt = sorted(methods, key=lambda m: -t[m] * (1 if better_hi else -1))
            W.append([rw.index(m) for m in methods])
            T.append([rt.index(m) for m in methods])
            if rw != rt:
                out["per_image"].append(dict(metric=q, img=im, whole=rw, tiled=rt,
                                             gap={m: round(t[m] - w[m], 4) for m in methods}))
        if W:
            tau = [float(kendalltau(a, b).correlation) for a, b in zip(W, T)]
            out["kendall"][q] = dict(mean_tau=float(np.nanmean(tau)),
                                     n=len(tau),
                                     n_flips=int(np.sum(np.isnan(tau) == False) - np.sum(tau == 1.0)))
    json.dump(out, open(f"{OUT}/stats_flip.json", "w"), indent=1)
    return out


def fig_tce_halo(by):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m in ORDER:
        if m not in by:
            continue
        v = [np.median(vals(by[m], "tce_halo", f"h{h}", "rmse")) for h in HALOS_H]
        if not np.any(v):
            continue
        ax.semilogy(HALOS_H, np.maximum(v, 1e-7), "o-", color=COL[m], label=m, lw=1.6, ms=4)
    ax.axhline(1 / 255, ls=":", c="k", lw=1)
    ax.text(HALOS_H[-1], 1 / 255 * 1.15, "1/255", ha="right", fontsize=7)
    ax.set_xlabel("overlap halo (px)")
    ax.set_ylabel("TCE (RMSE, log)")
    ax.set_title("Tile-consistency vs overlap halo (n=4)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/F1_tce_halo.png", dpi=300)
    plt.close(fig)


def fig_tce_n(by):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m in ORDER:
        if m not in by:
            continue
        for h, ls in ((0, "--"), (16, "-")):
            v = [np.median(vals(by[m], "tce_n", f"n{n}_h{h}", "rmse")) for n in NS]
            if not np.any(v):
                continue
            ax.semilogy([n * n for n in NS], np.maximum(v, 1e-7), ls, marker="o",
                        color=COL[m], ms=4, lw=1.4,
                        label=f"{m} halo {h}")
    ax.set_xlabel("number of tiles")
    ax.set_ylabel("TCE (RMSE, log)")
    ax.set_title("Tile-consistency vs grid fineness")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{FIG}/F3_tce_n.png", dpi=300)
    plt.close(fig)


def fig_bands(by):
    ms = [m for m in ORDER if m in by and "bands_n8_h16" in by[m][0]]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = np.arange(len(BANDS))
    w = 0.8 / max(1, len(ms))
    for i, m in enumerate(ms):
        B = np.median(np.array([r["bands_n8_h16"]["bands"] for r in by[m]]), 0)
        ax.bar(x + i * w, B, w, color=COL[m], label=m)
    ax.set_yscale("log")
    ax.set_xticks(x + w * (len(ms) - 1) / 2)
    ax.set_xticklabels(BANDS, fontsize=8)
    ax.set_xlabel("distance from nearest tile grid line (px)")
    ax.set_ylabel("mean |d| (log)")
    ax.set_title("Where the mismatch lives: seam bands vs tile interior")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/F2_bands.png", dpi=300)
    plt.close(fig)


def fig_blind(by):
    if "CA-AKF" not in by:
        return
    fig, axes = plt.subplots(1, len(QUAL), figsize=(7.4, 2.8))
    for ax, q in zip(axes, QUAL):
        w = vals(by["CA-AKF"], "qual_whole", q)
        t = vals(by["CA-AKF"], "qual_tiled", q)
        ax.plot(w, t, "o", color=COL["CA-AKF"], ms=4)
        lo, hi = min(w.min(), t.min()), max(w.max(), t.max())
        ax.plot([lo, hi], [lo, hi], "k:", lw=1)
        ax.set_title(q, fontsize=9)
        ax.set_xlabel("whole", fontsize=7)
        ax.set_ylabel("tiled", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_title("CA-AKF: quality metrics, whole vs tiled\n(identical: mismatch invisible)")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(f"{FIG}/F4_metrics_blind.png", dpi=300)
    plt.close(fig)


def fig_flip(by):
    fl = rank_flips(by)
    methods = [m for m in ORDER if m in by]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = np.arange(len(methods))
    d = [1e4 * np.median(vals(by[m], "qual_whole", "sr") -
                         vals(by[m], "qual_tiled", "sr")) for m in methods]
    ax.bar(x, d, 0.6, color=[COL[m] for m in methods])
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel("1e4 x (SR whole - SR tiled)")
    ax.set_title("Evaluation protocol changes the score (rank-flip probe)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/F5_rank_flip.png", dpi=300)
    plt.close(fig)
    return fl


if __name__ == "__main__":
    by = load(sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/sweep.jsonl")
    print(tables(by))
    os.makedirs(FIG, exist_ok=True)
    fig_tce_halo(by)
    fig_tce_n(by)
    fig_bands(by)
    fig_blind(by)
    fl = fig_flip(by)
    print("\n== rank inversions ==", json.dumps(fl["kendall"], indent=1))
    print("flips:", len(fl["per_image"]))
    json.dump(fl, open(f"{OUT}/stats_flip.json", "w"), indent=1)
