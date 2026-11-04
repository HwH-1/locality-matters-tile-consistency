"""Aggregate out/broad.jsonl: the halo ladder of tile-consistency error measured
on every image in the benchmark, with one pre-specified primary endpoint.

The primary protocol (Section 6.1-6.5) uses 10 images because it records the
full metric battery (~3 min/image).  This aggregation is the breadth check: the
same claim on the whole benchmark, single endpoint, no metric battery.  Multiplicity
is controlled by declaring ONE primary endpoint in advance -- the fraction of
pixels differing by more than one 8-bit level at the most generous halo we test --
and reporting the rest descriptively.

Writes out/broad.md and figures/F7_broad.png.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, "/data/tileconsistency/src")

OUT = "/data/tileconsistency/out"
FIG = "/data/tileconsistency/figures"
LEVEL = 1 / 255.0
PRIMARY = "h32"          # pre-specified: most generous halo
HALOS = ["h0", "h16", "h32"]


def load():
    recs = {}
    for p in sorted(glob.glob(f"{OUT}/broad*.jsonl")):
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            recs[(r["img"], r["method"])] = r
    return recs


def gvisable(recs):
    """GV ratios for the images where a spatial distribution of the deviation exists.

    gridline_visibility divides the error energy on tile borders by the error energy on a
    random pixel set.  If the deviation is at or below one quantisation step everywhere,
    both sums are float dust and the quotient is arbitrary; such images are excluded here
    rather than allowed to set a method's median.
    """
    out = []
    for r in recs:
        g = r.get("gv")
        h16 = r.get("tce_halo", {}).get("h16", {})
        if g and h16.get("frac1", 0.0) > 0.001:
            out.append(g["ratio"])
    return np.array(out, dtype=float) if out else np.array([], dtype=float)


def main():
    recs = load()
    if not recs:
        print("no records yet")
        return
    methods = sorted({k[1] for k in recs})
    per = {m: [r for (i, mm), r in recs.items() if mm == m] for m in methods}
    n_img = {m: len(v) for m, v in per.items()}
    n_union = len({i for (i, _m) in recs})

    lines = ["# Broad replication (Section 6.8)", ""]
    lines.append(f"Records: {sum(n_img.values())} across {len(methods)} methods, "
                 f"{n_union} distinct images (union over methods).  n=4 tile grid, "
                 f"{sorted(HALOS)} halos, 384x384.")
    lines.append("")
    lines.append("| method | images | median frac>1 level (h0) | (h16) | (h32) | "
                 "median rmse (h32) | images exact by h32 | images exact already at h16 | "
                 "median GV ratio (h16) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in methods:
        v = per[m]
        f = lambda h, k: np.array([r["tce_halo"][h][k] for r in v if h in r["tce_halo"]])
        ex = [r for r in v if r.get("halo_exact") is not None]
        gv = gvisable(v)
        gvcell = f"{np.median(gv):.1f} [{gv.size}]" if gv.size else f"n/a [0]"
        lines.append(
            f"| {m} | {len(v)} | {np.median(f('h0','frac1'))*100:.2f} % | "
            f"{np.median(f('h16','frac1'))*100:.2f} % | {np.median(f('h32','frac1'))*100:.2f} % | "
            f"{np.median(f('h32','rmse')):.6f} | {len(ex)}/{len(v)} | "
            f"{sum(1 for r in v if r.get('halo_exact') is not None and r['halo_exact'] <= 16)}/{len(v)} | "
            f"{gvcell} |")
    lines.append("")
    lines.append("")
    lines.append("GV (grid-line visibility) is only computed for images whose halo-16 deviation "
                 "covers more than 0.1 % of pixels: the ratio is the quotient of two error-energy "
                 "sums, and when the deviation sits at the quantisation floor the quotient is noise, "
                 "not a statement about where the error lives.  The count of such images is given in "
                 "brackets.")
    lines.append("")

    # paired primary endpoint: adaptive-vs-proposed on the same images
    def vals(m, h, k):
        return {r["img"]: r["tce_halo"][h][k] for r in per.get(m, []) if h in r["tce_halo"]}

    lines.append("## Paired tests on the pre-specified primary endpoint")
    lines.append("")
    lines.append("| pair | endpoint | n paired | median A | median B | A>B | Wilcoxon p |")
    lines.append("|---|---|---|---|---|---|---|")
    pairs = [("CA-AKF", "LRS-final"), ("SC-AKF", "LRS-final"), ("AKF", "LRS-final"),
             ("CA-AKF", "SC-AKF")]
    for a, b in pairs:
        if a not in per or b not in per:
            continue
        va, vb = vals(a, PRIMARY, "frac1"), vals(b, PRIMARY, "frac1")
        common = sorted(set(va) & set(vb))
        if len(common) < 6:
            continue
        x, y = np.array([va[c] for c in common]), np.array([vb[c] for c in common])
        d = x - y
        try:
            st, p = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            p = float("nan")
        lines.append(f"| {a} vs {b} | frac>1 level @{PRIMARY} | {len(common)} | "
                     f"{np.median(x)*100:.3f} % | {np.median(y)*100:.3f} % | "
                     f"{int((d > 0).sum())}/{len(common)} | {p:.3g} |")
    lines.append("")

    # secondary: does the deviation concentrate on the grid?  class-A vs class-B
    lines.append("## Secondary (descriptive): grid-line concentration of the deviation")
    lines.append("")
    for m in methods:
        gv = gvisable(per[m])
        if gv.size:
            lines.append(f"- {m}: median ratio {np.median(gv):.1f}; "
                         f"{int((gv > 4).sum())}/{gv.size} images with deviation energy "
                         f">4x concentrated on tile boundaries; "
                         f"{int((gv < 2).sum())}/{gv.size} with <2x (not hideable as a seam)")
    lines.append("")

    txt = "\n".join(lines)
    with open(f"{OUT}/broad.md", "w") as fh:
        fh.write(txt)
    print(txt)

    # figure: median rmse-vs-halo ladder per method
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [0, 16, 32]
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for m in methods:
            med = [np.median([r["tce_halo"][f"h{h}"]["frac1"] for r in per[m]
                              if f"h{h}" in r["tce_halo"]]) * 100 for h in xs]
            ax.plot(xs, med, marker="o", label=m)
        ax.set_xlabel("overlap halo (px)")
        ax.set_ylabel("median fraction of pixels\n> one 8-bit level (%)")
        ax.set_title(f"broad replication, n={n_union} images")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{FIG}/F7_broad.png", dpi=300)
        print("wrote figures/F7_broad.png")
    except Exception as exc:
        print("figure failed:", type(exc).__name__, exc)


if __name__ == "__main__":
    main()
