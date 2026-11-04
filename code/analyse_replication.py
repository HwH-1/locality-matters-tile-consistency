#!/usr/bin/env python3
"""Analyse the independent cross-platform replication cohort (23 images).

Produces: quality family with effect sizes and bootstrap CIs, the consistency
family, harness portability, and the cross-platform drift measured on the ten
images shared with the primary cohort. All numbers come from the raw records.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from config import RECORDS as OUT
NBOOT = 20000
SEED = 20261104
METRICS = ["sr", "ssim", "fsr", "ece", "scg", "ovt"]
HIGHER = {"sr", "ssim", "scg"}


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def by_method(rows, key="qual_whole"):
    d = {}
    for r in rows:
        d.setdefault(r["method"], {})[r["img"]] = r[key]
    return d


def analyse_pair(a, b, imgs, metric, rng):
    x = np.array([a[i][metric] for i in imgs], float)
    y = np.array([b[i][metric] for i in imgs], float)
    d = y - x
    sd = d.std(ddof=1)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(NBOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = float(wilcoxon(y, x, zero_method="wilcox", alternative="two-sided").pvalue)
    wins = int(np.sum(d > 0)) if metric in HIGHER else int(np.sum(d < 0))
    return dict(metric=metric, a_mean=x.mean(), a_sd=x.std(ddof=1),
                b_mean=y.mean(), b_sd=y.std(ddof=1), delta=d.mean(),
                dz=d.mean() / sd if sd > 0 else float("nan"),
                ci_lo=lo, ci_hi=hi, wins=wins, n=len(imgs), p=p)


def main():
    rows = load(OUT / "sweep_replication.jsonl")
    methods = sorted({r["method"] for r in rows})
    counts = {}
    for r in rows:
        counts[r["img"]] = counts.get(r["img"], 0) + 1
    imgs = sorted(i for i, c in counts.items() if c == len(methods))
    rows = [r for r in rows if r["img"] in set(imgs)]
    print("records:", len(rows), "images:", len(imgs), "methods:", methods)
    assert len(imgs) >= 10 and len(rows) == len(imgs) * len(methods), "incomplete cohort"

    q = by_method(rows, "qual_whole")
    qt = by_method(rows, "qual_tiled")
    tce = {}
    for r in rows:
        tce.setdefault(r["method"], {})[r["img"]] = r["tce_n"]["n8_h16"]

    rng = np.random.default_rng(SEED)
    quality = [analyse_pair(q["CA-AKF"], q["LRS-final"], imgs, m, rng) for m in METRICS]
    order = np.argsort([r["p"] for r in quality])
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(quality) - rank) * quality[idx]["p"])
        quality[idx]["p_holm"] = min(1.0, running)

    # harness portability: tiled minus whole, per method and metric
    portability = {}
    for m in methods:
        portability[m] = {}
        for k in METRICS:
            d = np.array([qt[m][i][k] - q[m][i][k] for i in imgs], float)
            portability[m][k] = dict(mean=d.mean(), max_abs=float(np.max(np.abs(d))),
                                     n_nonzero=int(np.sum(d != 0)))

    # consistency family at n=8, halo 16
    consistency = {}
    for m in methods:
        v = np.array([tce[m][i]["rmse"] for i in imgs], float)
        consistency[m] = dict(median=float(np.median(v)), mean=float(v.mean()),
                              frac1_median=float(np.median([tce[m][i]["frac1"] for i in imgs])))
    d_con = np.array([tce["CA-AKF"][i]["rmse"] - tce["LRS-final"][i]["rmse"] for i in imgs])
    p_con = float(wilcoxon([tce["CA-AKF"][i]["rmse"] for i in imgs],
                           [tce["LRS-final"][i]["rmse"] for i in imgs],
                           zero_method="wilcox", alternative="two-sided").pvalue)

    # cross-platform drift on the ten shared images
    prim = load(OUT / "sweep.jsonl")
    primf = load(OUT / "sweep_final.jsonl")
    old = {(r["img"], "CA-AKF"): r for r in prim if r["method"] == "CA-AKF"}
    old.update({(r["img"], "LRS-final"): r for r in primf if r["method"] == "LRS-final"})
    shared = [i for i in imgs if (i, "CA-AKF") in old]
    drift = {}
    for m in ["CA-AKF", "LRS-final"]:
        rel = []
        for i in shared:
            o = old[(i, m)]["tce_n"]["n8_h16"]["rmse"]
            v = tce[m][i]["rmse"]
            if o > 0:
                rel.append(abs(v - o) / o)
        drift[m] = dict(n=len(shared), median_rel=float(np.median(rel)),
                        max_rel=float(np.max(rel)))

    report = dict(images=imgs, methods=methods, bootstrap=NBOOT, seed=SEED,
                  quality=quality, portability=portability, consistency=consistency,
                  consistency_p=p_con, cross_platform_drift=drift,
                  shared_images=shared)
    (OUT / "replication_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n| metric | CA-AKF | LRS-AKF | delta | dz | 95% CI | win/{len(imgs)} | p | p_Holm |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in quality:
        print(f"| {r['metric']} | {r['a_mean']:.4f} +/- {r['a_sd']:.4f} | "
              f"{r['b_mean']:.4f} +/- {r['b_sd']:.4f} | {r['delta']:+.4f} | "
              f"{r['dz']:+.2f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
              f"{r['wins']}/{len(imgs)} | {r['p']:.3g} | {r['p_holm']:.3g} |")
    print("\nharness delta (mean, max abs):")
    for m in methods:
        big = max(portability[m].items(), key=lambda kv: abs(kv[1]["mean"]))
        print(f"  {m:12s} largest mean move: {big[0]} {big[1]['mean']:+.5f} "
              f"(max abs {big[1]['max_abs']:.5f})")
    print("\nconsistency n8h16 median rmse:", {k: round(v["median"], 8) for k, v in consistency.items()})
    print("consistency paired p:", p_con)
    print("cross-platform drift:", drift)
    print("\n-> data/records/replication_report.json")


if __name__ == "__main__":
    main()
