#!/usr/bin/env python3
"""Recompute the six-metric quality family with effect sizes and bootstrap CIs.

Inputs are the raw per-image records; no number is taken from the manuscript.
Outputs a Markdown table plus a JSON record for the paper build.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from config import RECORDS as ROOT
METRICS = ["sr", "ssim", "fsr", "ece", "scg", "ovt"]
HIGHER = {"sr", "ssim", "scg"}


def load(path, method):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["img"]: r["qual_whole"] for r in rows if r["method"] == method}


ca = load(ROOT / "sweep.jsonl", "CA-AKF")
lrs = load(ROOT / "sweep_final.jsonl", "LRS-final")
imgs = sorted(set(ca) & set(lrs))
assert len(imgs) == 10, f"expected 10 paired images, got {len(imgs)}: {imgs}"

rng = np.random.default_rng(20261104)
NBOOT = 20000
rows = []
for m in METRICS:
    a = np.array([ca[i][m] for i in imgs], dtype=float)
    b = np.array([lrs[i][m] for i in imgs], dtype=float)
    d = b - a
    mean_d, sd_d = d.mean(), d.std(ddof=1)
    # paired Cohen's dz on the signed difference
    dz = mean_d / sd_d if sd_d > 0 else float("nan")
    # exact two-sided Wilcoxon signed-rank p (n=10)
    p = float(wilcoxon(b, a, zero_method="wilcox", alternative="two-sided").pvalue)
    # paired bootstrap CI of the mean difference
    boot = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(NBOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    wins = int(np.sum(d > 0)) if m in HIGHER else int(np.sum(d < 0))
    rows.append(dict(metric=m, ca_mean=a.mean(), ca_sd=a.std(ddof=1),
                     lrs_mean=b.mean(), lrs_sd=b.std(ddof=1),
                     delta=mean_d, sd_d=sd_d, dz=dz, ci_lo=lo, ci_hi=hi,
                     wins=wins, p=p, n=len(imgs)))

# Holm across the six-metric family
order = np.argsort([r["p"] for r in rows])
running = 0.0
for rank, idx in enumerate(order):
    val = (len(rows) - rank) * rows[idx]["p"]
    running = max(running, val)
    rows[idx]["p_holm"] = min(1.0, running)

out = {"paired_images": imgs, "n": len(imgs), "bootstrap": NBOOT, "seed": 20261104,
       "rows": rows}
(ROOT / "quality_effects.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

print("| metric | CA-AKF | LRS-AKF | delta | dz | 95% CI of delta | win/10 | p | p_Holm |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['metric']} | {r['ca_mean']:.4f} +/- {r['ca_sd']:.4f} | "
          f"{r['lrs_mean']:.4f} +/- {r['lrs_sd']:.4f} | {r['delta']:+.4f} | "
          f"{r['dz']:+.2f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
          f"{r['wins']}/10 | {r['p']:.5f} | {r['p_holm']:.4f} |")
print("\nimages:", ", ".join(imgs))
