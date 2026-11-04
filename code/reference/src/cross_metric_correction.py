"""Holm correction across metrics, as a sensitivity analysis."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TEX = ROOT.parent / "02_manuscript" / "tables"
sys.path.insert(0, str(ROOT))

from src.metrics import METRICS  # noqa: E402


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    out = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        out[idx] = running
    return out


def main() -> None:
    tests = pd.read_csv(RES / "paired_tests.csv")
    akf = tests[tests.baseline == "akf"].set_index("metric").loc[METRICS].reset_index()
    akf["p_across_metrics"] = holm(akf.p_raw.to_numpy(float))
    akf.to_csv(RES / "cross_metric_holm.csv", index=False, float_format="%.6f")

    kuw = tests[tests.family == "kuwahara"].copy()
    kuw["p_across_family"] = np.nan
    for metric in METRICS:
        mask = kuw.metric == metric
        # Holm across all 7 metrics x 4 baselines within the Kuwahara family.
        pass
    kuw["p_across_family"] = holm(kuw.p_raw.to_numpy(float))

    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Holm correction across all seven metrics for the CA-AKF versus fixed-scale AKF comparison. The second column is the raw two-sided Wilcoxon $p$ value, the third is the correction used in the main text (within metric, across the four Kuwahara-family baselines), and the fourth corrects across the seven metrics as well.}",
        "\\label{tab:crossmetric}", "\\small",
        "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
        "Metric & $p_{\\mathrm{raw}}$ & $p_{\\mathrm{Holm}}$ (main) & $p_{\\mathrm{Holm}}$ (across metrics) \\\\",
        "\\midrule",
    ]
    for _, r in akf.iterrows():
        lines.append(f"{r.metric.upper()} & {r.p_raw:.4f} & {r.p_holm:.4f} & {r.p_across_metrics:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TEX / "table_crossmetric.tex").write_text("\n".join(lines) + "\n")

    six = akf[akf.metric != "ovt"].p_across_metrics.max()
    ovt = float(akf.loc[akf.metric == "ovt", "p_across_metrics"].iloc[0])
    fam_six = kuw[(kuw.baseline == "akf") & (kuw.metric != "ovt")].p_across_family.max()
    fam_ovt = float(kuw[(kuw.baseline == "akf") & (kuw.metric == "ovt")].p_across_family.iloc[0])
    print(akf[["metric", "p_raw", "p_holm", "p_across_metrics"]].round(5).to_string(index=False))
    print(f"six metrics max across-metric Holm p = {six:.5f}; OVT = {ovt:.5f}")
    print(f"family-wide Holm: six max = {fam_six:.5f}; OVT = {fam_ovt:.5f}")
    print("wrote cross_metric_holm.csv and table_crossmetric.tex")


if __name__ == "__main__":
    main()
