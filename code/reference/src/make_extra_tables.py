"""Generate supplementary tables from the result CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TEX = ROOT.parent / "02_manuscript" / "tables"
sys.path.insert(0, str(ROOT))

LABELS = {
    "classic": "Kuwahara (1976)",
    "sector": "Sector Kuwahara",
    "akf": "AKF (fixed scale)",
    "bilateral": "Bilateral",
    "rgf": "RGF",
    "guided": "Guided filter",
    "ca_akf": "CA-AKF (base)",
    "jlr_cakf": "JLR-CAKF (ours)",
    "jlr_junction": "JLR: junction only",
    "jlr_line": "JLR: line only",
    "jlr_global": "JLR: global blend",
}
MAIN = ["classic", "sector", "akf", "bilateral", "rgf", "guided", "ca_akf", "jlr_cakf"]


def per_image() -> None:
    df = pd.read_csv(RES / "metrics_all.csv")
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Per-image comparison of CA-AKF and JLR-CAKF for SR, ESR and ECE. Each row is one $256\times256$ test image; values are raw scores, not medians. FSR is omitted because it is undefined for two images.}",
        r"\label{tab:supp_perimage}", r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
        r"Image & SR$_{\mathrm{CA}}$ & SR$_{\mathrm{JLR}}$ & ESR$_{\mathrm{CA}}$ & ESR$_{\mathrm{JLR}}$ & ECE$_{\mathrm{CA}}$ & ECE$_{\mathrm{JLR}}$ \\",
        r"\midrule",
    ]
    for image in df.image.unique():
        row = {}
        for method in ("ca_akf", "jlr_cakf"):
            r = df[(df.image == image) & (df.method == method)].iloc[0]
            row[method] = r
        name = image.replace("_", r"\_")
        lines.append(
            f"{name} & {row['ca_akf'].sr:.3f} & {row['jlr_cakf'].sr:.3f} & "
            f"{row['ca_akf'].esr:.3f} & {row['jlr_cakf'].esr:.3f} & "
            f"{row['ca_akf'].ece:.2f} & {row['jlr_cakf'].ece:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TEX / "table_perimage.tex").write_text("\n".join(lines) + "\n")


def iqr() -> None:
    agg = pd.read_csv(RES / "aggregate.csv").set_index("method")
    metrics = [("sr", "SR$\\uparrow$"), ("ssim", "SSIM$\\uparrow$"),
               ("fsr", "FSR$\\downarrow$"), ("scg", "SCG$\\uparrow$"),
               ("esr", "ESR$\\to1$"), ("ece", "ECE$\\downarrow$"),
               ("ovt", "OVT$\\downarrow$")]
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Median and interquartile range for every method and metric on the twelve images (FSR: ten images). Cells are median [Q1, Q3].}",
        r"\label{tab:supp_iqr}", r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}l" + "c" * len(metrics) + r"@{}}", r"\toprule",
        "Method & " + " & ".join(h for _, h in metrics) + r" \\", r"\midrule",
    ]
    for m in MAIN:
        r = agg.loc[m]
        cells = []
        for key, _ in metrics:
            med = r[f"{key}_median"]
            q1 = r[f"{key}_q1"]
            q3 = r[f"{key}_q3"]
            d = 2 if key == "ece" else 3
            cells.append(f"{med:.{d}f} [{q1:.{d}f}, {q3:.{d}f}]")
        lines.append(f"{LABELS[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TEX / "table_iqr.tex").write_text("\n".join(lines) + "\n")


def crossmetric() -> None:
    df = pd.read_csv(RES / "cross_metric_holm.csv")
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Effect of the correction scope. For each metric, ``within'' counts baselines significant after Holm correction across the seven baselines; ``all'' counts those still significant when all 49 metric--baseline tests are corrected together. The last column is the smallest all-test adjusted $p$ for that metric.}",
        r"\label{tab:supp_cross}", r"\footnotesize",
        r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
        r"Metric & Within ($\leq 7$) & All ($\leq 49$) & min $p_{\mathrm{Holm,all}}$ \\", r"\midrule",
    ]
    names = {"sr": "SR", "ssim": "SSIM", "fsr": "FSR", "scg": "SCG", "esr": "ESR", "ece": "ECE", "ovt": "OVT"}
    for metric, g in df.groupby("metric", sort=False):
        within = int((g.p_holm < 0.05).sum())
        allsig = int((g.p_holm_all < 0.05).sum())
        lines.append(f"{names[metric]} & {within}/7 & {allsig}/7 & {g.p_holm_all.min():.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TEX / "table_crossmetric.tex").write_text("\n".join(lines) + "\n")


def sensitivity_compact() -> None:
    df = pd.read_csv(RES / "sensitivity_jlr.csv")
    params = [
        ("snr_min", r"$\tau$", "1.5--4"),
        ("line_percentile", "line percentile", "98--99.8"),
        ("w_max", r"$w_{\max}$", "0.6--1.0"),
        ("junction_dilation", "junction dilation", "1--4 px"),
        ("line_dilation", "line dilation", "1--3 px"),
        ("mask_sigma", r"mask $\sigma$", "0.5--2.5 px"),
    ]
    default = df[df.parameter == "default"].iloc[0]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Compact one-at-a-time sensitivity of the repair layer. Ranges span all tested values; the default is compared with the full grid. JC and LR are the junction and line scores of the structure benchmark (medians over twelve phantoms). The full grid is reported in the supplementary material.}",
        r"\label{tab:sensitivity_compact}", r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
        r"Parameter & Values & SR & ESR & ECE & JC & LR & FSR \\", r"\midrule",
    ]
    for key, label, rng in params:
        g = df[df.parameter == key]
        lines.append(
            f"{label} & {rng} & {g.sr.min():.3f}--{g.sr.max():.3f} & "
            f"{g.esr.min():.3f}--{g.esr.max():.3f} & {g.ece.min():.2f}--{g.ece.max():.2f} & "
            f"{g.junction_score.min():.3f}--{g.junction_score.max():.3f} & "
            f"{g.line_score.min():.3f}--{g.line_score.max():.3f} & "
            f"{g.fsr.min():.3f}--{g.fsr.max():.3f} \\\\"
        )
    lines += [
        r"\midrule",
        f"Default & -- & {default.sr:.3f} & {default.esr:.3f} & {default.ece:.2f} & "
        f"{default.junction_score:.3f} & {default.line_score:.3f} & {default.fsr:.3f} \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    (TEX / "table_sensitivity_compact.tex").write_text("\n".join(lines) + "\n")


def ablation_compact() -> None:
    df = pd.read_csv(RES / "ablation_tests.csv")
    names = {"jlr_junction": "junction only", "jlr_line": "line only", "jlr_global": "global blend"}

    def holm(p: np.ndarray) -> np.ndarray:
        order = np.argsort(p)
        m = len(p)
        adj = np.empty(m)
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, min(1.0, (m - rank) * p[idx]))
            adj[idx] = running
        return adj

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Paired ablation contrasts against the full model. $\Delta$ is the median paired difference (full minus variant); a positive SR/ESR/SSIM difference and a negative ECE difference favour the full model. The last column counts metrics with Holm-adjusted $p<0.05$ (seven tests per variant).}",
        r"\label{tab:supp_ablation_tests}", r"\footnotesize",
        r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
        r"Variant & $\Delta$SR & $\Delta$ESR & $\Delta$ECE & Significant \\", r"\midrule",
    ]
    for key in ("jlr_junction", "jlr_line", "jlr_global"):
        g = df[df.baseline == key].set_index("metric")
        p_holm = holm(g.p_raw.to_numpy(float))
        sig = int((p_holm < 0.05).sum())
        lines.append(
            f"{names[key]} & {g.loc['sr','median_diff']:+.3f} & "
            f"{g.loc['esr','median_diff']:+.3f} & {g.loc['ece','median_diff']:+.2f} & {sig}/7 \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TEX / "table_ablation_tests.tex").write_text("\n".join(lines) + "\n")


def runtime() -> None:
    df = pd.read_csv(RES / "runtime_scaling.csv")
    p = df.pivot(index="size", columns="method", values="median_s") * 1000.0
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Runtime scaling (median of three runs, single-thread CPU, milliseconds). Sizes are square; the exponent $\alpha$ is a log--log fit over the five sizes.}",
        r"\label{tab:runtime}", r"\footnotesize",
        r"\begin{tabular}{@{}rcccc@{}}", r"\toprule",
        r"Size & Kuwahara & AKF & CA-AKF & JLR-CAKF \\", r"\midrule",
    ]
    for size, row in p.iterrows():
        lines.append(f"${size}^2$ & {row['classic']:.1f} & {row['akf']:.0f} & {row['ca_akf']:.0f} & {row['jlr_cakf']:.0f} \\\\")
    lines += [r"\midrule"]
    exps = {}
    for m in ("classic", "akf", "ca_akf", "jlr_cakf"):
        g = df[df.method == m]
        exps[m] = np.polyfit(np.log(g.megapixels), np.log(g.median_s), 1)[0]
    lines.append(f"$\\alpha$ & {exps['classic']:.2f} & {exps['akf']:.2f} & {exps['ca_akf']:.2f} & {exps['jlr_cakf']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TEX / "table_runtime.tex").write_text("\n".join(lines) + "\n")


def mask_table() -> None:
    df = pd.read_csv(RES / "mask_stats.csv")
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Per-image repair-mask coverage in percent of pixels (threshold 0.01) and mean blend weight. ``Strong'' counts weights above 0.5. Values are per image; the last row is the median.}",
        r"\label{tab:mask_perimage}", r"\footnotesize",
        r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
        r"Image & Junction & Line & Combined & Strong & Mean weight \\", r"\midrule",
    ]
    for _, r in df.iterrows():
        name = r.image.replace("_", r"\_")
        lines.append(f"{name} & {r.junction_pct:.1f} & {r.line_pct:.1f} & {r.combined_pct:.1f} & {r.strong_pct:.1f} & {r.mean_weight_pct:.1f} \\\\")
    lines += [r"\midrule"]
    med = df.drop(columns="image").median()
    lines.append(f"Median & {med.junction_pct:.1f} & {med.line_pct:.1f} & {med.combined_pct:.1f} & {med.strong_pct:.1f} & {med.mean_weight_pct:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TEX / "table_mask.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    per_image()
    iqr()
    crossmetric()
    sensitivity_compact()
    ablation_compact()
    runtime()
    mask_table()
    print("wrote supplementary tables")


if __name__ == "__main__":
    main()
