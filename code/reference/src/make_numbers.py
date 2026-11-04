"""Aggregate results, run paired tests, and emit LaTeX tables/macros + a summary."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TEX = ROOT.parent / "02_manuscript" / "tables"
TEX.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from src.metrics import METRICS, METRIC_LABELS, aggregate, paired_tests  # noqa: E402

MAIN = ["classic", "sector", "akf", "bilateral", "rgf", "guided", "ca_akf"]
ABL = ["ab_no_adaptive", "ab_no_soft", "ab_iso_pre", "ca_akf"]
LOWER_BETTER = {"fsr", "ece", "ovt"}
TARGET_ONE = {"esr"}
LABELS = {
    "classic": "Kuwahara (1976)",
    "sector": "Sector Kuwahara",
    "akf": "AKF (fixed scale)",
    "bilateral": "Bilateral",
    "rgf": "RGF",
    "guided": "Guided filter",
    "ca_akf": "CA-AKF (ours)",
    "ab_no_adaptive": "Fixed scale",
    "ab_no_soft": "Hard minimum",
    "ab_iso_pre": "Isotropic pre-filter",
}

df = pd.read_csv(RES / "metrics_full.csv")
agg = aggregate(df).set_index("method")
FAMILIES = {
    "kuwahara": ["classic", "sector", "akf", "bilateral"],
    "reference": ["rgf", "guided"],
}
frames = []
for fam, bases in FAMILIES.items():
    t = paired_tests(df, "ca_akf", bases, METRICS)
    t["family"] = fam
    frames.append(t)
tests = pd.concat(frames, ignore_index=True)
tests.to_csv(RES / "paired_tests.csv", index=False, float_format="%.6f")
agg.to_csv(RES / "aggregate.csv", float_format="%.6f")

scaling = pd.read_csv(RES / "runtime_scaling.csv")
exponents = {}
for method, g in scaling.groupby("method"):
    x = np.log(g["megapixels"].to_numpy(float))
    y = np.log(g["median_s"].to_numpy(float))
    exponents[method] = float(np.polyfit(x, y, 1)[0])


def fmt(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}"


def cell(method: str, metric: str, bold: bool = False) -> str:
    a = agg.loc[method]
    med = fmt(a[f"{metric}_median"])
    if bold:
        med = f"\\textbf{{{med}}}"
    iqr = f"[{fmt(a[f'{metric}_q1'])}, {fmt(a[f'{metric}_q3'])}]"
    return (f"\\begin{{tabular}}{{@{{}}c@{{}}}}{med}\\\\[-1pt]"
            f"{{\\scriptsize {iqr}}}\\end{{tabular}}")


def best(methods: list[str], metric: str) -> str:
    vals = {m: agg.loc[m, f"{metric}_median"] for m in methods}
    if metric in TARGET_ONE:
        return min(vals, key=lambda m: abs(vals[m] - 1.0))
    if metric in LOWER_BETTER:
        return min(vals, key=lambda m: vals[m])
    return max(vals, key=lambda m: vals[m])


# ---------------- Table 2: main comparison ----------------
lines = []
lines.append("\\begin{table*}[t]")
lines.append("\\centering")
lines.append("\\caption{Quantitative comparison on the twelve test images. Medians over the twelve images, with the interquartile range below each median; the best value per column is in bold. RGF and the guided filter are modern edge-preserving reference smoothers rather than Kuwahara variants. Arrows indicate whether higher or lower is better; ESR is ideal at 1.}")
lines.append("\\label{tab:main}")
lines.append("\\footnotesize")
lines.append("\\begin{tabular}{@{}lcccccccc@{}}")
lines.append("\\toprule")
lines.append("Method & SR$\\uparrow$ & SSIM$\\uparrow$ & FSR$\\downarrow$ & SCG$\\uparrow$ & ESR$\\to 1$ & ECE$\\downarrow$ & OVT$\\downarrow$ & Time (s)$\\downarrow$ \\\\")
lines.append("\\midrule")
for m in MAIN:
    cells = [cell(m, k, bold=(best(MAIN, k) == m)) for k in METRICS]
    t = fmt(agg.loc[m, "runtime_s"], 3)
    if m == min(MAIN, key=lambda z: agg.loc[z, "runtime_s"]):
        t = f"\\textbf{{{t}}}"
    lines.append(f"{LABELS[m]} & " + " & ".join(cells) + f" & {t} \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table*}")
(TEX / "table_main.tex").write_text("\n".join(lines) + "\n")

# ---------------- Table 3: paired tests ----------------
metric_order = ["sr", "ssim", "fsr", "scg", "ece", "ovt"]
lines = []
lines.append("\\begin{table*}[t]")
lines.append("\\centering")
lines.append("\\caption{Paired differences (CA-AKF minus baseline) in median values over the twelve images, with Holm-adjusted Wilcoxon signed-rank significance computed within two baseline families (the four Kuwahara/edge-preserving baselines and the two reference smoothers). Positive values mean CA-AKF is larger. For FSR, ECE and OVT, negative values are better. $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}")
lines.append("\\label{tab:tests}")
lines.append("\\footnotesize")
lines.append("\\setlength{\\tabcolsep}{4pt}")
lines.append("\\begin{tabular}{@{}lcccccc@{}}")
lines.append("\\toprule")
lines.append("Metric & vs. Kuw. & vs. Sector & vs. AKF & vs. Bilateral & vs. RGF & vs. Guided \\\\")
lines.append("\\midrule")
base_cols = ["classic", "sector", "akf", "bilateral", "rgf", "guided"]
for met in metric_order:
    row = []
    for b in base_cols:
        r = tests[(tests.metric == met) & (tests.baseline == b)].iloc[0]
        n_star = 0
        if r.p_holm < 0.001:
            n_star = 3
        elif r.p_holm < 0.01:
            n_star = 2
        elif r.p_holm < 0.05:
            n_star = 1
        sign = "+" if r.median_diff >= 0 else "-"
        body = f"{sign}{abs(r.median_diff):.3f}"
        if n_star:
            row.append(f"${body}^{{{'*' * n_star}}}$")
        else:
            row.append(f"${body}$")
    lines.append(f"{METRIC_LABELS[met]} & " + " & ".join(row) + " \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table*}")
(TEX / "table_tests.tex").write_text("\n".join(lines) + "\n")

# ---------------- Table 4: ablation ----------------
lines = []
lines.append("\\begin{table}[t]")
lines.append("\\centering")
lines.append("\\caption{Ablation study. Medians over the twelve images; the best value per column is in bold.}")
lines.append("\\label{tab:ablation}")
lines.append("\\small")
lines.append("\\begin{tabular}{@{}lccccc@{}}")
lines.append("\\toprule")
lines.append("Variant & SR$\\uparrow$ & SSIM$\\uparrow$ & FSR$\\downarrow$ & SCG$\\uparrow$ & ECE$\\downarrow$ \\\\")
lines.append("\\midrule")
for m in ABL:
    cells = []
    for k in ["sr", "ssim", "fsr", "scg", "ece"]:
        v = fmt(agg.loc[m, f"{k}_median"])
        if best(ABL, k) == m:
            v = f"\\textbf{{{v}}}"
        cells.append(v)
    name = "CA-AKF (full)" if m == "ca_akf" else LABELS[m]
    lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")
(TEX / "table_ablation.tex").write_text("\n".join(lines) + "\n")

# ---------------- Table 5: runtime ----------------
lines = []
lines.append("\\begin{table}[t]")
lines.append("\\centering")
lines.append("\\caption{Median single-thread runtime at $256\\times256$ pixels and empirical scaling exponent $\\alpha$ from a log--log fit between 128$^2$ and 384$^2$ pixels. Relative runtime is normalised by the classic Kuwahara filter.}")
lines.append("\\label{tab:runtime}")
lines.append("\\small")
lines.append("\\begin{tabular}{@{}lccc@{}}")
lines.append("\\toprule")
lines.append("Method & 256$^2$ (ms) & $\\alpha$ & Relative \\\\")
lines.append("\\midrule")
tcl = agg.loc["classic", "runtime_s"]
for m in ["classic", "akf", "ca_akf"]:
    lines.append(f"{LABELS[m]} & {agg.loc[m, 'runtime_s']*1000:.0f} & {exponents[m]:.2f} & {agg.loc[m, 'runtime_s']/tcl:.1f}$\\times$ \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")
(TEX / "table_runtime.tex").write_text("\n".join(lines) + "\n")

# ---------------- Inline macros ----------------
mac = []
for m in list(dict.fromkeys(MAIN + ABL)):
    tag = {
        "classic": "Classic", "sector": "Sector", "akf": "AKF", "bilateral": "Bilateral",
        "rgf": "RGF", "guided": "Guided",
        "ca_akf": "CA", "ab_no_adaptive": "NoAdapt", "ab_no_soft": "NoSoft", "ab_iso_pre": "IsoPre",
    }[m]
    for k in METRICS:
        mac.append(f"\\newcommand{{\\{tag}{k.upper()}}}{{{fmt(agg.loc[m, f'{k}_median'])}}}")
    mac.append(f"\\newcommand{{\\{tag}Time}}{{{fmt(agg.loc[m, 'runtime_s'], 2)}}}")

for _, r in tests.iterrows():
    tag = {"classic": "Classic", "sector": "Sector", "akf": "AKF", "bilateral": "Bilateral",
           "rgf": "RGF", "guided": "Guided"}[r.baseline]
    mac.append(f"\\newcommand{{\\P{tag}{r.metric.upper()}}}{{{r.p_holm:.4f}}}")
    mac.append(f"\\newcommand{{\\D{tag}{r.metric.upper()}}}{{{r.median_diff:+.3f}}}")
    mac.append(f"\\newcommand{{\\R{tag}{r.metric.upper()}}}{{{r.rank_biserial:+.2f}}}")
for m, a in exponents.items():
    tag = {"classic": "Classic", "akf": "AKF", "ca_akf": "CA"}[m]
    mac.append(f"\\newcommand{{\\Alpha{tag}}}{{{a:.2f}}}")

(TEX / "numbers.tex").write_text("\n".join(mac) + "\n")

# ---------------- Human-readable summary ----------------
rows = []
for m in list(dict.fromkeys(MAIN + ABL)):
    a = agg.loc[m]
    rows.append(f"| {LABELS[m]} | {a['sr_median']:.3f} [{a['sr_q1']:.3f}, {a['sr_q3']:.3f}] | "
                f"{a['ssim_median']:.3f} | {a['fsr_median']:.3f} [{a['fsr_q1']:.3f}, {a['fsr_q3']:.3f}] | "
                f"{a['scg_median']:.3f} | {a['esr_median']:.3f} | {a['ece_median']:.2f} | {a['ovt_median']:.3f} | "
                f"{a['runtime_s']:.2f} |")
summary = [
    "# 结果摘要（由 results/metrics_full.csv 自动生成）",
    "",
    "| Method | SR | SSIM | FSR | SCG | ESR | ECE | OVT | Time (s) |",
    "|---|---|---|---|---|---|---|---|---|",
    *rows,
    "",
    "## 关键配对检验（CA-AKF vs 基线，Holm 校正）",
    "",
    "| Metric | Baseline | Δ median | p_holm | rank-biserial |",
    "|---|---|---|---|---|",
]
for _, r in tests.iterrows():
    summary.append(f"| {r.metric} | {r.baseline} | {r.median_diff:+.3f} | {r.p_holm:.4f} | {r.rank_biserial:+.2f} |")
summary += ["", "## 运行时间缩放指数", ""]
for m, a in exponents.items():
    summary.append(f"- {m}: alpha = {a:.2f}")
(RES / "results_summary.md").write_text("\n".join(summary) + "\n")

print(agg.loc[MAIN, ["sr_median", "ssim_median", "fsr_median", "scg_median", "esr_median", "ece_median", "ovt_median"]].round(3).to_string())
print("exponents:", exponents)
print("tables ->", TEX)
