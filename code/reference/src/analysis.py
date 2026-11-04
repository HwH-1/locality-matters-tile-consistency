"""Statistical analysis and LaTeX tables for the JLR-CAKF study."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TEX = ROOT.parent / "02_manuscript" / "tables"
sys.path.insert(0, str(ROOT))

from src.metrics import HIGHER_IS_BETTER, METRICS, paired_tests  # noqa: E402

TEX.mkdir(parents=True, exist_ok=True)
MAIN_ORDER = [
    "classic",
    "sector",
    "akf",
    "bilateral",
    "rgf",
    "guided",
    "ca_akf",
    "jlr_cakf",
]
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
METRIC_FMT = {
    "sr": "%.3f",
    "ssim": "%.3f",
    "fsr": "%.3f",
    "scg": "%.3f",
    "esr": "%.3f",
    "ece": "%.2f",
    "ovt": "%.3f",
}


def load() -> pd.DataFrame:
    main = pd.read_csv(RES / "metrics_main.csv")
    abl = pd.read_csv(RES / "metrics_ablation.csv")
    df = pd.concat([main, abl], ignore_index=True)
    df.to_csv(RES / "metrics_all.csv", index=False, float_format="%.6f")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in LABELS:
        g = df[df.method == method]
        if g.empty:
            continue
        row = {
            "method": method,
            "method_label": LABELS[method],
            "n": len(g),
            "runtime_ms": float(g.runtime_s.median() * 1000.0),
        }
        for m in METRICS:
            row[f"{m}_median"] = float(g[m].median())
            row[f"{m}_q1"] = float(g[m].quantile(0.25))
            row[f"{m}_q3"] = float(g[m].quantile(0.75))
            row[f"{m}_n"] = int(g[m].notna().sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "aggregate.csv", index=False, float_format="%.6f")
    return out


def wilcoxon_table(df: pd.DataFrame, proposed: str, baselines: list[str], out_name: str) -> pd.DataFrame:
    tests = paired_tests(df, proposed, baselines, METRICS)
    tests.to_csv(RES / out_name, index=False, float_format="%.6f")
    return tests


def cross_metric_holm(df: pd.DataFrame, proposed: str, baselines: list[str]) -> pd.DataFrame:
    tests = paired_tests(df, proposed, baselines, METRICS).copy()
    p = tests["p_raw"].to_numpy(float)
    valid = np.isfinite(p)
    adj = np.full_like(p, np.nan)
    signed = p.copy()
    # Holm across all (metric, baseline) tests, ignoring undefined ones.
    order = np.argsort(signed[valid])
    m = int(valid.sum())
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * signed[valid][idx])
        running = max(running, val)
        adj[np.where(valid)[0][idx]] = running
    tests["p_holm_all"] = adj
    tests.to_csv(RES / "cross_metric_holm.csv", index=False, float_format="%.6f")
    return tests


def holm_across_metrics(tests: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Holm correction over the seven metrics for one paired comparison.

    ``paired_tests`` corrects within each metric across baselines, which is the
    right family for the main comparison table.  A table that reports one
    baseline against seven metrics needs the other family: seven hypotheses.
    """
    sub = tests[tests.baseline == baseline].copy()
    p = sub["p_raw"].to_numpy(float)
    valid = np.isfinite(p)
    order = np.argsort(p[valid])
    m = int(valid.sum())
    adj = np.full_like(p, np.nan)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[valid][idx]))
        adj[np.where(valid)[0][idx]] = running
    sub["p_holm_metrics"] = adj
    sub.to_csv(RES / "stats_tests.csv", index=False, float_format="%.6f")
    return sub


def fmt(value: float, key: str) -> str:
    if not np.isfinite(value):
        return "--"
    return METRIC_FMT[key] % value


def main_table(agg: pd.DataFrame) -> None:
    # Compact median table; IQRs are reported in the supplement.
    cols = ["sr", "ssim", "fsr", "scg", "esr", "ece", "ovt"]
    head = [
        "SR$\\uparrow$",
        "SSIM$\\uparrow$",
        "FSR$\\downarrow$",
        "SCG$\\uparrow$",
        "ESR$\\to1$",
        "ECE$\\downarrow$",
        "OVT$\\downarrow$",
        "Time (ms)",
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Main comparison on twelve images (median; interquartile ranges are reported in the supplementary material). SR and SSIM measure source fidelity, FSR flat-region standard deviation (lower is flatter), SCG structure-tensor coherence gain, ESR edge-sharpness ratio (1 is ideal), ECE edge colour error, and OVT overshoot. FSR is undefined for images with fewer than 100 flat pixels (rocket and stripe pattern) and is therefore computed on ten images; all other metrics use twelve. Runtime is the median over images of a single timed run on one CPU thread at $256\times256$. Bold marks the best value among the abstraction methods (Kuwahara family, CA-AKF and JLR-CAKF); the reference smoothers RGF and guided are shown for context but target pixel fidelity rather than abstraction, so their values are not directly comparable.}",
        r"\label{tab:main}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l" + "c" * len(cols) + r"c@{}}",
        r"\toprule",
        "Method & " + " & ".join(head) + r" \\",
        r"\midrule",
    ]
    # Determine best among the Kuwahara family + ours, excluding reference smoothers.
    family = ["classic", "sector", "akf", "ca_akf", "jlr_cakf"]
    best_value: dict[str, float] = {}
    for m in cols:
        sub = agg[agg.method.isin(family)]
        if HIGHER_IS_BETTER[m] is True:
            idx = sub[f"{m}_median"].idxmax()
        elif HIGHER_IS_BETTER[m] is False:
            idx = sub[f"{m}_median"].idxmin()
        else:
            idx = (sub[f"{m}_median"] - 1.0).abs().idxmin()
        best_value[m] = float(sub.loc[idx, f"{m}_median"])
    for method in MAIN_ORDER:
        r = agg[agg.method == method].iloc[0]
        cells = []
        for m in cols:
            val = fmt(r[f"{m}_median"], m)
            if fmt(best_value[m], m) == val:
                val = r"\textbf{" + val + "}"
            cells.append(val)
        label = LABELS[method]
        if method == "jlr_cakf":
            label = r"\textbf{" + label + "}"
        lines.append(
            f"{label} & " + " & ".join(cells) + f" & {r['runtime_ms']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TEX / "table_main.tex").write_text("\n".join(lines) + "\n")


def ablation_table(agg: pd.DataFrame) -> None:
    cols = ["sr", "ssim", "fsr", "scg", "esr", "ece", "ovt"]
    head = [
        "SR$\\uparrow$",
        "SSIM$\\uparrow$",
        "FSR$\\downarrow$",
        "SCG$\\uparrow$",
        "ESR$\\to1$",
        "ECE$\\downarrow$",
        "OVT$\\downarrow$",
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Localisation and component ablations (median over twelve images). The global-blend control applies the same mean repair weight uniformly instead of at detected structures. FSR uses ten images; other metrics use twelve.}",
        r"\label{tab:ablation}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}l" + "c" * len(cols) + r"@{}}",
        r"\toprule",
        "Variant & " + " & ".join(head) + r" \\",
        r"\midrule",
    ]
    for method in ["ca_akf", "jlr_junction", "jlr_line", "jlr_global", "jlr_cakf"]:
        r = agg[agg.method == method].iloc[0]
        cells = [fmt(r[f"{m}_median"], m) for m in cols]
        label = LABELS[method]
        if method == "jlr_cakf":
            label = r"\textbf{" + label + "}"
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TEX / "table_ablation.tex").write_text("\n".join(lines) + "\n")


def stats_table(sub: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{JLR-CAKF versus its CA-AKF base: paired two-sided Wilcoxon signed-rank tests with Holm correction across the seven metrics. $\Delta$ is the median paired difference (ours minus base). Change is the difference of the two marginal medians relative to the baseline median. $r_{\mathrm{rb}}$ is the rank-biserial effect size. FSR uses the ten images for which it is defined. For FSR, ECE and OVT a negative $\Delta$ is better; for the other metrics a positive $\Delta$ is better.}",
        r"\label{tab:stats}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Metric & $\Delta$ & Change & $p_{\mathrm{Holm}}$ & $r_{\mathrm{rb}}$ \\",
        r"\midrule",
    ]
    labels = {
        "sr": "SR",
        "ssim": "SSIM",
        "fsr": "FSR",
        "scg": "SCG",
        "esr": "ESR",
        "ece": "ECE",
        "ovt": "OVT",
    }
    sub = sub.set_index("metric")
    for m in METRICS:
        r = sub.loc[m]
        delta = r.median_diff
        p = r.p_holm_metrics
        p_txt = f"{p:.4f}" if p >= 0.0001 else "$<10^{-4}$"
        lines.append(
            f"{labels[m]} & {delta:+.3f} & {r['pct_change']:+.1f}\\% & {p_txt} & {r.rank_biserial:+.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TEX / "table_stats.tex").write_text("\n".join(lines) + "\n")


def macros(agg: pd.DataFrame, stats: pd.DataFrame) -> None:
    a = agg.set_index("method")
    t = stats.set_index("metric")
    lines = [
        "% Auto-generated by src/analysis.py",
        f"\\newcommand{{\\JlrSr}}{{{a.loc['jlr_cakf','sr_median']:.3f}}}",
        f"\\newcommand{{\\CaSr}}{{{a.loc['ca_akf','sr_median']:.3f}}}",
        f"\\newcommand{{\\JlrSsim}}{{{a.loc['jlr_cakf','ssim_median']:.3f}}}",
        f"\\newcommand{{\\CaSsim}}{{{a.loc['ca_akf','ssim_median']:.3f}}}",
        f"\\newcommand{{\\JlrFsr}}{{{a.loc['jlr_cakf','fsr_median']:.3f}}}",
        f"\\newcommand{{\\CaFsr}}{{{a.loc['ca_akf','fsr_median']:.3f}}}",
        f"\\newcommand{{\\JlrEce}}{{{a.loc['jlr_cakf','ece_median']:.2f}}}",
        f"\\newcommand{{\\CaEce}}{{{a.loc['ca_akf','ece_median']:.2f}}}",
        f"\\newcommand{{\\JlrEsr}}{{{a.loc['jlr_cakf','esr_median']:.3f}}}",
        f"\\newcommand{{\\CaEsr}}{{{a.loc['ca_akf','esr_median']:.3f}}}",
        f"\\newcommand{{\\JlrScg}}{{{a.loc['jlr_cakf','scg_median']:.3f}}}",
        f"\\newcommand{{\\CaScg}}{{{a.loc['ca_akf','scg_median']:.3f}}}",
        f"\\newcommand{{\\JlrTime}}{{{a.loc['jlr_cakf','runtime_ms']:.0f}}}",
        f"\\newcommand{{\\CaTime}}{{{a.loc['ca_akf','runtime_ms']:.0f}}}",
        f"\\newcommand{{\\JlrSrp}}{{{t.loc['sr','p_holm_metrics']:.2g}}}",
        f"\\newcommand{{\\JlrEcep}}{{{t.loc['ece','p_holm_metrics']:.2g}}}",
    ]
    (TEX / "numbers.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    df = load()
    agg = aggregate(df)
    tests = wilcoxon_table(
        df, "jlr_cakf", ["classic", "sector", "akf", "bilateral", "rgf", "guided", "ca_akf"], "paired_tests.csv"
    )
    wilcoxon_table(
        df, "jlr_cakf", ["jlr_junction", "jlr_line", "jlr_global"], "ablation_tests.csv"
    )
    cross_metric_holm(
        df, "jlr_cakf", ["classic", "sector", "akf", "bilateral", "rgf", "guided", "ca_akf"]
    )
    main_table(agg)
    ablation_table(agg)
    stats = holm_across_metrics(tests, "ca_akf")
    stats_table(stats)
    macros(agg, stats)
    print(agg[["method", "sr_median", "ssim_median", "fsr_median", "scg_median", "ece_median"]].round(3).to_string(index=False))
    print("\nJLR vs CA-AKF")
    print(tests[tests.baseline == "ca_akf"][["metric", "median_diff", "pct_change", "p_holm", "rank_biserial", "n"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
