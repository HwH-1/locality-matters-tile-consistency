"""Held-out analysis, bootstrap confidence intervals, and supplementary tables."""
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

from src.metrics import METRICS, aggregate, paired_tests  # noqa: E402

TUNED = ["astronaut", "camera", "synth_checker", "synth_regions"]
MAIN = ["classic", "sector", "akf", "bilateral", "rgf", "guided", "ca_akf"]
LABELS = {
    "classic": "Kuwahara (1976)",
    "sector": "Sector Kuwahara",
    "akf": "AKF (fixed scale)",
    "bilateral": "Bilateral",
    "rgf": "RGF",
    "guided": "Guided filter",
    "ca_akf": "CA-AKF (ours)",
}
RNG = np.random.default_rng(20261025)
N_BOOT = 10000


def fmt(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}"


def boot_ci(values: np.ndarray, stat=np.median, n: int = N_BOOT) -> tuple[float, float]:
    idx = RNG.integers(0, len(values), size=(n, len(values)))
    vals = stat(values[idx], axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_diff_ci(x: np.ndarray, y: np.ndarray, n: int = N_BOOT) -> tuple[float, float]:
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    vals = np.median(x[idx] - y[idx], axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    df = pd.read_csv(RES / "metrics_full.csv")
    held = df[~df.image.isin(TUNED)].copy()

    # ---- Held-out aggregate and tests ----
    held_agg = aggregate(held).set_index("method")
    held_agg.to_csv(RES / "heldout_aggregate.csv", float_format="%.6f")
    base = ["classic", "sector", "akf", "bilateral", "rgf", "guided"]
    held_tests = paired_tests(held, "ca_akf", base, METRICS)
    held_tests.to_csv(RES / "heldout_tests.csv", index=False, float_format="%.6f")

    # ---- Bootstrap CIs (all 12 images) ----
    rows = []
    for m in MAIN:
        g = df[df.method == m]
        for met in METRICS:
            v = g[met].to_numpy(float)
            lo, hi = boot_ci(v)
            rows.append({"method": m, "metric": met, "median": float(np.median(v)),
                         "ci_lo": lo, "ci_hi": hi, "n": len(v)})
    boot = pd.DataFrame(rows)
    boot.to_csv(RES / "bootstrap_ci.csv", index=False, float_format="%.6f")

    drows = []
    for b in base:
        for met in METRICS:
            x = df[df.method == "ca_akf"].sort_values("image")[met].to_numpy(float)
            y = df[df.method == b].sort_values("image")[met].to_numpy(float)
            lo, hi = boot_diff_ci(x, y)
            drows.append({"baseline": b, "metric": met,
                          "median_diff": float(np.median(x - y)),
                          "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(drows).to_csv(RES / "bootstrap_diff_ci.csv", index=False, float_format="%.6f")

    # ---- Table S1: per-image CA-AKF vs fixed-scale AKF ----
    lines = [
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Per-image results for CA-AKF and the strongest Kuwahara-family baseline (fixed-scale AKF). Each cell lists CA-AKF / AKF. SR, SSIM and SCG: higher is better; FSR: lower is better.}",
        "\\label{tab:perimage}", "\\small",
        "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
        "Image & SR$\\uparrow$ & SSIM$\\uparrow$ & FSR$\\downarrow$ & SCG$\\uparrow$ \\\\",
        "\\midrule",
    ]
    order = ["astronaut", "brick", "camera", "chelsea", "coffee", "coins", "hubble", "rocket",
             "synth_checker", "synth_regions", "synth_stripes", "synth_wedges"]
    for im in order:
        g = df[df.image == im].set_index("method")
        cells = []
        for met in ["sr", "ssim", "fsr", "scg"]:
            cells.append(f"{fmt(g.loc['ca_akf', met])} / {fmt(g.loc['akf', met])}")
        marker = "$^{\\dagger}$" if im in TUNED else ""
        lines.append(f"{im.replace('_', r'\_')}{marker} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\smallskip\\raggedright\\footnotesize $^{\\dagger}$ Image used for parameter tuning (the remaining eight form the held-out set).",
              "\\end{table*}"]
    (TEX / "table_perimage.tex").write_text("\n".join(lines) + "\n")

    # ---- Table S2: held-out aggregate with Holm p vs CA-AKF ----
    lines = [
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Results on the eight held-out images only. Medians are reported, with the Holm-adjusted Wilcoxon $p$ value for the difference from CA-AKF in the last column (FSR, ECE, OVT: lower is better). ``--'' marks the proposed method.}",
        "\\label{tab:heldout}", "\\small",
        "\\begin{tabular}{@{}lcccccccc@{}}", "\\toprule",
        "Method & SR$\\uparrow$ & SSIM$\\uparrow$ & FSR$\\downarrow$ & SCG$\\uparrow$ & ESR$\\to 1$ & ECE$\\downarrow$ & OVT$\\downarrow$ & $p_{\\mathrm{Holm}}$ \\\\",
        "\\midrule",
    ]
    for m in MAIN:
        a = held_agg.loc[m]
        cells = [fmt(a[f"{k}_median"]) for k in METRICS]
        if m == "ca_akf":
            pcell = "--"
            name = LABELS[m]
        else:
            r = held_tests[(held_tests.metric == "fsr") & (held_tests.baseline == m)].iloc[0]
            pcell = f"{r.p_holm:.4f}" if r.p_holm >= 1e-4 else "$<0.0001$"
            name = LABELS[m]
        lines.append(f"{name} & " + " & ".join(cells) + f" & {pcell} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\par\\smallskip\\raggedright\\footnotesize The $p$ value refers to FSR, the primary abstraction metric; full per-metric tests are provided in the repository.",
              "\\end{table*}"]
    (TEX / "table_heldout.tex").write_text("\n".join(lines) + "\n")

    # ---- Table S3: bootstrap CIs for FSR and SCG ----
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Bootstrap 95\\% confidence intervals (10{,}000 image resamples) for the median FSR and SCG over the twelve images.}",
        "\\label{tab:bootstrap}", "\\small",
        "\\begin{tabular}{@{}lcc@{}}", "\\toprule",
        "Method & FSR$\\downarrow$ [95\\% CI] & SCG$\\uparrow$ [95\\% CI] \\\\",
        "\\midrule",
    ]
    for m in MAIN:
        b = boot[boot.method == m].set_index("metric")
        lines.append(f"{LABELS[m]} & {fmt(b.loc['fsr','median'])} [{fmt(b.loc['fsr','ci_lo'])}, {fmt(b.loc['fsr','ci_hi'])}] & "
                     f"{fmt(b.loc['scg','median'])} [{fmt(b.loc['scg','ci_lo'])}, {fmt(b.loc['scg','ci_hi'])}] \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TEX / "table_bootstrap.tex").write_text("\n".join(lines) + "\n")

    # ---- Console summaries ----
    print("Held-out (n=8) medians:")
    print(held_agg.loc[MAIN, ["sr_median", "ssim_median", "fsr_median", "scg_median",
                              "esr_median", "ece_median", "ovt_median"]].round(3).to_string())
    print("\nHeld-out tests (CA-AKF vs AKF):")
    print(held_tests[held_tests.baseline == "akf"][
        ["metric", "median_proposed", "median_baseline", "median_diff", "p_raw", "p_holm"]].round(4).to_string(index=False))
    print("\nBootstrap diff CI vs AKF:")
    print(pd.read_csv(RES / "bootstrap_diff_ci.csv").query("baseline == 'akf'").round(3).to_string(index=False))
    print("\nwrote heldout_*.csv, bootstrap_*.csv, table_perimage/heldout/bootstrap.tex")


if __name__ == "__main__":
    main()
