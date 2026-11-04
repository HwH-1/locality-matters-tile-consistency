"""Supplementary sensitivity table and Pareto figure."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TEX = ROOT.parent / "02_manuscript" / "tables"
FIG = ROOT.parent / "02_manuscript" / "figures"
PREV = ROOT / "figures"
sys.path.insert(0, str(ROOT))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

CHOSEN = {"classic": "r=4", "sector": "r=4", "akf": "r=5",
          "bilateral": "ss=8,sr=0.15", "rgf": "sc=0.05,ss=8",
          "guided": "r=4,eps=0.01"}
LABEL = {"classic": "Kuwahara", "sector": "Sector", "akf": "AKF",
         "bilateral": "Bilateral", "rgf": "RGF", "guided": "Guided"}
COLOR = {"classic": "#7f7f7f", "sector": "#56B4E9", "akf": "#0072B2",
         "bilateral": "#009E73", "rgf": "#CC79A7", "guided": "#8C564B"}


def main() -> None:
    grid = pd.read_csv(RES / "baseline_sweep_grid.csv")
    df = pd.read_csv(RES / "metrics_full.csv")
    pilot = {"camera", "astronaut", "synth_checker", "synth_regions"}
    ca = df[(df.method == "ca_akf") & (df.image.isin(pilot))][["ssim", "fsr"]].median()

    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Baseline parameter sweep on the four tuning images. For each baseline family the table gives the number of configurations, the range of median FSR and SSIM over those configurations, and the configuration used in the main experiment. No baseline setting dominates CA-AKF in the FSR--SSIM plane (Fig.~S1).}",
        "\\label{tab:sensitivity}", "\\footnotesize",
        "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
        "Baseline & Configs & FSR range & SSIM range & Used setting & FSR & SSIM \\\\",
        "\\midrule",
    ]
    for m in ["classic", "sector", "akf", "bilateral", "rgf", "guided"]:
        g = grid[grid.method == m]
        used = g[g.setting == CHOSEN[m]].iloc[0]
        lines.append(
            f"{LABEL[m]} & {len(g)} & {g.fsr.min():.3f}--{g.fsr.max():.3f} & "
            f"{g.ssim.min():.3f}--{g.ssim.max():.3f} & {CHOSEN[m].replace(',', ', ')} & "
            f"{used.fsr:.3f} & {used.ssim:.3f} \\\\")
    lines += ["\\midrule",
              f"CA-AKF & -- & -- & -- & $r=3,6$; $\\beta=8$ & {ca.fsr:.3f} & {ca.ssim:.3f} \\\\",
              "\\bottomrule", "\\end{tabular}",
              "\\par\\smallskip\\raggedright\\footnotesize Values are medians over the four tuning images. The CA-AKF row is shown for reference and was selected by the separate 20-configuration pilot described in the main text.",
              "\\end{table}"]
    (TEX / "table_sensitivity.tex").write_text("\n".join(lines) + "\n")

    # Pareto figure
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for m in ["classic", "sector", "akf"]:
        g = grid[grid.method == m].copy()
        g["r"] = g.setting.str.extract(r"r=(\d)").astype(int)
        g = g.sort_values("r")
        ax.plot(g.ssim, g.fsr, marker="o", ms=3.0, lw=0.9, color=COLOR[m], label=LABEL[m])
        for _, row in g.iterrows():
            ax.annotate(str(row.r), (row.ssim, row.fsr), fontsize=5.2,
                        xytext=(3, 1), textcoords="offset points", color=COLOR[m])
    for m in ["bilateral", "rgf", "guided"]:
        g = grid[grid.method == m]
        ax.scatter(g.ssim, g.fsr, s=10, marker="s", color=COLOR[m], alpha=0.75,
                   edgecolors="white", linewidths=0.3, label=LABEL[m], zorder=3)
    ax.scatter([ca.ssim], [ca.fsr], s=70, marker="*", color="#D55E00",
               edgecolors="black", linewidths=0.4, label="CA-AKF", zorder=5)
    ax.set_xlabel("SSIM (higher is better)")
    ax.set_ylabel("FSR (lower is better)")
    ax.set_xlim(0.42, 1.0)
    ax.set_ylim(0.0, 0.50)
    ax.tick_params(length=2)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, ncol=3, handlelength=1.0, columnspacing=0.8,
              loc="upper left", bbox_to_anchor=(0.0, 1.02))
    fig.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.20)
    fig.savefig(FIG / "FigS1_pareto.pdf")
    fig.savefig(PREV / "FigS1_pareto.png", dpi=300)
    plt.close(fig)

    # CA-AKF one-at-a-time sensitivity figure.
    # Ratios to the chosen setting keep both metrics on one axis and avoid
    # overlapping twin-axis labels in the narrow panels.
    sens = pd.read_csv(RES / "sensitivity_ca.csv")
    panels = [
        ("beta", r"$\beta$", 8.0), ("gamma", r"$\gamma$", 2.0),
        ("sigma_along", r"$\sigma_\parallel$", 2.5), ("sigma_t", r"$\rho$", 4.0),
        ("r_max", r"$r_{\max}$", 6.0), ("r_min", r"$r_{\min}$", 3.0),
        ("bins", "orientation bins", 12.0), ("sectors", "sectors", 8.0),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.4))
    for ax, (param, label, chosen) in zip(axes.ravel(), panels):
        g = sens[sens.parameter == param].sort_values("value")
        x = g.value.to_numpy(float)
        f0 = float(g.loc[g.value == chosen, "fsr_median"].iloc[0])
        s0 = float(g.loc[g.value == chosen, "ssim_median"].iloc[0])
        ax.plot(x, g.fsr_median / f0, marker="o", ms=3.0, lw=1.0,
                color="#D55E00", label="FSR (lower better)")
        ax.plot(x, g.ssim_median / s0, marker="s", ms=2.6, lw=0.9,
                color="#0072B2", label="SSIM (higher better)")
        ax.axhline(1.0, color="#777777", lw=0.6, ls="--")
        ax.axvline(chosen, color="#555555", lw=0.7, ls=":")
        ax.set_xlabel(label)
        ax.set_ylim(0.78, 2.60)
        ax.tick_params(length=2)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    axes[0, 0].set_ylabel("ratio to chosen setting")
    axes[1, 0].set_ylabel("ratio to chosen setting")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.01), handlelength=1.4, columnspacing=1.4)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.12,
                        wspace=0.32, hspace=0.55)
    fig.savefig(FIG / "FigS2_sensitivity.pdf")
    fig.savefig(PREV / "FigS2_sensitivity.png", dpi=300)
    plt.close(fig)
    print("wrote table_sensitivity.tex, FigS1_pareto and FigS2_sensitivity")


if __name__ == "__main__":
    main()
