"""Generate all manuscript and supplementary figures for JLR-CAKF."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "outputs"
FIG = ROOT.parent / "02_manuscript" / "figures"
PREV = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
PREV.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from src import filters  # noqa: E402
from src.data import load_all  # noqa: E402
from src.failure_probe import boundary_phantom, line_phantom, noise_phantom  # noqa: E402
from src.methods import METHODS, run_method  # noqa: E402
from src.repair import repair_weights  # noqa: E402
from src.structure import structure_tensor  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

C = {
    "classic": "#7f7f7f",
    "sector": "#56B4E9",
    "akf": "#0072B2",
    "bilateral": "#009E73",
    "rgf": "#CC79A7",
    "guided": "#8C564B",
    "ca_akf": "#D55E00",
    "jlr_cakf": "#882255",
    "jlr_junction": "#E69F00",
    "jlr_line": "#56B4E9",
    "jlr_global": "#999999",
}
LABEL = {
    "classic": "Kuwahara 1976",
    "sector": "Sector",
    "akf": "AKF",
    "bilateral": "Bilateral",
    "rgf": "RGF",
    "guided": "Guided",
    "ca_akf": "CA-AKF",
    "jlr_cakf": "JLR-CAKF",
    "jlr_junction": "JLR junction",
    "jlr_line": "JLR line",
    "jlr_global": "JLR global",
}

IMAGES = {im.name: im for im in load_all()}
CFG = next(m for m in METHODS if m.name == "jlr_cakf").params


def save(fig, stem: str, png_dpi: int = 300) -> None:
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(PREV / f"{stem}.png", dpi=png_dpi)
    plt.close(fig)
    print("wrote", stem, flush=True)


def load_out(name: str, method: str) -> np.ndarray:
    return np.asarray(
        Image.open(OUT / f"{name}__{method}.png").convert("RGB"), dtype=np.float32
    ) / 255.0


def hide(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def panel(ax, text, color="black"):
    ax.text(
        0.025,
        0.975,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color=color,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.2),
    )


def masks(name: str):
    return repair_weights(
        IMAGES[name].image,
        line_percentile=CFG["line_percentile"],
        junction_dilation=CFG["junction_dilation"],
        line_dilation=CFG["line_dilation"],
        mask_sigma=CFG["mask_sigma"],
        w_max=CFG["w_max"],
        snr_min=CFG["snr_min"],
        junction_min_distance=CFG["junction_min_distance"],
        junction_rel_threshold=CFG["junction_rel_threshold"],
    )


# ---------------------------------------------------------------------------
# Figure 1: pipeline
# ---------------------------------------------------------------------------
def figure1() -> None:
    name = "camera"
    img = IMAGES[name].image
    mj, ml, w = masks(name)
    base = load_out(name, "ca_akf")
    out = load_out(name, "jlr_cakf")

    fig = plt.figure(figsize=(7.0, 3.05))
    w_box, h_box = 0.20, 0.30
    xs = [0.012, 0.255, 0.498, 0.741]
    y_top, y_bot = 0.665, 0.095
    top = [
        ("Input $I$", img),
        ("CA-AKF base $B$", base),
        ("Local blend", out),
        ("Output $O$", out),
    ]
    for i, (title, im) in enumerate(top):
        x = xs[i]
        fig.patches.append(
            FancyBboxPatch(
                (x, y_top), w_box, h_box,
                boxstyle="round,pad=0.004,rounding_size=0.010",
                linewidth=0.7, edgecolor="#444444",
                facecolor="#f1e6ee" if i == 3 else "#f6f1ee",
                transform=fig.transFigure, zorder=1,
            )
        )
        fig.text(x + 0.007, y_top + h_box - 0.021, title, ha="left", va="top",
                 fontsize=6.7, fontweight="bold" if i in (0, 3) else "normal",
                 transform=fig.transFigure, zorder=5)
        ax = fig.add_axes([x + 0.016, y_top + 0.020, w_box - 0.032, h_box - 0.075])
        ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
        ax.set_zorder(3)
        hide(ax)

    # Reliability gate.
    fig.patches.append(
        FancyBboxPatch(
            (xs[0], y_bot), w_box, h_box,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            linewidth=0.7, edgecolor="#444444", facecolor="#eef3f7",
            transform=fig.transFigure, zorder=1,
        )
    )
    fig.text(xs[0] + 0.007, y_bot + h_box - 0.020, "Reliability gate $Q$", ha="left",
             va="top", fontsize=6.4, transform=fig.transFigure, zorder=5)
    from src.repair import reliability_map

    ax = fig.add_axes([xs[0] + 0.016, y_bot + 0.018, w_box - 0.032, h_box - 0.070])
    ax.imshow(reliability_map(img, CFG["snr_min"]), cmap="gray", interpolation="nearest")
    ax.set_zorder(3)
    hide(ax)

    # Wide gated-mask box.
    wide_x, wide_w = 0.255, 0.686
    fig.patches.append(
        FancyBboxPatch(
            (wide_x, y_bot), wide_w, h_box,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            linewidth=0.7, edgecolor="#444444", facecolor="#eef3f7",
            transform=fig.transFigure, zorder=1,
        )
    )
    fig.text(wide_x + 0.007, y_bot + h_box - 0.020,
             "Gated masks: junction ($M_J$), thin line ($M_L$), combined weight ($W$)",
             ha="left", va="top", fontsize=6.2, transform=fig.transFigure, zorder=5)
    mini = [
        ("$M_J$", img, mj),
        ("$M_L$", img, ml),
        ("$W$", img, w),
    ]
    for j, (lab, im, mask) in enumerate(mini):
        mx = wide_x + 0.018 + j * 0.222
        ax = fig.add_axes([mx, y_bot + 0.040, 0.195, h_box - 0.092])
        ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
        ax.imshow(mask, cmap="magma", alpha=0.62, interpolation="nearest")
        ax.set_zorder(3)
        hide(ax)
        fig.text(mx + 0.098, y_bot + 0.010, lab, ha="center", va="bottom",
                 fontsize=6.2, transform=fig.transFigure, zorder=5)

    def arrow(p0, p1, **kw):
        fig.patches.append(
            FancyArrowPatch(
                p0, p1, arrowstyle="-|>", mutation_scale=7, linewidth=0.7,
                color="#333333", transform=fig.transFigure, shrinkA=0, shrinkB=1,
                zorder=4, **kw,
            )
        )

    y_mid_top = y_top + h_box / 2
    for i in range(3):
        arrow((xs[i] + w_box + 0.003, y_mid_top), (xs[i + 1] - 0.003, y_mid_top))
    arrow((xs[0] + 0.04, y_top - 0.004), (xs[0] + 0.04, y_bot + h_box + 0.004))
    arrow((xs[0] + w_box, y_bot + h_box * 0.55), (wide_x - 0.002, y_bot + h_box * 0.55))
    arrow((xs[2] + 0.10, y_bot + h_box + 0.004), (xs[2] + 0.10, y_top - 0.004))

    fig.text(
        0.5, 0.012,
        "Repair is local: flat regions receive $W=0$ and remain identical to the CA-AKF base. "
        "Parameters: $\\tau=2.5$, $3\\times3$ top-hat, 99th percentile, dilation 3/2 px, mask $\\sigma=1.5$ px, $w_{\\max}=0.9$.",
        ha="center", va="bottom", fontsize=5.8, color="#333333", transform=fig.transFigure,
    )
    save(fig, "Fig1_pipeline")


# ---------------------------------------------------------------------------
# Figure 2: qualitative comparison
# ---------------------------------------------------------------------------
def figure2() -> None:
    rows = ["astronaut", "rocket", "synth_wedges", "hubble"]
    cols = [("source", "Input"), ("akf", "AKF"), ("ca_akf", "CA-AKF"), ("jlr_cakf", "JLR-CAKF")]
    fig, axes = plt.subplots(4, 4, figsize=(7.0, 6.55))
    for r, name in enumerate(rows):
        for c, (m, _) in enumerate(cols):
            ax = axes[r, c]
            im = IMAGES[name].image if m == "source" else load_out(name, m)
            ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
            hide(ax)
            if r == 0:
                ax.set_title(cols[c][1], fontsize=7.2, pad=3)
            if c == 0:
                ax.set_ylabel(
                    {"astronaut": "(a) astronaut", "rocket": "(b) rocket",
                     "synth_wedges": "(c) wedges", "hubble": "(d) Hubble"}[name],
                    fontsize=7.0, labelpad=3,
                )
            ax.set_xlabel("")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.955, bottom=0.01, wspace=0.035, hspace=0.05)
    save(fig, "Fig2_qualitative")


# ---------------------------------------------------------------------------
# Figure 3: mechanism details
# ---------------------------------------------------------------------------
def figure3() -> None:
    specs = [
        ("astronaut", "junction", (slice(35, 195), slice(35, 195)), "jlr_junction", "m_j"),
        ("rocket", "line", (slice(30, 190), slice(65, 225)), "jlr_line", "m_l"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(7.0, 2.95))
    for r, (name, kind, crop, single, which) in enumerate(specs):
        img = IMAGES[name].image
        mj, ml, _ = masks(name)
        ims = [
            (img[crop], "Input"),
            (load_out(name, "ca_akf")[crop], "CA-AKF"),
            (load_out(name, single)[crop], f"JLR {kind} only"),
            (load_out(name, "jlr_cakf")[crop], "JLR-CAKF"),
        ]
        for c, (im, title) in enumerate(ims):
            ax = axes[r, c]
            ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
            ax.set_title(title, fontsize=6.8, pad=2)
            hide(ax)
            panel(ax, "abcdefghij"[r * 5 + c])
        ax = axes[r, 4]
        mask = (mj if which == "m_j" else ml)[crop]
        ax.imshow(np.clip(img[crop], 0, 1), interpolation="nearest")
        ax.imshow(mask, cmap="magma", alpha=0.62, interpolation="nearest")
        ax.set_title("junction mask" if which == "m_j" else "line mask", fontsize=6.8, pad=2)
        hide(ax)
        panel(ax, "abcdefghij"[r * 5 + 4])
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.01, wspace=0.04, hspace=0.10)
    save(fig, "Fig3_details")


# ---------------------------------------------------------------------------
# Figure 4: structure benchmark
# ---------------------------------------------------------------------------
def figure4() -> None:
    df = pd.read_csv(RES / "structure_benchmark_summary.csv")
    order = [
        "jlr_cakf", "jlr_junction", "jlr_line", "jlr_global", "ca_akf",
        "guided", "rgf", "bilateral", "akf", "sector", "classic",
    ]
    labels = {
        "jlr_cakf": "JLR-CAKF (ours)", "jlr_junction": "JLR junction only",
        "jlr_line": "JLR line only", "jlr_global": "JLR global blend",
        "ca_akf": "CA-AKF", "guided": "Guided", "rgf": "RGF",
        "bilateral": "Bilateral", "akf": "AKF", "sector": "Sector", "classic": "Kuwahara",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    y = np.arange(len(order))
    for ax, kind, title in [
        (axes[0], "junction", "(a) junction score"),
        (axes[1], "line", "(b) line score"),
    ]:
        vals = []
        for m in order:
            r = df[(df.method == m) & (df.kind == kind)].iloc[0]
            vals.append(float(r["median"]))
        cols = [C[m] for m in order]
        ax.barh(y, vals, height=0.66, color=cols, alpha=0.82, edgecolor="#333333", linewidth=0.45)
        for yi, v in zip(y, vals):
            tx = 1.035 if v >= 0.85 else v + 0.015
            ax.text(tx, yi, f"{v:.3f}", va="center", ha="left", fontsize=5.8)
        ax.axvline(1.0, color="#666666", lw=0.7, ls="--")
        ax.set_xlim(0, 1.13)
        ax.set_xlabel("retained structure (1 = perfect)")
        ax.set_title(title, pad=3)
        ax.tick_params(length=2)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels([labels[m] for m in order])
        else:
            ax.tick_params(labelleft=False)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.235, right=0.985, top=0.90, bottom=0.14, wspace=0.08)
    save(fig, "Fig4_structure")


# ---------------------------------------------------------------------------
# Figure 5: metric distributions
# ---------------------------------------------------------------------------
def box(ax, df, metric, methods, ylabel, log=False):
    data = [df[df.method == m][metric].dropna().to_numpy(float) for m in methods]
    bp = ax.boxplot(
        data, widths=0.62, patch_artist=True, showfliers=False,
        medianprops=dict(color="black", lw=0.8),
        whiskerprops=dict(lw=0.55), capprops=dict(lw=0.55),
    )
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(C[m])
        patch.set_alpha(0.45)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.45)
    rng = np.random.default_rng(7)
    for i, (m, d) in enumerate(zip(methods, data), start=1):
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(d)), d, s=3.0, color=C[m],
                   edgecolors="none", alpha=0.9, zorder=3)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(["Kuwahara", "Sector", "AKF", "Bilateral", "RGF", "Guided", "CA-AKF", "JLR"], rotation=40, ha="right", fontsize=5.9)
    ax.tick_params(length=2, pad=1)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    if log:
        ax.set_yscale("log")


def figure5() -> None:
    df = pd.read_csv(RES / "metrics_all.csv")
    methods = ["classic", "sector", "akf", "bilateral", "rgf", "guided", "ca_akf", "jlr_cakf"]
    specs = [
        ("sr", "SR $\\uparrow$", False), ("ssim", "SSIM $\\uparrow$", False),
        ("fsr", "FSR $\\downarrow$", False), ("scg", "SCG $\\uparrow$", False),
        ("esr", "ESR $\\to1$", False), ("ece", "ECE $\\downarrow$", False),
        ("ovt", "OVT $\\downarrow$", False), ("runtime_s", "time (s, log)", True),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.55))
    for i, (ax, (met, lab, log)) in enumerate(zip(axes.ravel(), specs)):
        box(ax, df, met, methods, lab, log=log)
        panel(ax, "abcdefgh"[i])
    fig.subplots_adjust(left=0.075, right=0.995, top=0.965, bottom=0.285, wspace=0.40, hspace=0.82)
    save(fig, "Fig5_metrics")


# ---------------------------------------------------------------------------
# Figure 6: trade-offs
# ---------------------------------------------------------------------------
def figure6() -> None:
    agg = pd.read_csv(RES / "aggregate.csv").set_index("method")
    struct = pd.read_csv(RES / "structure_benchmark_summary.csv")
    jc = struct[struct.kind == "junction"].set_index("method")["median"]
    lr = struct[struct.kind == "line"].set_index("method")["median"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85))

    ax = axes[0]
    methods_a = ["classic", "sector", "akf", "bilateral", "rgf", "guided", "ca_akf", "jlr_cakf"]
    for m in methods_a:
        ax.scatter(agg.loc[m, "fsr_median"], agg.loc[m, "sr_median"],
                   s=46 if m == "jlr_cakf" else 20, color=C[m],
                   edgecolors="black" if m == "jlr_cakf" else "none",
                   linewidths=0.5, zorder=4)
    text_a = {
        "classic": (0.365, 0.535, "left"),
        "sector": (0.447, 0.445, "left"),
        "akf": (0.252, 0.425, "left"),
        "bilateral": (0.292, 1.005, "center"),
        "rgf": (0.326, 0.985, "left"),
        "guided": (0.225, 0.955, "right"),
        "ca_akf": (0.178, 0.565, "left"),
        "jlr_cakf": (0.150, 0.845, "right"),
    }
    for m, (tx, ty, ha) in text_a.items():
        ax.text(tx, ty, LABEL[m], fontsize=5.8, color=C[m], ha=ha, va="center")
    ax.set_xlabel("FSR $\\downarrow$ (lower is flatter)")
    ax.set_ylabel("SR $\\uparrow$")
    ax.set_title("(a) fidelity versus abstraction", pad=3)
    ax.tick_params(length=2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlim(0.13, 0.48)
    ax.set_ylim(0.38, 1.04)

    ax = axes[1]
    methods_b = ["classic", "sector", "akf", "bilateral", "rgf", "guided", "ca_akf", "jlr_cakf"]
    for m in methods_b:
        ax.scatter(lr[m], jc[m], s=64 if m == "jlr_cakf" else 20, color=C[m],
                   marker="*" if m == "jlr_cakf" else "o",
                   edgecolors="black" if m == "jlr_cakf" else "none",
                   linewidths=0.5, zorder=4)
    ax.annotate("JLR-CAKF", xy=(lr["jlr_cakf"], jc["jlr_cakf"]),
                xytext=(0.60, 0.80), fontsize=6.2, color=C["jlr_cakf"],
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C["jlr_cakf"]))
    ax.annotate("Guided", xy=(lr["guided"], jc["guided"]),
                xytext=(0.78, 0.995), fontsize=5.9, color=C["guided"],
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C["guided"]))
    ax.text(0.985, 1.035, "RGF / Bilateral", fontsize=5.9, color=C["rgf"], ha="right")
    ax.text(0.025, 0.665, "Sector", fontsize=5.9, color=C["sector"])
    ax.text(0.225, 0.775, "Kuwahara", fontsize=5.9, color=C["classic"])
    ax.text(0.045, 0.435, "AKF", fontsize=5.9, color=C["akf"])
    ax.text(0.090, 0.300, "CA-AKF", fontsize=5.9, color=C["ca_akf"])
    ax.set_xlabel("line score $\\uparrow$")
    ax.set_ylabel("junction score $\\uparrow$")
    ax.set_title("(b) structure benchmark", pad=3)
    ax.set_xlim(-0.02, 1.10)
    ax.set_ylim(0.0, 1.10)
    ax.tick_params(length=2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.17, wspace=0.30)
    save(fig, "Fig6_tradeoff")


# ---------------------------------------------------------------------------
# Figure 7: ablation
# ---------------------------------------------------------------------------
def figure7() -> None:
    agg = pd.read_csv(RES / "aggregate.csv").set_index("method")
    struct = pd.read_csv(RES / "structure_benchmark_summary.csv")
    jc = struct[struct.kind == "junction"].set_index("method")["median"]
    lr = struct[struct.kind == "line"].set_index("method")["median"]
    variants = ["ca_akf", "jlr_junction", "jlr_line", "jlr_global", "jlr_cakf"]
    labels = ["CA-AKF", "junction\nonly", "line\nonly", "global\nblend", "JLR-CAKF"]
    cols = [C[v] for v in variants]
    specs = [("sr", "SR $\\uparrow$"), ("esr", "ESR $\\to1$"), ("ece", "ECE $\\downarrow$"),
             ("struct_jc", "junction score $\\uparrow$"), ("struct_lr", "line score $\\uparrow$"),
             ("ssim", "SSIM $\\uparrow$")]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.7))
    for i, (ax, (key, lab)) in enumerate(zip(axes.ravel(), specs)):
        if key == "struct_jc":
            vals = np.array([jc[v] for v in variants])
        elif key == "struct_lr":
            vals = np.array([lr[v] for v in variants])
        else:
            vals = agg.loc[variants, f"{key}_median"].to_numpy(float)
        x = np.arange(len(variants))
        ax.bar(x, vals, width=0.66, color=cols, alpha=0.82, edgecolor="#333333", linewidth=0.45)
        for xi, v in zip(x, vals):
            ax.text(xi, v + (0.012 if key != "ece" else 0.5), f"{v:.2f}" if key == "ece" else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=5.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=5.9)
        ax.set_ylabel(lab)
        ax.tick_params(length=2, pad=1)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        if key == "ece":
            ax.set_ylim(0, 27)
        elif key == "ssim":
            ax.set_ylim(0, 0.86)
        elif key.startswith("struct_"):
            ax.set_ylim(0, 1.05)
        panel(ax, "abcdef"[i])
    fig.subplots_adjust(left=0.075, right=0.99, top=0.96, bottom=0.14, wspace=0.36, hspace=0.52)
    save(fig, "Fig7_ablation")


# ---------------------------------------------------------------------------
# Figure 8: runtime
# ---------------------------------------------------------------------------
def figure8() -> None:
    df = pd.read_csv(RES / "runtime_scaling.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.85))
    ax = axes[0]
    ticks = None
    for m in ["classic", "akf", "ca_akf", "jlr_cakf"]:
        g = df[df.method == m].sort_values("megapixels")
        a = np.polyfit(np.log(g.megapixels), np.log(g.median_s), 1)[0]
        ax.plot(g.megapixels, g.median_s, marker="o", ms=3.0, lw=1.0, color=C[m],
                label=f"{LABEL[m]} ($\\alpha$={a:.2f})")
        ticks = g.megapixels.tolist()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(round((v*1e6)**0.5))}$^2$" for v in ticks])
    ax.minorticks_off()
    ax.set_xlabel("image size (pixels)")
    ax.set_ylabel("runtime (s)")
    handles, labels = ax.get_legend_handles_labels()
    ax.tick_params(length=2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_title("(a) scaling", pad=3)

    ax = axes[1]
    p = df.pivot(index="size", columns="method", values="median_s")
    overhead = (p["jlr_cakf"] / p["ca_akf"] - 1.0) * 100.0
    ax.plot(p.index, overhead, marker="s", ms=3.2, lw=1.1, color=C["jlr_cakf"])
    for x, y in zip(p.index, overhead):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=5.8, color=C["jlr_cakf"])
    ax.set_xlabel("image size (pixels)")
    ax.set_ylabel("overhead over CA-AKF (%)")
    ax.set_ylim(0, 30)
    ax.set_xticks(p.index)
    ax.tick_params(length=2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_title("(b) repair overhead", pad=3)

    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), handlelength=1.2, columnspacing=1.1, fontsize=5.7)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.30, wspace=0.34)
    save(fig, "Fig8_runtime")


# ---------------------------------------------------------------------------
# Supplementary figures
# ---------------------------------------------------------------------------
def figureS1() -> None:
    df = pd.read_csv(RES / "sensitivity_jlr.csv")
    params = [
        ("snr_min", r"$\tau$ (SNR threshold)"),
        ("line_percentile", "line percentile"),
        ("w_max", r"$w_{\max}$"),
        ("junction_dilation", "junction dilation (px)"),
        ("line_dilation", "line dilation (px)"),
        ("mask_sigma", r"mask $\sigma$ (px)"),
    ]
    default = df[df.parameter == "default"].iloc[0]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6))
    for ax, (key, lab) in zip(axes.ravel(), params):
        g = df[df.parameter == key].sort_values("value")
        x = g.value.to_numpy(float)
        ax.plot(x, g.sr / default.sr, marker="o", ms=2.8, lw=1.0, color="#0072B2", label="SR")
        ax.plot(x, g.esr / default.esr, marker="s", ms=2.6, lw=1.0, color="#D55E00", label="ESR")
        ax.plot(x, default.ece / g.ece, marker="^", ms=2.6, lw=1.0, color="#009E73", label="ECE (inverted)")
        ax.axhline(1.0, color="#777777", lw=0.6, ls="--")
        ax.set_xlabel(lab)
        ax.tick_params(length=2)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.set_ylim(0.80, 1.20)
    axes[0, 0].set_ylabel("ratio to default")
    axes[1, 0].set_ylabel("ratio to default")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.01), handlelength=1.3, columnspacing=1.4)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.13, wspace=0.30, hspace=0.55)
    save(fig, "FigS1_sensitivity")


def figureS2() -> None:
    rows = ["astronaut", "rocket", "hubble"]
    cols = [("source", "Input"), ("bilateral", "Bilateral"), ("rgf", "RGF"), ("guided", "Guided")]
    fig, axes = plt.subplots(3, 4, figsize=(5.2, 3.75))
    for r, name in enumerate(rows):
        for c, (m, _) in enumerate(cols):
            ax = axes[r, c]
            im = IMAGES[name].image if m == "source" else load_out(name, m)
            ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
            hide(ax)
            if r == 0:
                ax.set_title(cols[c][1], fontsize=7.0, pad=2)
            if c == 0:
                ax.set_ylabel({"astronaut": "(a) astronaut", "rocket": "(b) rocket",
                               "hubble": "(c) Hubble"}[name], fontsize=7.0, labelpad=3)
    fig.subplots_adjust(left=0.10, right=0.995, top=0.93, bottom=0.01, wspace=0.03, hspace=0.04)
    save(fig, "FigS2_reference")


def figureS3() -> None:
    specs = {
        "classic": next(m for m in METHODS if m.name == "ca_akf"),
        "jlr": next(m for m in METHODS if m.name == "jlr_cakf"),
    }
    boundary, band = boundary_phantom()
    noise = noise_phantom()
    checker = IMAGES["synth_checker"].image
    hub = IMAGES["hubble"].image
    rows = [
        ("hard boundary", boundary),
        ("unstructured noise", noise),
        ("checkerboard", checker),
        ("dense point-like texture", hub),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(3.6, 4.8))
    for r, (name, img) in enumerate(rows):
        outs = [img, run_method(specs["classic"], img), run_method(specs["jlr"], img)]
        titles = ["Input", "CA-AKF", "JLR-CAKF"]
        for c, (im, t) in enumerate(zip(outs, titles)):
            ax = axes[r, c]
            ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
            hide(ax)
            if r == 0:
                ax.set_title(t, fontsize=6.8, pad=2)
            if c == 0:
                ax.set_ylabel(name, fontsize=6.5, labelpad=2)
    fig.subplots_adjust(left=0.16, right=0.995, top=0.94, bottom=0.01, wspace=0.03, hspace=0.05)
    save(fig, "FigS3_limits")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()
    figure8()
    figureS1()
    figureS2()
    figureS3()
