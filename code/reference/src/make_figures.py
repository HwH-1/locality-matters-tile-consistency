"""Generate all manuscript figures from real data and real filter outputs."""

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
from src.data import load_all, synth_checker  # noqa: E402
from src.methods import METHODS, run_method  # noqa: E402
from src.structure import structure_tensor  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8,
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
    "no_adaptive": "#0072B2",
    "no_soft": "#E69F00",
    "iso_pre": "#009E73",
}
LABEL = {
    "classic": "Kuwahara",
    "sector": "Sector",
    "akf": "AKF",
    "bilateral": "Bilateral",
    "rgf": "RGF",
    "guided": "Guided",
    "ca_akf": "CA-AKF",
    "ab_no_adaptive": "Fixed scale",
    "ab_no_soft": "Hard min.",
    "ab_iso_pre": "Isotropic pre.",
}

IMAGES = {im.name: im for im in load_all()}


def save(fig, stem: str, png_dpi: int = 300) -> None:
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(PREV / f"{stem}.png", dpi=png_dpi)
    plt.close(fig)
    print("wrote", stem, flush=True)


def load_out(name: str, method: str) -> np.ndarray:
    return np.asarray(Image.open(OUT / f"{name}__{method}.png").convert("RGB"), dtype=np.float32) / 255.0


def add_panel_label(ax, text, color="black"):
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=color,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
    )


def hide(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ----------------------------------------------------------------------------
# Figure 1: pipeline overview
# ----------------------------------------------------------------------------
def figure1() -> None:
    src = IMAGES["camera"].image
    st = structure_tensor(src)
    smooth = filters.anisotropic_presmooth(src, st["theta"], st["coherence"], 2.5, 12)
    theta = st["theta"]
    stats = {}
    for r in (3, 6):
        mu, var = filters.sector_statistics(smooth, r, theta, 8, 12)
        stats[r] = (mu, var, filters.combine_sectors(mu, var, "soft", 8.0))
    alpha = (1.0 - st["edge_energy"]) ** 2
    out = stats[3][2] * (1 - alpha[..., None]) + stats[6][2] * alpha[..., None]
    out = np.clip(out, 0, 1)
    mu6, var6, _ = stats[6]
    varmap = np.median(var6, axis=0)
    varmap = np.log1p(varmap * 500.0)
    varmap = (varmap - varmap.min()) / (varmap.max() - varmap.min() + 1e-8)

    fig = plt.figure(figsize=(7.0, 2.30))
    xs = np.linspace(0.008, 0.842, 6)
    w_box = 0.15
    y_box, h_box = 0.76, 0.19
    titles = [
        "Input $I$",
        "Structure tensor\n$\\theta$, $C$, $E$",
        "Coherence-adaptive\nanisotropic smoothing\n$\\sigma_\\parallel=2.5$",
        "Sector statistics\n$r\\in\\{3,6\\}$, $K=8$\n$\\mu_s$, $v_s$",
        "Soft selection\n$w_s\\propto e^{-\\beta\\,\\Delta v_s}$\n+ scale blend",
        "Stylized output",
    ]
    for i, (x, t) in enumerate(zip(xs, titles)):
        box = FancyBboxPatch(
            (x, y_box), w_box, h_box,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=0.7, edgecolor="#444444",
            facecolor="#eef3f8" if i not in (0, 5) else "#f4f4f4",
            transform=fig.transFigure,
        )
        fig.patches.append(box)
        fig.text(x + w_box / 2, y_box + h_box / 2, t, ha="center", va="center",
                 fontsize=5.9, linespacing=1.25, transform=fig.transFigure)
        if i < 5:
            arr = FancyArrowPatch(
                (x + w_box + 0.004, y_box + h_box / 2),
                (xs[i + 1] - 0.004, y_box + h_box / 2),
                arrowstyle="-|>", mutation_scale=7, linewidth=0.7, color="#333333",
                transform=fig.transFigure,
            )
            fig.patches.append(arr)

    # Thumbnails.
    thumb_y, thumb_h = 0.30, 0.36
    thumb_w = 0.115
    thumbs = []
    thumbs.append(src)
    hsv = np.zeros((*theta.shape, 3), dtype=np.float32)
    hsv[..., 0] = (theta % np.pi) / np.pi
    hsv[..., 1] = 0.9
    hsv[..., 2] = 0.25 + 0.75 * st["coherence"]
    thumbs.append(colors.hsv_to_rgb(hsv).astype(np.float32))
    thumbs.append(np.repeat(st["coherence"][..., None], 3, axis=-1))
    thumbs.append(smooth)
    thumbs.append(plt.get_cmap("inferno")(varmap)[..., :3].astype(np.float32))
    thumbs.append(out)
    caps = ["$I$", "$\\theta$ (hue), $C$ (value)", "$C$", "pre-smoothed", "sector variance", "CA-AKF"]
    for i, (x, im, cap) in enumerate(zip(xs, thumbs, caps)):
        ax = fig.add_axes([x + (w_box - thumb_w) / 2, thumb_y, thumb_w, thumb_h])
        ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
        hide(ax)
        fig.text(x + w_box / 2, thumb_y - 0.035, cap, ha="center", va="top",
                 fontsize=6.0, transform=fig.transFigure)

    # Parameter strip.
    fig.text(0.5, 0.035,
             "Fixed parameters: $\\sigma_\\perp=\\sigma_\\parallel/(1+3C)$, $\\beta=8$, $K=8$ sectors, 12 orientation bins, $\\sigma_t=4$; "
             "$r_{\\min}=3$, $r_{\\max}=6$.",
             ha="center", va="bottom", fontsize=6.0, color="#333333", transform=fig.transFigure)
    save(fig, "Fig1_pipeline")


# ----------------------------------------------------------------------------
# Figure 2: qualitative comparison
# ----------------------------------------------------------------------------
def figure2() -> None:
    rows = ["astronaut", "camera", "synth_wedges"]
    cols = [("source", "Input"), ("classic", "Kuwahara"), ("sector", "Sector"),
            ("akf", "AKF"), ("bilateral", "Bilateral"), ("ca_akf", "CA-AKF")]
    fig, axes = plt.subplots(3, 6, figsize=(7.0, 3.75))
    for r, name in enumerate(rows):
        for c, (m, _) in enumerate(cols):
            ax = axes[r, c]
            if m == "source":
                im = IMAGES[name].image
            else:
                im = load_out(name, m)
            ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if r == 0:
                ax.set_title(cols[c][1], fontsize=7.0, pad=2)
            if c == 0:
                ax.set_ylabel(
                    {"astronaut": "(a) astronaut", "camera": "(b) camera", "synth_wedges": "(c) wedges"}[name],
                    fontsize=7.0, labelpad=3,
                )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.94, bottom=0.01, wspace=0.03, hspace=0.03)
    save(fig, "Fig2_qualitative")


# ----------------------------------------------------------------------------
# Figure 3: mechanism
# ----------------------------------------------------------------------------
def figure3() -> None:
    src = IMAGES["astronaut"].image
    st = structure_tensor(src)
    r_map = 3.0 + 3.0 * (1.0 - st["edge_energy"]) ** 2
    akf = load_out("astronaut", "akf")
    ours = load_out("astronaut", "ca_akf")

    checker = synth_checker(size=256)
    chk_akf = load_out("synth_checker", "akf")
    chk_ours = load_out("synth_checker", "ca_akf")
    crop = (slice(64, 192), slice(64, 192))

    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.55))
    # top row
    panels = [
        (src, "source (astronaut)", None),
        (np.repeat(st["coherence"][..., None], 3, -1), "coherence $C$", "coherence"),
        (r_map, "adaptive radius $r(p)$", "radius"),
        (None, "orientation $\\theta$", "orientation"),
    ]
    for i, (im, title, kind) in enumerate(panels):
        ax = axes[0, i]
        if kind == "orientation":
            hsv = np.zeros((*st["theta"].shape, 3), dtype=np.float32)
            hsv[..., 0] = (st["theta"] % np.pi) / np.pi
            hsv[..., 1] = 0.9
            hsv[..., 2] = 0.25 + 0.75 * st["coherence"]
            ax.imshow(colors.hsv_to_rgb(hsv), interpolation="nearest")
        elif kind == "radius":
            imh = ax.imshow(im, cmap="viridis", vmin=3, vmax=6, interpolation="nearest")
            cb = fig.colorbar(imh, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=5.5, length=1.5)
        elif kind == "coherence":
            imh = ax.imshow(im[..., 0], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            cb = fig.colorbar(imh, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=5.5, length=1.5)
        else:
            ax.imshow(im, interpolation="nearest")
        ax.set_title(title, fontsize=7.0, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        add_panel_label(ax, "abcd"[i])

    # bottom row: checker details and profile
    for i, (im, title) in enumerate([
        (checker[crop], "checkerboard input"),
        (chk_akf[crop], "AKF (fixed scale)"),
        (chk_ours[crop], "CA-AKF (ours)"),
    ]):
        ax = axes[1, i]
        ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
        ax.set_title(title, fontsize=7.0, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        add_panel_label(ax, "efg"[i])

    ax = axes[1, 3]
    df = pd.read_csv(RES / "metrics_full.csv")
    p = df.pivot_table(index="image", columns="method", values="fsr")
    x = p["akf"].to_numpy(float)
    y = p["ca_akf"].to_numpy(float)
    below = int(np.sum(y < x))
    ax.scatter(x, y, s=8, color=C["ca_akf"], edgecolors="white", linewidths=0.3, zorder=3)
    lim = [0, max(x.max(), y.max()) * 1.08]
    ax.plot(lim, lim, color="#777777", lw=0.7, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("FSR, AKF")
    ax.set_ylabel("FSR, CA-AKF", labelpad=1)
    ax.set_title(f"paired FSR ({below}/12 below)", fontsize=7.0, pad=2)
    ax.tick_params(length=2)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    add_panel_label(ax, "h")

    fig.subplots_adjust(left=0.01, right=0.995, top=0.93, bottom=0.10, wspace=0.30, hspace=0.12)
    save(fig, "Fig3_mechanism")


# ----------------------------------------------------------------------------
# Figure 4: metric distributions
# ----------------------------------------------------------------------------
def _boxplot_metric(ax, df, metric, methods, ylabel, log=False, show_xlabels=True):
    data = [df[df.method == m][metric].to_numpy(float) for m in methods]
    bp = ax.boxplot(
        data, widths=0.62, patch_artist=True, showfliers=False,
        medianprops=dict(color="black", lw=0.8),
        whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
    )
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(C[m])
        patch.set_alpha(0.55)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.5)
    rng = np.random.default_rng(7)
    for i, (m, d) in enumerate(zip(methods, data), start=1):
        ax.scatter(
            i + rng.uniform(-0.12, 0.12, len(d)), d, s=3.2,
            color=C[m], edgecolors="none", alpha=0.9, zorder=3,
        )
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, len(methods) + 1))
    if show_xlabels:
        ax.set_xticklabels([LABEL[m] for m in methods], rotation=90, ha="center")
    else:
        ax.set_xticklabels([])
    ax.tick_params(length=2, pad=1)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if log:
        ax.set_yscale("log")


def figure4() -> None:
    df = pd.read_csv(RES / "metrics_full.csv")
    methods = ["classic", "sector", "akf", "bilateral", "rgf", "guided", "ca_akf"]
    specs = [
        ("sr", "SR $\\uparrow$", False),
        ("ssim", "SSIM $\\uparrow$", False),
        ("fsr", "FSR $\\downarrow$", False),
        ("scg", "SCG $\\uparrow$", False),
        ("esr", "ESR (ideal 1)", False),
        ("ece", "ECE $\\downarrow$", False),
        ("ovt", "OVT $\\downarrow$", False),
        ("runtime_s", "Time (s, log)", True),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.0))
    for i, (ax, (met, lab, log)) in enumerate(zip(axes.ravel(), specs)):
        _boxplot_metric(ax, df, met, methods, lab, log=log, show_xlabels=(i >= 4))
        add_panel_label(ax, "abcdefgh"[i])
    fig.subplots_adjust(left=0.07, right=0.995, top=0.97, bottom=0.22, wspace=0.34, hspace=0.55)
    save(fig, "Fig4_metrics")


# ----------------------------------------------------------------------------
# Figure 5: ablation
# ----------------------------------------------------------------------------
def figure5() -> None:
    df = pd.read_csv(RES / "metrics_full.csv")
    agg = df.groupby("method").agg(
        sr=("sr", "median"), sr_q1=("sr", lambda x: x.quantile(.25)), sr_q3=("sr", lambda x: x.quantile(.75)),
        ssim=("ssim", "median"), ssim_q1=("ssim", lambda x: x.quantile(.25)), ssim_q3=("ssim", lambda x: x.quantile(.75)),
        fsr=("fsr", "median"), fsr_q1=("fsr", lambda x: x.quantile(.25)), fsr_q3=("fsr", lambda x: x.quantile(.75)),
        scg=("scg", "median"), scg_q1=("scg", lambda x: x.quantile(.25)), scg_q3=("scg", lambda x: x.quantile(.75)),
    )
    variants = ["ab_no_adaptive", "ab_no_soft", "ab_iso_pre", "ca_akf"]
    colors_ = [C["no_adaptive"], C["no_soft"], C["iso_pre"], C["ca_akf"]]
    labels = ["Fixed scale", "Hard min.", "Isotropic pre.", "Full CA-AKF"]
    specs = [("sr", "SR $\\uparrow$"), ("ssim", "SSIM $\\uparrow$"),
             ("fsr", "FSR $\\downarrow$"), ("scg", "SCG $\\uparrow$")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1))
    x = np.arange(len(variants))
    for i, (ax, (met, lab)) in enumerate(zip(axes, specs)):
        add_panel_label(ax, "abcd"[i])
        med = agg.loc[variants, met].to_numpy()
        lo = med - agg.loc[variants, f"{met}_q1"].to_numpy()
        hi = agg.loc[variants, f"{met}_q3"].to_numpy() - med
        ax.bar(x, med, width=0.62, color=colors_, alpha=0.75, edgecolor="#333333", linewidth=0.5)
        ax.errorbar(x, med, yerr=np.vstack([lo, hi]), fmt="none", ecolor="#222222", elinewidth=0.6, capsize=1.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylabel(lab)
        ax.tick_params(length=2, pad=1)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.96, bottom=0.28, wspace=0.34)
    save(fig, "Fig5_ablation")


# ----------------------------------------------------------------------------
# Figure 6: runtime scaling
# ----------------------------------------------------------------------------
def figure6() -> None:
    df = pd.read_csv(RES / "runtime_scaling.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ticks, ticklabels = None, None
    for m in ["classic", "akf", "ca_akf"]:
        g = df[df.method == m]
        a = np.polyfit(np.log(g["megapixels"]), np.log(g["median_s"]), 1)[0]
        ax.plot(g["megapixels"], g["median_s"], marker="o", ms=2.8, lw=1.0, color=C[m],
                label=f"{LABEL[m]} ($\\alpha$={a:.2f})")
        ticks = g["megapixels"].tolist()
        ticklabels = [f"{int(round((mp*1e6)**0.5))}$^2$" for mp in ticks]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, rotation=0)
    ax.minorticks_off()
    ax.set_xlabel("image size (pixels)")
    ax.set_ylabel("runtime (s)")
    ax.legend(frameon=False, loc="upper left", handlelength=1.2)
    ax.tick_params(length=2)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.96, bottom=0.21)
    save(fig, "Fig6_runtime")


# ----------------------------------------------------------------------------
# Figure S3: qualitative comparison with the reference smoothers
# ----------------------------------------------------------------------------
def figureS3() -> None:
    rows = ["astronaut", "camera", "synth_wedges"]
    cols = [("source", "Input"), ("rgf", "RGF"),
            ("guided", "Guided filter"), ("ca_akf", "CA-AKF")]
    fig, axes = plt.subplots(3, 4, figsize=(5.2, 3.45))
    for r, name in enumerate(rows):
        for c, (m, _) in enumerate(cols):
            ax = axes[r, c]
            im = IMAGES[name].image if m == "source" else load_out(name, m)
            ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if r == 0:
                ax.set_title(cols[c][1], fontsize=7.0, pad=2)
            if c == 0:
                ax.set_ylabel(
                    {"astronaut": "(a) astronaut", "camera": "(b) camera",
                     "synth_wedges": "(c) wedges"}[name],
                    fontsize=7.0, labelpad=3,
                )
    fig.subplots_adjust(left=0.10, right=0.995, top=0.93, bottom=0.01, wspace=0.03, hspace=0.03)
    save(fig, "FigS3_reference")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    figureS3()
