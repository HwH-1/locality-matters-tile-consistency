"""Figure 8: the native-resolution single-artwork experiment (Section 6.7.1).

Two panels, one per question the section makes.

  (a) WHERE the whole-canvas pass stops existing.  Peak RSS for the four canvases the
      paper measures, grouped by canvas rather than plotted against megapixels,
      because two of the groups contain the same number of pixels and differ only in
      whether those pixels were assembled (the 2048^2 mosaic) or photographed
      (the 2048^2 crop of a 4000-px artwork).  Filled circles = whole-canvas pass,
      open symbols = tile-wise, crosses = a whole-canvas pass that did not complete.
      The two ceilings are drawn in: the 5.5 GiB RLIMIT_AS the native runs used, and
      the 8 GiB of the host, above which the kernel OOM killer takes the process.

  (b) WHETHER the class-A/class-B split survives native pixel density.  frac of
      pixels more than one 8-bit level from the whole-canvas reference, against the
      overlap halo, on the 2048^2 crop of the artwork.  A literal 0 % is drawn at the
      axis floor and annotated, because zero is the claim being made.

Inputs are records only -- out/native.jsonl (this experiment), out/mem.jsonl,
out/mem_manual.jsonl and out/mem_4096.jsonl (the mosaic numbers of Section 6.7) --
so the figure is a re-plot of measured data, not a new computation.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT = "/data/tileconsistency/out"
FIG = "/data/tileconsistency/figures/F8_native.png"
CAP_MB = 5632.0        # 5.5 GiB RLIMIT_AS used by run_native.py
HOST_MB = 8192.0       # sandbox host

COLOUR = {"AKF": "tab:grey", "CA-AKF": "tab:orange", "SC-AKF": "tab:red",
          "LRS-final": "tab:blue", "LRS-AKF": "tab:blue"}
MARK = {"whole": "o", "tiled-n4": "s", "tiled-n8": "^", "aborted": "x"}


def rows(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def collect():
    """-> {group: [(method, kind, peak_mb, note)]}, ladder dict for panel (b)."""
    mem = rows(f"{OUT}/mem.jsonl") + rows(f"{OUT}/mem_manual.jsonl")
    m4096 = rows(f"{OUT}/mem_4096.jsonl")
    nat = rows(f"{OUT}/native.jsonl")

    groups = {"2048$^2$ mosaic\n(assembled)": [], "4096$^2$ mosaic": [],
              "2048$^2$ crop\n(native density)": [], "4000$\\times$3853\n(native)": []}

    for r in mem:                                   # 2048^2 mosaic, halo 48
        kind = "whole" if r["mode"] == "whole" else f"tiled-n{r['n']}"
        groups["2048$^2$ mosaic\n(assembled)"].append(
            (r["method"], kind, r["peak_mb"], f"{r['filter_s']:.0f} s"))
    for r in m4096:
        groups["4096$^2$ mosaic"].append(
            ("CA-AKF", f"tiled-n{r['n']}", r["peak_mb"], f"{r['filter_s']:.0f} s"))
    # the 4096^2 whole pass is not a number: it was killed by the kernel (exit 137)
    groups["4096$^2$ mosaic"].append(
        ("CA-AKF", "aborted", HOST_MB, "rc=137, kernel OOM"))

    for r in nat:
        g = ("2048$^2$ crop\n(native density)" if "crop" in r["img"]
             else "4000$\\times$3853\n(native)")
        if r["mode"] == "whole":
            if r.get("feasible"):
                groups[g].append((r["method"], "whole", r["peak_mb"],
                                  f"{r['filter_s']:.0f} s"))
            else:
                groups[g].append((r["method"], "aborted", CAP_MB,
                                  (r.get("error") or "MemoryError").split(":")[0]))
        elif r["mode"] == "tiled":
            groups[g].append((r["method"], "tiled-n4", r["peak_mb"],
                              f"h{r['halo']}, {r['filter_s']:.0f} s"))

    lad = {}
    for r in nat:
        if r["mode"] == "tce" and "crop" in r["img"]:
            lad.setdefault(r["method"], []).append((r["halo"], 100.0 * r["frac1"],
                                                    r["maxd"] * 255.0))
    for v in lad.values():
        v.sort()
    return groups, lad


def main():
    groups, lad = collect()
    gl = list(groups)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.9))

    # ---- (a) peak RSS by canvas group ------------------------------------------
    width = 0.22
    for gi, g in enumerate(gl):
        for method, kind, peak, note in groups[g]:
            x = gi + (list(MARK).index(kind) - 1.5) * width
            if kind == "aborted":
                ax1.plot(x, peak, marker="x", ms=11, mew=2.5, ls="none",
                         color=COLOUR[method])
                ax1.annotate("aborted", (x, peak), textcoords="offset points",
                             xytext=(0, 10), ha="center", fontsize=6.5)
            elif kind == "whole":
                ax1.plot(x, peak, marker="o", ms=8, ls="none", color=COLOUR[method])
            else:
                ax1.plot(x, peak, marker=MARK[kind], ms=8, ls="none", mfc="white",
                         mec=COLOUR[method], mew=1.8)
    ax1.axhline(CAP_MB, color="k", ls="--", lw=1)
    ax1.axhline(HOST_MB, color="tab:red", ls=":", lw=1.2)
    ax1.text(-0.5, CAP_MB + 160, "5.5 GiB address-space cap (native runs)", fontsize=7,
             ha="left")
    ax1.text(3.45, HOST_MB + 160, "8 GiB host -- above this the kernel kills the process",
             fontsize=7, color="tab:red", ha="right")
    ax1.set_xticks(range(len(gl)))
    ax1.set_xticklabels(gl, fontsize=8)
    ax1.set_xlim(-0.55, len(gl) - 0.45)
    ax1.set_ylim(0, 9600)
    ax1.set_ylabel("peak RSS of the pass (MB)")
    ax1.set_title("(a) the whole-canvas pass stops existing before the tiled one does",
                  fontsize=9, pad=8)
    ax1.grid(alpha=.3, axis="y")
    handles = [Line2D([], [], marker="o", ls="none", ms=8, color="k",
                      label="whole-canvas pass"),
               Line2D([], [], marker="s", ls="none", ms=8, mfc="white", mec="k",
                      label="tile-wise, n=4"),
               Line2D([], [], marker="^", ls="none", ms=8, mfc="white", mec="k",
                      label="tile-wise, n=8"),
               Line2D([], [], marker="x", ls="none", ms=10, mew=2.5, color="k",
                      label="whole-canvas pass did not complete")]
    handles += [Line2D([], [], marker="None", ls="none",
                       label="  " + m) for m in ("AKF (fixed radius)",
                                                 "CA-AKF (global P95)",
                                                 "SC-AKF (global quantile)",
                                                 "LRS-AKF (ours)")]
    ax1.legend(handles=handles, fontsize=6.8, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, -0.14), framealpha=.9, handlelength=1.1,
               columnspacing=1.0)

    # ---- (b) native-density ladder on the 2048^2 crop --------------------------
    # floor: a literal 0 % cannot be drawn on a log axis, so exact rungs are parked
    # at this height and labelled "0". The two exact methods are separated by the
    # marker/line style and by the side of the point their label sits on.
    floor = 2e-3
    NAME = {"AKF": "AKF (fixed radius)", "CA-AKF": "CA-AKF (global P95)",
            "LRS-final": "LRS-AKF (ours)"}
    STYLE = {"AKF": ("s", "--", "white", 10), "CA-AKF": ("D", "-", "tab:orange", 8),
             "LRS-final": ("o", "-", "tab:blue", -15)}
    for m, v in sorted(lad.items()):
        mk, ls, mfc, dy = STYLE[m]
        ys = [max(f, floor) for _, f, _ in v]
        ax2.plot([h for h, _, _ in v], ys, marker=mk, ls=ls, ms=6, lw=1.7, mew=1.6,
                 mfc=mfc, color=COLOUR[m], label=NAME[m])
        for (h, f, mx), yy in zip(v, ys):
            txt = "0" if f == 0.0 else (f"{f:.2f}" if f >= 0.01 else f"{f:.1e}")
            ax2.annotate(txt, (h, yy),
                         textcoords="offset points", xytext=(0, dy), ha="center",
                         fontsize=6.5, color=COLOUR[m])
    ax2.axhline(1e-2, color="k", ls="--", lw=1)
    ax2.text(17.5, 1.15e-2, "0.01 % of pixels = 419 px of 4.19 M", fontsize=7)
    ax2.text(17.5, 3.2e-4, '"0" = bit-exact; the height of those points is a plotting '
             'floor,\nnot a measured value', fontsize=6.5)
    ax2.set_yscale("log")
    ax2.set_ylim(2.5e-4, 3e1)
    ax2.set_xlabel("overlap halo (px)")
    ax2.set_ylabel("frac of pixels $>1$ 8-bit level (%)")
    ax2.set_title("(b) at native pixel density the bounded-support curves fall "
                  "through the floor;\nthe global-quantile one does not", fontsize=9,
                  pad=8)
    ax2.grid(alpha=.3, which="both")
    ax2.legend(fontsize=7.5, loc="center right")

    fig.tight_layout()
    fig.savefig(FIG, dpi=300, bbox_inches="tight")
    print("wrote", FIG, os.path.getsize(FIG), "bytes")
    for g in gl:
        print(f"  {g.replace(chr(10), ' '):32s} {len(groups[g]):2d} points")
    print("  ladder:", {m: [(h, round(f, 5)) for h, f, _ in v] for m, v in lad.items()})


if __name__ == "__main__":
    main()
