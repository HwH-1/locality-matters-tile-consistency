"""Figure 6: the same deployment configuration on a large canvas, one class-A
baseline and one class-B case side by side.

run_giga.py saves whole-canvas renders and normalised difference crops, which is
not a legible figure: the difference panels are auto-scaled per method, so a
sub-threshold deviation looks as loud as a real one.  This script re-runs the
two interesting methods on the canvas and draws three panels on a SHARED scale:
whole-image crop, tile-wise crop (same pixels), and the |difference| at 40x
amplification clipped at one 8-bit level.

Usage: python3 make_f6.py --n 4 --halo 48 --methods CA-AKF,LRS-final
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/data/tileconsistency/src")
import cv2
cv2.setNumThreads(4)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import tc

ROOT = "/data/tileconsistency"
FIG = f"{ROOT}/figures"
LEVEL = 1.0 / 255.0


def crop(img, y, x, s):
    return np.clip(img[y:y + s, x:x + s], 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canvas", default=f"{ROOT}/canvas_2048.png")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--halo", type=int, default=48)
    ap.add_argument("--tile", type=int, default=512, help="output crop size")
    ap.add_argument("--methods", default="CA-AKF,LRS-final")
    ap.add_argument("--y-frac", type=float, default=0.10,
                    help="crop origin as a fraction of the canvas, so the crop "
                         "lands on a busy region rather than a margin")
    ap.add_argument("--out", default=f"{FIG}/F6_canvas.png")
    a = ap.parse_args()

    big = np.ascontiguousarray(cv2.imread(a.canvas, cv2.IMREAD_COLOR)[:, :, ::-1]
                               .astype(np.float32) / 255.0)
    H, W = big.shape[:2]
    y0, x0 = int(H * a.y_frac), int(W * a.y_frac)
    cands = {"CA-AKF": tc.m_ca, "SC-AKF": tc.m_sc,
             "LRS-final": lambda x: tc.m_lrs(x, c=1.0, rad=8.0, mode="geo"),
             "AKF": tc.m_akf}
    names = a.methods.split(",")

    t0 = time.time()
    summary = {"canvas": os.path.basename(a.canvas), "n": a.n, "halo": a.halo,
               "crop": [y0, x0, a.tile], "methods": {}}
    fig, axes = plt.subplots(len(names), 3, figsize=(9.4, 3.2 * len(names)),
                             squeeze=False)
    for row, name in enumerate(names):
        fn = cands[name]
        whole = fn(big)
        tiled = tc.tiled(fn, big, a.n, a.halo)
        m = tc.tce(whole, tiled)
        gv = tc.gridline_visibility(whole, tiled, a.n)
        d = np.abs(whole - tiled).max(axis=-1)
        summary["methods"][name] = dict(
            tce={k: round(v, 7) for k, v in m.items()},
            gv=round(gv["gv"], 4), ratio=round(gv["ratio"], 2),
            crop_dev_levels=dict(max=round(float(d[y0:y0 + a.tile, x0:x0 + a.tile].max()) / LEVEL, 3),
                                 frac1=round(float((d[y0:y0 + a.tile, x0:x0 + a.tile] > LEVEL).mean()), 5)),
            frac1_full=round(m["frac1"], 6))
        axes[row][0].imshow(crop(whole, y0, x0, a.tile))
        axes[row][1].imshow(crop(tiled, y0, x0, a.tile))
        axes[row][2].imshow(np.clip(d[y0:y0 + a.tile, x0:x0 + a.tile] / (40 * LEVEL), 0, 1),
                            vmin=0, vmax=1)
        for col in range(3):
            axes[row][col].set_axis_off()
            if row == 0:
                axes[row][col].set_title(
                    ["whole canvas", f"tile-wise (n={a.n}, halo {a.halo})",
                     "|difference| x40 (white = 1 level)"][col], fontsize=10)
        axes[row][0].set_ylabel(name, fontsize=11)
        print(f"  {name}: rmse={m['rmse']:.7f} frac1={m['frac1']:.5f} "
              f"ratio={gv['ratio']:.1f}  ({time.time()-t0:.0f}s)", flush=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    with open(f"{ROOT}/out/f6_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print("wrote", a.out, f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
