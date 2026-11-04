#!/usr/bin/env python3
"""Output gallery: input / CA-AKF / LRS-AKF, with a flat-region zoom and a
difference map. Generated from the same operators and the same image as the
quality table; no intermediate arrays are needed."""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import TC as _TC, REFPARENT as _REF, FIGDIR as _FIG, IMAGES as _IMG
os.environ.setdefault("TC_SRC", str(_TC))
os.environ.setdefault("CAAKF_REF", str(_REF))
sys.path.insert(0, str(_TC))
import tc  # noqa: E402

OUT = _FIG
IMG = os.environ.get("TC_GALLERY_IMAGE", str(_IMG / "met/met_0189655.jpg"))

img = tc.load(IMG, 384)
ca = tc.m_ca(img)
lrs = tc.m_lrs(img, c=1.0, rad=8.0, mode="geo")
print("shapes", img.shape, ca.shape, lrs.shape,
      "ranges", float(ca.min()), float(ca.max()), float(lrs.min()), float(lrs.max()))

# choose the flattest 96x96 window by local gradient energy of the input
g = np.mean(np.abs(np.diff(img, axis=1))[:383, :, :]
            + np.abs(np.diff(img, axis=0))[:, :383, :], axis=2)
g = np.pad(g, ((0, 1), (0, 1)), mode="edge")
k = 96
cs = np.cumsum(np.cumsum(g, 0), 1)
tot = cs[k - 1:, k - 1:] - np.pad(cs[:-k, k - 1:], ((1, 0), (0, 0))) \
      - np.pad(cs[k - 1:, :-k], ((0, 0), (1, 0))) + np.pad(cs[:-k, :-k], ((1, 0), (1, 0)))
y, x = np.unravel_index(np.argmin(tot), tot.shape)
print("flat window at", x, y, "energy", float(tot.min()))
box = (slice(y, y + k), slice(x, x + k))

diff = np.max(np.abs(ca - lrs), axis=2)

fig = plt.figure(figsize=(10.2, 9.4))
gs = GridSpec(4, 3, figure=fig, hspace=0.24, wspace=0.06,
              height_ratios=[1.0, 0.92, 0.80, 0.10])
panels = [(img, "Input"), (ca, "CA-AKF (global-quantile reference)"),
          (lrs, "LRS-AKF (ours, bounded support)")]
for j, (a, title) in enumerate(panels):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(np.clip(a, 0, 1))
    ax.set_title(title, fontsize=9.5, pad=5)
    ax.add_patch(plt.Rectangle((x, y), k, k, fill=False, edgecolor="#e8b90b", linewidth=1.6))
    ax.set_xticks([]); ax.set_yticks([])
    if j == 0:
        ax.set_ylabel("whole image", fontsize=9)

for j, (a, _) in enumerate(panels):
    ax = fig.add_subplot(gs[1, j])
    ax.imshow(np.clip(a[box], 0, 1))
    ax.set_xticks([]); ax.set_yticks([])
    if j == 0:
        ax.set_ylabel("flat-region zoom", fontsize=9)

axd = fig.add_subplot(gs[2, 1])
im = axd.imshow(diff, cmap="magma", vmin=0, vmax=float(np.percentile(diff, 99.5)))
axd.set_xticks([]); axd.set_yticks([])
axd.set_title("per-pixel difference $|\\mathrm{CA}-\\mathrm{LRS}|$", fontsize=9, pad=5)
for j, txt in [(0, "CA-AKF uses one threshold" + chr(10) + "for the whole image"),
               (2, "LRS-AKF uses a threshold" + chr(10) + "from a bounded neighbourhood")]:
    ax = fig.add_subplot(gs[2, j])
    ax.axis("off")
    ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=9, wrap=True)

cax = fig.add_subplot(gs[3, :])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("difference on a [0,1] scale (99.5th percentile clipped)", fontsize=8)
cb.ax.tick_params(labelsize=7)

fig.savefig(OUT / "F10_gallery.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "F10_gallery.pdf", bbox_inches="tight")
print("wrote", OUT / "F10_gallery.png", "and .pdf")
