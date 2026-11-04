"""Tile-consistency machinery for content-adaptive abstraction filtering.

Terminology used throughout (paper-facing):
  Class-A mismatch  support truncation.  Caused by evaluating a bounded-support
                    operator on a tile without enough context.  Fixable exactly by
                    an overlap halo >= operator support.
  Class-B mismatch  statistic non-locality.  Caused by a parameter that is a
                    functional of the WHOLE image (global quantile / percentile).
                    No finite halo fixes it, because the parameter depends on
                    pixels arbitrarily far away.

TCE (tile-consistency error) = || f(I) - tiled(f, I, n, halo) ||, reported as
RMSE, max|d|, and the fraction of pixels differing by more than 1/255.

All operators here work on float32 RGB in [0,1], shape (H,W,3).
"""

from __future__ import annotations

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.environ.get("CAAKF_REF",
                                  "/tmp/jlr/TheVisualComputer_JLR/03_experiments"))
try:
    from src import filters, methods                       # noqa: E402
    from src.structure import structure_tensor             # noqa: E402
    from src.metrics import compute_metrics                # noqa: E402
except ImportError as exc:                                  # pragma: no cover
    raise ImportError(
        "tc.py needs the CA-AKF reference implementation (filters.py, methods.py, "
        "structure.py, metrics.py). Point CAAKF_REF at the 03_experiments directory that "
        "contains them, e.g. CAAKF_REF=/path/to/03_experiments. Original error: %s" % exc
    ) from exc

cv2.setNumThreads(4)

K, BINS = 8, 12
LEVELS = (2, 4, 6, 8, 10)
CFG_CA = dict(radius=5, sectors=K, bins=BINS, combine="soft", beta=8.0,
              sigma_along=2.5, r_min=3, r_max=6, gamma=2.0,
              adaptive=True, sigma_t=4.0, sigma_g=1.0)
CFG_AKF = dict(radius=5, sectors=K, bins=BINS, combine="soft", beta=8.0,
               sigma_along=2.5, adaptive=False, sigma_t=4.0, sigma_g=1.0)


# ---------------------------------------------------------------- statistics --
def dispersion(img, sigma_t=4.0, sigma_g=1.0, sigma_along=2.5):
    """Everything the scale criterion needs, computed once (all local)."""
    st = structure_tensor(img, sigma_g=sigma_g, sigma_t=sigma_t)
    sm = filters.anisotropic_presmooth(img, st["theta"], st["coherence"],
                                       sigma_along, BINS)
    stats = [filters.sector_statistics(sm, r, st["theta"], K, BINS)
             for r in LEVELS]
    vm = np.stack([v.min(0) for _, v in stats])          # (L,H,W)
    return st, stats, vm


def _select(vm, tau):
    """Largest level whose dispersion still fits under the threshold tau."""
    ok = vm <= tau
    sel = np.clip(np.count_nonzero(np.logical_and.accumulate(ok, axis=0), axis=0)
                  - 1, 0, len(vm) - 1)
    return sel.astype(np.int64)


def _compose(stats, sel):
    out = np.zeros(stats[0][0].shape[1:4], np.float32)
    for i, (mu, var) in enumerate(stats):
        po = filters.combine_sectors(mu, var, mode="soft", beta=6.0)
        m = sel == i
        out[m] = po[m]
    return out


# ------------------------------------------------------------------ methods --
def m_akf(img):
    """Anisotropic Kuwahara, fixed radius -- local by construction (control)."""
    return methods.m_akf(img, CFG_AKF)


def m_ca(img):
    """Published CA-AKF.  Two global statistics: P95(l1) and a global quantile."""
    return methods.m_ours(img, CFG_CA)


def m_sc(img, c=4.0, p=0.25):
    """Our earlier scale-criterion variant.  STILL global: Q_p over the image."""
    _, stats, vm = dispersion(img)
    tau = c * float(np.quantile(vm[0], p)) + 1e-12
    return _compose(stats, _select(vm, tau))


def m_lrs(img, c=4.0, rad=16.0, mode="mean"):
    """LRS-AKF: the global quantile is replaced by a BOUNDED-SUPPORT local
    reference of the base-scale dispersion.  Homogeneous of degree 2 in the
    image (up to a 1e-12 floor), hence contrast-equivariant, and exactly
    tile-consistent once halo >= operator support + rad.

    mode: mean = box mean (cheapest, exactly bounded),
          geo  = exp(box mean of log)   (robust to outlier patches),
          med  = local median (rank statistic, bounded by the window).
    """
    _, stats, vm = dispersion(img)
    k = int(2 * round(rad) + 1)
    v0 = np.ascontiguousarray(vm[0])
    if mode == "mean":
        ref = cv2.blur(v0, (k, k), borderType=cv2.BORDER_REFLECT)
    elif mode == "geo":
        ref = np.exp(cv2.blur(np.log(v0 + 1e-12), (k, k),
                              borderType=cv2.BORDER_REFLECT))
    elif mode == "med":
        ref = cv2.medianBlur(v0, k)
    tau = c * ref + 1e-12
    return _compose(stats, _select(vm, tau))


METHODS = {
    "AKF":  m_akf,
    "CA-AKF": m_ca,
    "SC-AKF": m_sc,
    "LRS-AKF": lambda x: m_lrs(x, c=4.0, rad=16.0, mode="mean"),
}


# ------------------------------------------------------------------ tiling ----
def tiled(fn, img, n, halo):
    """Evaluate fn tile-by-tile with an overlap halo, keep only tile interiors."""
    h, w = img.shape[:2]
    th, tw = h // n, w // n
    out = np.empty_like(img)
    for i in range(n):
        for j in range(n):
            y0, x0 = i * th, j * tw
            y1 = h if i == n - 1 else y0 + th
            x1 = w if j == n - 1 else x0 + tw
            top, bot = min(halo, y0), min(halo, h - y1)
            lef, rig = min(halo, x0), min(halo, w - x1)
            sub = img[y0 - top:y1 + bot, x0 - lef:x1 + rig]
            r = fn(sub)
            out[y0:y1, x0:x1] = r[top:top + (y1 - y0), lef:lef + (x1 - x0)]
    return out


def tile_grid(img, n):
    """Same as tiled() but ALSO returns the tile grid so seams can be located."""
    h, w = img.shape[:2]
    return tiled(lambda x: x, img, n, 0), (h // n, w // n)


# ------------------------------------------------------------------ metrics ---
def tce(whole, part):
    d = np.abs(whole.astype(np.float64) - part.astype(np.float64))
    dl = d.max(axis=-1)
    return dict(rmse=float(np.sqrt((d ** 2).mean())),
                maxd=float(d.max()),
                meand=float(d.mean()),
                frac1=float((dl > 1.0 / 255.0).mean()),
                frac4=float((dl > 4.0 / 255.0).mean()))


def bands_for_grid(whole, part, n):
    """For a n x n grid, band[k] = mean deviation of pixels whose distance to the
    nearest grid line falls in [edges[k], edges[k+1])."""
    h, w = whole.shape[:2]
    th, tw = h // n, w // n
    d = np.abs(whole - part).max(axis=-1)
    ys = np.arange(h)
    xs = np.arange(w)
    dy = np.min(np.stack([np.abs(ys - g * th) for g in range(n + 1)]), axis=0)
    dx = np.min(np.stack([np.abs(xs - g * tw) for g in range(n + 1)]), axis=0)
    dist = np.minimum(dy[:, None], dx[None, :])
    edges = (0, 2, 6, 12, 24, 9999)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (dist >= a) & (dist < b)
        out.append(float(d[m].mean()) if m.any() else 0.0)
    return out, edges


def gridline_visibility(whole, part, n):
    """How much of the tiled-vs-whole deviation energy sits exactly ON the grid.
    High value == seams are structurally visible, not just numerically different."""
    h, w = whole.shape[:2]
    th, tw = h // n, w // n
    d = np.abs(whole - part).max(axis=-1)
    rows = np.isin(np.arange(h), [g * th for g in range(1, n)])
    cols = np.isin(np.arange(w), [g * tw for g in range(1, n)])
    on = np.zeros((h, w), bool)
    on[rows, :] = True
    on[:, cols] = True
    e_on = float((d[on] ** 2).sum())
    e_tot = float((d ** 2).sum()) + 1e-20
    frac_px = float(on.mean())
    return dict(gv=float(e_on / e_tot), ratio=float((e_on / e_tot) / frac_px))


def contrast_equivariance(fn, img, gains=(0.5, 2.0)):
    """CEE: || f(g.I) - g.f(I) || / (g * RMSE(f(I))).  0 iff contrast-equivariant.
    Scaling is applied WITHOUT clipping, so the only remaining asymmetry is the
    operator itself.  An operator whose threshold is a whole-image quantile cannot
    be 0 here; a homogeneous bounded-support operator is 0 up to float error."""
    base = fn(img)
    rmse_b = float(np.sqrt((base ** 2).mean())) + 1e-12
    res = {}
    for g in gains:
        y = fn(np.ascontiguousarray((img * g).astype(np.float32)))
        res[float(g)] = float(np.sqrt(((y - g * base) ** 2).mean()) / (g * rmse_b))
    return res


def tau_of_sc(img, c=4.0, p=0.25):
    """The published global criterion: c * Q_p of the base dispersion map."""
    _, _, vm = dispersion(img)
    return c * float(np.quantile(vm[0], p))


def tau_drift(tau_of, img, n):
    """Per-tile value of a global statistic, normalised by its whole-image value.
    Direct mechanism probe for class-B mismatch."""
    h, w = img.shape[:2]
    th, tw = h // n, w // n
    ref = tau_of(img)
    vals = np.array([tau_of(img[i * th:(i + 1) * th, j * tw:(j + 1) * tw])
                     for i in range(n) for j in range(n)])
    r = vals / ref if ref > 0 else vals / (vals.mean() + 1e-12)
    return dict(ref=float(ref), ratio_min=float(r.min()), ratio_max=float(r.max()),
                spread=float(r.max() / (r.min() + 1e-12)),
                frac_out10=float((np.abs(r - 1) > 0.10).mean()))


def load(path, size=512):
    img = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
    if size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return np.ascontiguousarray(img)
