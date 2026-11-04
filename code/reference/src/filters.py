"""Low-level filtering primitives: oriented Gaussians, sector masks, soft-min."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .structure import orientation_weights

cv2.setNumThreads(1)
_BORDER = cv2.BORDER_REFLECT


def _filter(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img, -1, kernel, borderType=_BORDER)


def rotated_gaussian(
    sigma_along: float, sigma_across: float, theta: float, truncate: float = 3.0
) -> np.ndarray:
    radius = int(np.ceil(truncate * max(sigma_along, sigma_across)))
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    X, Y = np.meshgrid(ax, ax)
    Xr = X * np.cos(theta) + Y * np.sin(theta)
    Yr = -X * np.sin(theta) + Y * np.cos(theta)
    g = np.exp(-0.5 * (Xr**2 / sigma_along**2 + Yr**2 / sigma_across**2))
    g /= g.sum()
    return g.astype(np.float32)


def sector_mask(radius: int, sector: int, n_sectors: int, theta: float) -> np.ndarray:
    """Binary circular-wedge mask, normalised to unit sum."""
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    X, Y = np.meshgrid(ax, ax)
    Xr = X * np.cos(theta) + Y * np.sin(theta)
    Yr = -X * np.sin(theta) + Y * np.cos(theta)
    r = np.hypot(Xr, Yr)
    phi = np.arctan2(Yr, Xr)
    width = 2.0 * np.pi / n_sectors
    phi = (phi - sector * width) % (2.0 * np.pi)
    mask = ((r <= radius + 1e-6) & (phi <= width)).astype(np.float32)
    if mask.sum() == 0:  # degenerate radius
        mask[radius, radius] = 1.0
    return mask / mask.sum()


def _blend_orientations(bank: np.ndarray, theta: np.ndarray, n_bins: int) -> np.ndarray:
    """bank: (n_bins, H, W, C) -> (H, W, C), linearly blended by local theta."""
    i0, i1, frac = orientation_weights(theta, n_bins)
    b0 = np.take_along_axis(bank, i0[None, ..., None], axis=0)[0]
    b1 = np.take_along_axis(bank, i1[None, ..., None], axis=0)[0]
    f = frac[..., None]
    return b0 * (1.0 - f) + b1 * f


def anisotropic_presmooth(
    img: np.ndarray,
    theta: np.ndarray,
    coherence: np.ndarray,
    sigma_along: float = 2.5,
    n_bins: int = 12,
) -> np.ndarray:
    """Oriented smoothing whose elongation follows local coherence.

    Three anisotropy levels (isotropic, medium, strong) are rendered and
    blended according to the local coherence.  Orientation is interpolated
    linearly between the two nearest filter banks.
    """
    levels = (0.0, 0.5, 1.0)
    outs = []
    for a in levels:
        sigma_across = sigma_along / (1.0 + 3.0 * a)
        kernels = [
            rotated_gaussian(sigma_along, sigma_across, m * np.pi / n_bins)
            for m in range(n_bins)
        ]
        bank = np.stack([_filter(img, k) for k in kernels], axis=0)
        outs.append(_blend_orientations(bank, theta, n_bins))
    u = np.clip(coherence, 0.0, 1.0) * (len(levels) - 1)
    j0 = np.floor(u).astype(np.int32)
    j1 = np.minimum(j0 + 1, len(levels) - 1)
    w = (u - j0)[..., None]
    stack = np.stack(outs, axis=0)  # (3, H, W, C)
    hi = np.take_along_axis(stack, j0[None, ..., None], axis=0)[0]
    lo = np.take_along_axis(stack, j1[None, ..., None], axis=0)[0]
    return (hi * (1.0 - w) + lo * w).astype(np.float32)


def isotropic_presmooth(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    return np.stack(
        [gaussian_filter(img[..., c], sigma=sigma, mode="reflect") for c in range(3)],
        axis=-1,
    ).astype(np.float32)


def sector_statistics(
    img: np.ndarray,
    radius: int,
    theta: np.ndarray,
    n_sectors: int = 8,
    n_bins: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Orientation-interpolated sector means and variances.

    Returns
    -------
    means : (K, H, W, C) float32
    variances : (K, H, W) float32  (variance of the channel mean)
    """
    h, w, c = img.shape
    sq = img * img
    means = np.zeros((n_sectors, h, w, c), dtype=np.float32)
    variances = np.zeros((n_sectors, h, w), dtype=np.float32)

    for m in range(n_bins):
        theta_m = m * np.pi / n_bins
        # Orientation weight of this bank for every pixel (two neighbours).
        pos = (theta % np.pi) / (np.pi / n_bins)
        i0 = np.floor(pos).astype(np.int32) % n_bins
        frac = (pos - np.floor(pos)).astype(np.float32)
        wm = np.where(i0 == m, 1.0 - frac, 0.0).astype(np.float32)
        wm = np.where((i0 + 1) % n_bins == m, wm + frac, wm)
        active = wm > 1e-4
        if not np.any(active):
            continue
        wm = wm[..., None]
        for s in range(n_sectors):
            k = sector_mask(radius, s, n_sectors, theta_m)
            mean_s = _filter(img, k)
            mean2_s = _filter(sq, k)
            mean_lum = mean_s.mean(axis=-1)
            mean2_lum = mean2_s.mean(axis=-1)
            var_s = np.maximum(mean2_lum - mean_lum * mean_lum, 0.0)
            means[s] += mean_s * wm
            variances[s] += var_s * wm[..., 0]
    return means, variances


def combine_sectors(
    means: np.ndarray, variances: np.ndarray, mode: str = "soft", beta: float = 8.0
) -> np.ndarray:
    k = means.shape[0]
    if mode == "hard":
        idx = np.argmin(variances, axis=0)
        return np.take_along_axis(means, idx[None, ..., None], axis=0)[0]
    vmin = variances.min(axis=0, keepdims=True)
    scale = np.median(variances, axis=0, keepdims=True) + 1e-6
    logits = -beta * (variances - vmin) / scale
    logits -= logits.max(axis=0, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=0, keepdims=True)
    return (means * w[..., None]).sum(axis=0)


def classic_kuwahara(img: np.ndarray, radius: int = 4) -> np.ndarray:
    """Original four-quadrant Kuwahara filter."""
    h, w, c = img.shape
    sq = img * img
    win = 2 * radius + 1
    means, vars_ = [], []
    for qy in (0, 1):
        for qx in (0, 1):
            m = np.zeros((win, win), dtype=np.float32)
            ys = slice(0, radius + 1) if qy == 0 else slice(radius, win)
            xs = slice(0, radius + 1) if qx == 0 else slice(radius, win)
            m[ys, xs] = 1.0
            m /= m.sum()
            mu = _filter(img, m)
            mu2 = _filter(sq, m)
            means.append(mu)
            vars_.append(np.maximum(mu2.mean(-1) - mu.mean(-1) ** 2, 0.0))
    idx = np.argmin(np.stack(vars_, 0), axis=0)
    stack = np.stack(means, 0)
    return np.take_along_axis(stack, idx[None, ..., None], axis=0)[0]


def bilateral(img: np.ndarray, d: int = 13, sigma_color: float = 0.15, sigma_space: float = 8.0):
    out = cv2.bilateralFilter(
        img.astype(np.float32), d, sigma_color, sigma_space, borderType=_BORDER
    )
    return out.astype(np.float32)
