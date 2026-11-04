"""Structure-tensor analysis used by the proposed filter."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def luminance(img: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance of an RGB image in [0, 1]."""
    return (
        0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    ).astype(np.float32)


def structure_tensor(
    img: np.ndarray, sigma_g: float = 1.0, sigma_t: float = 4.0
) -> dict[str, np.ndarray]:
    """Return orientation, coherence and edge energy of an RGB image.

    theta is the *tangent* orientation (edge direction, radians in [0, pi)).
    coherence is (l1 - l2) / (l1 + l2) in [0, 1].
    edge_energy is l1 normalised by its 95th percentile, clipped to [0, 1].
    """
    L = luminance(img)
    gy = gaussian_filter(L, sigma=sigma_g, order=(1, 0), mode="reflect")
    gx = gaussian_filter(L, sigma=sigma_g, order=(0, 1), mode="reflect")

    jxx = gaussian_filter(gx * gx, sigma=sigma_t, mode="reflect")
    jyy = gaussian_filter(gy * gy, sigma=sigma_t, mode="reflect")
    jxy = gaussian_filter(gx * gy, sigma=sigma_t, mode="reflect")

    tmp = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy**2, 0.0))
    l1 = 0.5 * (jxx + jyy + tmp)
    l2 = 0.5 * (jxx + jyy - tmp)
    l1 = np.maximum(l1, 0.0)
    l2 = np.maximum(l2, 0.0)

    coherence = ((l1 - l2) / (l1 + l2 + 1e-8)).astype(np.float32)
    # Major eigenvector points across the edge; rotate to the tangent.
    theta_grad = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy + 1e-12)
    theta = (theta_grad + np.pi / 2.0).astype(np.float32)

    p95 = float(np.percentile(l1, 95.0)) + 1e-12
    edge_energy = np.clip(l1 / p95, 0.0, 1.0).astype(np.float32)
    return {
        "theta": theta,
        "coherence": coherence,
        "edge_energy": edge_energy,
        "l1": l1.astype(np.float32),
        "l2": l2.astype(np.float32),
        "luminance": L,
    }


def orientation_weights(theta: np.ndarray, n_bins: int):
    """Nearest two orientation bins and their linear interpolation weights."""
    pos = (theta % np.pi) / (np.pi / n_bins)
    i0 = np.floor(pos).astype(np.int32) % n_bins
    frac = (pos - np.floor(pos)).astype(np.float32)
    return i0, (i0 + 1) % n_bins, frac
