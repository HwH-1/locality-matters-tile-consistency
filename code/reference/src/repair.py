"""Structure-repair layer for Kuwahara abstraction.

The layer detects two structures that sector averaging destroys:
four-way (and T-) junctions, where colour is mixed, and one-pixel lines,
where contrast is eroded.  A high-fidelity reference is blended back only
inside soft masks around those structures, so flat regions are untouched.

Both detectors are gated by a local gradient-to-noise reliability map.  The
gate rejects isolated noise peaks and fine stochastic texture that would
otherwise be amplified by a purely percentile-based detector.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.feature import corner_harris, peak_local_max
from skimage.morphology import (
    black_tophat,
    dilation,
    disk,
    footprint_rectangle,
    white_tophat,
)

from .structure import luminance

cv2.setNumThreads(1)


def reliability_map(img: np.ndarray, snr_min: float = 2.5) -> np.ndarray:
    """Boolean map of pixels whose gradient is significant against local noise.

    The noise level is estimated robustly from the residual of a 3x3 median
    filter.  A median filter removes one-pixel lines and impulse noise from the
    estimate, so the gate does not suppress the structures it is meant to
    protect.  The gradient magnitude is lightly smoothed before comparison.
    """
    ratios = []
    for c in range(img.shape[2]):
        channel = img[..., c]
        residual = channel - median_filter(channel, size=3)
        sigma = 1.4826 * median_filter(np.abs(residual), size=9)
        gy, gx = np.gradient(channel)
        grad = np.hypot(gx, gy)
        ratios.append(gaussian_filter(grad, sigma=1.0) / (sigma + 1e-6))
    return np.maximum.reduce(ratios) >= snr_min


def _normalise_mask(mask: np.ndarray) -> np.ndarray:
    peak = float(mask.max())
    if peak <= 0.0:
        return np.zeros_like(mask, dtype=np.float32)
    return (mask / peak).astype(np.float32)


def junction_mask(
    img: np.ndarray,
    dilation_radius: int = 2,
    sigma: float = 1.0,
    snr_min: float = 2.5,
    min_distance: int = 2,
    rel_threshold: float = 0.05,
    reliability: np.ndarray | None = None,
) -> np.ndarray:
    """Soft mask of Harris corner peaks, normalised to [0, 1].

    Harris corners respond to two-dimensional structure (junctions and
    corners) but only weakly to straight edges.  Instead of thresholding the
    whole response map -- which, on a piecewise-constant phantom, would select
    the zero background -- we keep local maxima only.  This makes the mask
    genuinely junction-specific.
    """
    response = np.maximum.reduce(
        [
            np.maximum(corner_harris(img[..., c], method="k", k=0.05), 0.0)
            for c in range(img.shape[2])
        ]
    )
    peaks = peak_local_max(
        response,
        min_distance=min_distance,
        threshold_rel=rel_threshold,
        exclude_border=3,
    )
    mask = np.zeros(response.shape, dtype=bool)
    if peaks.size:
        if reliability is None:
            reliability = reliability_map(img, snr_min=snr_min)
        keep = np.array(
            [
                bool(reliability[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2].any())
                for y, x in peaks
            ],
            dtype=bool,
        )
        peaks = peaks[keep]
        if peaks.size:
            mask[tuple(peaks.T)] = True
    if dilation_radius > 0:
        mask = dilation(mask, disk(dilation_radius))
    mask = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return _normalise_mask(mask)


def line_mask(
    img: np.ndarray,
    percentile: float = 99.0,
    min_threshold: float = 0.08,
    dilation_radius: int = 1,
    sigma: float = 1.0,
    snr_min: float = 2.5,
    reliability: np.ndarray | None = None,
) -> np.ndarray:
    """Soft mask of thin bright or dark lines.

    A 3x3 morphological top-hat (bright lines) and black-hat (dark lines)
    respond to structures narrower than the footprint and are close to zero
    on a straight step edge.  The threshold is the larger of an absolute
    contrast floor and an image-dependent percentile, which keeps the mask
    non-empty on synthetic line phantoms.  A gradient-to-noise gate removes
    responses caused by stochastic noise.
    """
    footprint = footprint_rectangle((3, 3))
    response = np.maximum.reduce(
        [
            np.maximum(
                white_tophat(img[..., c], footprint),
                black_tophat(img[..., c], footprint),
            )
            for c in range(img.shape[2])
        ]
    )
    threshold = max(min_threshold, float(np.percentile(response, percentile)))
    mask = response >= threshold
    if reliability is None:
        reliability = reliability_map(img, snr_min=snr_min)
    mask &= reliability
    if dilation_radius > 0:
        mask = dilation(mask, disk(dilation_radius))
    mask = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return _normalise_mask(mask)


def repair_weights(
    img: np.ndarray,
    line_percentile: float = 99.0,
    junction_dilation: int = 2,
    line_dilation: int = 1,
    mask_sigma: float = 1.0,
    w_max: float = 0.9,
    snr_min: float = 2.5,
    junction_min_distance: int = 2,
    junction_rel_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (junction mask, line mask, combined weight)."""
    reliability = reliability_map(img, snr_min=snr_min)
    mj = junction_mask(
        img,
        dilation_radius=junction_dilation,
        sigma=mask_sigma,
        snr_min=snr_min,
        min_distance=junction_min_distance,
        rel_threshold=junction_rel_threshold,
        reliability=reliability,
    )
    ml = line_mask(
        img,
        percentile=line_percentile,
        dilation_radius=line_dilation,
        sigma=mask_sigma,
        snr_min=snr_min,
        reliability=reliability,
    )
    w = 1.0 - (1.0 - mj) * (1.0 - ml)
    w = np.clip(w, 0.0, w_max)
    return mj, ml, w


def structure_repair(
    img: np.ndarray,
    base: np.ndarray,
    line_percentile: float = 99.0,
    junction_dilation: int = 2,
    line_dilation: int = 1,
    mask_sigma: float = 1.0,
    w_max: float = 0.9,
    snr_min: float = 2.5,
    junction_min_distance: int = 2,
    junction_rel_threshold: float = 0.05,
    mode: str = "both",
) -> np.ndarray:
    """Blend the input back into an abstracted image inside structure masks.

    mode: "both", "junction", "line", or "global".  The global variant uses
    the spatial mean of the combined weight, i.e. the same average repair
    strength without localisation.
    """
    mj, ml, w = repair_weights(
        img,
        line_percentile=line_percentile,
        junction_dilation=junction_dilation,
        line_dilation=line_dilation,
        mask_sigma=mask_sigma,
        w_max=w_max,
        snr_min=snr_min,
        junction_min_distance=junction_min_distance,
        junction_rel_threshold=junction_rel_threshold,
    )
    if mode == "junction":
        w = mj
    elif mode == "line":
        w = ml
    elif mode == "global":
        w = np.full_like(w, float(w.mean()))
    elif mode != "both":
        raise ValueError("unknown repair mode: %s" % mode)
    # The cap applies to every mode, so that the single-structure ablations
    # differ from the full model only in which detector is active.
    w = np.clip(w, 0.0, w_max)
    weight = w[..., None]
    out = (1.0 - weight) * base + weight * img
    return np.clip(out, 0.0, 1.0).astype(np.float32)
