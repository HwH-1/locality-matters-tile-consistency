"""Method registry for the comparison and ablation study."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from . import filters, repair
from .structure import structure_tensor


@dataclass
class MethodSpec:
    name: str
    label: str
    family: str
    params: dict
    fn: Callable[[np.ndarray, dict], np.ndarray]
    description: str = ""


def _std_ctx(img: np.ndarray, st: dict) -> dict:
    return {"img": img, "st": st}


def _prepare(
    img: np.ndarray,
    sigma_along: float = 2.5,
    n_bins: int = 12,
    sigma_t: float = 4.0,
    sigma_g: float = 1.0,
) -> tuple[dict, np.ndarray]:
    st = structure_tensor(img, sigma_g=sigma_g, sigma_t=sigma_t)
    smooth = filters.anisotropic_presmooth(
        img, st["theta"], st["coherence"], sigma_along=sigma_along, n_bins=n_bins
    )
    return st, smooth


def m_classic(img: np.ndarray, cfg: dict) -> np.ndarray:
    return filters.classic_kuwahara(img, radius=cfg["radius"])


def m_sector(img: np.ndarray, cfg: dict) -> np.ndarray:
    st = structure_tensor(img)
    theta = np.zeros_like(st["theta"]) if not cfg["steered"] else st["theta"]
    mu, var = filters.sector_statistics(
        img, cfg["radius"], theta, n_sectors=cfg["sectors"], n_bins=cfg["bins"]
    )
    return filters.combine_sectors(mu, var, mode=cfg["combine"], beta=cfg["beta"])


def m_akf(img: np.ndarray, cfg: dict) -> np.ndarray:
    st, smooth = _prepare(
        img, cfg["sigma_along"], cfg["bins"],
        cfg.get("sigma_t", 4.0), cfg.get("sigma_g", 1.0),
    )
    mu, var = filters.sector_statistics(
        smooth, cfg["radius"], st["theta"], n_sectors=cfg["sectors"], n_bins=cfg["bins"]
    )
    return filters.combine_sectors(mu, var, mode="hard", beta=cfg["beta"])


def m_ours(img: np.ndarray, cfg: dict) -> np.ndarray:
    st, smooth = _prepare(
        img, cfg["sigma_along"], cfg["bins"],
        cfg.get("sigma_t", 4.0), cfg.get("sigma_g", 1.0),
    )
    theta = st["theta"]
    outs = []
    for r in (cfg["r_min"], cfg["r_max"]):
        mu, var = filters.sector_statistics(
            smooth, r, theta, n_sectors=cfg["sectors"], n_bins=cfg["bins"]
        )
        outs.append(filters.combine_sectors(mu, var, mode=cfg["combine"], beta=cfg["beta"]))
    if cfg["adaptive"]:
        alpha = ((1.0 - st["edge_energy"]) ** cfg.get("gamma", 2.0))[..., None]
        out = outs[0] * (1.0 - alpha) + outs[1] * alpha
    else:
        out = outs[0]
    return out


def m_akf_multiscale(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Three-scale AKF with hard sector selection and an edge-energy blend.

    The small, middle and large scales use the same anisotropic pre-smoothing
    and the same hard minimum as fixed-scale AKF.  Their outputs are blended
    by a quadratic function of edge energy, so flat regions receive the large
    scale and edges the small one.  This is a strong multi-scale Kuwahara
    baseline: it has more scales than CA-AKF but no soft sector pooling.
    """
    st, smooth = _prepare(
        img, cfg["sigma_along"], cfg["bins"],
        cfg.get("sigma_t", 4.0), cfg.get("sigma_g", 1.0),
    )
    outs = []
    for r in cfg["radii"]:
        mu, var = filters.sector_statistics(
            smooth, r, st["theta"], n_sectors=cfg["sectors"], n_bins=cfg["bins"]
        )
        outs.append(filters.combine_sectors(mu, var, mode="hard", beta=cfg["beta"]))
    e = st["edge_energy"][..., None]
    w_small = e**2
    w_mid = 2.0 * e * (1.0 - e)
    w_large = (1.0 - e) ** 2
    return w_small * outs[0] + w_mid * outs[1] + w_large * outs[2]


def m_bilateral(img: np.ndarray, cfg: dict) -> np.ndarray:
    return filters.bilateral(img, cfg["d"], cfg["sigma_color"], cfg["sigma_space"])


def m_rgf(img: np.ndarray, cfg: dict) -> np.ndarray:
    out = cv2.ximgproc.rollingGuidanceFilter(
        img.astype(np.float32),
        d=cfg["d"],
        sigmaColor=cfg["sigma_color"],
        sigmaSpace=cfg["sigma_space"],
        numOfIter=cfg["iterations"],
    )
    return out


def m_guided(img: np.ndarray, cfg: dict) -> np.ndarray:
    out = cv2.ximgproc.guidedFilter(
        guide=img.astype(np.float32),
        src=img.astype(np.float32),
        radius=cfg["radius"],
        eps=cfg["eps"],
    )
    return out


def _ca_config(cfg: dict) -> dict:
    return {
        "r_min": cfg["r_min"],
        "r_max": cfg["r_max"],
        "sectors": cfg["sectors"],
        "bins": cfg["bins"],
        "combine": "soft",
        "beta": cfg["beta"],
        "sigma_along": cfg["sigma_along"],
        "adaptive": True,
        "gamma": cfg["gamma"],
    }


def m_jlr(img: np.ndarray, cfg: dict) -> np.ndarray:
    """CA-AKF followed by local junction/line repair."""
    base = m_ours(img, _ca_config(cfg))
    return repair.structure_repair(
        img, base,
        line_percentile=cfg["line_percentile"],
        junction_dilation=cfg["junction_dilation"],
        line_dilation=cfg["line_dilation"],
        mask_sigma=cfg["mask_sigma"],
        w_max=cfg["w_max"],
        snr_min=cfg["snr_min"],
        junction_min_distance=cfg["junction_min_distance"],
        junction_rel_threshold=cfg["junction_rel_threshold"],
        mode="both",
    )


def m_jlr_ablation(img: np.ndarray, cfg: dict) -> np.ndarray:
    base = m_ours(img, _ca_config(cfg))
    return repair.structure_repair(
        img, base,
        line_percentile=cfg["line_percentile"],
        junction_dilation=cfg["junction_dilation"],
        line_dilation=cfg["line_dilation"],
        mask_sigma=cfg["mask_sigma"],
        w_max=cfg["w_max"],
        snr_min=cfg["snr_min"],
        junction_min_distance=cfg["junction_min_distance"],
        junction_rel_threshold=cfg["junction_rel_threshold"],
        mode=cfg["mode"],
    )


METHODS: list[MethodSpec] = [
    MethodSpec(
        "classic",
        "Kuwahara (1976)",
        "baseline",
        {"radius": 4},
        m_classic,
        "Four-quadrant Kuwahara, r = 4",
    ),
    MethodSpec(
        "sector",
        "Sector Kuwahara",
        "baseline",
        {"radius": 4, "sectors": 8, "bins": 12, "combine": "hard", "beta": 8.0, "steered": False},
        m_sector,
        "Eight fixed circular sectors, r = 4, hard minimum",
    ),
    MethodSpec(
        "akf",
        "AKF (fixed scale)",
        "baseline",
        {
            "radius": 5,
            "sectors": 8,
            "bins": 12,
            "combine": "hard",
            "beta": 8.0,
            "sigma_along": 2.5,
        },
        m_akf,
        "Anisotropic pre-smoothing + steered sector Kuwahara, r = 5, hard minimum",
    ),
    MethodSpec(
        "bilateral",
        "Bilateral filter",
        "baseline",
        {"d": 13, "sigma_color": 0.15, "sigma_space": 8.0},
        m_bilateral,
        "Edge-preserving smoothing baseline",
    ),
    MethodSpec(
        "rgf",
        "RGF",
        "reference",
        {"d": 0, "sigma_color": 0.05, "sigma_space": 8.0, "iterations": 4},
        m_rgf,
        "Rolling guidance filter as a modern edge-preserving reference smoother",
    ),
    MethodSpec(
        "guided",
        "Guided filter",
        "reference",
        {"radius": 4, "eps": 0.01},
        m_guided,
        "Self-guided filter as a modern edge-preserving reference smoother",
    ),
    MethodSpec(
        "ca_akf",
        "CA-AKF (base)",
        "baseline",
        {
            "r_min": 3,
            "r_max": 6,
            "sectors": 8,
            "bins": 12,
            "combine": "soft",
            "beta": 8.0,
            "sigma_along": 2.5,
            "adaptive": True,
            "gamma": 2.0,
        },
        m_ours,
        "Coherence-adaptive scale + soft sector selection",
    ),
    MethodSpec(
        "ab_no_adaptive",
        "Ablation: fixed scale",
        "ablation",
        {
            "r_min": 5,
            "r_max": 5,
            "sectors": 8,
            "bins": 12,
            "combine": "soft",
            "beta": 8.0,
            "sigma_along": 2.5,
            "adaptive": False,
            "gamma": 2.0,
        },
        m_ours,
        "Soft selection, fixed r = 5",
    ),
    MethodSpec(
        "ab_no_soft",
        "Ablation: hard minimum",
        "ablation",
        {
            "r_min": 3,
            "r_max": 6,
            "sectors": 8,
            "bins": 12,
            "combine": "hard",
            "beta": 8.0,
            "sigma_along": 2.5,
            "adaptive": True,
            "gamma": 2.0,
        },
        m_ours,
        "Adaptive scale, hard sector minimum",
    ),
    MethodSpec(
        "ab_iso_pre",
        "Ablation: isotropic pre-filter",
        "ablation",
        {
            "r_min": 3,
            "r_max": 6,
            "sectors": 8,
            "bins": 12,
            "combine": "soft",
            "beta": 8.0,
            "sigma_along": 2.5,
            "adaptive": True,
            "gamma": 2.0,
        },
        None,  # assigned below
        "Adaptive scale + soft selection with isotropic pre-filtering",
    ),
]

METHODS += [
    MethodSpec(
        "jlr_cakf",
        "JLR-CAKF (ours)",
        "proposed",
        {
            "r_min": 3, "r_max": 6, "sectors": 8, "bins": 12,
            "beta": 8.0, "sigma_along": 2.5, "gamma": 2.0,
            "line_percentile": 99.0,
            "junction_dilation": 3, "line_dilation": 2,
            "mask_sigma": 1.5, "w_max": 0.9,
            "snr_min": 2.5, "junction_min_distance": 2,
            "junction_rel_threshold": 0.05,
        },
        m_jlr,
        "CA-AKF with gated junction and thin-line repair",
    ),
    MethodSpec(
        "jlr_junction",
        "Ablation: junction only",
        "ablation",
        {
            "r_min": 3, "r_max": 6, "sectors": 8, "bins": 12,
            "beta": 8.0, "sigma_along": 2.5, "gamma": 2.0,
            "line_percentile": 99.0,
            "junction_dilation": 3, "line_dilation": 2,
            "mask_sigma": 1.5, "w_max": 0.9,
            "snr_min": 2.5, "junction_min_distance": 2,
            "junction_rel_threshold": 0.05, "mode": "junction",
        },
        m_jlr_ablation,
        "Repair at junctions only",
    ),
    MethodSpec(
        "jlr_line",
        "Ablation: line only",
        "ablation",
        {
            "r_min": 3, "r_max": 6, "sectors": 8, "bins": 12,
            "beta": 8.0, "sigma_along": 2.5, "gamma": 2.0,
            "line_percentile": 99.0,
            "junction_dilation": 3, "line_dilation": 2,
            "mask_sigma": 1.5, "w_max": 0.9,
            "snr_min": 2.5, "junction_min_distance": 2,
            "junction_rel_threshold": 0.05, "mode": "line",
        },
        m_jlr_ablation,
        "Repair at thin lines only",
    ),
    MethodSpec(
        "jlr_global",
        "Ablation: global blend",
        "ablation",
        {
            "r_min": 3, "r_max": 6, "sectors": 8, "bins": 12,
            "beta": 8.0, "sigma_along": 2.5, "gamma": 2.0,
            "line_percentile": 99.0,
            "junction_dilation": 3, "line_dilation": 2,
            "mask_sigma": 1.5, "w_max": 0.9,
            "snr_min": 2.5, "junction_min_distance": 2,
            "junction_rel_threshold": 0.05, "mode": "global",
        },
        m_jlr_ablation,
        "Global blend with the same mean repair weight",
    ),
]


def _m_iso(img: np.ndarray, cfg: dict) -> np.ndarray:
    st = structure_tensor(img)
    smooth = filters.isotropic_presmooth(img, sigma=1.5)
    outs = []
    for r in (cfg["r_min"], cfg["r_max"]):
        mu, var = filters.sector_statistics(
            smooth, r, st["theta"], n_sectors=cfg["sectors"], n_bins=cfg["bins"]
        )
        outs.append(filters.combine_sectors(mu, var, mode=cfg["combine"], beta=cfg["beta"]))
    alpha = ((1.0 - st["edge_energy"]) ** cfg.get("gamma", 2.0))[..., None]
    return outs[0] * (1.0 - alpha) + outs[1] * alpha


next(m for m in METHODS if m.name == "ab_iso_pre").fn = _m_iso



def run_method(spec: MethodSpec, img: np.ndarray) -> np.ndarray:
    out = spec.fn(img, spec.params)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
