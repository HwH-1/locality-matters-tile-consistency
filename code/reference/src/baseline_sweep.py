"""Baseline parameter sweep on the four pilot images.

This is a fairness check, not a replacement for the main experiment: it asks
whether the fixed settings used for the baselines in the paper are reasonable,
and how much each baseline moves when its main parameter is varied.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))

from src import filters  # noqa: E402
from src.data import load_all  # noqa: E402
from src.metrics import METRICS, compute_metrics  # noqa: E402
from src.structure import structure_tensor  # noqa: E402

cv2.setNumThreads(1)

PILOT = {"camera", "astronaut", "synth_checker", "synth_regions"}


def rgf(img: np.ndarray, sigma_color: float, sigma_space: float, iters: int = 4) -> np.ndarray:
    out = cv2.ximgproc.rollingGuidanceFilter(
        img.astype(np.float32), d=0, sigmaColor=sigma_color,
        sigmaSpace=sigma_space, numOfIter=iters,
    )
    return np.clip(out, 0, 1)


def guided(img: np.ndarray, radius: int, eps: float) -> np.ndarray:
    out = cv2.ximgproc.guidedFilter(
        guide=img.astype(np.float32), src=img.astype(np.float32),
        radius=radius, eps=eps,
    )
    return np.clip(out, 0, 1)


def configs():
    for r in (3, 4, 5, 6):
        yield "classic", f"r={r}", {"radius": r}
    for r in (3, 4, 5, 6):
        yield "sector", f"r={r}", {"radius": r, "sectors": 8, "bins": 12,
                                    "combine": "hard", "beta": 8.0, "steered": False}
    for r in (3, 4, 5, 6):
        yield "akf", f"r={r}", {"radius": r, "sectors": 8, "bins": 12,
                                "combine": "hard", "beta": 8.0, "sigma_along": 2.5}
    for ss in (5.0, 8.0, 11.0):
        for sr in (0.10, 0.15, 0.20):
            yield "bilateral", f"ss={ss:g},sr={sr:g}", {"d": 13, "sigma_color": sr, "sigma_space": ss}
    for sc in (0.05, 0.10, 0.20):
        for ss in (8.0, 12.0):
            yield "rgf", f"sc={sc:g},ss={ss:g}", {"sigma_color": sc, "sigma_space": ss}
    for rad in (4, 8):
        for eps in (1e-3, 1e-2):
            yield "guided", f"r={rad},eps={eps:g}", {"radius": rad, "eps": eps}


def apply(method: str, cfg: dict, img: np.ndarray, st: dict) -> np.ndarray:
    if method == "classic":
        return filters.classic_kuwahara(img, radius=cfg["radius"])
    if method == "sector":
        theta = np.zeros_like(st["theta"])
        mu, var = filters.sector_statistics(img, cfg["radius"], theta,
                                            n_sectors=cfg["sectors"], n_bins=cfg["bins"])
        return filters.combine_sectors(mu, var, mode=cfg["combine"], beta=cfg["beta"])
    if method == "akf":
        smooth = filters.anisotropic_presmooth(
            img, st["theta"], st["coherence"], sigma_along=cfg["sigma_along"], n_bins=cfg["bins"]
        )
        mu, var = filters.sector_statistics(smooth, cfg["radius"], st["theta"],
                                            n_sectors=cfg["sectors"], n_bins=cfg["bins"])
        return filters.combine_sectors(mu, var, mode="hard", beta=cfg["beta"])
    if method == "bilateral":
        return filters.bilateral(img, cfg["d"], cfg["sigma_color"], cfg["sigma_space"])
    if method == "rgf":
        return rgf(img, cfg["sigma_color"], cfg["sigma_space"])
    if method == "guided":
        return guided(img, cfg["radius"], cfg["eps"])
    raise ValueError(method)


def main() -> None:
    images = [im for im in load_all() if im.name in PILOT]
    rows = []
    for im in images:
        st = structure_tensor(im.image)
        for method, label, cfg in configs():
            out = apply(method, cfg, im.image, st)
            met = compute_metrics(im.image, out, st_src=st)
            rows.append({"image": im.name, "method": method, "setting": label, **met})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "baseline_sweep.csv", index=False, float_format="%.6f")
    agg = df.groupby(["method", "setting"]).agg(
        sr=("sr", "median"), ssim=("ssim", "median"), fsr=("fsr", "median"),
        scg=("scg", "median"), esr=("esr", "median"), ece=("ece", "median"),
        ovt=("ovt", "median"),
    ).reset_index()
    agg.to_csv(RES / "baseline_sweep_grid.csv", index=False, float_format="%.6f")
    for method, g in agg.groupby("method"):
        print("\n==", method)
        print(g.sort_values("fsr").to_string(index=False))


if __name__ == "__main__":
    main()
