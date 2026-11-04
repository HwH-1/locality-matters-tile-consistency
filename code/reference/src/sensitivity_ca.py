"""One-at-a-time sensitivity of CA-AKF on the four tuning images."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TEX = ROOT.parent / "02_manuscript" / "tables"
sys.path.insert(0, str(ROOT))

from src.data import load_all  # noqa: E402
from src.methods import m_ours  # noqa: E402
from src.metrics import METRICS, compute_metrics  # noqa: E402
from src.structure import structure_tensor  # noqa: E402

cv2.setNumThreads(1)

TUNED = {"camera", "astronaut", "synth_checker", "synth_regions"}
BASE = {
    "r_min": 3, "r_max": 6, "sectors": 8, "bins": 12, "combine": "soft",
    "beta": 8.0, "sigma_along": 2.5, "adaptive": True, "gamma": 2.0,
    "sigma_t": 4.0, "sigma_g": 1.0,
}
SWEEPS = [
    ("beta", [2.0, 4.0, 8.0, 16.0, 32.0]),
    ("gamma", [1.0, 1.5, 2.0, 3.0, 4.0]),
    ("sigma_along", [1.5, 2.0, 2.5, 3.0, 4.0]),
    ("sigma_t", [2.0, 3.0, 4.0, 6.0, 8.0]),
    ("r_max", [4, 5, 6, 7, 8]),
    ("r_min", [2, 3, 4]),
    ("bins", [6, 8, 12, 16]),
    ("sectors", [6, 8, 12]),
]
PRETTY = {
    "beta": r"$\beta$", "gamma": r"$\gamma$", "sigma_along": r"$\sigma_\parallel$",
    "sigma_t": r"$\rho$", "r_max": r"$r_{\max}$", "r_min": r"$r_{\min}$",
    "bins": "orientation bins", "sectors": "sectors",
}


def main() -> None:
    images = [im for im in load_all() if im.name in TUNED]
    print("images:", [im.name for im in images], flush=True)
    rows = []
    for param, values in SWEEPS:
        for val in values:
            cfg = {**BASE, param: val}
            if param == "r_min":
                cfg["r_max"] = 6
            if param == "r_max":
                cfg["r_min"] = 3
            for im in images:
                st = structure_tensor(im.image, sigma_g=cfg["sigma_g"], sigma_t=cfg["sigma_t"])
                t0 = time.perf_counter()
                out = np.clip(m_ours(im.image, cfg), 0, 1).astype(np.float32)
                dt = time.perf_counter() - t0
                met = compute_metrics(im.image, out, st_src=st)
                rows.append({"parameter": param, "value": val, "image": im.name,
                             "runtime_s": dt, **met})
            print(f"{param:12s} {val:>6} done", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(RES / "sensitivity_ca_raw.csv", index=False, float_format="%.6f")
    agg = raw.groupby(["parameter", "value"]).agg(
        runtime_s=("runtime_s", "median"),
        **{f"{m}_median": (m, "median") for m in METRICS},
        **{f"{m}_q1": (m, lambda x: x.quantile(0.25)) for m in METRICS},
        **{f"{m}_q3": (m, lambda x: x.quantile(0.75)) for m in METRICS},
    ).reset_index()
    agg.to_csv(RES / "sensitivity_ca.csv", index=False, float_format="%.6f")

    # Compact table: range per parameter plus the chosen setting.
    chosen = {"beta": 8.0, "gamma": 2.0, "sigma_along": 2.5, "sigma_t": 4.0,
              "r_max": 6, "r_min": 3, "bins": 12, "sectors": 8}
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{One-at-a-time sensitivity of CA-AKF on the four tuning images. Each row varies one parameter while the others stay at the chosen value. Ranges are minima and maxima of the median over the four images; the chosen setting is marked in the last column.}",
        "\\label{tab:ca-sensitivity}", "\\small",
        "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
        "Parameter & Values & FSR range$\\downarrow$ & SSIM range$\\uparrow$ & Chosen \\\\",
        "\\midrule",
    ]
    for param, values in SWEEPS:
        g = agg[agg.parameter == param]
        lines.append(
            f"{PRETTY[param]} & {len(values)} & {g.fsr_median.min():.3f}--{g.fsr_median.max():.3f} & "
            f"{g.ssim_median.min():.3f}--{g.ssim_median.max():.3f} & {chosen[param]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TEX / "table_sensitivity_ca.tex").write_text("\n".join(lines) + "\n")

    print("\nRanges:")
    for param, values in SWEEPS:
        g = agg[agg.parameter == param]
        best = g.loc[g.fsr_median.idxmin()]
        print(f"{param:12s} FSR {g.fsr_median.min():.3f}-{g.fsr_median.max():.3f} "
              f"(best at {best.value}), SSIM {g.ssim_median.min():.3f}-{g.ssim_median.max():.3f}")
    print("\nwrote sensitivity_ca.csv, sensitivity_ca_raw.csv, table_sensitivity_ca.tex")


if __name__ == "__main__":
    main()
