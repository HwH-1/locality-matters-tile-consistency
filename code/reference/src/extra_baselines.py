"""Additional strong baselines: AKF at r=7 and a three-scale hard AKF.

These are supplementary comparisons, not part of the seven-method main table.
They answer two obvious objections: (i) why not simply use a larger fixed
window, and (ii) why not use more scales with the same hard sector rule.
"""
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
from src.methods import m_akf, m_akf_multiscale, run_method, MethodSpec  # noqa: E402
from src.metrics import METRICS, compute_metrics, paired_tests  # noqa: E402
from src.structure import structure_tensor  # noqa: E402

cv2.setNumThreads(1)

AKF7 = MethodSpec(
    "akf7", "AKF r=7", "baseline",
    {"radius": 7, "sectors": 8, "bins": 12, "combine": "hard", "beta": 8.0,
     "sigma_along": 2.5, "sigma_t": 4.0, "sigma_g": 1.0},
    m_akf, "Fixed-scale AKF at r=7, hard minimum",
)
MS = MethodSpec(
    "ms_akf", "Multi-scale AKF", "baseline",
    {"radii": (3, 5, 7), "sectors": 8, "bins": 12, "combine": "hard", "beta": 8.0,
     "sigma_along": 2.5, "sigma_t": 4.0, "sigma_g": 1.0},
    m_akf_multiscale, "Three-scale AKF (r=3,5,7), hard sectors, edge-energy blend",
)
LABELS = {
    "akf": "AKF, r = 5",
    "akf7": "AKF, r = 7",
    "ms_akf": "Multi-scale AKF, r = 3, 5, 7",
    "ca_akf": "CA-AKF (ours)",
}


def main() -> None:
    images = load_all()
    rows = []
    for im in images:
        st = structure_tensor(im.image)
        for spec in (AKF7, MS):
            t0 = time.perf_counter()
            out = run_method(spec, im.image)
            dt = time.perf_counter() - t0
            met = compute_metrics(im.image, out, st_src=st)
            rows.append({"image": im.name, "method": spec.name,
                         "method_label": spec.label, "runtime_s": dt, **met})
            print(f"{im.name:16s} {spec.name:8s} {dt:5.2f}s  "
                  f"FSR={met['fsr']:.3f} SSIM={met['ssim']:.3f}", flush=True)
    extra = pd.DataFrame(rows)
    extra.to_csv(RES / "extra_baselines.csv", index=False, float_format="%.6f")

    main = pd.read_csv(RES / "metrics_full.csv")
    main = main[main.method.isin(["akf", "ca_akf"])]
    combined = pd.concat(
        [main[["image", "method", "runtime_s", *METRICS]], extra[["image", "method", "runtime_s", *METRICS]]],
        ignore_index=True,
    )
    tests = paired_tests(combined, "ca_akf", ["akf7", "ms_akf"], METRICS)
    tests.to_csv(RES / "extra_tests.csv", index=False, float_format="%.6f")

    agg = combined.groupby("method")[["runtime_s", *METRICS]].median()
    agg.to_csv(RES / "extra_aggregate.csv", float_format="%.6f")
    print("\nMedians:")
    print(agg.loc[["akf", "akf7", "ms_akf", "ca_akf"]].round(3).to_string())
    print("\nTests vs CA-AKF:")
    print(tests[["metric", "baseline", "median_diff", "p_raw", "p_holm", "rank_biserial"]].round(4).to_string(index=False))

    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Additional strong baselines on the twelve images. AKF at $r=7$ tests whether a larger fixed window suffices; the three-scale AKF uses $r=3,5,7$ with hard sector selection and an edge-energy blend. Medians are reported; lower is better for FSR, ECE and OVT, higher for SR, SSIM and SCG, and ESR is ideal at one.}",
        "\\label{tab:extra}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.6pt}",
        "\\begin{tabular}{@{}lcccccccc@{}}", "\\toprule",
        "Method & SR$\\uparrow$ & SSIM$\\uparrow$ & FSR$\\downarrow$ & SCG$\\uparrow$ & ESR$\\to1$ & ECE$\\downarrow$ & OVT$\\downarrow$ & Time (s)$\\downarrow$ \\\\",
        "\\midrule",
    ]
    for m in ["akf", "akf7", "ms_akf", "ca_akf"]:
        a = agg.loc[m]
        cells = [f"{a[k]:.3f}" for k in METRICS]
        lines.append(f"{LABELS[m]} & " + " & ".join(cells) + f" & {a.runtime_s:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TEX / "table_extra.tex").write_text("\n".join(lines) + "\n")
    print("\nwrote extra_baselines.csv, extra_tests.csv, extra_aggregate.csv, table_extra.tex")


if __name__ == "__main__":
    main()
