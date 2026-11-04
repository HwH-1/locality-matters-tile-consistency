"""Resolution check at 512 x 512 on five natural images."""
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

from src.data import NATURAL, _resize_crop, _to_rgb_float  # noqa: E402
from src.methods import METHODS, run_method  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.structure import structure_tensor  # noqa: E402

cv2.setNumThreads(1)
SIZE = 512
NAMES = ["coffee", "brick", "hubble", "astronaut", "chelsea"]
METHODS_SUB = ["classic", "sector", "akf", "bilateral", "rgf", "guided", "ca_akf", "jlr_cakf"]
LABELS = {
    "classic": "Kuwahara (1976)",
    "sector": "Sector Kuwahara",
    "akf": "AKF (fixed scale)",
    "bilateral": "Bilateral",
    "rgf": "RGF",
    "guided": "Guided filter",
    "ca_akf": "CA-AKF (base)",
    "jlr_cakf": "JLR-CAKF (ours)",
}


def main() -> None:
    source = {n: fn for n, fn, _ in NATURAL}
    specs = {m.name: m for m in METHODS}
    rows = []
    for name in NAMES:
        img = _resize_crop(_to_rgb_float(source[name]()), size=SIZE)
        st = structure_tensor(img)
        for m in METHODS_SUB:
            t0 = time.perf_counter()
            out = run_method(specs[m], img)
            dt = time.perf_counter() - t0
            met = compute_metrics(img, out, st_src=st)
            rows.append({"image": name, "method": m, "runtime_s": dt, **met})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "resolution_512.csv", index=False, float_format="%.6f")
    agg = df.groupby("method")[["sr", "ssim", "fsr", "scg", "esr", "ece", "ovt", "runtime_s"]].median()
    print(agg.round(3).to_string())

    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Resolution check: median results over five natural images resized to $512\\times512$, with the same pixel-based parameters as the $256\\times256$ experiment.}",
        "\\label{tab:resolution}", "\\footnotesize",
        "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
        "Method & SR$\\uparrow$ & SSIM$\\uparrow$ & FSR$\\downarrow$ & SCG$\\uparrow$ & ESR$\\to1$ & ECE$\\downarrow$ & Time (s)$\\downarrow$ \\\\",
        "\\midrule",
    ]
    for m in METHODS_SUB:
        r = agg.loc[m]
        lines.append(f"{LABELS[m]} & {r.sr:.3f} & {r.ssim:.3f} & {r.fsr:.3f} & {r.scg:.3f} & {r.esr:.3f} & {r.ece:.2f} & {r.runtime_s:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TEX / "table_resolution.tex").write_text("\n".join(lines) + "\n")
    print("wrote resolution_512.csv and table_resolution.tex")


if __name__ == "__main__":
    main()
