"""Quantitative probes for the four qualitative failure modes.

Each probe is a deterministic synthetic image isolating one behaviour:
  junction_chroma     -- chroma retention at a four-colour junction (higher is better)
  line_retention      -- retention of one-pixel lines (higher is better)
  boundary_mixing     -- colour mixing within +/-2 px of a hard boundary (lower is better)
  spurious_coherence  -- coherence imposed on unstructured noise (lower is better)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TEX = ROOT.parent / "02_manuscript" / "tables"
sys.path.insert(0, str(ROOT))

from src.methods import METHODS, run_method  # noqa: E402
from src.structure import luminance, structure_tensor  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402

cv2.setNumThreads(1)
SIZE = 256
METHOD_ORDER = ["classic", "sector", "akf", "bilateral", "rgf", "guided", "ca_akf", "jlr_cakf", "jlr_junction", "jlr_line", "jlr_global"]
LABELS = {
    "classic": "Kuwahara (1976)",
    "sector": "Sector Kuwahara",
    "akf": "AKF (fixed scale)",
    "bilateral": "Bilateral",
    "rgf": "RGF",
    "guided": "Guided filter",
    "ca_akf": "CA-AKF (base)",
    "jlr_cakf": "JLR-CAKF (ours)",
    "jlr_junction": "JLR, junctions only",
    "jlr_line": "JLR, lines only",
    "jlr_global": "JLR, global blend",
}


def junction_phantom() -> tuple[np.ndarray, np.ndarray, float]:
    img = np.zeros((SIZE, SIZE, 3), np.float32)
    quad = [
        (slice(0, SIZE // 2), slice(0, SIZE // 2), (0.85, 0.15, 0.15)),
        (slice(0, SIZE // 2), slice(SIZE // 2, None), (0.15, 0.65, 0.25)),
        (slice(SIZE // 2, None), slice(0, SIZE // 2), (0.15, 0.25, 0.80)),
        (slice(SIZE // 2, None), slice(SIZE // 2, None), (0.85, 0.75, 0.15)),
    ]
    for ys, xs, col in quad:
        img[ys, xs] = np.array(col, np.float32)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    centre = (np.abs(yy - 128) <= 2) & (np.abs(xx - 128) <= 2)
    palette = np.array([q[2] for q in quad], np.float32)
    palette_chroma = float(np.ptp(palette, axis=1).mean())
    return img, centre, palette_chroma


def line_phantom() -> tuple[np.ndarray, np.ndarray]:
    img = np.full((SIZE, SIZE, 3), 0.10, np.float32)
    cols = np.arange(16, SIZE, 16)
    img[:, cols] = 0.90
    mask = np.zeros((SIZE, SIZE), bool)
    mask[20:-20, cols] = True
    return img, mask


def boundary_phantom() -> tuple[np.ndarray, np.ndarray]:
    img = np.zeros((SIZE, SIZE, 3), np.float32)
    img[:, : SIZE // 2] = np.array([0.90, 0.10, 0.10], np.float32)
    img[:, SIZE // 2 :] = np.array([0.10, 0.10, 0.90], np.float32)
    band = np.zeros((SIZE, SIZE), bool)
    band[16:-16, SIZE // 2 - 2 : SIZE // 2 + 2] = True
    return img, band


def noise_phantom() -> np.ndarray:
    rng = np.random.default_rng(20261025)
    base = np.array([0.45, 0.50, 0.55], np.float32)
    return np.clip(base + rng.normal(0, 0.18, (SIZE, SIZE, 3)).astype(np.float32), 0, 1)


def test_models() -> dict:
    jimg, centre, palette_chroma = junction_phantom()
    limg, lmask = line_phantom()
    bimg, band = boundary_phantom()
    nimg = noise_phantom()
    return {
        "junction": (jimg, centre, palette_chroma),
        "line": (limg, lmask),
        "boundary": (bimg, band),
        "noise": nimg,
    }


def metrics_for(method: str, data: dict) -> dict:
    spec = {m.name: m for m in METHODS}[method]

    jimg, centre, palette_chroma = data["junction"]
    jout = run_method(spec, jimg)
    core = jout[centre].reshape(-1, 3)
    src_core = jimg[centre].reshape(-1, 3)
    src_spread = float(
        np.max(np.linalg.norm(src_core[:, None, :] - src_core[None, :, :], axis=2))
    )
    jc = float(
        np.max(np.linalg.norm(core[:, None, :] - core[None, :, :], axis=2))
        / (src_spread + 1e-12)
    )

    limg, lmask = data["line"]
    lout = run_method(spec, limg)
    ls = luminance(limg)
    lo = luminance(lout)
    bg = median_filter(ls, size=9)
    lr = float(
        (lo[lmask] - bg[lmask]).mean() / ((ls[lmask] - bg[lmask]).mean() + 1e-12)
    )

    bimg, band = data["boundary"]
    bout = run_method(spec, bimg)
    red = np.array([0.90, 0.10, 0.10], np.float32)
    blue = np.array([0.10, 0.10, 0.90], np.float32)
    dr = np.linalg.norm(bout - red, axis=-1)
    db = np.linalg.norm(bout - blue, axis=-1)
    bcm = float(np.minimum(dr, db)[band].mean() * 100.0)

    nimg = data["noise"]
    nout = run_method(spec, nimg)
    st_s = structure_tensor(nimg)
    st_o = structure_tensor(nout)
    m = st_s["coherence"] < 0.10
    spc = float((st_o["coherence"][m].mean() - st_s["coherence"][m].mean()) * 100.0)

    return {"junction_chroma": jc, "line_retention": lr,
            "boundary_mixing": bcm, "spurious_coherence": spc}


def main() -> None:
    data = test_models()
    rows = []
    for method in METHOD_ORDER:
        met = metrics_for(method, data)
        rows.append({"method": method, **met})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "failure_probe.csv", index=False, float_format="%.6f")
    print(df.round(4).to_string(index=False))

    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{Failure-mode probes on deterministic synthetic images. Junction chroma: chroma retained at a four-colour junction, relative to the source palette (higher is better). Line retention: luminance retained at one-pixel line centres (higher is better). Boundary mixing: mean distance to the nearer source colour within $\\pm2$ px of a hard boundary, $\\times100$ (lower is better). Spurious coherence: mean coherence imposed on unstructured noise, $\\times100$ (lower is better).}",
        "\\label{tab:failure}", "\\footnotesize",
        "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
        "Method & Junction chroma$\\uparrow$ & Line$\\uparrow$ & Mixing$\\downarrow$ & Spurious coh.$\\downarrow$ \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(f"{LABELS[r.method]} & {r.junction_chroma:.3f} & {r.line_retention:.3f} & "
                     f"{r.boundary_mixing:.2f} & {r.spurious_coherence:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TEX / "table_failure.tex").write_text("\n".join(lines) + "\n")
    print("wrote failure_probe.csv and table_failure.tex")


if __name__ == "__main__":
    main()
