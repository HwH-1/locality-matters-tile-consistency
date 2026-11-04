"""Deterministic structure benchmark: six junctions and six thin-line images.

The benchmark isolates the two structures that the repair layer targets.
Junction score: ratio between the largest colour distance observed in a 5x5
core of the output and that of the source (1 = fully retained, 0 = fully
mixed).  Line score: contrast retained at the line centres relative to a local
median background (1 = fully retained).

The images are generated here from fixed seeds and contain no natural texture,
so the scores are reproducible and do not depend on a pretrained detector.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TEX = ROOT.parent / "02_manuscript" / "tables"
sys.path.insert(0, str(ROOT))

from src.methods import METHODS, run_method  # noqa: E402
from src.structure import luminance  # noqa: E402

cv2.setNumThreads(1)
SIZE = 256
METHOD_ORDER = [
    "classic",
    "sector",
    "akf",
    "bilateral",
    "rgf",
    "guided",
    "ca_akf",
    "jlr_cakf",
    "jlr_junction",
    "jlr_line",
    "jlr_global",
]
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
PALETTE = np.array(
    [
        [0.85, 0.15, 0.15],
        [0.15, 0.65, 0.25],
        [0.15, 0.25, 0.80],
        [0.85, 0.75, 0.15],
    ],
    dtype=np.float32,
)


def _core_spread(img: np.ndarray, centre: tuple[int, int], radius: int = 2) -> float:
    y, x = centre
    core = img[y - radius : y + radius + 1, x - radius : x + radius + 1].reshape(-1, 3)
    d = np.linalg.norm(core[:, None, :] - core[None, :, :], axis=2)
    return float(d.max())


def junction_phantom(kind: str) -> tuple[np.ndarray, tuple[int, int], float]:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    cy = cx = SIZE // 2
    img = np.zeros((SIZE, SIZE, 3), np.float32)
    cols = PALETTE.copy()
    if kind == "low":
        cols = 0.5 + (cols - 0.5) * 0.35
    if kind in ("axis", "low"):
        for c, (y0, y1, x0, x1) in zip(
            cols,
            [(0, cy, 0, cx), (0, cy, cx, SIZE), (cy, SIZE, 0, cx), (cy, SIZE, cx, SIZE)],
        ):
            img[y0:y1, x0:x1] = c
    elif kind == "diag":
        a = (yy - cy) + (xx - cx)
        b = (yy - cy) - (xx - cx)
        masks = [(a <= 0) & (b <= 0), (a <= 0) & (b > 0), (a > 0) & (b <= 0), (a > 0) & (b > 0)]
        for m, c in zip(masks, cols):
            img[m] = c
    elif kind == "T":
        img[:cy, :] = cols[0]
        img[cy:, :cx] = cols[1]
        img[cy:, cx:] = cols[2]
    elif kind == "Y":
        ang = np.arctan2(yy - cy, xx - cx)
        k = ((ang + np.pi) / (2 * np.pi / 3)).astype(int) % 3
        for i, c in enumerate(cols[:3]):
            img[k == i] = c
    elif kind == "cross":
        for c, (y0, y1, x0, x1) in zip(
            cols,
            [(0, cy, 0, cx), (0, cy, cx, SIZE), (cy, SIZE, 0, cx), (cy, SIZE, cx, SIZE)],
        ):
            img[y0:y1, x0:x1] = c
        img[cy - 1 : cy + 1, :] = [0.95, 0.95, 0.95]
        img[:, cx - 1 : cx + 1] = [0.95, 0.95, 0.95]
    else:
        raise ValueError(kind)
    centre = (cy, cx)
    return img, centre, _core_spread(img, centre)


def line_phantom(kind: str) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    img = np.full((SIZE, SIZE, 3), 0.10, np.float32)
    mask = np.zeros((SIZE, SIZE), bool)
    if kind == "v1":
        cols = np.arange(16, SIZE, 16)
        img[:, cols] = 0.78
        mask[20:-20, cols] = True
    elif kind == "d1":
        for k in range(-SIZE, SIZE, 16):
            band = np.abs((xx - yy) - k) <= 0
            img[band] = 0.78
            mask[band] = True
        mask[:20] = mask[-20:] = False
        mask[:, :20] = mask[:, -20:] = False
    elif kind == "curve":
        r = np.hypot(yy - 128, xx - 128)
        for rad in (18, 36, 54, 72, 90, 108):
            band = np.abs(r - rad) < 0.75
            img[band] = 0.78
            mask[band] = True
    elif kind == "dark":
        img[:] = 0.90
        cols = np.arange(16, SIZE, 16)
        img[:, cols] = 0.15
        mask[20:-20, cols] = True
    elif kind == "w2":
        cols = np.arange(16, SIZE, 16)
        img[:, cols] = 0.78
        img[:, cols + 1] = 0.78
        mask[20:-20, cols] = True
        mask[20:-20, cols + 1] = True
    elif kind == "low":
        img[:] = 0.46
        cols = np.arange(16, SIZE, 16)
        img[:, cols] = 0.54
        mask[20:-20, cols] = True
    else:
        raise ValueError(kind)
    return img, mask


def junction_score(out: np.ndarray, centre: tuple[int, int], source_spread: float) -> float:
    return _core_spread(out, centre) / (source_spread + 1e-12)


def line_score(out: np.ndarray, mask: np.ndarray, src: np.ndarray) -> float:
    lo = luminance(out)
    ls = luminance(src)
    bg = median_filter(ls, size=9)
    numerator = float((lo[mask] - bg[mask]).mean())
    denominator = float((ls[mask] - bg[mask]).mean())
    return numerator / (denominator + 1e-12)


def main() -> None:
    TEX.mkdir(parents=True, exist_ok=True)
    junctions = {
        "X (axis)": junction_phantom("axis"),
        "X (diagonal)": junction_phantom("diag"),
        "T": junction_phantom("T"),
        "Y": junction_phantom("Y"),
        "cross + lines": junction_phantom("cross"),
        "X (low contrast)": junction_phantom("low"),
    }
    lines = {
        "vertical 1 px": line_phantom("v1"),
        "diagonal 1 px": line_phantom("d1"),
        "curved 1 px": line_phantom("curve"),
        "dark 1 px": line_phantom("dark"),
        "vertical 2 px": line_phantom("w2"),
        "low contrast": line_phantom("low"),
    }
    specs = {m.name: m for m in METHODS}
    rows = []
    for method in METHOD_ORDER:
        spec = specs[method]
        for name, (img, centre, spread) in junctions.items():
            out = run_method(spec, img)
            rows.append(
                {
                    "method": method,
                    "method_label": LABELS[method],
                    "kind": "junction",
                    "phantom": name,
                    "score": junction_score(out, centre, spread),
                }
            )
        for name, (img, mask) in lines.items():
            out = run_method(spec, img)
            rows.append(
                {
                    "method": method,
                    "method_label": LABELS[method],
                    "kind": "line",
                    "phantom": name,
                    "score": line_score(out, mask, img),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(RES / "structure_benchmark.csv", index=False, float_format="%.6f")
    summary = (
        df.groupby(["method", "method_label", "kind"])["score"]
        .agg(["median", "min", "max"])
        .reset_index()
    )
    summary.to_csv(RES / "structure_benchmark_summary.csv", index=False, float_format="%.6f")
    print(df.pivot_table(index="method", columns="kind", values="score", aggfunc="median").round(3).to_string())

    order = [m for m in METHOD_ORDER if m in set(df.method)]
    jmed = df[df.kind == "junction"].groupby("method")["score"].median()
    lmed = df[df.kind == "line"].groupby("method")["score"].median()
    lines_tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Structure benchmark on twelve deterministic phantoms (six junctions, six thin-line images). Junction score is the colour spread retained in a $5\times5$ core relative to the source; line score is the contrast retained at line centres relative to a local background. Both are in $[0,1]$ with 1 denoting perfect retention; higher is better.}",
        r"\label{tab:structure}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Method & Junction score$\uparrow$ & Line score$\uparrow$ \\",
        r"\midrule",
    ]
    for method in order:
        label = LABELS[method]
        if method == "jlr_cakf":
            label = r"\textbf{" + label + "}"
        lines_tex.append(f"{label} & {jmed[method]:.3f} & {lmed[method]:.3f} \\\\")
    lines_tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TEX / "table_structure.tex").write_text("\n".join(lines_tex) + "\n")
    print("wrote structure_benchmark.csv and table_structure.tex")


if __name__ == "__main__":
    main()
