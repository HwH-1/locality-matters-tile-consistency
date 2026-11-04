"""Evaluation metrics, aggregation and paired statistical tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
from skimage.metrics import structural_similarity

from .structure import luminance, structure_tensor

METRICS = ["sr", "ssim", "fsr", "scg", "esr", "ece", "ovt"]
HIGHER_IS_BETTER = {
    "sr": True,
    "ssim": True,
    "fsr": False,
    "scg": True,
    "esr": None,  # closest to 1 is best
    "ece": False,
    "ovt": False,
}
METRIC_LABELS = {
    "sr": "Structure retention (SR)",
    "ssim": "SSIM",
    "fsr": "Flat-region std. ratio (FSR)",
    "scg": "Coherence gain (SCG)",
    "esr": "Edge sharpness ratio (ESR)",
    "ece": "Edge colour error (ECE)",
    "ovt": "Overshoot (OVT)",
}


def _gradmag(L: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(L)
    return np.hypot(gx, gy).astype(np.float32)


def _local_std(L: np.ndarray, size: int = 5) -> np.ndarray:
    m = uniform_filter(L, size=size, mode="reflect")
    m2 = uniform_filter(L * L, size=size, mode="reflect")
    return np.sqrt(np.maximum(m2 - m * m, 0.0)).astype(np.float32)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_metrics(
    src: np.ndarray,
    out: np.ndarray,
    st_src: dict | None = None,
    min_flat_pixels: int = 100,
) -> dict:
    if st_src is None:
        st_src = structure_tensor(src)
    Ls = luminance(src)
    Lo = luminance(out)
    gs = _gradmag(Ls)
    go = _gradmag(Lo)

    sr = _pearson(gs, go)
    ssim = float(
        structural_similarity(
            src, out, channel_axis=2, data_range=1.0, gaussian_weights=False
        )
    )

    # Flat regions: low coherence and low edge energy in the source.  FSR is
    # left undefined when the source contains fewer than 100 flat pixels, too
    # few for a stable ratio (for example, a fully periodic stripe image).
    l1 = st_src["l1"]
    flat = (st_src["coherence"] < 0.10) & (l1 <= np.percentile(l1, 25.0))
    if flat.sum() == 0:
        flat = l1 <= np.percentile(l1, 10.0)
    if flat.sum() < min_flat_pixels:
        fsr = float("nan")
    else:
        std_s = _local_std(Ls)[flat].mean()
        std_o = _local_std(Lo)[flat].mean()
        fsr = float(std_o / (std_s + 1e-8))

    # Coherence gain on source-coherent pixels.
    st_out = structure_tensor(out)
    coh_mask = st_src["coherence"] > 0.5
    scg = float(
        (st_out["coherence"][coh_mask].mean() - st_src["coherence"][coh_mask].mean())
        if coh_mask.any()
        else 0.0
    )

    # Edge sharpness and edge colour error at the strongest source gradients.
    edge = gs >= np.percentile(gs, 95.0)
    esr = float(go[edge].mean() / (gs[edge].mean() + 1e-8))
    diff = np.sqrt(((out - src) ** 2).sum(axis=-1))
    ece = float(diff[edge].mean() * 100.0)

    # Local overshoot of luminance outside the 5x5 source envelope.
    hi = maximum_filter(Ls, size=5, mode="reflect")
    lo = minimum_filter(Ls, size=5, mode="reflect")
    over = np.maximum(Lo - hi, 0.0) + np.maximum(lo - Lo, 0.0)
    ovt = float(over.mean() * 100.0)

    return {
        "sr": sr,
        "ssim": ssim,
        "fsr": fsr,
        "scg": scg,
        "esr": esr,
        "ece": ece,
        "ovt": ovt,
    }


def aggregate(df: pd.DataFrame, group: str = "method") -> pd.DataFrame:
    rows = []
    for (method,), g in df.groupby([group]):
        row = {group: method, "n": len(g), "runtime_s": float(g["runtime_s"].median())}
        for m in METRICS:
            row[f"{m}_median"] = float(g[m].median())
            row[f"{m}_q1"] = float(g[m].quantile(0.25))
            row[f"{m}_q3"] = float(g[m].quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def _rank_biserial(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    pos = r[d > 0].sum()
    neg = r[d < 0].sum()
    return float((pos - neg) / (pos + neg + 1e-12))


def paired_tests(
    df: pd.DataFrame, proposed: str, baselines: list[str], metrics: list[str]
) -> pd.DataFrame:
    rows = []
    for m in metrics:
        pvals = []
        for b in baselines:
            x = df[df.method == proposed].sort_values("image")[m].to_numpy(float)
            y = df[df.method == b].sort_values("image")[m].to_numpy(float)
            valid = np.isfinite(x) & np.isfinite(y)
            xv, yv = x[valid], y[valid]
            d = xv - yv
            if len(d) == 0:
                p = float("nan")
            elif np.allclose(d, 0):
                p = 1.0
            else:
                try:
                    p = float(
                        stats.wilcoxon(
                            xv, yv, zero_method="wilcox", alternative="two-sided", method="auto"
                        ).pvalue
                    )
                except ValueError:
                    p = 1.0
            pvals.append(p)
            rows.append(
                {
                    "metric": m,
                    "baseline": b,
                    "median_proposed": float(np.nanmedian(x)),
                    "median_baseline": float(np.nanmedian(y)),
                    "median_diff": float(np.median(d)),
                    "pct_change": float(
                        100.0
                        * (np.nanmedian(x) - np.nanmedian(y))
                        / (abs(np.nanmedian(y)) + 1e-12)
                    ),
                    "p_raw": p,
                    "rank_biserial": _rank_biserial(d),
                    "n": int(len(d)),
                }
            )
        # Holm correction across baselines within this metric.
        p = np.array(pvals)
        order = np.argsort(p)
        m_holm = len(p)
        adj = np.empty_like(p)
        running = 0.0
        for rank, idx in enumerate(order):
            val = min(1.0, (m_holm - rank) * p[idx])
            running = max(running, val)
            adj[idx] = running
        for row, pa in zip(rows[-len(baselines):], adj):
            row["p_holm"] = float(pa)
    return pd.DataFrame(rows)
