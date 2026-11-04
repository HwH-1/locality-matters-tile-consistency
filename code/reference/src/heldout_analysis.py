"""Held-out check and bootstrap confidence intervals."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TEX = ROOT.parent / "02_manuscript" / "tables"
sys.path.insert(0, str(ROOT))
from src.metrics import METRICS  # noqa: E402

DEV = {"astronaut", "camera", "synth_checker", "synth_regions"}


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    order = np.argsort(p[valid])
    m = int(valid.sum())
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[valid][idx])
        running = max(running, val)
        out[np.where(valid)[0][idx]] = running
    return out


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 20000, seed: int = 20261027):
    d = x - y
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.array([np.median(rng.choice(d, size=len(d), replace=True)) for _ in range(n_boot)])
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main() -> None:
    df = pd.read_csv(RES / "metrics_all.csv")
    rows = []
    for metric in METRICS:
        for split, images in [
            ("all", set(df.image.unique())),
            ("heldout", set(df.image.unique()) - DEV),
        ]:
            sub = df[df.image.isin(images)]
            x = sub[sub.method == "jlr_cakf"].set_index("image")[metric]
            y = sub[sub.method == "ca_akf"].set_index("image")[metric]
            x, y = x.align(y, join="inner")
            valid = x.notna() & y.notna()
            x, y = x[valid], y[valid]
            d = (x - y).to_numpy(float)
            if len(d) == 0:
                p = np.nan
            elif np.allclose(d, 0):
                p = 1.0
            else:
                p = float(stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)
            lo, hi = bootstrap_ci(x.to_numpy(float), y.to_numpy(float))
            rows.append({"split": split, "metric": metric, "n": len(d),
                         "median_diff": float(np.median(d)),
                         "ci_low": lo, "ci_high": hi, "p_raw": p,
                         "p_holm": np.nan})
    out = pd.DataFrame(rows)
    for split in ["all", "heldout"]:
        mask = out.split == split
        out.loc[mask, "p_holm"] = holm(out.loc[mask, "p_raw"].to_numpy(float))
    out.to_csv(RES / "heldout_bootstrap.csv", index=False, float_format="%.6f")

    sub = out[out.split == "heldout"].set_index("metric")
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Secondary-subset check on the eight images not used in the parameter pilot. This is not a pristine hold-out: the sensitivity analysis in Section~\ref{sec:sensitivity} reports medians over all twelve images, and the four pilot images remain in the main tables. Entries are median paired differences (JLR-CAKF minus CA-AKF), percentile bootstrap 95\% confidence intervals (20,000 resamples) and Holm-corrected $p$ values across the seven metrics. FSR uses six images of the subset.}",
        r"\label{tab:heldout}", r"\footnotesize",
        r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
        r"Metric & $n$ & $\Delta$ [95\% CI] & $p_{\mathrm{Holm}}$ & \\", r"\midrule",
    ]
    labels = {"sr":"SR","ssim":"SSIM","fsr":"FSR","scg":"SCG","esr":"ESR","ece":"ECE","ovt":"OVT"}
    for m in METRICS:
        r = sub.loc[m]
        p = r.p_holm
        p_txt = "--" if not np.isfinite(p) else (f"{p:.4f}" if p >= 1e-4 else "$<10^{-4}$")
        lines.append(f"{labels[m]} & {int(r.n)} & {r.median_diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] & {p_txt} & \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    # Remove the empty fifth column from the header/rows cleanly.
    text = "\n".join(lines).replace("& \\\\", "\\\\").replace(" & p_{\\mathrm{Holm}} & \\\\", " & $p_{\\mathrm{Holm}}$ \\\\")
    (TEX / "table_heldout.tex").write_text(text + "\n")
    print(out[out.split == "heldout"][["metric","n","median_diff","ci_low","ci_high","p_raw","p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
