"""Coverage statistics of the junction, line and combined repair masks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))

from src.data import load_all  # noqa: E402
from src.methods import METHODS  # noqa: E402
from src.repair import repair_weights  # noqa: E402


def main() -> None:
    cfg = next(m for m in METHODS if m.name == "jlr_cakf").params
    rows = []
    for im in load_all():
        mj, ml, w = repair_weights(
            im.image,
            line_percentile=cfg["line_percentile"],
            junction_dilation=cfg["junction_dilation"],
            line_dilation=cfg["line_dilation"],
            mask_sigma=cfg["mask_sigma"],
            w_max=cfg["w_max"],
            snr_min=cfg["snr_min"],
            junction_min_distance=cfg["junction_min_distance"],
            junction_rel_threshold=cfg["junction_rel_threshold"],
        )
        total = float(mj.size)
        rows.append(
            {
                "image": im.name,
                "junction_pct": 100.0 * float((mj > 0.01).sum()) / total,
                "line_pct": 100.0 * float((ml > 0.01).sum()) / total,
                "combined_pct": 100.0 * float((w > 0.01).sum()) / total,
                "strong_pct": 100.0 * float((w > 0.5).sum()) / total,
                "mean_weight_pct": 100.0 * float(w.mean()),
                "max_weight": float(w.max()),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RES / "mask_stats.csv", index=False, float_format="%.4f")
    print(df.round(2).to_string(index=False))
    print("\nmedian:", df.drop(columns="image").median().round(3).to_dict())


if __name__ == "__main__":
    main()
