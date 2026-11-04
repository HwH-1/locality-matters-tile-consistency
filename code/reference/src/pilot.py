"""Parameter pilot on four images (not a final evaluation)."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_all  # noqa: E402
from src.methods import m_ours  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.structure import structure_tensor  # noqa: E402

BASE = {
    "r_min": 3,
    "r_max": 6,
    "sectors": 8,
    "bins": 12,
    "combine": "soft",
    "beta": 8.0,
    "sigma_along": 2.5,
    "adaptive": True,
}

configs = []
for gamma in (1.0, 2.0):
    for beta in (8.0, 12.0, 16.0):
        for rmin, rmax in ((3, 6), (3, 7), (4, 7)):
            configs.append({"gamma": gamma, "beta": beta, "r_min": rmin, "r_max": rmax, "combine": "soft"})
configs.append({"gamma": 1.0, "beta": 8.0, "r_min": 3, "r_max": 6, "combine": "hard"})
configs.append({"gamma": 2.0, "beta": 8.0, "r_min": 3, "r_max": 6, "combine": "hard"})

names = {"camera", "astronaut", "synth_checker", "synth_regions"}
images = [im for im in load_all() if im.name in names]
print("pilot images:", [im.name for im in images], flush=True)

rows = []
for cfg in configs:
    for im in images:
        st = structure_tensor(im.image)
        # gamma is applied to the scale blend only; monkey-patch by computing here.
        cfg_full = {**BASE, **{k: v for k, v in cfg.items() if k != "gamma"}}
        st2, smooth = None, None
        from src import filters

        smooth = filters.anisotropic_presmooth(
            im.image, st["theta"], st["coherence"], sigma_along=cfg_full["sigma_along"],
            n_bins=cfg_full["bins"]
        )
        outs = []
        for r in (cfg_full["r_min"], cfg_full["r_max"]):
            mu, var = filters.sector_statistics(
                smooth, r, st["theta"], n_sectors=cfg_full["sectors"], n_bins=cfg_full["bins"]
            )
            outs.append(filters.combine_sectors(mu, var, mode=cfg_full["combine"], beta=cfg_full["beta"]))
        alpha = (1.0 - st["edge_energy"]) ** cfg["gamma"]
        out = outs[0] * (1.0 - alpha[..., None]) + outs[1] * alpha[..., None]
        out = np.clip(out, 0, 1)
        met = compute_metrics(im.image, out, st_src=st)
        rows.append({"gamma": cfg["gamma"], "beta": cfg["beta"], "r_min": cfg["r_min"],
                     "r_max": cfg["r_max"], "combine": cfg["combine"], "image": im.name, **met})
    print(cfg, "done", flush=True)

df = pd.DataFrame(rows)
agg = df.groupby(["gamma", "beta", "r_min", "r_max", "combine"]).agg(
    sr=("sr", "median"), ssim=("ssim", "median"), fsr=("fsr", "median"),
    scg=("scg", "median"), esr=("esr", "median"), ece=("ece", "median"), ovt=("ovt", "median")
).reset_index()
pd.set_option("display.width", 200)
print(agg.sort_values(["fsr", "sr"], ascending=[True, False]).to_string(index=False))
agg.to_csv(ROOT / "results" / "pilot_grid.csv", index=False)
df.to_csv(ROOT / "results" / "pilot_raw.csv", index=False)
