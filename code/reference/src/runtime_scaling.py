"""Runtime scaling with image size (single thread)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))

from src.data import synth_regions  # noqa: E402
from src.methods import METHODS  # noqa: E402

SIZES = [128, 192, 256, 320, 384]
REPEATS = 3
TARGETS = ["classic", "akf", "ca_akf", "jlr_cakf"]

rows = []
base = synth_regions(size=max(SIZES))
for size in SIZES:
    img = np.asarray(
        Image.fromarray((base * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS),
        dtype=np.float32,
    ) / 255.0
    for name in TARGETS:
        spec = next(m for m in METHODS if m.name == name)
        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            spec.fn(img, spec.params)
            times.append(time.perf_counter() - t0)
        med = float(np.median(times))
        rows.append({"size": size, "megapixels": size * size / 1e6, "method": name, "median_s": med,
                     "min_s": min(times), "max_s": max(times)})
        print(f"{size:4d} {name:10s} {med:.3f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RES / "runtime_scaling.csv", index=False, float_format="%.6f")
(RES / "runtime_meta.json").write_text(json.dumps({"sizes": SIZES, "repeats": REPEATS,
                                                   "opencv_threads": 1}, indent=2))
print(df.to_string(index=False))
