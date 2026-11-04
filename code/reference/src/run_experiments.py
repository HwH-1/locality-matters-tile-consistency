"""Run the full comparison and ablation study.

Usage:
    python -m src.run_experiments                # full study (12 images x 8 methods)
    python -m src.run_experiments --quick        # 2 images, smoke test
    python -m src.run_experiments --methods ca_akf akf
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy
import skimage

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "outputs"
LOGS = RES / "logs"

sys.path.insert(0, str(ROOT))
from src.data import load_all, save_rgb  # noqa: E402
from src.methods import METHODS, run_method  # noqa: E402
from src.metrics import METRICS, compute_metrics  # noqa: E402
from src.structure import structure_tensor  # noqa: E402


def log(msg: str, fh=None) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def environment() -> dict:
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": cpu,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_image": skimage.__version__,
        "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    images = load_all()
    if args.quick:
        images = images[:2]
    if args.images:
        images = [im for im in images if im.name in set(args.images)]
    methods = METHODS
    if args.methods:
        methods = [m for m in METHODS if m.name in set(args.methods)]

    log_path = LOGS / f"run_{args.tag}.log"
    with log_path.open("w", encoding="utf-8") as fh:
        log(f"methods: {[m.name for m in methods]}", fh)
        log(f"images: {[im.name for im in images]}", fh)
        env = environment()
        log(json.dumps(env, indent=2), fh)
        (RES / f"environment_{args.tag}.json").write_text(json.dumps(env, indent=2))

        rows = []
        for im in images:
            st = structure_tensor(im.image)
            for spec in methods:
                t0 = time.perf_counter()
                out = run_method(spec, im.image)
                dt = time.perf_counter() - t0
                assert out.shape == im.image.shape and np.isfinite(out).all()
                path = OUT / f"{im.name}__{spec.name}.png"
                save_rgb(path, out)
                met = compute_metrics(im.image, out, st_src=st)
                row = {
                    "image": im.name,
                    "kind": im.kind,
                    "method": spec.name,
                    "method_label": spec.label,
                    "family": spec.family,
                    "runtime_s": dt,
                    **met,
                }
                rows.append(row)
                log(
                    f"{im.name:16s} {spec.name:12s} {dt:6.2f}s  "
                    + "  ".join(f"{k}={met[k]:.4f}" for k in METRICS),
                    fh,
                )
        df = pd.DataFrame(rows)
        csv_path = RES / f"metrics_{args.tag}.csv"
        df.to_csv(csv_path, index=False, float_format="%.6f")
        log(f"saved {csv_path} ({len(df)} rows)", fh)


if __name__ == "__main__":
    main()
