"""Per-mode peak memory and wall time on a large canvas.

run_giga.py measures whole and tiled in the same process, so the reported peak
is the process peak (dominated by whichever pass is larger) and cannot separate
the two modes.  This script runs ONE mode in ONE process and reports its own
peak RSS and wall time, which is the number an implementer actually needs:
"can my pipeline hold this canvas, and what does it cost to do it whole vs
tiled?"

Usage:
  python3 run_mem.py --canvas canvas_2048.png --mode whole --method CA-AKF
  python3 run_mem.py --canvas canvas_2048.png --mode tiled  --method CA-AKF --n 4 --halo 48
"""
import argparse
import glob
import json
import os
import resource
import sys
import time
import numpy as np

sys.path.insert(0, "/data/tileconsistency/src")
import cv2
cv2.setNumThreads(4)
import tc

FIG = "/data/tileconsistency/figures"


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canvas", required=True)
    ap.add_argument("--mode", choices=["whole", "tiled"], required=True)
    ap.add_argument("--method", required=True,
                    choices=["CA-AKF", "SC-AKF", "LRS-final", "AKF"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--halo", type=int, default=48)
    ap.add_argument("--out", default="/data/tileconsistency/out/mem.jsonl")
    a = ap.parse_args()

    t0 = time.time()
    big = np.ascontiguousarray(
        cv2.imread(a.canvas, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0)
    load_s = time.time() - t0
    array_mb = round(big.nbytes / 1e6, 1)

    methods = {"CA-AKF": tc.m_ca,
               "SC-AKF": tc.m_sc,
               "LRS-final": lambda x: tc.m_lrs(x, c=1.0, rad=8.0, mode="geo"),
               "AKF": tc.m_akf}
    fn = methods[a.method]

    rec = dict(canvas=os.path.basename(a.canvas), shape=list(big.shape),
               array_mb=array_mb, load_s=round(load_s, 2), method=a.method,
               mode=a.mode, n=a.n, halo=a.halo)
    t0 = time.time()
    try:
        if a.mode == "whole":
            out = fn(big)
        else:
            out = tc.tiled(fn, big, a.n, a.halo)
        rec["ok"] = True
        rec["filter_s"] = round(time.time() - t0, 2)
        rec["out_rms"] = round(float(np.sqrt((out.astype(np.float64) ** 2).mean())), 6)
        # save the output for the visual panel (8-bit PNG)
        os.makedirs(FIG, exist_ok=True)
        tag = f"{a.method}_{a.mode}_n{a.n}_h{a.halo}"
        cv2.imwrite(f"{FIG}/canvas_{tag}.png",
                    (np.clip(out, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        # consistency vs the whole-canvas pass is measured by the caller from the
        # saved outputs; here we only report this run's own footprint
    except MemoryError:
        rec["ok"] = False
        rec["note"] = "MemoryError"
        rec["filter_s"] = round(time.time() - t0, 2)
    rec["peak_mb"] = round(peak_mb(), 1)
    print(json.dumps(rec), flush=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
