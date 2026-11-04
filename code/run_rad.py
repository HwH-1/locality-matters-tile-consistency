"""Bias -- memory -- halo trade-off for LRS-AKF.

The local reference window radius `rad` is the ONLY knob that decides how big a
halo a tiling pipeline must carry to be exact.  Small rad = cheap halos but a
narrow reference; large rad = closer to the (non-local) global quantile it
replaces.  This script measures the frontier: quality vs required halo, per rad.
"""
import argparse
import glob
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/data/tileconsistency/src")
import cv2
cv2.setNumThreads(1)
import tc
from src.metrics import compute_metrics

RADS = (8, 16, 32, 64)
HALOS = (0, 8, 16, 24, 32, 40)
N = 4


def one(path, size, rads):
    img = tc.load(path, size)
    out = []
    for rad in rads:
        fn = lambda x, r=rad: tc.m_lrs(x, c=1.0, rad=float(r), mode="geo")
        whole = fn(img)
        rec = dict(img=os.path.basename(path), rad=rad, size=size,
                   qual={k: round(v, 4) for k, v in compute_metrics(img, whole).items()})
        rec["tce_halo"] = {}
        for h in HALOS:
            rec["tce_halo"][f"h{h}"] = tc.tce(whole, tc.tiled(fn, img, N, h))
        # the halo at which the tiled result is bit-exact for good
        ex = [h for h in HALOS if rec["tce_halo"][f"h{h}"]["frac1"] == 0.0
              and rec["tce_halo"][f"h{h}"]["maxd"] < 1 / 255]
        rec["halo_exact"] = min(ex) if ex else None
        out.append(rec)
        print(f"  {os.path.basename(path)} rad={rad}: "
              f"halo_exact={rec['halo_exact']} "
              f"rmse@h32={rec['tce_halo']['h32']['rmse']:.7f} "
              f"sr={rec['qual'].get('sr'):.3f} ece={rec['qual'].get('ece'):.2f}",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--rads", default="8,16,32,64")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="/data/tileconsistency/out/rad.jsonl")
    a = ap.parse_args()
    rads = tuple(int(v) for v in a.rads.split(","))
    paths = sorted(glob.glob("/data/abstrabench/images/met/*.jpg"))[: a.limit]
    from multiprocessing import Pool
    with open(a.out, "w") as fh:
        with Pool(min(a.workers, len(paths))) as pool:
            for rows in pool.starmap(one, [(p, a.size, rads) for p in paths]):
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()


if __name__ == "__main__":
    main()
