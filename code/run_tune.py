"""Quality calibration for LRS-AKF.

The local reference is a MEAN of the base dispersion, while the published
criterion is a low QUANTILE of the same map, so the same c is not comparable:
mean > p25 and the operator over-selects large radii.  Find (c, rad) that
matches or beats the published quality metrics, then re-measure TCE for it.
"""
import argparse
import glob
import json
import os
import sys
import itertools
from multiprocessing import Pool

sys.path.insert(0, "/data/tileconsistency/src")
import cv2
cv2.setNumThreads(1)
import tc
from src.metrics import compute_metrics

QUAL_KEYS = ("sr", "ssim", "fsr", "ece", "scg", "ovt")


def one(args):
    path, size, c, rad, mode = args
    img = tc.load(path, size)
    out = tc.m_lrs(img, c=c, rad=rad, mode=mode)
    m = compute_metrics(img, out)
    rec = dict(img=os.path.basename(path), c=c, rad=rad, mode=mode,
               **{k: round(float(m[k]), 4) for k in QUAL_KEYS if k in m})
    print(f"  {os.path.basename(path)} c={c} rad={rad} {mode}: " +
          " ".join(f"{k}={rec[k]:.3f}" for k in QUAL_KEYS if k in rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--cs", default="1.0,1.5,2.0,3.0,4.0")
    ap.add_argument("--rads", default="8,16,32")
    ap.add_argument("--modes", default="mean,geo")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="/data/tileconsistency/out/tune.jsonl")
    a = ap.parse_args()
    paths = sorted(glob.glob("/data/abstrabench/images/met/*.jpg"))[: a.limit]
    jobs = [(p, a.size, float(c), float(r), mo)
            for c, r, mo in itertools.product(a.cs.split(","), a.rads.split(","),
                                              a.modes.split(","))
            for p in paths]
    with open(a.out, "w") as fh:
        with Pool(a.workers) as pool:
            for rec in pool.imap_unordered(one, jobs):
                fh.write(json.dumps(rec) + "\n")
                fh.flush()


if __name__ == "__main__":
    main()
