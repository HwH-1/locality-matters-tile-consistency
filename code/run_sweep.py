"""Main tile-consistency sweep.

For every (method, image) we record:
  * TCE vs tile count n        (halo = 0 and halo = 16)
  * TCE vs overlap halo        (fixed n = 4)
  * border-band decomposition  (does the deviation die out away from the grid?)
  * grid-line visibility       (how much of the deviation energy sits ON the grid?)
  * quality metrics whole vs tiled (do current metrics even notice?)
  * contrast-equivariance error (class-B signature #2)
Writes JSON-lines to out/sweep.jsonl; aggregate with analyse.py.
"""
import argparse
import glob
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.environ.get("TC_SRC", "/data/tileconsistency/src"))
import cv2
cv2.setNumThreads(1)
import tc
from src.metrics import compute_metrics

NS = (2, 4, 8)
HALOS_N = (0, 16)
HALOS_H = (0, 8, 16, 24, 32)
N_HALO = 4
BAND_N = 8

ADAPTIVE = {
    "CA-AKF": tc.m_ca,
    "SC-AKF": tc.m_sc,
    "LRS-AKF": lambda x: tc.m_lrs(x, c=4.0, rad=16.0, mode="mean"),
    "LRS-final": lambda x: tc.m_lrs(x, c=1.0, rad=8.0, mode="geo"),
    "LRS-geo": lambda x: tc.m_lrs(x, c=4.0, rad=16.0, mode="geo"),
    "AKF": tc.m_akf,
}


def one_image(path, size, names, do_cee):
    img = tc.load(path, size)
    rows = []
    for name in names:
        fn = ADAPTIVE[name]
        t0 = time.time()
        whole = fn(img)
        rec = dict(img=os.path.basename(path), size=size, method=name,
                   t_whole=round(time.time() - t0, 1))
        mw = compute_metrics(img, whole)
        rec["qual_whole"] = {k: round(v, 4) for k, v in mw.items()}
        # TCE vs tile count
        rec["tce_n"] = {}
        outs = {}
        for n in NS:
            for h in HALOS_N:
                p = tc.tiled(fn, img, n, h)
                rec["tce_n"][f"n{n}_h{h}"] = tc.tce(whole, p)
                if (n, h) == (N_HALO, 16):
                    outs["n4_h16"] = p
                if (n, h) == (BAND_N, 16):
                    b, edges = tc.bands_for_grid(whole, p, n)
                    rec["bands_n8_h16"] = dict(edges=list(edges), bands=b)
                    rec["gv_n8_h16"] = tc.gridline_visibility(whole, p, n)
        # TCE vs halo at fixed n
        rec["tce_halo"] = {}
        for h in HALOS_H:
            if h in HALOS_N:
                rec["tce_halo"][f"h{h}"] = rec["tce_n"][f"n{N_HALO}_h{h}"]
                continue
            rec["tce_halo"][f"h{h}"] = tc.tce(whole, tc.tiled(fn, img, N_HALO, h))
        # quality of the tiled result -- the "metrics don't see it" control
        rec["qual_tiled"] = {k: round(v, 4) for k, v in
                             compute_metrics(img, outs["n4_h16"]).items()}
        rec["gv_n4_h16"] = tc.gridline_visibility(whole, outs["n4_h16"], N_HALO)
        b, edges = tc.bands_for_grid(whole, outs["n4_h16"], N_HALO)
        rec["bands_n4_h16"] = dict(edges=list(edges), bands=b)
        if do_cee:
            rec["cee"] = {k: round(v, 6) for k, v in
                          tc.contrast_equivariance(fn, img).items()}
        rec["t"] = round(time.time() - t0, 1)
        rows.append(rec)
        print(f"  {os.path.basename(path)} {name}: {rec['t']}s "
              f"rmse(n8,h16)={rec['tce_n']['n8_h16']['rmse']:.6f} "
              f"frac1={rec['tce_n']['n8_h16']['frac1']:.4f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/data/abstrabench/images/met/*.jpg")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N images; lets a replication set be disjoint "
                         "from the primary set without re-running it")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--methods", default="CA-AKF,SC-AKF,LRS-AKF,AKF")
    ap.add_argument("--out", default="/data/tileconsistency/out/sweep.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-cee", action="store_true")
    ap.add_argument("--serial", action="store_true")
    a = ap.parse_args()
    names = a.methods.split(",")
    paths = sorted(glob.glob(a.glob))[a.offset: a.offset + a.limit]
    jobs = [(p, a.size, names, not a.no_cee) for p in paths]
    if a.serial:
        with open(a.out, "w") as fh:
            for j in jobs:
                for r in one_image(*j):
                    fh.write(json.dumps(r) + "\n")
        return
    from multiprocessing import Pool
    with open(a.out, "w") as fh:
        with Pool(a.workers) as pool:
            for rows in pool.starmap(one_image, jobs):
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()


if __name__ == "__main__":
    main()
