"""Does the threshold multiplier c affect tile consistency?

c rescales the local reference tau(x) = c * R(x).  It changes the *value* of the
parameter, hence the look of the output, but it must not change the *support* of
the parameter chain, hence it must not change the halo at which tiling becomes
exact.  This script is the direct test of that claim: same rad, different c,
measure exactness halo and quality separately.
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

HALOS = (0, 8, 16, 24, 32)
N = 4


def one(path, size, rads, cs):
    img = tc.load(path, size)
    out = []
    for rad in rads:
        for c in cs:
            fn = lambda x, r=rad, cc=c: tc.m_lrs(x, c=cc, rad=float(r), mode="geo")
            whole = fn(img)
            rec = dict(img=os.path.basename(path), rad=rad, c=c, size=size,
                       qual={k: round(v, 4) for k, v in compute_metrics(img, whole).items()})
            rec["tce_halo"] = {}
            for h in HALOS:
                rec["tce_halo"][f"h{h}"] = tc.tce(whole, tc.tiled(fn, img, N, h))
            ex = [h for h in HALOS if rec["tce_halo"][f"h{h}"]["frac1"] == 0.0
                  and rec["tce_halo"][f"h{h}"]["maxd"] < 1 / 255]
            rec["halo_exact"] = min(ex) if ex else None
            out.append(rec)
            print(f"  {os.path.basename(path)} rad={rad} c={c}: "
                  f"halo_exact={rec['halo_exact']} "
                  f"rmse@h32={rec['tce_halo']['h32']['rmse']:.7f} "
                  f"sr={rec['qual'].get('sr'):.3f} fsr={rec['qual'].get('fsr'):.3f} "
                  f"ece={rec['qual'].get('ece'):.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--rads", default="8")
    ap.add_argument("--cs", default="1,1.5,2,3")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="/data/tileconsistency/out/c.jsonl")
    a = ap.parse_args()
    rads = tuple(int(v) for v in a.rads.split(","))
    cs = tuple(float(v) for v in a.cs.split(","))
    paths = sorted(glob.glob("/data/abstrabench/images/met/*.jpg"))[: a.limit]
    from multiprocessing import Pool
    with open(a.out, "w") as fh:
        with Pool(min(a.workers, len(paths))) as pool:
            for rows in pool.starmap(one, [(p, a.size, rads, cs) for p in paths]):
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()


if __name__ == "__main__":
    main()
