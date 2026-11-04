"""Broad tile-consistency replication over the whole image set.

The primary protocol (run_sweep.py) records the full metric battery, which costs
~3 min per image; that is affordable for 10 images and not for 145.  This script
keeps only what the central claim depends on -- the halo ladder of the
tile-consistency error at a fixed tile count, plus how the deviation energy is
distributed relative to the tile grid -- and runs it on every image in the
benchmark.  Resume-safe: records already in --out are skipped, so a partial run
is still usable.

Usage:
  python3 run_broad.py --out out/broad.jsonl --workers 2
"""
import argparse
import glob
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, "/data/tileconsistency/src")
import cv2
cv2.setNumThreads(1)
import tc

ADAPTIVE = {
    "CA-AKF": tc.m_ca,
    "SC-AKF": tc.m_sc,
    "LRS-final": lambda x: tc.m_lrs(x, c=1.0, rad=8.0, mode="geo"),
    "AKF": tc.m_akf,
}


def one_image(path, size, names, n, halos, gv_halo):
    img = tc.load(path, size)
    rows = []
    for name in names:
        fn = ADAPTIVE[name]
        t0 = time.time()
        whole = fn(img)
        rec = dict(img=os.path.basename(path), size=size, method=name,
                   n=n, t=round(time.time() - t0, 1))
        rec["tce_halo"] = {}
        outs = {}
        for h in halos:
            p = tc.tiled(fn, img, n, h)
            rec["tce_halo"][f"h{h}"] = tc.tce(whole, p)
            if h == gv_halo:
                outs["gv"] = p
        exact = [h for h in halos
                 if rec["tce_halo"][f"h{h}"]["maxd"] < 1 / 255
                 and rec["tce_halo"][f"h{h}"]["frac1"] == 0.0]
        rec["halo_exact"] = min(exact) if exact else None
        rec["gv"] = tc.gridline_visibility(whole, outs["gv"], n) if outs else None
        rec["t"] = round(time.time() - t0, 1)
        rows.append(rec)
        print(f"  {os.path.basename(path)} {name}: {rec['t']}s "
              f"exact_halo={rec['halo_exact']} "
              f"rmse@h{halos[-1]}={rec['tce_halo'][f'h{halos[-1]}']['rmse']:.7f} "
              f"frac1={rec['tce_halo'][f'h{halos[-1]}']['frac1']:.5f}", flush=True)
    return rows


def _job(job):
    """Pool target: unpack the argument tuple. Module-level so it pickles."""
    return one_image(*job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/data/abstrabench/images/met/*.jpg")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--halos", default="0,16,32")
    ap.add_argument("--gv-halo", type=int, default=16)
    ap.add_argument("--methods", default="CA-AKF,SC-AKF,LRS-final")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default="/data/tileconsistency/out/broad.jsonl")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    names = a.methods.split(",")
    halos = [int(v) for v in a.halos.split(",")]
    paths = sorted(glob.glob(a.glob))[a.offset:]
    if a.limit:
        paths = paths[: a.limit]

    done = set()
    # The done-set is read from every sibling broad*.jsonl, not only from --out.
    # The run has been restarted into new output files after sandbox reboots; a
    # resume that looks at its own file alone silently recomputes finished images.
    for rec_file in sorted(glob.glob(os.path.join(
            os.path.dirname(os.path.abspath(a.out)), "broad*.jsonl"))):
        with open(rec_file) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                done.add((r["img"], r["method"]))
    if done:
        print(f"resume: {len(done)} (image, method) records present", flush=True)
        paths = [p for p in paths
                 if any((os.path.basename(p), m) not in done for m in names)]

    jobs = [(p, a.size, names, a.n, halos, a.gv_halo) for p in paths]
    mode = "a" if done else "w"
    print(f"{len(paths)} images x {len(names)} methods queued", flush=True)
    from multiprocessing import Pool
    with open(a.out, mode) as fh:
        with Pool(a.workers) as pool:
            # chunksize=1 + unordered: every finished image is written immediately,
            # so the run is usable and resumable at any moment.
            for rows in pool.imap_unordered(_job, jobs, chunksize=1):
                for r in rows:
                    if (r["img"], r["method"]) in done:
                        continue
                    fh.write(json.dumps(r) + "\n")
                    done.add((r["img"], r["method"]))
                fh.flush()


if __name__ == "__main__":
    main()
