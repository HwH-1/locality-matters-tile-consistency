"""Is each OpenCV implementation even a function of pixel content?

Section 6.9 attributes a halo plateau to one of two very different causes:

  (i)  the algorithm's decision statistic is nonlocal (class B, Prop. 2), or
  (ii) the implementation's output is not determined by the pixel values it is
       handed, so tiled evaluation differs from whole-canvas evaluation for a
       reason no overlap setting could address.

(ii) has to be tested separately, and it is cheap to test: call the function twice
on arrays with identical content but different memory geometry -- a fresh copy, and
a non-contiguous view of a wider buffer.  A content-determined function returns the
same bytes.  Anything else is an implementation artefact and must not be reported as
evidence about the algorithm.

Writes out/cv_det.jsonl: one record per (image, method).  Two whole-canvas calls per
pair, so the whole 144-image pool costs about 15 minutes.
"""
import argparse
import glob
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/data/tileconsistency/src")
sys.path.insert(0, "/data/tileconsistency")
import cv2
cv2.setNumThreads(1)
import tc
from run_cv import METHODS, to_u8


def stats(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    return int((d > 0).sum()), int(d.max())


def one_image(path, size, names):
    img = tc.load(path, size)
    u8 = to_u8(img)
    h, w = u8.shape[:2]
    wide = np.zeros((h, w + 64, 3), np.uint8)
    wide[:, :w] = u8
    view = wide[:, :w]                       # identical values, step = (w+64)*3
    recs = []
    for name in names:
        fn = METHODS[name][0]
        base = fn(u8)
        fresh = fn(np.array(u8, copy=True))
        nd_f, mx_f = stats(base, fresh)
        nd_v, mx_v = stats(base, fn(view))
        # and the same call twice on the identical array (pure determinism control)
        nd_s, mx_s = stats(base, fn(u8))
        recs.append(dict(img=os.path.basename(path), size=size, method=name,
                         ndiff_fresh_copy=nd_f, maxd_fresh_copy=mx_f,
                         ndiff_noncontig_view=nd_v, maxd_noncontig_view=mx_v,
                         ndiff_repeat=nd_s, maxd_repeat=mx_s))
        print(f"  {os.path.basename(path)} {name:12s} fresh={nd_f:6d}/{mx_f:3d} "
              f"view={nd_v:6d}/{mx_v:3d} repeat={nd_s}/{mx_s}", flush=True)
    return recs


def _job(j):
    return one_image(*j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-records", default="/data/tileconsistency/out/broad_all.jsonl")
    ap.add_argument("--dirs", default="/data/abstrabench/images/met,"
                                      "/data/abstrabench/images/synthetic,"
                                      "/data/abstrabench/images/skimage")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/data/tileconsistency/out/cv_det.jsonl")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    names = a.methods.split(",")
    want = sorted({r["img"] for r in (json.loads(l) for l in open(a.from_records))})
    by_name = {}
    for d in a.dirs.split(","):
        for p in sorted(glob.glob(os.path.join(d, "*"))):
            by_name.setdefault(os.path.basename(p), p)
    paths = [by_name[n] for n in want if n in by_name]
    if a.limit:
        paths = paths[:a.limit]

    done = set()
    for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(a.out)),
                                           "cv_det*.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r["img"], r["method"]))
    if done:
        paths = [p for p in paths if any((os.path.basename(p), m) not in done
                                         for m in names)]
        print(f"resume: {len(done)} records present", flush=True)

    jobs = [(p, a.size, names) for p in paths]
    print(f"{len(paths)} images x {len(names)} methods queued", flush=True)
    from multiprocessing import Pool
    with open(a.out, "a" if done else "w") as fh:
        with Pool(a.workers) as pool:
            for recs in pool.imap_unordered(_job, jobs, chunksize=1):
                for r in recs:
                    if (r["img"], r["method"]) in done:
                        continue
                    fh.write(json.dumps(r) + "\n")
                    done.add((r["img"], r["method"]))
                fh.flush()


if __name__ == "__main__":
    main()
