"""Third-party published implementations under the same tile-consistency protocol.

Every method in Section 6.1-6.8 is an implementation *we* wrote, which leaves an
obvious objection: the class-A/class-B taxonomy might be a description of our own
code rather than of operators in general.  This script applies the identical
protocol (load at 384 px, n=4 grid, halo ladder, TCE against the whole-canvas
reference, exactness predicate max|d| < 1/255 and frac>1level == 0) to public
implementations shipped in OpenCV / OpenCV contrib, each of which corresponds to a
published algorithm we did not write and cannot tune.

Two differences from our own pipeline are deliberate and are recorded per row:

  * these functions take and return uint8, so a deviation of one 8-bit level is a
    hard integer; the "one level" predicate is the native resolution of the method
    rather than a display convention;
  * each method is run at ONE setting, taken from the library's own documentation or
    the paper's recommended value, chosen before the ladder was measured (the
    settings and the predicted exact halo are in PRED below, so the prediction is
    auditable against the measurement).

Predictions (support of the whole decision chain, from the algorithm as published):
  bilateral  d=9            -> 5  (window radius)
  guided     radius=16,eps   -> 32 (box averages applied twice: a and b, then output)
  nlm        7/21 template/search -> 30 (search radius + template radius + prefilter box)
  pm-diffusion 20 iters      -> 20 (one pixel of stencil propagation per iteration)
  am-filter  sigma_s=9       -> 36 (3*sigma_s kernel support)
  rgf        sigmaSpace=15, 3 iters + global affine -> NONE (whole-image least-squares
                                    correction step is nonlocal by construction)
  l0-smooth                  -> NONE (global gradient-domain solve)

Usage:
  python3 run_cv.py --out out/cv_all.jsonl --workers 2
"""
import argparse
import glob
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, "/data/tileconsistency/src")
sys.path.insert(0, "/data/tileconsistency")
import cv2
cv2.setNumThreads(1)
import cv2.ximgproc as X
import tc
from src.metrics import compute_metrics

# ---------------------------------------------------------------- methods ----
# Each entry: (name, callable, predicted exact halo, setting string, citation key)
def _bilateral(x):
    return cv2.bilateralFilter(x, 9, 75, 75)


def _guided(x):
    g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    return X.guidedFilter(g, x, 16, (0.1 * 255) ** 2)


def _nlm(x):
    return cv2.fastNlMeansDenoisingColored(x, None, 10, 10, 7, 21)


def _pmdiff(x):
    return X.anisotropicDiffusion(x, 0.5, 0.2, 20)


def _am(x):
    return X.amFilter(x, x, 9.0, 0.1)


def _rgf(x):
    # keyword form required: the second positional argument of the one-line binding
    # is `dst`, not `d`.  d=-1 lets OpenCV derive the window from sigmaSpace, which is
    # the self-consistent setting (a hand-set d truncates the guidance kernel).
    return X.rollingGuidanceFilter(x, d=-1, sigmaColor=35.0,
                                   sigmaSpace=15.0, numOfIter=3)


def _l0(x):
    return X.l0Smooth(x, 0.02, 0.7)


METHODS = {
    "bil-9":      (_bilateral, 5,   "d=9, sColor=75, sSpace=75",
                   "tomasi1998bilateral"),
    "guided-16":  (_guided,    32,  "radius=16, eps=(0.1*255)^2",
                   "he2010guided"),
    "nlm-7-21":   (_nlm,       30,  "h=10, hColor=10, template=7, search=21",
                   "buades2005nlm"),
    "pmdiff-20":  (_pmdiff,    20,  "alpha=0.5, K=0.2, iters=20",
                   "perona1991diffusion"),
    "am-9":       (_am,        36,  "sigma_s=9, sigma_r=0.1",
                   "gastal2012amf"),
    "rgf-15":     (_rgf,       None, "d=-1, sigmaColor=35, sigmaSpace=15, iters=3",
                   "zhang2014rollingguidance"),
    "l0-0.02":    (_l0,        None, "lambda=0.02, gradient=0.7",
                   "xu2011l0smooth"),
}

L1 = 1.0 / 255.0


def to_u8(img):
    """float RGB [0,1] (what tc.load returns) -> uint8 BGR (what OpenCV wants)."""
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)[:, :, ::-1].copy()


def to_f(x):
    """uint8 BGR -> float RGB [0,1], so TCE and the quality metrics use the paper's units."""
    return x[:, :, ::-1].astype(np.float32) / 255.0


def one_image(path, size, names, n, halos, gv_halo):
    img = tc.load(path, size)
    u8 = to_u8(img)
    rows = []
    for name in names:
        fn, pred, setting, cite = METHODS[name]
        t0 = time.time()
        whole_u8 = fn(u8)
        rec = dict(img=os.path.basename(path), size=size, method=name, n=n,
                   setting=setting, cite=cite, pred_exact_halo=pred,
                   t=round(time.time() - t0, 1))
        rec["tce_halo"] = {}
        outs = {}
        for h in halos:
            p = tc.tiled(fn, u8, n, h)
            rec["tce_halo"][f"h{h}"] = tc.tce(to_f(whole_u8), to_f(p))
            if h == gv_halo:
                outs["gv"] = p
            if h == halos[-1]:
                outs["last"] = p
        exact = [h for h in halos
                 if rec["tce_halo"][f"h{h}"]["maxd"] < L1
                 and rec["tce_halo"][f"h{h}"]["frac1"] == 0.0]
        rec["halo_exact"] = min(exact) if exact else None
        rec["pred_holds"] = (None if pred is None or rec["halo_exact"] is None
                             else bool(rec["halo_exact"] <= pred))
        rec["gv"] = (tc.gridline_visibility(to_f(whole_u8), to_f(outs["gv"]), n)
                     if "gv" in outs else None)
        rec["qual_whole"] = {k: round(v, 4) for k, v in
                             compute_metrics(img, to_f(whole_u8)).items()}
        rec["qual_tiled"] = {k: round(v, 4) for k, v in
                             compute_metrics(img, to_f(outs["last"])).items()}
        rec["dev_levels"] = round(float(np.abs(whole_u8.astype(np.int16)
                                              - u8.astype(np.int16)).mean()), 3)
        rec["t"] = round(time.time() - t0, 1)
        rows.append(rec)
        print(f"  {os.path.basename(path)} {name}: {rec['t']}s exact_halo={rec['halo_exact']} "
              f"pred={pred} frac1@h{halos[-1]}={rec['tce_halo'][f'h{halos[-1]}']['frac1']:.5f}",
              flush=True)
    return rows


def _job(job):
    return one_image(*job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-records", default="/data/tileconsistency/out/broad_all.jsonl",
                    help="use exactly the image names already measured in the breadth run")
    ap.add_argument("--dirs", default="/data/abstrabench/images/met,"
                                      "/data/abstrabench/images/synthetic,"
                                      "/data/abstrabench/images/skimage")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--halos", default="0,2,4,8,16,24,32,48")
    ap.add_argument("--gv-halo", type=int, default=16)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/data/tileconsistency/out/cv_all.jsonl")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    names = a.methods.split(",")
    for m in names:
        assert m in METHODS, f"unknown method {m}"
    halos = [int(v) for v in a.halos.split(",")]

    # Same images as the breadth run, so every row is paired with an existing one.
    want = sorted({r["img"] for r in
                   (json.loads(l) for l in open(a.from_records))})
    by_name = {}
    for d in a.dirs.split(","):
        for p in sorted(glob.glob(os.path.join(d, "*"))):
            by_name.setdefault(os.path.basename(p), p)
    paths = [by_name[n] for n in want if n in by_name]
    missing = [n for n in want if n not in by_name]
    if missing:
        print(f"WARNING: {len(missing)} images not found: {missing[:5]}", flush=True)
    if a.limit:
        paths = paths[:a.limit]

    done = set()
    for rec_file in sorted(glob.glob(os.path.join(
            os.path.dirname(os.path.abspath(a.out)), "cv*.jsonl"))):
        for line in open(rec_file):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r["img"], r["method"]))
    if done:
        print(f"resume: {len(done)} (image, method) records present", flush=True)
        paths = [p for p in paths if any((os.path.basename(p), m) not in done
                                         for m in names)]

    jobs = [(p, a.size, names, a.n, halos, a.gv_halo) for p in paths]
    print(f"{len(paths)} images x {len(names)} methods queued", flush=True)
    mode = "a" if done else "w"
    from multiprocessing import Pool
    with open(a.out, mode) as fh:
        with Pool(a.workers) as pool:
            for rows in pool.imap_unordered(_job, jobs, chunksize=1):
                for r in rows:
                    if (r["img"], r["method"]) in done:
                        continue
                    fh.write(json.dumps(r) + "\n")
                    done.add((r["img"], r["method"]))
                fh.flush()


if __name__ == "__main__":
    main()
