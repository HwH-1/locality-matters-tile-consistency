"""Native-resolution large-image test on single CC0 artworks (no mosaic assembly).

Why this exists: Section 6.7's canvas is *assembled* from served files that cap at
1024 px, so a reviewer can object that no single native large image was ever run.
Here the input is one Met original at 4000x3853 / 3630x4000 (manifest licence CC0,
sha1 recorded), and the questions are the two the paper's deployment claim rests on:

  (1) FEASIBILITY -- does the whole-canvas pass run at all on this host?  The child
      process sets RLIMIT_AS so an infeasible pass raises MemoryError *inside our
      process* instead of inviting the kernel OOM killer onto whichever job happens
      to be running.

  (2) HALO BEHAVIOUR at native scale, without a whole-image reference.  With no
      reference, a tiled-vs-whole TCE is impossible, so we use the Cauchy form:
      the same partition evaluated at halo h1 and h2 (h2 >> h1) is compared to
      itself.  A class-A (local) operator's two passes agree to float dust once
      h1 >= support; a class-B operator's do not, at any h1, because the parameter
      is a functional of the tile window rather than of the image.  This is stated
      as what it is -- a self-consistency test, weaker than TCE, usable where TCE
      cannot be computed at all.

One mode per process, always: peak RSS is only attributable to the pass that made it.

usage:
  python3 run_native.py --self <args>        (child; do not call directly)
  python3 run_native.py --img PATH --out out/native.jsonl [--cap 5.5]
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time

sys.path.insert(0, "/data/tileconsistency/src")

CAP_GB = 5.5
METHODS = ("CA-AKF", "SC-AKF", "LRS-final", "AKF")


def _peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _child(img_path, method, mode, n, halo, cap):
    import numpy as np  # noqa: F401  (kept for parity with parent env)
    import cv2
    cv2.setNumThreads(1)
    import tc

    if cap:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        lim = int(cap * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (lim, hard if hard != -1 else lim))

    fn = {"CA-AKF": tc.m_ca, "SC-AKF": tc.m_sc, "AKF": tc.m_akf,
          "LRS-final": lambda x: tc.m_lrs(x, c=1.0, rad=8.0, mode="geo")}[method]
    img = tc.load(img_path, None)
    rec = dict(img=os.path.basename(img_path), method=method, mode=mode, n=n,
               halo=halo, height=img.shape[0], width=img.shape[1],
               array_mb=round(img.nbytes / 1e6, 1))
    t0 = time.time()
    try:
        if mode == "whole":
            out = fn(img)
            rec["ref"] = "whole"
        else:
            out = tc.tiled(fn, img, n, halo)
    except (MemoryError, Exception) as exc:  # noqa: B014 - rlimit raises MemoryError
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        rec["feasible"] = False
        rec["filter_s"] = round(time.time() - t0, 1)
        rec["peak_mb"] = round(_peak_mb(), 1)
        print(json.dumps(rec), flush=True)
        return
    rec["feasible"] = True
    rec["filter_s"] = round(time.time() - t0, 1)
    rec["peak_mb"] = round(_peak_mb(), 1)
    import os as _os
    stem = _os.path.splitext(_os.path.basename(img_path))[0]
    np.save(f"/tmp/native_{stem}_{method.replace('-', '')}_{mode}_{n}_{halo}.npy", out)
    print(json.dumps(rec), flush=True)


def spawn(args):
    """Run one (method, mode) pass in a fresh process and return its record."""
    cmd = [sys.executable, os.path.abspath(__file__), "--self",
           "--img", args.img, "--method", args.method, "--mode", args.mode,
           "--n", str(args.n), "--halo", str(args.halo), "--cap", str(args.cap)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if not line:
        return dict(img=os.path.basename(args.img), method=args.method, mode=args.mode,
                    n=args.n, halo=args.halo, feasible=False,
                    error=f"no record (rc={p.returncode}) {p.stderr[-200:]}",
                    wall_s=round(time.time() - t0, 1))
    rec = json.loads(line[-1])
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["rc"] = p.returncode
    return rec


def pair_stats(out_a, out_b, n, mode, extra):
    """Deviation between two passes over the same pixels, with seam concentration."""
    import tc
    d = tc.tce(out_a, out_b)
    d["gv_ratio"] = tc.gridline_visibility(out_a, out_b, n)["ratio"]
    rec = dict(mode=mode, n=n, **extra)
    rec.update({k: round(float(v), 8) for k, v in d.items()})
    return rec


def path_for(img, method, mode, n, halo):
    stem = os.path.splitext(os.path.basename(img))[0]
    return f"/tmp/native_{stem}_{method.replace('-', '')}_{mode}_{n}_{halo}.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self", action="store_true")
    ap.add_argument("--img")
    ap.add_argument("--out", default="/data/tileconsistency/out/native.jsonl")
    ap.add_argument("--method")
    ap.add_argument("--mode")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--halo", type=int, default=32)
    ap.add_argument("--cap", type=float, default=CAP_GB)
    ap.add_argument("--ladder", default="32,64")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--whole", choices=["yes", "no"], default="yes",
                    help="attempt the whole-canvas pass (feasibility + TCE reference)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the .npy outputs (needed when phases are run separately)")
    a = ap.parse_args()

    if a.__dict__.get("self"):
        _child(a.img, a.method, a.mode, a.n, a.halo, a.cap)
        return

    if not a.img:
        ap.error("--img required in orchestrator mode")
    halos = [] if a.ladder == "none" else [int(x) for x in a.ladder.split(",")]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    def emit(rec):
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    for m in a.methods.split(","):
        import numpy as np
        whole_p = None
        if a.whole == "yes":
            wr = spawn(argparse.Namespace(img=a.img, method=m, mode="whole", n=a.n,
                                          halo=0, cap=a.cap))
            emit(wr)
            cand = path_for(a.img, m, "whole", a.n, 0)
            if wr.get("feasible") and os.path.exists(cand):
                whole_p = cand
        elif os.path.exists(path_for(a.img, m, "whole", a.n, 0)):
            # reference produced by an earlier phase of this same image and partition
            whole_p = path_for(a.img, m, "whole", a.n, 0)
        tiled_p = {}
        for h in halos:
            rec = spawn(argparse.Namespace(img=a.img, method=m, mode="tiled", n=a.n,
                                           halo=h, cap=a.cap))
            rec["halo"] = h
            emit(rec)
            cand = path_for(a.img, m, "tiled", a.n, h)
            if rec.get("feasible") and os.path.exists(cand):
                tiled_p[h] = cand
        if whole_p:
            ref = np.load(whole_p)
            for h, p in sorted(tiled_p.items()):
                emit(dict(img=os.path.basename(a.img), method=m,
                          **pair_stats(ref, np.load(p), a.n, "tce",
                                        dict(reference="whole", halo=h))))
        hs = sorted(tiled_p)
        for h1, h2 in zip(hs, hs[1:]):
            emit(dict(img=os.path.basename(a.img), method=m,
                      **pair_stats(np.load(tiled_p[h1]), np.load(tiled_p[h2]), a.n,
                                   "cauchy", dict(halos=[h1, h2]))))
        if not a.keep:
            for p in list(tiled_p.values()) + ([whole_p] if whole_p else []):
                try:
                    os.remove(p)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
