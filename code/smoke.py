import sys, time, glob, numpy as np
sys.path.insert(0, "/data/tileconsistency/src")
import tc

paths = sorted(glob.glob("/data/abstrabench/images/met/*.jpg"))
print("images:", len(paths), flush=True)
img = tc.load(paths[0], 512)
t0 = time.time()
CASES = [("AKF", tc.m_akf),
         ("LRS-mean", lambda x: tc.m_lrs(x, c=4.0, rad=16., mode="mean")),
         ("SC-AKF", tc.m_sc),
         ("CA-AKF", tc.m_ca)]
for name, fn in CASES:
    w = fn(img)
    print(f"\n== {name}  whole {time.time()-t0:.1f}s", flush=True)
    for n in (4, 8):
        for halo in (0, 16, 32, 48, 64):
            p = tc.tiled(fn, img, n, halo)
            m = tc.tce(w, p)
            print(f"   n={n:2d} halo={halo:2d} rmse={m['rmse']:.6f} "
                  f"maxd={m['maxd']:.4f} frac1={m['frac1']:.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    print("   CEE:", {k: round(v, 6) for k, v in tc.contrast_equivariance(fn, img).items()},
          flush=True)
print(f"\ntotal {time.time()-t0:.0f}s", flush=True)
