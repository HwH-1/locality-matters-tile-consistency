"""Gigapixel-canvas experiment.

Claim to support: tile-wise processing is not an optional optimisation for large
canvases -- it is forced by memory.  So an operator whose output changes under
tiling is wrong-by-default on exactly the inputs that need it most.

Measured per configuration:
  * peak RSS (resource.getrusage ru_maxrss, kB)
  * wall time
  * TCE between the whole-canvas result and the tile-wise result
  * grid-line visibility of the seam energy
  * a saved 512px crop straddling a seam, for the visual figure
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
cv2.setNumThreads(1)
import tc

OUT = "/data/tileconsistency/out"
FIG = "/data/tileconsistency/figures"


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build_canvas(tile_px, grid, srcs, path):
    """Mosaic of public-domain images -- a stand-in for a gigapixel scan."""
    h = w = tile_px * grid
    canvas = np.zeros((h, w, 3), np.uint8)
    for i in range(grid):
        for j in range(grid):
            img = cv2.imread(srcs[(i * grid + j) % len(srcs)], cv2.IMREAD_COLOR)
            img = cv2.resize(img, (tile_px, tile_px), interpolation=cv2.INTER_LANCZOS4)
            canvas[i * tile_px:(i + 1) * tile_px,
                   j * tile_px:(j + 1) * tile_px] = img
    cv2.imwrite(path, canvas)
    return path


def run(fn, img, tag, rec):
    t0 = time.time()
    out = fn(img)
    rec[tag] = dict(s=round(time.time() - t0, 2), peak_mb=round(rss_mb(), 1),
                    out_rms=float(np.sqrt((out.astype(np.float64) ** 2).mean())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=512)
    ap.add_argument("--grid", type=int, default=4)          # canvas = tiles x grid
    ap.add_argument("--tile-counts", default="2,4")
    ap.add_argument("--halos", default="16,48")
    ap.add_argument("--try-4096", action="store_true")
    ap.add_argument("--only", default=None,
                    help="run a single method in this process (clean peak-RSS)")
    ap.add_argument("--out", default=f"{OUT}/giga.json")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    srcs = sorted(glob.glob("/data/abstrabench/images/met/*.jpg"))
    canvas_path = f"/data/tileconsistency/canvas_{a.tiles * a.grid}.png"
    if not os.path.exists(canvas_path):
        build_canvas(a.tiles, a.grid, srcs, canvas_path)
    print("canvas:", canvas_path, flush=True)
    big = (cv2.imread(canvas_path, cv2.IMREAD_COLOR)[:, :, ::-1]
           .astype(np.float32) / 255.0)
    big = np.ascontiguousarray(big)
    print(f"size {big.shape}  array {big.nbytes/1e6:.0f} MB", flush=True)

    methods = {"CA-AKF": tc.m_ca,
               "SC-AKF": tc.m_sc,
               "LRS-final": lambda x: tc.m_lrs(x, c=1.0, rad=8.0, mode="geo"),
               "AKF": tc.m_akf}
    if a.only:
        methods = {a.only: methods[a.only]}
    rec = {"canvas": os.path.basename(canvas_path),
           "shape": list(big.shape), "array_mb": round(big.nbytes / 1e6, 1)}
    for name, fn in methods.items():
        r = {}
        whole = run(fn, big, "whole", r)
        cv2.imwrite(f"{FIG}/whole_{name}.png", (np.clip(whole, 0, 1) * 255)
                    .astype(np.uint8)[:, :, ::-1])
        for n in [int(v) for v in a.tile_counts.split(",")]:
            for h in [int(v) for v in a.halos.split(",")]:
                t = tc.tiled(fn, big, n, h)
                m = tc.tce(whole, t)
                gv = tc.gridline_visibility(whole, t, n)
                key = f"n{n}_h{h}"
                r[key] = dict(**{k: round(v, 6) for k, v in m.items()},
                              gv=round(gv["gv"], 4), ratio=round(gv["ratio"], 1),
                              peak_mb=round(rss_mb(), 1))
                # seam crop: 512 px centred on the middle vertical grid line
                w = big.shape[1]
                cx = (n // 2) * (w // n)
                crop = np.abs(whole - t).max(axis=-1)
                crop = crop[:, max(0, cx - 256):cx + 256]
                cv2.imwrite(f"{FIG}/seam_{name}_{key}.png",
                            np.clip(crop / max(crop.max(), 1e-6) * 255, 0, 255)
                            .astype(np.uint8))
                print(f"  {name} {key}: rmse={m['rmse']:.6f} "
                      f"frac1={m['frac1']:.4f} gv={gv['gv']:.3f} "
                      f"peak={r[key]['peak_mb']}MB", flush=True)
        # seam crop as a rendered difference panel on the tiling the pipeline uses
        rec[name] = r
    if a.try_4096:
        p = "/data/tileconsistency/canvas_4096.png"
        if not os.path.exists(p):
            build_canvas(512, 8, srcs, p)
        try:
            im = np.ascontiguousarray(cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1]
                                      .astype(np.float32) / 255.0)
            print(f"4096 array {im.nbytes/1e6:.0f} MB; attempting whole pass", flush=True)
            r = {"array_mb": round(im.nbytes / 1e6, 1)}
            tc.m_ca(im)
            r["whole_ok"] = True
        except MemoryError:
            r["whole_ok"] = False
            r["note"] = "MemoryError on 4096x4096 whole pass"
        r["peak_mb"] = round(rss_mb(), 1)
        rec["attempt4096"] = r
        print("4096:", json.dumps(r), flush=True)
    out = a.out if not a.only else a.out.replace(".json", "_" + a.only + ".json")
    json.dump(rec, open(out, "w"), indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
