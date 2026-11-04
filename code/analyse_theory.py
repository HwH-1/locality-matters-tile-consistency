"""Numerical verification for the two theory results added for the top-tier pass.

P3 (rank structure of the decision statistic — and the limit of that claim).
The AKF criterion accumulates rank-1 outer products of the image gradient,
    M(x) = sum_q w_q(x) g(q) g(q)^T ,   g = grad I ,
so M(x) is 2x2 symmetric PSD: rank(M) <= 2 always, rank(M) = 1 exactly when the
window's gradient directions are collinear, and the rank-1 truncation is available
in closed form,
    ||M - lambda_max u u^T||_2 = lambda_min ,  relative error = lambda_min/lambda_max.
This measures the distribution of that relative error on real images.  It is written
to falsify a "the covariance is low-rank" claim if it is false, which is why the
exact-rank-deficiency column is reported next to the tolerance-based ones.

P4 (cost of tiling, as an upper bound with a fitted additive term).
With tile side T and halo h, a tiled pass evaluates (T+2h)^2 positions to emit T^2
pixels, so
    W_tiled <= (1 + 2h/T)^2 * W_whole + N * c0 ,
c0 being the per-tile fixed cost (reflect padding, moment-array allocation, output
copy) and N = n^2 the tile count.  We fit c0 per method from the measured whole/tiled
pairs at n=4 and n=8 and check the bound, and we check the memory shape
peak_tiled vs canvas array + tiles * (T+2h)^2.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/data/tileconsistency/src")
import cv2
cv2.setNumThreads(1)
import tc

OUT = "/data/tileconsistency/out"
IMG_GLOB = "/data/abstrabench/images/met/*.jpg"


def coherence_map(img, rad):
    """Eigen-spectrum of the box-accumulated gradient outer product."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    k = 2 * rad + 1
    box = lambda a: cv2.boxFilter(a, -1, (k, k), normalize=False,
                                  borderType=cv2.BORDER_REFLECT)
    jxx, jyy, jxy = box(gx * gx), box(gy * gy), box(gx * gy)
    tr = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0))
    lam_hi, lam_lo = (tr + disc) / 2.0, (tr - disc) / 2.0
    nz = lam_hi > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = np.where(nz, lam_lo / np.maximum(lam_hi, 1e-300), np.nan)
    return lam_hi, lam_lo, coh[nz], nz


def p3(paths, rad):
    rows = []
    for path in paths:
        img = tc.load(path, 384)
        hi, lo, c, nz = coherence_map(img, rad)
        tot = float(c.size)
        rows.append(dict(
            img=os.path.basename(path),
            frac_rank1_le_001=float((c <= 0.01).mean()),
            frac_rank1_le_005=float((c <= 0.05).mean()),
            frac_gt_050=float((c > 0.50).mean()),
            median_coh=float(np.median(c)),
            p90_coh=float(np.percentile(c, 90)),
            frac_exact_rank1=float((lo[nz] == 0.0).mean()),
            frac_active=float(nz.mean()),
        ))
    return rows


def load_mem():
    """Prefer whole/tiled pairs recorded in the SAME session file.

    The two files are different host-load sessions (CA-AKF's whole pass is 143.9 s
    in mem.jsonl and 92.5 s in mem_manual.jsonl).  A cost *ratio* is only meaningful
    within one session, so mem.jsonl wins on conflict and mem_manual fills only keys
    that are otherwise absent -- those rows get a cross_session flag.
    """
    rec, src = {}, {}
    for f in ("mem.jsonl", "mem_manual.jsonl"):
        p = f"{OUT}/{f}"
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            k = (r["method"], r["mode"], r.get("n"), r.get("halo"))
            if k in rec:
                continue
            rec[k] = r
            src[k] = f
    for k, r in rec.items():
        r["_src"] = src[k]
    return rec


def p4(rec):
    canvas_px = 2048
    array_mb = 50.3
    rows = []
    for m in sorted({k[0] for k in rec}):
        w = rec.get((m, "whole", 4, 48)) or next(
            (v for k, v in rec.items() if k[0] == m and k[1] == "whole"), None)
        if not w:
            continue
        for n in (4, 8):
            t = next((v for k, v in rec.items()
                      if k[0] == m and k[1] == "tiled" and k[2] == n), None)
            if not t:
                continue
            h = t.get("halo", 48)
            T = canvas_px // n
            N = n * n
            bound = (1.0 + 2.0 * h / T) ** 2
            ratio = t["filter_s"] / w["filter_s"]
            # c0 per TILE: the additive term is charged once per tile, so the
            # divisor is the tile count N = n^2, not the grid side n.
            c0 = max((t["filter_s"] - bound * w["filter_s"]) / N, 0.0)
            rows.append(dict(method=m, n=n, tiles=N, tile_side=T, halo=h,
                             whole_s=round(w["filter_s"], 1),
                             tiled_s=round(t["filter_s"], 1),
                             same_session=bool(w["_src"] == t["_src"]),
                             bound_mult=round(bound, 4),
                             measured_ratio=round(ratio, 4),
                             mult_bound_holds=bool(ratio <= bound),
                             c0_per_tile_s=round(c0, 4),
                             peak_whole_mb=w["peak_mb"],
                             peak_tiled_mb=t["peak_mb"],
                             mem_ratio=round(w["peak_mb"] / t["peak_mb"], 2),
                             array_mb=array_mb,
                             tile_working_set_mb=round((T + 2 * h) ** 2 * 3 * 8 / 1e6, 2)))
    return rows


def main():
    paths = sorted(glob.glob(IMG_GLOB))[:10]
    rad = 8
    res = {"protocol": f"{len(paths)} Met images @384, support rad={rad}",
           "p3_coherence": p3(paths, rad),
           "p4_cost": p4(load_mem())}
    with open(f"{OUT}/theory.json", "w") as fh:
        json.dump(res, fh, indent=1)

    a = np.array([r["frac_rank1_le_001"] for r in res["p3_coherence"]])
    b = np.array([r["frac_rank1_le_005"] for r in res["p3_coherence"]])
    c = np.array([r["frac_gt_050"] for r in res["p3_coherence"]])
    d = np.array([r["frac_exact_rank1"] for r in res["p3_coherence"]])
    print("# P3 — rank-1 relative error (= coherence lambda_min/lambda_max) per image")
    print("| image | coh<=1% | coh<=5% | coh>50% | median | p90 | exact rank-1 | active px |")
    print("|---|---|---|---|---|---|---|---|")
    for r in res["p3_coherence"]:
        print(f"| {r['img']} | {r['frac_rank1_le_001']*100:.1f} % | "
              f"{r['frac_rank1_le_005']*100:.1f} % | {r['frac_rank2_gt_050' if 'frac_rank2_gt_050' in r else 'frac_gt_050']*100:.1f} % | "
              f"{r['median_coh']:.3f} | {r['p90_coh']:.3f} | {r['frac_exact_rank1']*100:.1f} % | "
              f"{r['frac_active']*100:.1f} % |")
    print(f"\nrange over images: coh<=1% in {a.min()*100:.1f}–{a.max()*100:.1f} %, "
          f"coh<=5% in {b.min()*100:.1f}–{b.max()*100:.1f} %, "
          f"coh>50% in {c.min()*100:.1f}–{c.max()*100:.1f} %, "
          f"exact rank-1 in {d.min()*100:.1f}–{d.max()*100:.1f} %")

    print("\n# P4 — measured tiled/whole cost vs (1+2h/T)^2")
    print("| method | n | T | halo | bound | measured | c0/tile (s) | whole MB | tiled MB | ratio | tile WS MB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["p4_cost"]:
        print(f"| {r['method']} | {r['n']} | {r['tile_side']} | {r['halo']} | "
              f"{r['bound_mult']:.3f} | {r['measured_ratio']:.3f} | {r['c0_per_tile_s']:.3f} | "
              f"{r['peak_whole_mb']:.0f} | {r['peak_tiled_mb']:.0f} | {r['mem_ratio']:.1f} | "
              f"{r['tile_working_set_mb']:.1f} |")
    print(f"\nwrote {OUT}/theory.json")


if __name__ == "__main__":
    main()
