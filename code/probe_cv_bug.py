"""Diagnostic probe: why OpenCV's Perona-Malik diffusion is not tile-consistent.

run_cv.py measures the halo ladder of seven published third-party implementations.
One of them -- cv::ximgproc::anisotropicDiffusion (Perona & Malik, TPAMI 1991), a
purely local 8-neighbour stencil -- does not converge to the whole-image result at
ANY halo, which is the class-B signature, even though Proposition 1 says a bounded
support should give exactness.  This script isolates why, in four controlled tests,
so the claim in Section 6.9 rests on a reproducer rather than on a hunch.

The explanation the source gives (opencv_contrib modules/ximgproc/src/anisodiff.cpp):
the iteration loop does

    copyMakeBorder(t == 0 ? src0 : src, temp0x, 1, 1, 1, 1, BORDER_REPLICATE);

where, for t > 0, `src` is `temp0` -- a Rect view INTO temp0x (lines 276-277, 282,
287-288 of the 5.0.0 tag).  From the second iteration on, the border-replication
step reads from and writes into the same buffer, so the intermediate state depends
on how the copy walks the memory, i.e. on the array's geometry.  Consequences this
script checks:

  T1  iterations = 1 must be bit-exact tile-wise at a whole-image halo,
      iterations >= 2 must not be (the self-overlap needs one swap to appear);
  T2  the deviation must NOT decrease with the halo -- it is not truncation;
  T3  the same pixels, padded to a different row stride, must render differently
      (geometry, not content);
  T4  the effect must survive cv2.setUseOptimized(False), i.e. it is not a SIMD
      artefact that a build flag would excuse.

Writes out/cv_bug.jsonl (one record per test/step).  Cost: well under a minute.
"""
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, "/data/tileconsistency/src")
import cv2
cv2.setNumThreads(1)
import cv2.ximgproc as X

L1 = 1.0 / 255.0


def tiled(fn, img, n, halo, contiguous=False):
    """tc.tiled(), repeated here with an optional ascontiguousarray() so that tile
    contiguity can be tested as a variable rather than assumed away."""
    h, w = img.shape[:2]
    th, tw = h // n, w // n
    out = np.empty_like(img)
    for i in range(n):
        for j in range(n):
            y0, x0 = i * th, j * tw
            y1 = h if i == n - 1 else y0 + th
            x1 = w if j == n - 1 else x0 + tw
            top, bot = min(halo, y0), min(halo, h - y1)
            lef, rig = min(halo, x0), min(halo, w - x1)
            sub = img[y0 - top:y1 + bot, x0 - lef:x1 + rig]
            if contiguous:
                sub = np.ascontiguousarray(sub)
            r = fn(sub)
            out[y0:y1, x0:x1] = r[top:top + (y1 - y0), lef:lef + (x1 - x0)]
    return out


def diff_stats(whole, part):
    d = np.abs(whole.astype(np.int16) - part.astype(np.int16))
    dl = d.max(axis=2)
    ys, xs = np.nonzero(dl > 0)
    return dict(ndiff=int(dl.size and (dl > 0).sum() or 0),
                maxd=int(dl.max()),
                frac=float((dl > 0).mean()),
                frac1=float((dl > 1).mean()),
                row_of_first_diff=int(ys[0]) if len(ys) else None,
                n_rows_with_diff=int(len(set(ys.tolist()))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", default="/data/abstrabench/images/met/met_0100017.jpg")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="/data/tileconsistency/out/cv_bug.jsonl")
    a = ap.parse_args()

    u8 = cv2.imread(a.img, cv2.IMREAD_COLOR)
    assert u8 is not None, a.img
    h, w = u8.shape[:2]
    recs = []

    def emit(rec):
        recs.append(rec)
        print(json.dumps(rec), flush=True)

    emit(dict(test="meta", img=os.path.basename(a.img), height=h, width=w,
              n=a.n, opencv=cv2.__version__,
              src="opencv_contrib-5.0.0/modules/ximgproc/src/anisodiff.cpp:276-288"))

    # ---- T1 / T2: iterations vs halo ladder -------------------------------------
    for it in (1, 2, 3, 5, 10, 20, 40):
        fn = (lambda x, it=it: X.anisotropicDiffusion(x, 0.5, 0.2, it))
        whole = fn(u8)
        for halo in (0, 2, 4, 8, 16, 24, 32, 48, 96, max(h, w)):
            st = diff_stats(whole, tiled(fn, u8, a.n, halo))
            emit(dict(test="ladder", iterations=it, halo=halo, **st))

    # ---- T3: identical content, different memory geometry -----------------------
    # A naive "pad the image with black and compare" test is confounded, because the
    # padded array has a different *border*, and a diffusion stencil legitimately
    # reads its neighbourhood.  Two confound-free versions:
    #   (a) contiguity -- same values, same shape, different row step: a view into a
    #       wider buffer vs a contiguous array;
    #   (b) replication -- extend the array to the right by replicating its last
    #       column, which preserves the border-replicated values exactly, and compare
    #       only columns farther from the new right edge than the stencil can reach.
    for it in (1, 2, 5, 20, 40):
        fn = (lambda x, it=it: X.anisotropicDiffusion(x, 0.5, 0.2, it))
        big = np.zeros((h, w + 64, 3), np.uint8)
        big[:, :w] = u8
        view = big[:, :w]                       # same values, step = (w+64)*3
        st = diff_stats(fn(u8), fn(view))
        emit(dict(test="contiguity", iterations=it,
                  step_a=w * 3, step_b=(w + 64) * 3, **st))

        ext = np.concatenate([u8, np.repeat(u8[:, -1:], 64, axis=1)], axis=1)
        margin = it + 2                          # stencil reach from the new border
        o_a = fn(u8)[:, :max(w - margin, 1)]
        o_b = fn(ext)[:, :max(w - margin, 1)]
        st = diff_stats(o_a, o_b)
        emit(dict(test="replication", iterations=it, margin_excluded=margin,
                  compare_cols=int(o_a.shape[1]), **st))

    # ---- T4: SIMD off -----------------------------------------------------------
    for opt in (True, False):
        cv2.setUseOptimized(opt)
        fn = (lambda x: X.anisotropicDiffusion(x, 0.5, 0.2, 2))
        st = diff_stats(fn(u8), tiled(fn, u8, a.n, max(h, w)))
        emit(dict(test="optimized", set_use_optimized=bool(opt), iterations=2,
                  halo=max(h, w), **st))
    cv2.setUseOptimized(True)

    # ---- control: a local method without the self-overlap, same harness ---------
    fn_bil = lambda x: cv2.bilateralFilter(x, 9, 75, 75)
    for halo in (4, 8, 48):
        emit(dict(test="control_bilateral", halo=halo,
                  **diff_stats(fn_bil(u8), tiled(fn_bil, u8, a.n, halo))))
    big = np.zeros((h, w + 64, 3), np.uint8)
    big[:, :w] = u8
    emit(dict(test="control_bilateral", what="contiguity",
              **diff_stats(fn_bil(u8), fn_bil(big[:, :w]))))
    ext = np.concatenate([u8, np.repeat(u8[:, -1:], 64, axis=1)], axis=1)
    emit(dict(test="control_bilateral", what="replication", margin_excluded=7,
              **diff_stats(fn_bil(u8)[:, :w - 7], fn_bil(ext)[:, :w - 7])))

    # ---- T5: identical content, freshly allocated buffers -----------------------
    # If the output were a function of pixel values, four independent copies of the
    # same array would render identically.  They do not (OpenCV 5.0.0, this host).
    outs = []
    for k in range(4):
        c = np.array(u8, copy=True)
        outs.append((X.anisotropicDiffusion(c, 0.5, 0.2, 5), c.ctypes.data % 64))
    base = outs[0][0]
    for o, align in outs:
        st = diff_stats(base, o)
        emit(dict(test="fresh_copies", iterations=5, ptr_mod_64=int(align), **st))
    emit(dict(test="fresh_copies_control_bilateral",
              **diff_stats(cv2.bilateralFilter(np.array(u8, copy=True), 9, 75, 75),
                           cv2.bilateralFilter(np.array(u8, copy=True), 9, 75, 75))))

    # ---- T6: does the deviation need real image content at all? -----------------
    # Two-block synthetic images, tiled n=4 at halo 32 (support at 5 iterations is
    # 5 px, so truncation cannot explain anything at this halo).  `hard 0/255` can
    # index the conductance table at its maximum; `hard 1/254` cannot.
    rng = np.random.default_rng(7)

    def _hard(lo, hi):
        a = np.zeros((256, 256, 3), np.uint8)
        a[:, 128:] = hi
        return a

    def _noisy(lo, hi):
        a = _hard(lo, hi)
        n = rng.integers(lo, hi + 1, (256, 256, 3)).astype(np.uint8)
        return np.where(rng.random((256, 256, 1)) < 0.3, n, a)

    for tag, arr in (("hard 0/255", _hard(0, 255)), ("hard 1/254", _hard(1, 254)),
                     ("noisy 0/255", _noisy(0, 255)), ("noisy 1/254", _noisy(1, 254))):
        for it in (5, 20):
            fn = (lambda x, it=it: X.anisotropicDiffusion(x, 0.5, 0.2, it))
            emit(dict(test="synthetic", arr=tag, iterations=it, halo=32,
                      **diff_stats(fn(arr), tiled(fn, arr, 4, 32))))
            fnb = lambda x: cv2.bilateralFilter(x, 9, 75, 75)
            emit(dict(test="synthetic_control_bilateral", arr=tag, halo=32,
                      **diff_stats(fnb(arr), tiled(fnb, arr, 4, 32))))

    with open(a.out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(recs)} records -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
