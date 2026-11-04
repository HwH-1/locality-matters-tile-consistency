"""Rebuild the two native-resolution test images and their provenance record.

The large-canvas experiment of Section 6.7 needs pixel content at *native* density:
the served Met files are normalised to a longest side of 1024 px, so a 4000-px
artwork down-sampled to 1024 px is not a large image, it is a small one with a large
number of pixels.  Two CC0 originals were therefore retrieved at full size and are
kept in abstrabench/images_orig; this script derives from them, deterministically:

  <stem>_full.png      lossless PNG of the decoded JPEG (verified byte-equal to
                       cv2.imread(orig), i.e. no resampling anywhere)
  <stem>_crop2048.png  centred 2048x2048 window of the same decoded pixels

and rewrites abstrabench/images_native/provenance.json so that every file the paper
uses carries its own SHA-1: the retrieved JPEG, the derived full PNG, the derived
crop, plus the crop offset and the shape.  `--verify` recomputes all of it into a
temporary directory and fails loudly if any digest moves.

Usage:
  python3 make_native.py            # write PNGs + provenance.json
  python3 make_native.py --verify   # rebuild elsewhere and compare digests
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ORIG = "/data/abstrabench/images_orig"
NATIVE = "/data/abstrabench/images_native"
STEMS = ("met_0436104", "met_0436140")
CROP = 2048


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def derive(stem, dst_dir):
    """Return (full_path, crop_path, offset) having written both PNGs in dst_dir."""
    src = os.path.join(ORIG, f"{stem}.jpg")
    px = cv2.imread(src, cv2.IMREAD_COLOR)
    assert px is not None, src
    h, w = px.shape[:2]
    assert h >= CROP and w >= CROP, (stem, px.shape)
    y, x = (h - CROP) // 2, (w - CROP) // 2
    full = os.path.join(dst_dir, f"{stem}_full.png")
    crop = os.path.join(dst_dir, f"{stem}_crop2048.png")
    assert cv2.imwrite(full, px), full
    assert cv2.imwrite(crop, px[y:y + CROP, x:x + CROP]), crop
    # the saved PNG must decode to exactly the pixels the JPEG decoder produced
    assert np.array_equal(cv2.imread(full, cv2.IMREAD_COLOR), px), full
    assert np.array_equal(cv2.imread(crop, cv2.IMREAD_COLOR), px[y:y + CROP, x:x + CROP]), crop
    return full, crop, (int(y), int(x)), (int(h), int(w))


def build():
    rec = []
    for stem in STEMS:
        full, crop, off, shape = derive(stem, NATIVE)
        rec.append(dict(
            src=f"images_orig/{stem}.jpg",
            src_sha1=sha1(os.path.join(ORIG, f"{stem}.jpg")),
            full=f"images_native/{stem}_full.png",
            full_sha1=sha1(full),
            crop=f"images_native/{stem}_crop2048.png",
            crop_sha1=sha1(crop),
            crop_offset_yx=list(off),
            crop_size=[CROP, CROP],
            decoded_shape=[shape[0], shape[1], 3],
            mb=round(os.path.getsize(full) / 1e6, 1),
            derivation="cv2.imread(orig JPEG, IMREAD_COLOR) -> lossless PNG; "
                       "crop = centred 2048x2048 window of those exact pixels",
            licence="Metropolitan Museum of Art Open Access, CC0",
        ))
    with open(os.path.join(NATIVE, "provenance.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    for r in rec:
        print(f"{r['src']}: src {r['src_sha1'][:12]} full {r['full_sha1'][:12]} "
              f"crop {r['crop_sha1'][:12]} offset={r['crop_offset_yx']}")
    print(f"-> {os.path.join(NATIVE, 'provenance.json')}")


def verify():
    prev = {r["src"]: r for r in json.load(open(os.path.join(NATIVE, "provenance.json")))}
    tmp = tempfile.mkdtemp(prefix="native_verify_")
    bad = []
    for stem in STEMS:
        full, crop, off, shape = derive(stem, tmp)
        key = f"images_orig/{stem}.jpg"
        p = prev.get(key)
        if p is None:
            bad.append(f"{key}: not in provenance.json")
            continue
        for fld, path in (("full_sha1", full), ("crop_sha1", crop)):
            if p[fld] != sha1(path):
                bad.append(f"{key} {fld}: recorded {p[fld][:12]} rebuilt {sha1(path)[:12]}")
        if p["crop_offset_yx"] != [off[0], off[1]]:
            bad.append(f"{key}: crop offset moved {p['crop_offset_yx']} -> {off}")
    print("digests reproduce exactly" if not bad else "MISMATCH:\n  " + "\n  ".join(bad))
    subprocess.run(["rm", "-rf", tmp])
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    sys.exit(verify() if a.verify else (build() or 0))
