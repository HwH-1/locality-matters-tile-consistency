"""Add real-world strata to AbstraBench through the Met's official API.

Why this script exists: the v1 `/objects?departmentIds=` route used by
fetch_met2.py returns EVERY object in a department, so finding public-domain
objects with images means one metadata request per candidate (39,583 for
Photographs).  That hammered the API into a Cloudflare 403 on 2026-11-03.  The
v1.1 search endpoint (v1 /search was retired 2026-10-01) filters
`hasImages=true&isPublicArt=true` server-side, so a candidate list is already
license-clean before a single object is opened.

Acquisition is deliberately non-optional-looking: the selection rule is "the
first `target` objects of the API's default ordering for this department that
survive the resolution floor", recorded here so a reviewer can re-run it and get
the same ids.  Requests are serialised with a fixed spacing and a contact UA.

Usage:
  python3 scripts/fetch_strata.py --dept 19 --target 45
"""
import argparse
import csv
import hashlib
import io
import json
import os
import time
import urllib.request
from PIL import Image

API = "https://collectionapi.metmuseum.org/public/collection/v1"
S11 = "https://collectionapi.metmuseum.org/public/collection/v1.1/search"
ROOT = "/data/abstrabench"
OUTDIR = f"{ROOT}/images/met"
CSV_PATH = f"{ROOT}/manifest_met.csv"
UA = {"User-Agent": "AbstraBench/1.0 (academic benchmark; contact: research@example.org)"}
SPACING = 0.35          # seconds between requests -- politeness, not speed
FIELDS = ["id", "file", "source", "source_id", "page_url", "image_url", "title", "medium",
          "department", "object_date", "license", "license_evidence", "attribution",
          "orig_width", "orig_height", "width", "height", "bytes", "sha1"]
MIN_LONGEST = 768
MAX_LONGEST = 1024


def get(url, tries=3):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                         timeout=60) as r:
                time.sleep(SPACING)
                return r.read()
        except Exception as exc:
            if t == tries - 1:
                print(f"    give up {url}: {type(exc).__name__}", flush=True)
                return None
            time.sleep(2.0 * (t + 1))
    return None


def search_ids(dept, page=250, cap=4000):
    ids, offset = [], 0
    while len(ids) < cap:
        time.sleep(SPACING)
        b = get(f"{S11}?departmentId={dept}&hasImages=true&isPublicArt=true&q=*&limit={page}&offset={offset}")
        if not b:
            break
        d = json.loads(b)
        new = d.get("objectIDs") or []
        if not new:
            break
        ids += [str(i) for i in new]
        if len(ids) >= int(d.get("total") or 0):
            break
        offset += page
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dept", type=int, required=True)
    ap.add_argument("--target", type=int, default=45)
    ap.add_argument("--min-file-kb", type=int, default=40)
    a = ap.parse_args()

    done = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH) as fh:
            for r in csv.DictReader(fh):
                done.add(r["source_id"])
    new = not os.path.exists(CSV_PATH)
    fh = open(CSV_PATH, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        wr.writeheader()

    cands = [i for i in search_ids(a.dept) if i not in done]
    print(f"dept {a.dept}: {len(cands)} public-domain candidates with images "
          f"({len(done)} objects already collected)", flush=True)
    got = scanned = 0
    for oid in cands:
        if got >= a.target:
            break
        scanned += 1
        b = get(f"{API}/objects/{oid}")
        if not b:
            continue
        rec = json.loads(b)
        if not rec.get("isPublicDomain") or not rec.get("primaryImage"):
            continue
        data = get(rec["primaryImage"])
        if not data or len(data) < a.min_file_kb * 1024:
            continue
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
        except Exception:
            continue
        w, h = im.size
        if max(w, h) < MIN_LONGEST:
            continue
        im = im.convert("RGB")
        if max(w, h) > MAX_LONGEST:
            s = MAX_LONGEST / max(w, h)
            im = im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))),
                           Image.LANCZOS)
        path = f"{OUTDIR}/met_{int(oid):07d}.jpg"
        im.save(path, "JPEG", quality=92, optimize=True)
        raw = open(path, "rb").read()
        wr.writerow(dict(
            id=f"met_{int(oid):07d}", file=os.path.relpath(path, ROOT), source="met",
            source_id=str(oid),
            page_url=rec.get("objectURL") or f"{API}/objects/{oid}",
            image_url=rec["primaryImage"], title=(rec.get("title") or "")[:120],
            medium=(rec.get("medium") or "")[:80], department=rec.get("department", ""),
            object_date=(rec.get("objectDate") or "")[:40], license="CC0",
            license_evidence="isPublicDomain=true via "
                             "v1.1 search(hasImages=true,isPublicArt=true)",
            attribution=f"The Metropolitan Museum of Art, CC0 "
                        f"({(rec.get('creditLine') or '')[:50]})",
            orig_width=w, orig_height=h, width=im.size[0], height=im.size[1],
            bytes=len(raw), sha1=hashlib.sha1(raw).hexdigest()))
        fh.flush()
        got += 1
        if got % 10 == 0 or got == a.target:
            print(f"  [{got:3d}/{a.target}] {oid} {(rec.get('title') or '')[:50]}",
                  flush=True)
    fh.close()
    print(f"dept {a.dept}: kept {got} images after scanning {scanned} candidates",
          flush=True)


if __name__ == "__main__":
    main()
