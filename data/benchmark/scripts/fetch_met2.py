"""Fetch CC0 images from the Met Open Access API (parallel, resume-safe)."""
import json, os, sys, time, hashlib, csv, io
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

API   = "https://collectionapi.metmuseum.org/public/collection/v1"
OUT   = "/data/abstrabench/images/met"
CSV   = "/data/abstrabench/manifest_met.csv"
UA    = {"User-Agent": "AbstraBench/1.0 (academic benchmark; contact: research@example.org)"}
TARGET_PER_DEPT = {"Photographs": 55, "Drawings and Prints": 45, "Textiles": 25, "Islamic Art": 20}
WORKERS = 6
FIELDS = ["id","file","source","source_id","page_url","image_url","title","medium",
          "department","object_date","license","license_evidence","attribution",
          "orig_width","orig_height","width","height","bytes","sha1"]

def get(url, tries=4):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:
            if t == tries - 1:
                return None
            time.sleep(0.8 * (t + 1))
    return None

def meta(oid):
    b = get(f"{API}/objects/{oid}")
    if not b:
        return None
    try:
        rec = json.loads(b)
    except Exception:
        return None
    if not rec.get("isPublicDomain") or not rec.get("primaryImage"):
        return None
    return rec

def render(rec):
    url = rec["primaryImage"]
    oid = int(rec["objectID"])
    b = get(url)
    if not b:
        return None
    try:
        im = Image.open(io.BytesIO(b)); im.load()
    except Exception:
        return None
    w, h = im.size
    if max(w, h) < 768:
        return None
    im = im.convert("RGB")
    if max(w, h) > 1024:
        s = 1024.0 / max(w, h)
        im = im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.LANCZOS)
    p = os.path.join(OUT, f"met_{oid:07d}.jpg")
    im.save(p, "JPEG", quality=92, optimize=True)
    data = open(p, "rb").read()
    return dict(file=os.path.relpath(p, "/data/abstrabench"), orig_width=w, orig_height=h,
                width=im.size[0], height=im.size[1], bytes=len(data),
                sha1=hashlib.sha1(data).hexdigest())

def row(rec, m):
    oid = int(rec["objectID"])
    return dict(id=f"met_{oid:07d}", source="met", source_id=str(oid),
                page_url=rec.get("objectURL", ""), image_url=rec["primaryImage"],
                title=(rec.get("title") or "")[:120], medium=(rec.get("medium") or "")[:80],
                department=rec.get("department", ""), object_date=rec.get("objectDate", ""),
                license="CC0",
                license_evidence=f"isPublicDomain=true; rights={str(rec.get('rightsAndReproduction',''))[:60]}",
                attribution=f"The Metropolitan Museum of Art, CC0 ({str(rec.get('creditLine',''))[:50]})",
                **m)

def main():
    done = set()
    if os.path.exists(CSV):
        with open(CSV) as f:
            for r in csv.DictReader(f):
                done.add(r["source_id"])
    print(f"resume: {len(done)} already collected", flush=True)
    new = not os.path.exists(CSV)
    f = open(CSV, "a", newline="")
    wr = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        wr.writeheader()

    depts = json.loads(get(API + "/departments") or b"{}")
    byname = {d["displayName"]: d["departmentId"] for d in depts.get("departments", [])}
    pool = ThreadPoolExecutor(WORKERS)
    for dname, target in TARGET_PER_DEPT.items():
        did = byname.get(dname)
        if did is None:
            print("missing dept", dname, flush=True); continue
        ids = json.loads(get(f"{API}/objects?departmentIds={did}") or b"{}")
        obj = [str(i) for i in (ids.get("objectIDs") or []) if str(i) not in done]
        print(f"\n== {dname} (dept {did}): {len(obj)} candidates, target {target}", flush=True)
        n = got = 0
        CH = 48
        while n < target and obj:
            chunk, obj = obj[:CH], obj[CH:]
            recs = [r for r in pool.map(meta, chunk) if r]
            todo = recs[: target - n]
            for rec, m in zip(todo, pool.map(render, todo)):
                got += 1
                if m is None:
                    continue
                wr.writerow(row(rec, m)); f.flush(); n += 1
                print(f"  [{n:3d}/{target}] {rec['objectID']} {str(rec.get('title'))[:52]}", flush=True)
        print(f"== {dname}: {n} images kept", flush=True)
    f.close()

if __name__ == "__main__":
    main()
