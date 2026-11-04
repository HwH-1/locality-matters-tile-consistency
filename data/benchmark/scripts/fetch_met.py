"""Fetch CC0 images from the Met Open Access API. Resume-safe, rate-limited."""
import json, os, sys, time, hashlib, csv, io
import urllib.request
from PIL import Image

API   = "https://collectionapi.metmuseum.org/public/collection/v1"
OUT   = "/data/abstrabench/images/met"
CACHE = "/data/abstrabench/cache"
CSV   = "/data/abstrabench/manifest_met.csv"
UA    = {"User-Agent": "AbstraBench/1.0 (academic benchmark; contact: research@example.org)"}
TARGET_PER_DEPT = {"European Paintings": 45, "Photographs": 55, "Drawings and Prints": 45, "Textiles": 25}
FIELDS = ["id","file","source","source_id","page_url","image_url","title","medium",
          "department","object_date","license","license_evidence","attribution",
          "orig_width","orig_height","width","height","bytes","sha1"]

def get(url, tries=4):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return r.read()
        except Exception as e:
            if t == tries-1: print("  FAIL", url, e, flush=True); return None
            time.sleep(1.5*(t+1))
    return None

def jget(url):
    b = get(url)
    return json.loads(b) if b else None

def save_image(url, oid, title, med):
    b = get(url)
    if not b: return None
    try:
        im = Image.open(io.BytesIO(b)); im.load()
    except Exception as e:
        print("  undecodable", oid, e, flush=True); return None
    w, h = im.size
    if max(w, h) < 768: return None
    im = im.convert("RGB")
    if max(w, h) > 1024:
        s = 1024.0/max(w, h); im = im.resize((max(1,int(round(w*s))), max(1,int(round(h*s)))), Image.LANCZOS)
    fn = f"met_{oid:07d}.jpg"
    p = os.path.join(OUT, fn)
    im.save(p, "JPEG", quality=92, optimize=True)
    return dict(file=os.path.relpath(p, "/data/abstrabench"), orig_width=w, orig_height=h,
                width=im.size[0], height=im.size[1], bytes=os.path.getsize(p),
                sha1=hashlib.sha1(open(p,"rb").read()).hexdigest())

def main():
    done = set()
    if os.path.exists(CSV):
        with open(CSV) as f:
            for row in csv.DictReader(f): done.add(row["source_id"])
    new = not os.path.exists(CSV)
    f = open(CSV, "a", newline="")
    wr = csv.DictWriter(f, fieldnames=FIELDS); new and wr.writeheader()

    depts = jget(API + "/departments")
    if not depts: sys.exit("cannot list departments")
    byname = {d["displayName"]: d["departmentId"] for d in depts["departments"]}
    print("departments:", {k: byname[k] for k in TARGET_PER_DEPT if k in byname}, flush=True)

    for dname, target in TARGET_PER_DEPT.items():
        did = byname.get(dname)
        if did is None: print("missing dept", dname, flush=True); continue
        ids = jget(f"{API}/objects?departmentIds={did}")
        if not ids: continue
        obj = ids.get("objectIDs") or []
        print(f"\n== {dname} (dept {did}): {len(obj)} objects, target {target}", flush=True)
        n = 0
        for oid in obj:
            if n >= target: break
            if str(oid) in done: n += 1; continue
            rec = jget(f"{API}/objects/{oid}"); time.sleep(0.15)
            if not rec or not rec.get("isPublicDomain") or not rec.get("primaryImage"):
                continue
            meta = save_image(rec["primaryImage"], oid, rec.get("title"), rec.get("medium"))
            if not meta: continue
            row = dict(id=f"met_{oid:07d}", source="met", source_id=str(oid),
                       page_url=rec.get("objectURL",""), image_url=rec["primaryImage"],
                       title=(rec.get("title") or "")[:120], medium=(rec.get("medium") or "")[:80],
                       department=rec.get("departmentTitle",""), object_date=rec.get("objectDate",""),
                       license="CC0",
                       license_evidence=f"isPublicDomain=true; rights={rec.get('rightsAndReproduction','')[:60]}",
                       attribution=f"The Metropolitan Museum of Art, CC0 ({rec.get('creditLine','')[:50]})",
                       **meta)
            wr.writerow(row); f.flush(); n += 1
            print(f"  [{n:3d}/{target}] {oid} {str(rec.get('title'))[:55]}", flush=True)
        print(f"== {dname}: {n} images", flush=True)
    f.close()

if __name__ == "__main__":
    main()
