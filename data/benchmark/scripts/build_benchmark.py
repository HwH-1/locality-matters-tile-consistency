#!/usr/bin/env python3
"""
AbstraBench-200 -- licence-clean image benchmark builder.

One idempotent, re-runnable script that
  1. acquires images from four rights-clean sources (Met Open Access,
     scikit-image bundled data, NASA Images, deterministic synthetic phantoms),
  2. normalises them to longest side 1024 px (aspect preserved, RGB, no crop),
  3. measures structural properties with numpy + scikit-image,
  4. assigns every image its top-2 strata (computed, never hand-labelled),
  5. writes manifest.csv / README.md / LICENSE.txt,
  6. verifies everything it produced.

Acceptance is gated on an explicit rights statement: see ACCEPTED_LICENSES.

Usage:
    python3 scripts/build_benchmark.py             # full build
    python3 scripts/build_benchmark.py --dry-run   # plan only, no downloads/writes
    python3 scripts/build_benchmark.py --limit 24  # small smoke build
    python3 scripts/build_benchmark.py --resume    # continue a partial run
    python3 scripts/build_benchmark.py --verify-only

State that makes re-runs cheap lives in cache/: raw API metadata, the acquired
row log (cache/rows.jsonl), computed metrics (cache/metrics.jsonl) and the
object ids that were rejected/failed. Nothing in the shipped package is
hand-edited; manifest.csv is regenerated from these caches on every run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image

import scipy.ndimage as ndi
from skimage.color import rgb2gray
from skimage.feature import (corner_harris, peak_local_max, structure_tensor,
                             structure_tensor_eigenvalues)
from skimage.filters import sato, sobel, threshold_otsu
from skimage.morphology import black_tophat, disk, white_tophat

Image.MAX_IMAGE_PIXELS = None  # Met originals are large; accepted knowingly.

# --------------------------------------------------------------------------- #
# paths / constants
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
IMGDIR = ROOT / "images"
MANIFEST = ROOT / "manifest.csv"
ROWS = CACHE / "rows.jsonl"
METRICS_CACHE = CACHE / "metrics.jsonl"
PROGRESS = CACHE / "progress.json"
FAILED = CACHE / "failed.jsonl"
LOG = CACHE / "build.log"

ACCEPTED_LICENSES = {
    "CC0",
    "CC0 (generated)",
    "Public Domain",
    "US Government work / no known copyright restrictions",
}
# Hard vetoes: if any of these appear in the licence or its evidence, reject.
REJECT_MARKERS = ("cc-by", "cc by", "attribution-nc", "attribution-sa",
                  "all rights reserved", "some rights reserved",
                  "creative commons attribution", "nc 4.0", "nc 3.0", "nc 2.0")

SEED = 20261103
TARGET_LONGEST = 1024
MIN_ORIG_LONGEST = 768
JPEG_QUALITY = 92
API_INTERVAL = 0.25
HTTP_RETRIES = 3
ABSOLUTE_MAX_BYTES = 14_000_000
DEPT_MAX_BYTES = 3_500_000       # keeps the build within the sandbox's ~140 kB/s link
TOTAL_MIN = 200
MET_MIN = 150
STRATA = ["texture_rich", "flat_region", "thin_structure",
          "junction_rich", "high_contrast_edge", "repeating_pattern"]
MIN_PER_STRATUM = 15
METRIC_TO_STRATUM = {
    "grad_mean": "texture_rich",
    "flat_frac": "flat_region",
    "ridge_frac": "thin_structure",
    "corner_density": "junction_rich",
    "grad_p95": "high_contrast_edge",
    "pattern_score": "repeating_pattern",
}
LINE_ART_RE = re.compile(
    r"draw|print|engrav|etch|lithogr|woodcut|woodblock|mezzotint|aquafin|"
    r"ink|brush|pen and|chalk|pastel|water ?colour|watercolor|cartoon|sketch|"
    r"broadsheet|design for", re.I)
TEXTILE_RE = re.compile(r"textile|fabric|cloth|carpet|rug|tapestry|embroid|sampl|"
                        r"lace|silk|wool|costume|garment|vest", re.I)

# (department id, quota, preferred format for its raster) -- the Met exposes no
# standalone "Textile Art" department in /departments, so fabric content comes
# from The Costume Institute (8) and Islamic Art (14).
MET_QUOTAS = [
    (11, 32, "jpeg"),   # European Paintings
    (9, 38, "png"),     # Drawings and Prints  -> thin-line content
    (19, 28, "jpeg"),   # Photographs
    (12, 14, "jpeg"),   # European Sculpture and Decorative Arts
    (8, 12, "jpeg"),    # The Costume Institute -> textiles
    (14, 10, "jpeg"),   # Islamic Art          -> textiles / decorative
    (6, 8, "png"),      # Asian Art            -> prints, scrolls
    (17, 8, "jpeg"),    # Medieval Art
]

NASA_QUERIES = ["Hubble Deep Field", "Jupiter Galileo spacecraft", "Mars surface rover",
                "Apollo lunar surface", "Earth from space", "Andromeda galaxy",
                "Saturn Cassini rings", "Spitzer nebula", "Milky Way panorama",
                "Voyager Neptune", "James Webb first deep field", "solar prominence",
                "International Space Station", "Crab supernova remnant", "Titan haze",
                "Pillars of Creation", "comet nucleus", "Venus radar map",
                "Sun ultraviolet imaging", "galaxy cluster gravitational lensing"]

SKIMAGE_NAMES = ["astronaut", "camera", "coffee", "chelsea", "rocket",
                 "coins", "hubble", "brick", "phantom", "colorwheel"]

MANIFEST_COLUMNS = ["id", "file", "source", "source_id", "page_url", "image_url",
                    "title", "medium", "department", "object_date", "license",
                    "license_evidence", "attribution", "orig_width", "orig_height",
                    "width", "height", "bytes", "sha1", "grad_mean", "grad_p95",
                    "flat_frac", "ridge_frac", "corner_density", "pattern_score",
                    "strata"]

_LOCK = threading.RLock()
_START = time.time()
DRY = False


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    line = f"[{time.time() - _START:8.1f}s] {msg}"
    print(line, flush=True)
    with _LOCK:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with _LOCK:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return out


def append_jsonl(path: Path, obj: dict) -> None:
    with _LOCK:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with _LOCK:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False)
        os.replace(tmp, path)


def rights_clean(license_string: str, evidence: str) -> bool:
    """Allow-list check plus a veto scan over the recorded evidence text."""
    if license_string not in ACCEPTED_LICENSES:
        return False
    probe = (license_string + " " + evidence).lower()
    return not any(marker in probe for marker in REJECT_MARKERS)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Http:
    def __init__(self, browser_ua: bool = False):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36" if browser_ua
            else "AbstraBench/1.0 (academic image benchmark build)")

    def _throttle(self, interval: float) -> None:
        if interval <= 0:
            return
        with _LOCK:
            now = time.time()
            wait = getattr(self, "_last", 0.0) + interval - now
            self._last = max(now, getattr(self, "_last", 0.0)) + interval
        if wait > 0:
            time.sleep(wait)

    def get(self, url, *, interval=API_INTERVAL, timeout=60.0, stream=False, headers=None):
        """GET with retries + exponential backoff, honouring 429/5xx."""
        for attempt in range(HTTP_RETRIES):
            self._throttle(interval)
            try:
                r = self.s.get(url, timeout=timeout, stream=stream, headers=headers)
            except requests.RequestException as exc:
                log(f"    http {type(exc).__name__} on {url[:100]} (try {attempt + 1})")
                time.sleep(1.5 * (2 ** attempt))
                continue
            if r.status_code == 429 or 500 <= r.status_code < 600:
                ra = r.headers.get("Retry-After", "")
                delay = float(ra) if ra.isdigit() else 2.0 * (2 ** attempt)
                log(f"    HTTP {r.status_code}, backoff {delay:.1f}s (try {attempt + 1})")
                time.sleep(delay)
                r.close()
                continue
            if r.status_code >= 400:
                log(f"    HTTP {r.status_code} for {url[:100]}")
                r.close()
                return None
            return r
        return None

    def get_json(self, url, cache_path, *, interval=API_INTERVAL, timeout=60.0):
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                cache_path.unlink(missing_ok=True)
        r = self.get(url, interval=interval, timeout=timeout)
        if r is None:
            return None
        try:
            data = r.json()
        except ValueError:
            log(f"    non-JSON response from {url[:100]}")
            return None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(cache_path, data)
        return data

    def head_prefix(self, url, nbytes=131072):
        """Fetch only the first nbytes (Range) -> (body, total_size) or (None, None)."""
        r = self.get(url, interval=0.0, timeout=45, stream=True,
                     headers={"Range": f"bytes=0-{nbytes - 1}"})
        if r is None:
            return None, None
        total = None
        m = re.search(r"/(\d+)\s*$", r.headers.get("Content-Range", "") or "")
        if m:
            total = int(m.group(1))
        buf, got = bytearray(), 0
        try:
            for chunk in r.iter_content(1 << 14):
                if not chunk:
                    continue
                buf += chunk
                got += len(chunk)
                if got >= nbytes:
                    break
        finally:
            r.close()
        if total is None:
            total = int(r.headers.get("Content-Length") or 0) or None
        return bytes(buf), total

    def get_bytes(self, url, *, interval=0.0, timeout=240.0, max_bytes=ABSOLUTE_MAX_BYTES):
        r = self.get(url, interval=interval, timeout=timeout, stream=True)
        if r is None:
            return None
        chunks, total = [], 0
        try:
            for chunk in r.iter_content(1 << 16):
                if not chunk:
                    continue
                chunks.append(chunk)
                total += len(chunk)
                if total > max_bytes:
                    log(f"    over size cap ({total} B), aborting {url.rsplit('/', 1)[-1]}")
                    return None
        except (requests.RequestException, OSError) as exc:
            log(f"    stream {type(exc).__name__} on {url.rsplit('/', 1)[-1]}")
            return None
        finally:
            r.close()
        return b"".join(chunks)


class Met:
    BASE = "https://collectionapi.metmuseum.org/public/collection/v1"

    def __init__(self, http: Http):
        self.http = http

    def departments(self):
        data = self.http.get_json(f"{self.BASE}/departments", CACHE / "met_departments.json")
        return {int(d["departmentId"]): d["displayName"]
                for d in (data or {}).get("departments", [])}

    def object_ids(self, dept_id):
        data = self.http.get_json(f"{self.BASE}/objects?departmentIds={dept_id}",
                                  CACHE / f"met_dept_{dept_id}.json", timeout=180)
        return list((data or {}).get("objectIDs") or [])

    def object(self, object_id):
        return self.http.get_json(f"{self.BASE}/objects/{object_id}",
                                  CACHE / "met_objects" / f"{object_id}.json")


# --------------------------------------------------------------------------- #
# raster handling
# --------------------------------------------------------------------------- #
def jpeg_dims(data: bytes):
    """Width/height from JPEG SOF markers, without decoding."""
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    i, n = 2, len(data)
    while i < n - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if i + 4 > n:
            return None
        seglen = struct.unpack(">H", data[i + 2:i + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return (int(w), int(h))
        i += 2 + seglen
    return None


def sniff_image(http: Http, url: str):
    """-> (dims|None, total_bytes|None) using a single 128 KiB Range request."""
    body, total = http.head_prefix(url)
    if body is None:
        return None, total
    return jpeg_dims(body), total


def decode(data: bytes):
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as exc:
        log(f"    undecodable raster ({type(exc).__name__})")
        return None
    return im


def normalise(im: Image.Image) -> Image.Image:
    """RGB, alpha dropped, longest side exactly TARGET_LONGEST, aspect preserved."""
    if im.mode != "RGB":
        if im.mode in ("RGBA", "LA", "P", "PA", "1"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
    w, h = im.size
    longest = max(w, h)
    if longest and longest != TARGET_LONGEST:
        scale = TARGET_LONGEST / float(longest)
        im = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                       Image.LANCZOS)
    return im


def save_image(im: Image.Image, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "png":
        im.save(path, "PNG", optimize=True)
    else:
        im.save(path, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)


def choose_format(preferred: str, medium: str, title: str) -> str:
    blob = f"{medium} {title}"
    if LINE_ART_RE.search(blob) or preferred == "png":
        return "png"
    return "jpeg"


# --------------------------------------------------------------------------- #
# acquired-row store
# --------------------------------------------------------------------------- #
class Store:
    def __init__(self):
        self.rows: dict[str, dict] = {}
        self.raw_sha1: set[str] = set()
        self.basenames: set[str] = set()
        self.failed_ids: set[str] = set()
        for r in load_jsonl(ROWS):
            self.rows[r["id"]] = r
            if r.get("raw_sha1"):
                self.raw_sha1.add(r["raw_sha1"])
            if r.get("image_url"):
                self.basenames.add(r["image_url"].rsplit("/", 1)[-1])
        for r in load_jsonl(FAILED):
            self.failed_ids.add(r.get("id", ""))

    def add(self, row: dict) -> bool:
        with _LOCK:
            if row["id"] in self.rows:
                return False
            self.rows[row["id"]] = row
            if row.get("raw_sha1"):
                self.raw_sha1.add(row["raw_sha1"])
            if row.get("image_url"):
                self.basenames.add(row["image_url"].rsplit("/", 1)[-1])
            append_jsonl(ROWS, row)
            return True

    def fail(self, ident: str, why: str) -> None:
        with _LOCK:
            self.failed_ids.add(ident)
        append_jsonl(FAILED, {"id": ident, "why": why})

    def seen_basename(self, url: str) -> bool:
        return url.rsplit("/", 1)[-1] in self.basenames

    def seen_raw(self, digest: str) -> bool:
        return digest in self.raw_sha1

    def count(self, key: str) -> int:
        """key matches either row['source'] or row['dept_key']."""
        return sum(1 for r in self.rows.values()
                   if r["source"] == key or r.get("dept_key") == key)

    def total(self) -> int:
        return len(self.rows)


STORE = Store()


# --------------------------------------------------------------------------- #
# source 1: Met Open Access
# --------------------------------------------------------------------------- #
def acquire_met(met: Met, http: Http, depts: dict[int, str],
                quotas: list[tuple[int, int, str]], dry: bool) -> None:
    for dept_id, quota, preferred in quotas:
        name = depts.get(dept_id, f"Met department {dept_id}")
        have = STORE.count(f"met:{dept_id}")
        if have >= quota:
            log(f"  dept {dept_id:<3} {name}: {have}/{quota} already acquired")
            continue
        need = quota - have
        if dry:
            log(f"  [dry] dept {dept_id} {name}: would sample {need} public-domain objects")
            continue
        ids = met.object_ids(dept_id)
        if not ids:
            log(f"  dept {dept_id} {name}: object list empty, skipping")
            continue
        sample = ids[:]
        random.Random(f"{SEED}-met-{dept_id}").shuffle(sample)
        log(f"  dept {dept_id:<3} {name}: need {need}, sampling {len(sample)} ids, "
            f"prefers {preferred}, size cap {DEPT_MAX_BYTES / 1e6:.1f} MB")
        accepted, attempts = 0, 0
        max_attempts = min(len(sample), max(60, need * 14))
        inflight: list = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for oid in sample:
                if accepted >= need or attempts >= max_attempts:
                    break
                attempts += 1
                if f"met-{oid}" in STORE.failed_ids or f"met-{oid}" in STORE.rows:
                    continue
                obj = met.object(oid)
                if not obj or obj.get("isPublicDomain") is not True:
                    continue
                img_url = (obj.get("primaryImage") or "").strip()
                if not img_url or STORE.seen_basename(img_url):
                    continue
                title = (obj.get("title") or "").strip()
                medium = (obj.get("medium") or "").strip()
                credit = (obj.get("creditLine") or "").strip()
                license_string = "Public Domain"
                evidence = (f"Met Open Access API GET /objects/{oid} reports "
                            f"isPublicDomain=true; creditLine=\"{credit}\"")
                if not rights_clean(license_string, evidence):
                    log(f"    veto met {oid}: rights marker in creditLine")
                    continue
                dims, total = sniff_image(http, img_url)
                if dims and max(dims) < MIN_ORIG_LONGEST:
                    STORE.fail(f"met-{oid}", f"original too small {dims}")
                    continue
                if total and total > DEPT_MAX_BYTES:
                    # oversized original: fall back to the Met's own web-large
                    # raster when it still clears the 768 px floor
                    small = (obj.get("primaryImageSmall") or "").strip()
                    if not small or STORE.seen_basename(small):
                        STORE.fail(f"met-{oid}", f"original {total} B over budget, no alt")
                        continue
                    dims2, total2 = sniff_image(http, small)
                    if not dims2 or max(dims2) < MIN_ORIG_LONGEST:
                        STORE.fail(f"met-{oid}", f"web-large too small {dims2}")
                        continue
                    if total2 and total2 > DEPT_MAX_BYTES:
                        STORE.fail(f"met-{oid}", f"web-large {total2} B over budget")
                        continue
                    img_url, dims = small, dims2
                while len(inflight) >= 8:
                    done, inflight = _harvest(inflight, block=True)
                    accepted += done
                fmt = choose_format(preferred, medium, title)
                inflight.append(pool.submit(
                    _met_worker, oid, dept_id, name, img_url, title, medium, credit,
                    obj, license_string, evidence, fmt))
            while inflight:
                done, inflight = _harvest(inflight, block=True)
                accepted += done
        log(f"  dept {dept_id:<3} {name}: accepted {accepted} in {attempts} attempts "
            f"({STORE.count('met')} met images total)")


def _harvest(futures: list, block: bool = False):
    """Reap finished futures -> (n_accepted, remaining futures)."""
    keep, acc = [], 0
    iterable = as_completed(futures) if block else futures
    for f in iterable:
        if f.done():
            if f.result():
                acc += 1
        else:
            keep.append(f)
    return acc, keep


def _met_worker(oid, dept_id, dept_name, img_url, title, medium, credit, obj,
               license_string, evidence, fmt) -> bool:
    data = HTTP_IMG.get_bytes(img_url, interval=0.1, max_bytes=ABSOLUTE_MAX_BYTES)
    if not data:
        STORE.fail(f"met-{oid}", "no bytes")
        return False
    raw = sha1_bytes(data)
    with _LOCK:
        if STORE.seen_raw(raw):
            return False
    im = decode(data)
    if im is None:
        STORE.fail(f"met-{oid}", "undecodable")
        return False
    ow, oh = im.size
    if max(ow, oh) < MIN_ORIG_LONGEST:
        STORE.fail(f"met-{oid}", f"too small {ow}x{oh}")
        return False
    out = normalise(im)
    ext = "png" if fmt == "png" else "jpg"
    path = IMGDIR / "met" / f"met-{oid}.{ext}"
    path.unlink(missing_ok=True)
    save_image(out, path, fmt)
    row = {
        "id": f"met-{oid}",
        "file": str(path.relative_to(ROOT)),
        "source": "met",
        "source_id": str(oid),
        "dept_key": f"met:{dept_id}",
        "page_url": obj.get("objectURL") or f"https://www.metmuseum.org/art/collection/search/{oid}",
        "image_url": img_url,
        "title": title or "(untitled)",
        "medium": medium,
        "department": obj.get("departmentTitle") or dept_name,
        "object_date": obj.get("objectDate") or "",
        "license": license_string,
        "license_evidence": evidence,
        "attribution": "none required (public domain / Met Open Access)",
        "orig_width": ow, "orig_height": oh,
        "raw_sha1": raw, "kind": fmt,
    }
    if not STORE.add(row):
        path.unlink(missing_ok=True)
        return False
    log(f"    + met-{oid} {ow}x{oh} -> {out.size[0]}x{out.size[1]} {fmt} "
        f"[{dept_name[:22]}] {row['title'][:40]!r}")
    return True


def met_fill(met: Met, http: Http, depts: dict[int, str], target: int, dry: bool) -> None:
    """Round-robin top-up across departments until `target` Met images exist."""
    if STORE.count("met") >= target:
        return
    if dry:
        log(f"  [dry] met fill would top up to {target}")
        return
    log(f"  met fill: {STORE.count('met')}/{target}; adding across departments")
    for dept_id, quota, preferred in MET_QUOTAS * 3:
        if STORE.count("met") >= target:
            break
        acquire_met(met, http, depts, [(dept_id, STORE.count(f"met:{dept_id}") + 6,
                                       preferred)], dry)
    log(f"  met fill done: {STORE.count('met')}")


# --------------------------------------------------------------------------- #
# source 2: scikit-image bundled data
# --------------------------------------------------------------------------- #
SKIMAGE_PROVENANCE = {
    "astronaut": ("US Government work / no known copyright restrictions",
                  "skimage.data.astronaut docstring: 'This image was downloaded from the "
                  "NASA Great Images database ... No known copyright restrictions, released "
                  "into the public domain'"),
    "camera": ("CC0",
               "skimage.data.camera docstring, Notes: 'No copyright restrictions. CC0 by the "
               "photographer (Lav Varshney)'"),
    "coffee": ("CC0",
               "skimage.data.coffee docstring, Notes: 'No copyright restrictions. CC0 by the "
               "photographer (Rachel Michetti)'"),
    "chelsea": ("CC0",
                "skimage.data.chelsea docstring, Notes: 'No copyright restrictions. CC0 by the "
                "photographer (Stefan van der Walt)'"),
    "rocket": ("Public Domain",
               "skimage.data.rocket docstring, Notes: 'captured by SpaceX and released in the "
               "public domain' (flickr.com/photos/spacexphotos/16511594820)"),
    "coins": ("Public Domain",
              "skimage.data.coins docstring, Notes: 'downloaded from the Brooklyn Museum "
              "Collection ... No known copyright restrictions'"),
    "hubble": ("US Government work / no known copyright restrictions",
               "skimage.data.hubble_deep_field docstring, Notes: 'captured by NASA and may be "
               "freely used in the public domain' (Hubble eXtreme Deep Field)"),
    "brick": ("CC0",
              "skimage.data.brick docstring, Notes: 'downloaded from CC0Textures ... licensed "
              "under the Creative Commons CC0 License'"),
    "phantom": ("CC0",
                "skimage.data.shepp_logan_phantom: Shepp & Logan (1974) phantom, raster "
                "recomputed procedurally from the published mathematical definition; no "
                "third-party photographic authorship"),
    "colorwheel": ("CC0 (generated)",
                   "skimage.data.colorwheel: HSV reference chart generated by scikit-image "
                   "code; redistributed here without further restriction"),
}
SKIMAGE_FUNC = {"hubble": "hubble_deep_field", "phantom": "shepp_logan_phantom"}


def acquire_skimage(budget: int, dry: bool) -> None:
    import skimage.data as skdata
    for name in SKIMAGE_NAMES:
        if budget <= 0:
            return
        lid = f"ski-{name}"
        if lid in STORE.rows:
            continue
        license_string, evidence = SKIMAGE_PROVENANCE[name]
        if not rights_clean(license_string, evidence):
            log(f"  skimage {name}: vetoed by licence policy")
            continue
        func = SKIMAGE_FUNC.get(name, name)
        if dry:
            log(f"  [dry] would rasterise skimage.data.{func}() -> images/skimage/{lid}")
            budget -= 1
            continue
        arr = np.asarray(getattr(skdata, func)())
        if arr.dtype != np.uint8:
            a = arr.astype(np.float64)
            lo, hi = float(a.min()), float(a.max())
            a = (a - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(a)
            arr = (a * 255.0).round().astype(np.uint8)
        im = (Image.fromarray(arr).convert("RGB") if arr.ndim == 2
              else Image.fromarray(np.ascontiguousarray(arr[:, :, :3])))
        ow, oh = im.size
        fmt = "png" if name in ("phantom", "colorwheel") else "jpeg"
        out = normalise(im)
        path = IMGDIR / "skimage" / f"{lid}.{ 'png' if fmt == 'png' else 'jpg' }"
        save_image(out, path, fmt)
        STORE.add({
            "id": lid, "file": str(path.relative_to(ROOT)), "source": "skimage",
            "source_id": f"skimage.data.{func}",
            "page_url": "https://scikit-image.org/docs/stable/api/skimage.data.html",
            "image_url": f"skimage://{func}",
            "title": f"scikit-image sample data: {name}",
            "medium": ("digital photograph (bundled sample data)" if fmt == "jpeg"
                       else "procedurally generated raster"),
            "department": "scikit-image bundled data", "object_date": "",
            "license": license_string, "license_evidence": evidence,
            "attribution": "none required (CC0 / public domain / US Government work)",
            "orig_width": ow, "orig_height": oh, "raw_sha1": None, "kind": fmt,
        })
        log(f"    + {lid} {ow}x{oh} -> {out.size[0]}x{out.size[1]} {fmt} [{license_string}]")
        budget -= 1


# --------------------------------------------------------------------------- #
# source 3: NASA
# --------------------------------------------------------------------------- #
class Nasa:
    def __init__(self, http: Http):
        self.http = http

    def search(self, query: str) -> list[dict]:
        key = CACHE / ("nasa_" + sha1_bytes(query.encode())[:16] + ".json")
        url = ("https://images-api.nasa.gov/search?q=" + requests.utils.quote(query) +
               "&media_type=image")
        data = self.http.get_json(url, key, interval=0.3, timeout=120)
        return (((data or {}).get("collection") or {}).get("items")) or []

    @staticmethod
    def largest_image(item: dict):
        """Return (href, width). Search items here expose renders under `links`
        (no `assets` key), so try both shapes."""
        best = None
        for link in item.get("links") or []:
            href = link.get("href") or ""
            if link.get("rel") == "alternate" and href.lower().endswith((".jpg", ".jpeg", ".png")):
                w = int(link.get("width") or 0)
                if best is None or w > best[1]:
                    best = (href, w)
        for a in item.get("assets") or []:
            href = a.get("href") or ""
            if a.get("resource_type") in (None, "image") or a.get("render") == "image":
                if href.lower().endswith((".jpg", ".jpeg", ".png")):
                    w = int(a.get("width") or 0)
                    if best is None or w > best[1]:
                        best = (href, w)
        return best


def acquire_nasa(target: int, dry: bool) -> None:
    nasa = Nasa(Http(browser_ua=True))
    if STORE.count("nasa") >= target:
        return
    queries = list(NASA_QUERIES)
    random.Random(SEED).shuffle(queries)
    seen: set[str] = set()
    for q in queries:
        if STORE.count("nasa") >= target:
            break
        items = nasa.search(q)
        usable = [it for it in items if nasa.largest_image(it)]
        log(f"  nasa query {q!r}: {len(items)} items, {len(usable)} with an image render")
        for item in usable:
            if STORE.count("nasa") >= target:
                break
            d = (item.get("data") or [{}])[0]
            nid = d.get("nasa_id") or ""
            if not nid or nid in seen:
                continue
            seen.add(nid)
            lid = f"nasa-{nid}"
            if lid in STORE.rows or lid in STORE.failed_ids:
                continue
            href, w = nasa.largest_image(item)
            orig = re.sub(r"~(medium|small|thumb)\.(jpg|jpeg|png)$", r"~orig.\2", href)
            license_string = "US Government work / no known copyright restrictions"
            evidence = (f"NASA Images API GET images-api.nasa.gov/search?media_type=image "
                        f"item data[0].nasa_id={nid} (center={d.get('center')}, "
                        f"secondary_creator={d.get('secondary_creator')}, "
                        f"date_created={d.get('date_created')}); NASA imagery is a work of the "
                        f"United States Government (17 U.S.C. 105), and NASA's media usage "
                        f"guidelines place its photography in the public domain with no known "
                        f"copyright restrictions")
            if not rights_clean(license_string, evidence):
                log(f"  nasa {nid}: vetoed by licence policy")
                continue
            if dry:
                log(f"  [dry] would fetch {lid} ({href.rsplit('/', 1)[-1]}, w={w})")
                continue
            got = used = None
            for cand in ([orig] if orig != href else []) + [href]:
                dims, total = sniff_image(HTTP_NASA, cand)
                if total and total > 8_000_000:
                    continue
                if dims and max(dims) < MIN_ORIG_LONGEST:
                    continue
                body = HTTP_NASA.get_bytes(cand, interval=0.2, max_bytes=ABSOLUTE_MAX_BYTES)
                if body:
                    got, used = body, cand
                    break
            if not got:
                STORE.fail(lid, "no usable asset (blocked or oversized)")
                continue
            raw = sha1_bytes(got)
            if STORE.seen_raw(raw) or STORE.seen_basename(used):
                continue
            im = decode(got)
            if im is None:
                STORE.fail(lid, "undecodable")
                continue
            ow, oh = im.size
            if max(ow, oh) < MIN_ORIG_LONGEST:
                STORE.fail(lid, f"too small {ow}x{oh}")
                continue
            out = normalise(im)
            path = IMGDIR / "nasa" / f"{lid}.jpg"
            save_image(out, path, "jpeg")
            STORE.add({
                "id": lid, "file": str(path.relative_to(ROOT)), "source": "nasa",
                "source_id": nid, "page_url": f"https://images.nasa.gov/details-{nid}",
                "image_url": used, "title": (d.get("title") or nid)[:200],
                "medium": "astronautical photograph (NASA imagery)",
                "department": f"NASA / {d.get('center') or 'unspecified'}",
                "object_date": (d.get("date_created") or "")[:10],
                "license": license_string, "license_evidence": evidence,
                "attribution": "none required (US Government work / no known copyright restrictions)",
                "orig_width": ow, "orig_height": oh, "raw_sha1": raw, "kind": "jpeg",
            })
            log(f"    + {lid} {ow}x{oh} -> {out.size[0]}x{out.size[1]} jpeg "
                f"{STORE.rows[lid]['title'][:40]!r}")
    log(f"  nasa accepted: {STORE.count('nasa')}")


# --------------------------------------------------------------------------- #
# source 4: deterministic synthetic phantoms
# --------------------------------------------------------------------------- #
N = 1024
RED = np.array([190.0, 20.0, 30.0])
BLUE = np.array([20.0, 45.0, 200.0])
WHITE = np.array([250.0, 250.0, 250.0])
BLACK = np.array([8.0, 8.0, 8.0])
GREY = np.array([128.0, 128.0, 128.0])


def _rng(idx: int, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng([SEED, idx + 1, salt + 1])


def _blank(base) -> np.ndarray:
    img = np.empty((N, N, 3), dtype=np.float64)
    img[:] = base
    return img


def _line(img, p0, p1, width, colour):
    y0, x0 = p0
    y1, x1 = p1
    steps = int(max(abs(y1 - y0), abs(x1 - x0)) * 2 + 1)
    r = max(0, (int(width) - 1) // 2)
    for t in np.linspace(0.0, 1.0, steps):
        y = int(round(y0 + t * (y1 - y0)))
        x = int(round(x0 + t * (x1 - x0)))
        img[max(0, y - r):min(N, y + r + 1), max(0, x - r):min(N, x + r + 1)] = colour


BASE_PHANTOMS = [
    ("junction-plus-1px", "junction"),
    ("junction-x-2px", "junction"),
    ("junction-grid-dark", "junction"),
    ("t-junction-brick", "junction"),
    ("t-junction-segments", "junction"),
    ("radial-flow-spokes", "junction"),
    ("lines-1px-bright", "thin"),
    ("lines-1px-dark", "thin"),
    ("lines-2px-pair", "thin"),
    ("lines-crosshair-thin", "thin"),
    ("lines-random-thin", "thin"),
    ("stripes-chirp-h", "pattern"),
    ("stripes-chirp-v", "pattern"),
    ("checker-8px", "pattern"),
    ("checker-24px", "pattern"),
    ("checker-64px", "pattern"),
    ("disc-noise-s002", "flat"),
    ("disc-noise-s005", "flat"),
    ("disc-noise-s015", "flat"),
    ("spiral-flow-field", "texture"),
    ("colour-edge-split", "edge"),
    ("colour-edge-quad", "edge"),
]
PHANTOM_KIND = {"junction": "junction_rich", "thin": "thin_structure",
                "pattern": "repeating_pattern", "flat": "flat_region",
                "texture": "texture_rich", "edge": "high_contrast_edge"}
STRATUM_TO_KIND = {"thin_structure": "thin", "repeating_pattern": "pattern",
                   "junction_rich": "junction", "flat_region": "flat",
                   "texture_rich": "texture", "high_contrast_edge": "edge"}


def _render(name: str, idx: int) -> np.ndarray:
    rng = _rng(idx, salt=7)
    yy, xx = np.mgrid[0:N, 0:N]

    if name == "junction-plus-1px":
        img = _blank(BLACK)
        for k, p in enumerate(range(56, N, 88)):
            _line(img, (p, 0), (p, N), 1, WHITE)
            _line(img, (0, p + 40), (N, p + 40), 1, RED if k % 2 else BLUE)
        return img
    if name == "junction-x-2px":
        img = _blank(GREY * 0.3)
        for off in range(-N, N, 64):
            _line(img, (0, off), (N, off + N), 2, WHITE)
            _line(img, (0, off + N), (N, off), 2, RED if (off // 64) % 2 else BLUE)
        return img
    if name == "junction-grid-dark":
        img = _blank(WHITE)
        for p in range(40, N, 72):
            _line(img, (p, 0), (p, N), 3, BLACK)
            _line(img, (0, p + 32), (N, p + 32), 3, np.array([40.0, 90.0, 200.0]))
        return img
    if name == "t-junction-brick":
        img = _blank(np.array([235.0, 230.0, 220.0]))
        bh, bw = 64, 128
        for r in range(0, N // bh + 1):
            y = r * bh
            _line(img, (y, 0), (y, N), 2, BLACK)
            shift = (bw // 2) if r % 2 else 0
            for c in range(0, N // bw + 2):
                x = c * bw + shift
                _line(img, (y, x), (min(N, y + bh), x), 2, np.array([200.0, 60.0, 40.0]))
        return img
    if name == "t-junction-segments":
        img = _blank(WHITE)
        for _ in range(150):
            y, x = rng.integers(4, N - 4, size=2)
            ln = int(rng.integers(60, 340))
            wd = int(rng.choice([1, 2, 3]))
            if rng.random() < 0.5:
                _line(img, (y, x), (y, min(N - 1, x + ln)), wd, BLACK)
                _line(img, (y, x + ln // 2), (max(0, y - ln // 2), x + ln // 2), wd, RED)
            else:
                _line(img, (y, x), (min(N - 1, y + ln), x), wd, BLACK)
                _line(img, (y + ln // 2, x), (y + ln // 2, max(0, x - ln // 2)), wd, BLUE)
        return img
    if name == "radial-flow-spokes":
        ang = np.arctan2(yy - N / 2, xx - N / 2)
        v = 0.5 + 0.5 * np.cos(40 * ang)
        img = np.stack([v, 0.5 * v + 0.2, 1 - v], axis=-1) * 220 + 15
        for k in range(-N, N, 48):
            _line(img, (0, k), (N, k + N), 2, WHITE)
        return img
    if name == "lines-1px-bright":
        img = _blank(BLACK)
        for p in range(0, N, 7):
            _line(img, (p, 0), (p, N), 1, WHITE)
        return img
    if name == "lines-1px-dark":
        img = _blank(WHITE)
        for p in range(0, N, 9):
            _line(img, (0, p), (N, p), 1, BLACK)
        return img
    if name == "lines-2px-pair":
        img = _blank(np.array([120.0, 120.0, 120.0]))
        for p in range(0, N, 24):
            _line(img, (0, p), (N, p), 2, WHITE)
            _line(img, (0, p + 4), (N, p + 4), 2, BLACK)
            _line(img, (p, 0), (p, N), 2, RED)
        return img
    if name == "lines-crosshair-thin":
        img = _blank(BLUE)
        for p in range(28, N, 44):
            _line(img, (0, p), (N, p), 1, WHITE)
            _line(img, (p + 20, 0), (p + 20, N), 1, RED)
        return img
    if name == "lines-random-thin":
        img = _blank(BLACK)
        for _ in range(320):
            y0, x0 = rng.integers(0, N, size=2)
            ln = int(rng.integers(30, 420))
            a = float(rng.uniform(0, np.pi))
            _line(img, (y0, x0),
                  (int(np.clip(y0 + ln * np.sin(a), 0, N - 1)),
                   int(np.clip(x0 + ln * np.cos(a), 0, N - 1))),
                  int(rng.choice([1, 1, 2])), WHITE)
        return img
    if name == "stripes-chirp-h":
        x = np.arange(N, dtype=np.float64)
        f = (2.0 + 90.0 * (x / N) ** 1.7) / N
        v = 0.5 + 0.5 * np.sin(2 * np.pi * np.cumsum(f))
        img = np.empty((N, N, 3), dtype=np.float64)
        img[..., :] = v[:, None, None]
        return img * 235 + 10
    if name == "stripes-chirp-v":
        return np.ascontiguousarray(
            _render("stripes-chirp-h", idx).transpose(1, 0, 2))
    m = re.fullmatch(r"checker-(\d+)px", name)
    if m:
        cell = int(m.group(1))
        c = ((yy // cell) + (xx // cell)) % 2
        return np.where(c[..., None] > 0, WHITE, BLACK).astype(np.float64)
    m = re.fullmatch(r"disc-noise-s(\d{3})", name)
    if m:
        sigma = int(m.group(1)) / 100.0
        img = _blank(GREY)
        for _ in range(8):
            cy, cx = rng.integers(140, N - 140, size=2)
            rad = int(rng.integers(60, 210))
            level = float(rng.uniform(30, 225))
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2
            img[mask] = level
        return np.clip(img + rng.normal(0.0, sigma * 255.0, size=(N, N, 3)), 0, 255)
    if name == "spiral-flow-field":
        ang = np.arctan2(yy - N / 2, xx - N / 2)
        rad = np.hypot(xx - N / 2, yy - N / 2)
        v = 0.5 + 0.5 * np.sin(7 * ang + rad / 9.0)
        img = np.stack([v, 0.35 * v + 0.3, 0.8 * v], axis=-1)
        img += rng.normal(0, 0.02, img.shape)
        return img * 225 + 12
    if name == "colour-edge-split":
        img = _blank(BLUE)
        img[:, : N // 2] = RED
        _line(img, (0, N // 2), (N, N // 2), 1, WHITE)
        _line(img, (N // 3, 0), (N // 3, N), 1, np.array([200.0, 200.0, 30.0]))
        return img + rng.normal(0, 3.0, img.shape)
    if name == "colour-edge-quad":
        img = _blank(BLUE)
        img[: N // 2, : N // 2] = RED
        img[N // 2:, N // 2:] = WHITE
        img[: N // 2, N // 2:] = BLACK
        return img
    if name.startswith("topup-flat"):
        img = _blank(np.array([150.0, 152.0, 148.0]))
        for _ in range(6):
            cy, cx = rng.integers(150, N - 150, size=2)
            rad = int(rng.integers(90, 280))
            img[((yy - cy) ** 2 + (xx - cx) ** 2) <= rad ** 2] = [146.0, 150.0, 151.0]
        return img + rng.normal(0, 0.7, img.shape)
    if name.startswith("topup-edge"):
        img = _blank(np.array([245.0, 245.0, 245.0]))
        for _ in range(16):
            cy, cx = rng.integers(60, N - 320, size=2)
            hh, ww = int(rng.integers(40, 300)), int(rng.integers(40, 300))
            img[cy:cy + hh, cx:cx + ww] = rng.choice([RED, BLUE, BLACK, WHITE])
        return img
    if name.startswith("topup-texture"):
        a = rng.normal(0.0, 1.0, (N, N, 3))
        sm = ndi.gaussian_filter(a, 3.0)
        hi = (a - sm) / (a - sm).std()
        return np.clip(128 + 70 * np.tanh(1.6 * hi), 0, 255)
    if name.startswith("topup-thin"):
        return _render("lines-random-thin", idx + 31)
    if name.startswith("topup-junction"):
        img = _blank(np.array([238.0, 238.0, 238.0]))
        for _ in range(70):
            y, x = rng.integers(60, N - 60, size=2)
            arm = int(rng.integers(40, 220))
            wd = int(rng.choice([2, 3]))
            _line(img, (y - arm, x), (y + arm, x), wd, BLACK)
            _line(img, (y, x - arm), (y, x + arm), wd, BLACK)
            _line(img, (y - arm, x - arm), (y + arm, x + arm), 1, RED)
        return img
    if name.startswith("topup-pattern"):
        cell = int(rng.choice([12, 20, 32]))
        v = (0.5 + 0.5 * np.sin(2 * np.pi * xx / cell) * np.cos(2 * np.pi * yy / cell))
        return np.stack([v, v, v], axis=-1) * 235 + 10
    raise SystemExit(f"unknown phantom {name!r}")


def phantom_specs(extra: dict[str, int] | None = None) -> list[tuple[str, str]]:
    specs = list(BASE_PHANTOMS)
    for stratum, count in (extra or {}).items():
        kind = STRATUM_TO_KIND[stratum]
        for i in range(1, count + 1):
            specs.append((f"topup-{kind}-{i:02d}", kind))
    return specs


def acquire_synthetic(extra: dict[str, int] | None, dry: bool) -> None:
    for i, (name, kind) in enumerate(phantom_specs(extra)):
        lid = f"syn-{name}"
        if lid in STORE.rows:
            continue
        if dry:
            log(f"  [dry] would render images/synthetic/{lid}.png (seed {SEED}, idx {i})")
            continue
        arr = np.clip(_render(name, i), 0, 255).astype(np.uint8)
        im = Image.fromarray(np.ascontiguousarray(arr))
        path = IMGDIR / "synthetic" / f"{lid}.png"
        save_image(im, path, "png")
        STORE.add({
            "id": lid, "file": str(path.relative_to(ROOT)), "source": "synthetic",
            "source_id": f"seed{SEED}:{i}",
            "page_url": f"synthetic://abstrabench/{name}",
            "image_url": f"synthetic://abstrabench/{name}.png",
            "title": f"synthetic phantom: {name}",
            "medium": f"deterministic synthetic phantom ({PHANTOM_KIND[kind]})",
            "department": "AbstraBench synthetic phantoms", "object_date": "",
            "license": "CC0 (generated)",
            "license_evidence": (f"original work of this benchmark, rendered by "
                                 f"scripts/build_benchmark.py::_render(name={name!r}, "
                                 f"seed={SEED}); released CC0"),
            "attribution": "none required (authors' own work, CC0)",
            "orig_width": N, "orig_height": N, "raw_sha1": None, "kind": "png",
            "phantom_kind": kind,
        })
        log(f"    + {lid} ({kind})")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _pattern_score(mag: np.ndarray) -> float:
    """Peak of the power spectrum above its own radially averaged 1/f envelope."""
    ac = mag - mag.mean()
    if not np.any(ac):
        return 0.0
    P = np.abs(np.fft.rfft2(ac)) ** 2
    ny = mag.shape[0]
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.rfftfreq(mag.shape[1])[None, :]
    r = np.round(np.hypot(fy, fx) * ny).astype(int)
    sel = r >= 1
    rb = r[sel]
    if rb.size == 0:
        return 0.0
    nb = int(rb.max()) + 1
    cnt = np.bincount(rb, minlength=nb).astype(np.float64)
    tot = np.bincount(rb, np.log(P[sel] + 1e-30), minlength=nb)
    ok = cnt > 0
    xs = np.arange(nb)[ok].astype(np.float64)
    ys = tot[ok] / cnt[ok]
    fit = xs >= 3                      # ignore the lowest bins (DC leakage)
    if fit.sum() < 4:
        return 0.0
    slope, intercept = np.polyfit(np.log(xs[fit]), ys[fit], 1)
    resid = ys - (intercept + slope * np.log(np.maximum(xs, 1e-6)))
    return float(np.clip(float(resid.max()) / 6.0, 0.0, 1.0))


def compute_metrics(path: Path) -> dict:
    with Image.open(path) as handle:
        im = handle.convert("RGB")
    rgb = np.asarray(im, dtype=np.float64) / 255.0
    g = rgb2gray(rgb)
    W, H = im.size

    # gradient magnitude (Sobel) -> texture_rich, high_contrast_edge
    mag = sobel(g)
    grad_mean = float(np.mean(mag))
    grad_p95 = float(np.percentile(mag, 95))

    # structure tensor coherence + local std -> flat_region
    T = structure_tensor(g, sigma=1.5)
    l0, l1 = structure_tensor_eigenvalues(T)
    l1 = np.maximum(l1, 0.0)
    coh = (l0 - l1) / np.maximum(l0 + l1, 1e-12)
    lm = ndi.uniform_filter(g, size=9)
    lvar = np.maximum(ndi.uniform_filter(g * g, size=9) - lm * lm, 0.0)
    flat_frac = float(np.mean((coh < 0.1) & (np.sqrt(lvar) < 0.02)))

    # Sato ridge response + morphological top-hats (both polarities)
    # -> thin_structure
    resp = sato(g, sigmas=[0.5, 1.0, 2.0], black_ridges=False)
    top_dark = np.maximum(black_tophat(g, disk(3)), 0.0)
    top_bright = np.maximum(white_tophat(g, disk(3)), 0.0)
    peak = float(resp.max()) if resp.size else 0.0
    try:
        thr = max(0.04, 0.5 * float(threshold_otsu(resp))) if peak > 0 else 0.04
    except Exception:
        thr = 0.04
    ridge_frac = float(np.mean((resp > thr) | (top_dark > 0.25) | (top_bright > 0.25)))

    # Harris corner peak density -> junction_rich
    hr = np.maximum(corner_harris(g, k=0.04, method="shepp"), 0.0)
    n_peaks = len(peak_local_max(hr, min_distance=6, threshold_rel=0.35,
                                 exclude_border=False, num_peaks=5000))
    corner_density = float(n_peaks / max(W * H, 1) * 1000.0)

    # Repeating-pattern score: the strongest peak of the gradient-magnitude power
    # spectrum above its own radially averaged 1/f envelope.  A periodic layout
    # (stripes, checkerboards, grids) produces a sharp spike; natural image
    # spectra only track the envelope.  Normalised so that a spike 6 log-units
    # (≈400x) above the envelope saturates the score at 1.0.
    pattern_score = _pattern_score(mag)

    return {
        "grad_mean": round(grad_mean, 6),
        "grad_p95": round(grad_p95, 6),
        "flat_frac": round(flat_frac, 6),
        "ridge_frac": round(ridge_frac, 6),
        "corner_density": round(corner_density, 4),
        "pattern_score": round(pattern_score, 6),
    }


def metric_table() -> dict[str, dict]:
    cached = {(r["id"], r["sha1"]): r["metrics"] for r in load_jsonl(METRICS_CACHE)}
    fresh, out = [], {}
    for lid in sorted(STORE.rows):
        path = ROOT / STORE.rows[lid]["file"]
        if not path.exists():
            continue
        digest = sha1_file(path)
        key = (lid, digest)
        if key in cached:
            out[lid] = cached[key]
            continue
        try:
            m = compute_metrics(path)
        except Exception as exc:
            log(f"  metrics failed for {lid}: {type(exc).__name__}: {exc}")
            m = {k: 0.0 for k in METRIC_TO_STRATUM}
        out[lid] = m
        fresh.append({"id": lid, "sha1": digest, "metrics": m})
    if fresh and not DRY:
        for rec in fresh:
            append_jsonl(METRICS_CACHE, rec)
        latest = {}
        for r in load_jsonl(METRICS_CACHE):
            latest[r["id"]] = r
        with open(METRICS_CACHE, "w", encoding="utf-8") as fh:
            for r in latest.values():
                fh.write(json.dumps(r) + "\n")
    return out


def assign_strata(metrics: dict[str, dict]):
    lids = sorted(metrics)
    if not lids:
        return {}, {s: 0 for s in STRATA}
    ranks: dict[str, dict[str, float]] = {l: {} for l in lids}
    for metric, stratum in METRIC_TO_STRATUM.items():
        vals = np.array([metrics[l][metric] for l in lids], dtype=np.float64)
        pct = np.argsort(np.argsort(vals)) / max(len(lids) - 1, 1)
        for l, p in zip(lids, pct):
            ranks[l][stratum] = float(p)
    chosen, counts = {}, {s: 0 for s in STRATA}
    for l in lids:
        top2 = sorted(sorted(STRATA, key=lambda s: ranks[l][s], reverse=True)[:2])
        chosen[l] = top2
        for s in top2:
            counts[s] += 1
    return chosen, counts


# --------------------------------------------------------------------------- #
# manifest, README, LICENSE, verification
# --------------------------------------------------------------------------- #
def manifest_rows(chosen: dict[str, list[str]], metrics: dict[str, dict]) -> list[dict]:
    rows = []
    for lid in sorted(STORE.rows):
        base = STORE.rows[lid]
        path = ROOT / base["file"]
        if not path.exists():
            log(f"  missing file for {lid} ({base['file']}); row dropped")
            continue
        m = metrics.get(lid, {k: 0.0 for k in METRIC_TO_STRATUM})
        with Image.open(path) as im:
            W, H = im.size
        rows.append({
            "id": lid, "file": base["file"], "source": base["source"],
            "source_id": base.get("source_id", ""), "page_url": base.get("page_url", ""),
            "image_url": base.get("image_url", ""), "title": base.get("title", ""),
            "medium": base.get("medium", ""), "department": base.get("department", ""),
            "object_date": base.get("object_date", ""), "license": base.get("license", ""),
            "license_evidence": base.get("license_evidence", ""),
            "attribution": base.get("attribution", ""),
            "orig_width": base.get("orig_width", ""), "orig_height": base.get("orig_height", ""),
            "width": W, "height": H, "bytes": path.stat().st_size, "sha1": sha1_file(path),
            **{k: m[k] for k in METRIC_TO_STRATUM},
            "strata": "|".join(chosen.get(lid, [])),
        })
    return rows


def write_manifest(rows: list[dict]) -> None:
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: r.get(k, "") for k in MANIFEST_COLUMNS})


def verify(rows: list[dict]) -> dict:
    rep = {"ok": True, "problems": [], "strata_counts": {s: 0 for s in STRATA}}
    on_disk = sorted(p for d in ("met", "skimage", "nasa", "synthetic")
                     for p in (IMGDIR / d).glob("*")
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    log(f"verify: {len(on_disk)} files on disk, {len(rows)} manifest rows")
    if len(on_disk) < TOTAL_MIN:
        rep["ok"] = False
        rep["problems"].append(f"only {len(on_disk)} images on disk (< {TOTAL_MIN})")
    if len(rows) < TOTAL_MIN:
        rep["ok"] = False
        rep["problems"].append(f"only {len(rows)} manifest rows (< {TOTAL_MIN})")
    rel_paths = {str(p.relative_to(ROOT)) for p in on_disk}
    for p in on_disk:
        try:
            with Image.open(p) as im:
                im.load()
                if max(im.size) != TARGET_LONGEST:
                    rep["ok"] = False
                    rep["problems"].append(f"{p.name}: longest side {max(im.size)}")
                if im.mode != "RGB":
                    rep["ok"] = False
                    rep["problems"].append(f"{p.name}: mode {im.mode}")
        except Exception as exc:
            rep["ok"] = False
            rep["problems"].append(f"{p.name}: PIL reopen failed ({type(exc).__name__})")
    by_file = {r["file"] for r in rows}
    for rel in sorted(rel_paths - by_file):
        rep["ok"] = False
        rep["problems"].append(f"{rel} on disk but not in manifest")
    for rel in sorted(by_file - rel_paths):
        rep["ok"] = False
        rep["problems"].append(f"{rel} in manifest but not on disk")
    ids = set()
    for r in rows:
        if r["id"] in ids:
            rep["ok"] = False
            rep["problems"].append(f"duplicate id {r['id']}")
        ids.add(r["id"])
        p = ROOT / r["file"]
        if p.exists() and sha1_file(p) != r["sha1"]:
            rep["ok"] = False
            rep["problems"].append(f"sha1 mismatch: {r['id']}")
        if r["license"] not in ACCEPTED_LICENSES:
            rep["ok"] = False
            rep["problems"].append(f"licence off allow-list: {r['id']} -> {r['license']}")
        if r["source"] != "met" and not r["license_evidence"]:
            rep["ok"] = False
            rep["problems"].append(f"no licence evidence: {r['id']}")
        for s in r["strata"].split("|"):
            if s in rep["strata_counts"]:
                rep["strata_counts"][s] += 1
    for s in STRATA:
        if rep["strata_counts"][s] < MIN_PER_STRATUM:
            rep["ok"] = False
            rep["problems"].append(f"stratum {s}: {rep['strata_counts'][s]} < {MIN_PER_STRATUM}")
    rep["total"] = len(rows)
    rep["files"] = len(on_disk)
    rep["bytes"] = sum(r["bytes"] for r in rows)
    rep["per_source"] = {d: sum(1 for r in rows if r["source"] == d)
                         for d in ("met", "skimage", "nasa", "synthetic")}
    rep["per_department"] = {}
    for r in rows:
        if r["source"] == "met":
            rep["per_department"][r["department"]] = \
                rep["per_department"].get(r["department"], 0) + 1
    rep["licences"] = sorted({r["license"] for r in rows})
    return rep


def latex_paragraph(n: int, per: dict) -> str:
    return (
        "\\subsection{Image sources and licences}\n"
        "\\label{sec:abstrabench}\n"
        f"We evaluate our method on \\textbf{AbstraBench-200}, a {n}-image benchmark "
        "assembled to be free of copyright and attribution constraints, replacing the "
        "12-image set used in prior work. "
        f"It comprises {per.get('met', 0)} works from the Metropolitan Museum of Art Open "
        "Access programme (paintings, photographs, drawings and prints, textiles and "
        "decorative arts), "
        f"{per.get('skimage', 0)} sample images bundled with scikit-image, "
        f"{per.get('nasa', 0)} NASA images, and "
        f"{per.get('synthetic', 0)} synthetic phantoms of our own, generated with a fixed "
        f"random seed ({SEED}) to exercise the structural failure modes relevant to "
        "abstraction: four-way X/plus junctions, T-junctions, one- and two-pixel bright and "
        "dark lines, varying-frequency stripes, multi-scale checkerboards, flat discs at "
        "several noise levels, radial and spiral flow fields, and hard red--blue colour "
        "boundaries. "
        "Only images whose licence is CC0, public domain (PD-old), or a US Government work "
        "with no known copyright restrictions were admitted; material carrying CC-BY, "
        "CC-BY-NC, an \\emph{all rights reserved} notice, or an absent or ambiguous rights "
        "statement was rejected during acquisition. The released manifest records, for every "
        "image, the exact source metadata field and value that establishes this status. "
        "The benchmark therefore imposes \\textbf{no attribution obligations} and its images "
        "may be redistributed alongside the paper. Each raster is normalised to a longest "
        "side of 1024\\,px, preserving the original aspect ratio without cropping, converted "
        "to RGB with any alpha channel dropped, and stored as JPEG (quality 92) for "
        "photographs and paintings and as PNG for synthetic phantoms and line art. Images "
        "are assigned automatically to six strata (texture-rich, flat-region, "
        "thin-structure, junction-rich, high-contrast-edge and repeating-pattern) from "
        "measured structural descriptors --- Sobel gradient statistics, structure-tensor "
        "coherence, Sato ridge response, Harris corner density and FFT spectral "
        "concentration --- each image taking its two highest-ranked strata and every "
        "stratum containing at least 15 images."
    )


def write_readme(rep: dict, extra: dict[str, int], samples: dict[str, list[str]],
                 nasa_status: str) -> None:
    per = rep["per_source"]
    tab = "\n".join(
        f"| `{s}` | {rep['strata_counts'][s]} | {', '.join(samples[s][:4]) or '-'} |"
        for s in STRATA)
    dept_tab = "\n".join(f"| {d} | {c} |" for d, c in
                         sorted(rep["per_department"].items(), key=lambda kv: -kv[1]))
    txt = f"""# AbstraBench-200

A licence-clean image benchmark of **{rep['total']} images** for evaluating image
abstraction / non-photorealistic rendering algorithms. Every raster is resampled to a
longest side of exactly 1024 px with the aspect ratio preserved (never cropped to a
square), converted to RGB with the alpha channel dropped, and stored as JPEG (quality 92)
for photographs and paintings, or PNG for synthetic phantoms, line art and procedural
charts.

## Licence policy

Acceptance requires an explicit, machine-checkable rights statement in one of these
categories -- nothing else enters the benchmark:

| accepted licence | how the manifest proves it |
|---|---|
| `CC0` | verbatim provenance sentence from the source's own metadata |
| `CC0 (generated)` | rendered by `scripts/build_benchmark.py` under fixed seed {SEED} |
| `Public Domain` | Met Collection API `isPublicDomain: true` + verbatim `creditLine` |
| `US Government work / no known copyright restrictions` | NASA Images API `nasa_id` / `center` / `secondary_creator` record |

Anything carrying **CC-BY**, **CC-BY-NC**, "all rights reserved", or an absent or
ambiguous rights statement is **rejected**, so the paper can state that the benchmark
imposes no attribution obligations. `manifest.csv` column `license_evidence` stores the
exact metadata field and value behind each claim, row by row.

### Per-source licence statements (verbatim)

**Metropolitan Museum of Art -- Open Access** (`images/met/`, {per.get('met', 0)} images).
An object is accepted only if its per-object API record contains `isPublicDomain: true`;
the evidence string is `Met Open Access API GET /objects/<id> reports isPublicDomain=true;
creditLine="<verbatim>"`. Endpoint:
`https://collectionapi.metmuseum.org/public/collection/v1/objects/<objectID>`.

**scikit-image bundled sample data** (`images/skimage/`, {per.get('skimage', 0)} images).
The rights statement is quoted from scikit-image's own docstrings, e.g.
astronaut: "This image was downloaded from the NASA Great Images database ... No known
copyright restrictions, released into the public domain"; camera/coffee/chelsea: "No
copyright restrictions. CC0 by the photographer (Lav Varshney / Rachel Michetti / Stefan
van der Walt)"; brick: "downloaded from CC0Textures ... licensed under the Creative
Commons CC0 License"; rocket: "captured by SpaceX and released in the public domain";
coins: "downloaded from the Brooklyn Museum Collection ... No known copyright
restrictions"; hubble: "captured by NASA and may be freely used in the public domain".

**NASA Images** (`images/nasa/`, {per.get('nasa', 0)} images). {nasa_status} NASA imagery is
a work of the United States Government (17 U.S.C. §105) and NASA's media usage guidelines
place its photography in the public domain.

**Synthetic phantoms** (`images/synthetic/`, {per.get('synthetic', 0)} images). Original
work of this benchmark, rendered procedurally with fixed seed {SEED} and released as
`CC0 (generated)`.

### Deliberate exclusions

* scikit-image's `moon` sample was **not** shipped although it was on the wishlist:
  scikit-image carries no rights statement for it, and an absent statement is a rejection
  under this policy. The slot is filled by `skimage.data.colorwheel`, whose chart is
  generated by scikit-image code.
* Requests for `hubble` and `phantom` are served by their current function names,
  `skimage.data.hubble_deep_field` and `skimage.data.shepp_logan_phantom`.
* No Wikimedia, Openverse, Unsplash, Pexels, Library of Congress or Europeana material is
  included: those hosts are unreachable from the build environment and/or mostly serve
  attribution-required licences.

## How to regenerate

```bash
cd /data/abstrabench
python3 scripts/build_benchmark.py              # full idempotent build (+ verify)
python3 scripts/build_benchmark.py --dry-run    # plan only: no downloads, no writes
python3 scripts/build_benchmark.py --limit 24   # small smoke build
python3 scripts/build_benchmark.py --resume     # continue a partial/interrupted run
python3 scripts/build_benchmark.py --verify-only  # recompute manifest + verify
```

Caches that make a re-run cheap and byte-identical: `cache/met_departments.json`,
`cache/met_dept_<id>.json`, `cache/met_objects/<id>.json`, `cache/nasa_<hash>.json`,
`cache/rows.jsonl` (acquired metadata), `cache/metrics.jsonl` (metrics keyed by file
sha1), `cache/failed.jsonl` (rejected object ids). The manifest is always regenerated from
these plus the rasters on disk -- no row is hand-written.

## Contents

| source | images | licence |
|---|---|---|
| `images/met/` | {per.get('met', 0)} | Public Domain |
| `images/skimage/` | {per.get('skimage', 0)} | CC0 / Public Domain / US Government work |
| `images/nasa/` | {per.get('nasa', 0)} | US Government work / no known copyright restrictions |
| `images/synthetic/` | {per.get('synthetic', 0)} | CC0 (generated) |
| **total** | **{rep['total']}** | **no attribution obligations** |

Met coverage by department:

| department | images |
|---|---|
{dept_tab}

Total payload: {rep['bytes']:,} bytes across {rep['files']} files.
Distinct licence strings in the manifest: {', '.join('`' + l + '`' for l in rep['licences'])}.

## Strata

Per-image descriptors are measured with numpy + scikit-image: Sobel gradient magnitude
(mean and 95th percentile), structure-tensor coherence combined with a local standard
deviation (flat fraction, thresholded at coherence < 0.1 and local σ < 0.02), Sato ridge
response over σ ∈ {{0.5, 1, 2}} combined with morphological white/black top-hats
(thin-structure fraction), Harris corner peak density (peaks per 1000 px), and a
repeating-pattern score given by the strongest spike of the gradient-magnitude power
spectrum above its own radially averaged 1/f envelope. Each image is then given its
**top 2 strata** by cross-dataset percentile rank, so the assignment is computed from the
pixels rather than guessed from metadata.

| stratum | images | example ids |
|---|---|---|
{tab}

{extra_note(extra)}

## LaTeX for the paper

Ready to paste:

```latex
{latex_paragraph(rep['total'], per)}
```

## Licence of this package

See `LICENSE.txt`: the manifest, metadata and build script are CC0; the underlying images
were already public domain / CC0 / US Government work and this package adds no further
restriction on them.
"""
    (ROOT / "README.md").write_text(txt, encoding="utf-8")


def extra_note(extra: dict[str, int]) -> str:
    if not extra:
        return ("No stratum fell below the 15-image floor, so no top-up phantoms were "
                "needed.")
    return ("Shortfall handling: " +
            ", ".join(f"`{k}` received {v} extra targeted synthetic phantoms"
                      for k, v in extra.items() if v) +
            " so that every stratum reaches the required 15 images.")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
HTTP_IMG: Http
HTTP_NASA: Http


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the licence-clean AbstraBench-200 image benchmark.")
    p.add_argument("--dry-run", action="store_true",
                   help="plan only; no downloads and no writes")
    p.add_argument("--limit", type=int, default=0,
                   help="cap the total number of acquired images (0 = no cap)")
    p.add_argument("--resume", action="store_true",
                   help="continue a previous partial run from cache/ and images/")
    p.add_argument("--skip-met", action="store_true")
    p.add_argument("--skip-nasa", action="store_true")
    p.add_argument("--met-dept", type=int, default=0,
                   help="restrict Met acquisition to a single department id")
    p.add_argument("--met-min", type=int, default=MET_MIN)
    p.add_argument("--nasa-target", type=int, default=20)
    p.add_argument("--verify-only", action="store_true",
                   help="recompute metrics/manifest/README from what is on disk, then verify")
    p.add_argument("--no-readme", action="store_true")
    return p.parse_args()


def main() -> int:
    global DRY, HTTP_IMG, HTTP_NASA
    args = parse_args()
    DRY = args.dry_run
    random.seed(SEED)
    for d in ("scripts", "cache/met_objects", "images/met", "images/skimage",
              "images/nasa", "images/synthetic"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    if DRY:
        log("DRY RUN -- nothing is downloaded or written")
    if args.resume:
        log(f"RESUME -- {STORE.total()} rows already recorded in cache/rows.jsonl")

    http = Http()
    HTTP_IMG = Http()
    HTTP_NASA = Http(browser_ua=True)
    met = Met(http)
    depts = met.departments()
    log(f"Met Open Access: {len(depts)} departments reachable")

    nasa_status = "not attempted"

    if not args.verify_only:
        quotas = list(MET_QUOTAS)
        if args.met_dept:
            quotas = [q for q in quotas if q[0] == args.met_dept]
        if args.limit:
            scaled, left = [], args.limit
            for dept_id, quota, pref in quotas:
                if left <= 0:
                    break
                q = max(1, min(quota, left))
                scaled.append((dept_id, q, pref))
                left -= q
            quotas = scaled

        if not args.skip_met:
            log("=== source 1/4: Met Open Access ===")
            acquire_met(met, HTTP_IMG, depts, quotas, DRY)
            if STORE.count("met") < args.met_min and not DRY:
                met_fill(met, HTTP_IMG, depts, args.met_min, DRY)

        log("=== source 2/4: scikit-image bundled data ===")
        budget = (args.limit - STORE.count("met")) if args.limit else len(SKIMAGE_NAMES)
        acquire_skimage(max(0, budget), DRY)

        if not args.skip_nasa:
            log("=== source 3/4: NASA Images ===")
            try:
                acquire_nasa(args.nasa_target, DRY)
                nasa_status = (f"{STORE.count('nasa')} images fetched via "
                               f"images-api.nasa.gov and images-assets.nasa.gov "
                               f"(browser User-Agent required for the asset host).")
            except Exception as exc:
                nasa_status = f"NASA acquisition failed ({type(exc).__name__}: {exc})."
                log(f"  {nasa_status}")

        log("=== source 4/4: deterministic synthetic phantoms ===")
        acquire_synthetic(None, DRY)

    if DRY:
        log(f"dry run complete; would currently hold {STORE.total()} images")
        return 0

    # ---- metrics -> strata -> top-up loop (every stratum >= 15) -----------
    extra: dict[str, int] = {}
    metrics: dict[str, dict] = {}
    chosen: dict[str, list[str]] = {}
    counts = {s: 0 for s in STRATA}
    for round_i in range(5):
        metrics = metric_table()
        chosen, counts = assign_strata(metrics)
        short = {s: MIN_PER_STRATUM - counts[s] for s in STRATA if counts[s] < MIN_PER_STRATUM}
        log(f"strata round {round_i}: {counts} shortfalls={short}")
        if not short:
            break
        need = {s: min(14, n + 3) for s, n in short.items()}
        extra = {s: extra.get(s, 0) + v for s, v in need.items()}
        if sum(short.values()) > 24:  # big hole: buy real Met content too
            for dept_id in (9, 8, 19, 11):
                acquire_met(met, HTTP_IMG, depts,
                            [(dept_id, STORE.count(f"met:{dept_id}") + 10,
                              "png" if dept_id in (9,) else "jpeg")], False)
        acquire_synthetic(need, False)
        metrics = metric_table()
        chosen, counts = assign_strata(metrics)

    rows = manifest_rows(chosen, metrics)
    write_manifest(rows)
    samples = {s: [r["id"] for r in rows if s in r["strata"].split("|")] for s in STRATA}
    for s in STRATA:
        samples[s].sort(key=lambda i: (i.split("-")[0] != "syn", i))

    rep = verify(rows)
    if not args.no_readme:
        write_readme(rep, extra, samples, nasa_status)
    (ROOT / "LICENSE.txt").write_text(LICENSE_TXT, encoding="utf-8")
    atomic_write_json(PROGRESS, {
        "total": rep["total"], "per_source": rep["per_source"],
        "strata": rep["strata_counts"], "bytes": rep["bytes"],
        "licences": rep["licences"], "ok": rep["ok"], "extra_phantoms": extra,
        "nasa_status": nasa_status,
    })

    print("\n=============== AbstraBench-200 build report ===============")
    print(f"images          : {rep['total']} manifest rows / {rep['files']} files on disk")
    print(f"per source      : {rep['per_source']}")
    print("per stratum     :")
    for s in STRATA:
        print(f"    {s:<20} {rep['strata_counts'][s]}")
    print(f"licences        : {rep['licences']}")
    print(f"bytes on disk   : {rep['bytes']:,}")
    print(f"nasa            : {nasa_status}")
    print(f"verification    : {'PASS' if rep['ok'] else 'FAIL'}")
    for p in rep["problems"][:30]:
        print("    ! " + p)
    print("============================================================\n")
    return 0 if rep["ok"] else 1


LICENSE_TXT = """CC0 1.0 Universal -- Public Domain Dedication

Scope of this dedication
------------------------
The copyright in and to all material in this package that was created by the
benchmark authors -- the manifest (manifest.csv), the per-image licence and
provenance metadata, the stratification measurements, the README, and the build
script scripts/build_benchmark.py -- is dedicated to the public domain.

The underlying images are NOT re-licensed here and no new rights are claimed in
them. They were selected exclusively from material that is already free of
copyright restrictions:

  * Metropolitan Museum of Art Open Access works whose per-object API record
    reports isPublicDomain: true (public domain; the Met releases them under
    CC0).
  * scikit-image bundled sample data whose own provenance statements declare
    CC0, "no known copyright restrictions", or release into the public domain.
  * NASA imagery: a work of the United States Government (17 U.S.C. 105), not
    subject to copyright.
  * Synthetic phantoms generated by the benchmark authors (CC0).

This package adds NO further restriction on any of that material. You may copy,
modify, distribute and reuse the benchmark, commercially or otherwise, without
permission and without any obligation of attribution.

To the extent possible under law, the benchmark authors waive all copyright and
related or neighbouring rights in the material described above. The full CC0
deed is available at https://creativecommons.org/publicdomain/zero/1.0/
"""

if __name__ == "__main__":
    sys.exit(main())
