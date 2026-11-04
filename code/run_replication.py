#!/usr/bin/env python3
"""Re-run the independent cross-platform replication cohort (23 images).

Writes sweep_replication_rerun.jsonl so the published records are never
overwritten. Run from the code/ directory after installing requirements.
"""
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

from config import CODE, TC, REFPARENT, RECORDS, IMAGES

os.environ["TC_SRC"] = str(TC)
os.environ["CAAKF_REF"] = str(REFPARENT)
sys.path.insert(0, str(CODE))

from run_sweep import one_image  # noqa: E402

OUT = RECORDS / "sweep_replication_rerun.jsonl"
METHODS = ["CA-AKF", "SC-AKF", "LRS-final", "AKF"]
WORKERS = 4


def primary_images():
    names = set()
    for line in (RECORDS / "sweep.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            names.add(json.loads(line)["img"])
    return names


def job(args):
    return one_image(*args)


def main():
    primary = primary_images()
    all_paths = sorted(glob.glob(str(IMAGES / "met" / "*.jpg")))
    shared = [p for p in all_paths if os.path.basename(p) in primary]
    extra = [p for p in all_paths if os.path.basename(p) not in primary][:13]
    paths = shared + extra
    assert len(paths) == 23, "expected 10 shared + 13 new Met images"
    jobs = [(p, 384, METHODS, False) for p in paths]
    t0 = time.time()
    with open(OUT, "w", encoding="utf-8") as fh, Pool(WORKERS) as pool:
        for i, rows in enumerate(pool.imap_unordered(job, jobs), 1):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 5 == 0 or i == len(jobs):
                print("[%d/%d] elapsed %.0fs" % (i, len(jobs), time.time() - t0), flush=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
