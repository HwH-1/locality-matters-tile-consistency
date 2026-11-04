import glob
import os
import zipfile

ARCH = sorted(glob.glob('/data/CA-AKF_project_ALL_*.zip'))
names = set()
for a in ARCH:
    for i in zipfile.ZipFile(a).infolist():
        if not i.is_dir():
            names.add(i.filename)
print("parts:", len(ARCH), "| total distinct entries:", len(names),
      "| on-disk size:", round(sum(os.path.getsize(a) for a in ARCH) / 2**20), "MB")


def scan(root, skip=('__pycache__',)):
    out = []
    for dp, dn, fn in os.walk(root):
        if any(s in dp for s in skip):
            continue
        for f in fn:
            if f.endswith('.pyc'):
                continue
            out.append(os.path.join(dp, f))
    return out


for r in ['tileconsistency', 'abstrabench', 'consistency', 'reposition', 'zoom']:
    files = scan('/data/' + r)
    miss = [os.path.relpath(p, '/data') for p in files
            if os.path.relpath(p, '/data') not in names]
    print(f"[root] {r:<15} {len(files):>4} files  missing={len(miss)} {miss[:3]}")

checks = [
    ('JLR reference implementation', '/tmp/jlr/TheVisualComputer_JLR', 'external_JLR_reference_implementation/'),
    ('OpenCV contrib source read', '/tmp/opencv_contrib-5.0.0', 'opencv_source_read/'),
    ('OpenCV main source read', '/tmp/oc-main', 'opencv_main_source_read/'),
    ('3rd-party AKF baselines', '/tmp/ext', 'external_baseline_implementations/'),
    ('Crossref query dumps', '/tmp/xr', 'literature_query_dumps/'),
    ('your uploaded inputs', '/uploads', 'user_uploads/'),
]
print()
for label, src, pref in checks:
    files = scan(src)
    pool = {k.replace(pref, '') for k in names if k.startswith(pref)}
    miss = [os.path.relpath(p, src) for p in files
            if os.path.relpath(p, src) not in pool]
    print(f"[ext ] {label:<28} {len(files):>4} files  missing={len(miss)} {[os.path.basename(x) for x in miss[:3]]}")

npy = glob.glob('/tmp/*.npy')
print(f"\n[skip] derived .npy arrays not shipped: {len(npy)} files, "
      f"{round(sum(os.path.getsize(p) for p in npy)/2**20)} MB (regenerable: run_native.py)")
leftovers = [os.path.relpath(p, '/tmp') for p in scan('/tmp')
             if os.path.basename(p) not in {k.split('/')[-1] for k in names}]
groups = {}
for x in leftovers:
    groups[x.split('/')[0]] = groups.get(x.split('/')[0], 0) + 1
print("[skip] /tmp entries not shipped:", dict(sorted(groups.items(), key=lambda t: -t[1])))
