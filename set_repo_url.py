#!/usr/bin/env python3
"""Fill the public repository URL (and optional Zenodo DOI) after pushing.

Usage: python set_repo_url.py https://github.com/<user>/<repo> [10.5281/zenodo.xxxx]
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if len(sys.argv) < 2:
    sys.exit("usage: python set_repo_url.py <repo-url> [doi]")
url = sys.argv[1].rstrip("/")
doi = sys.argv[2] if len(sys.argv) > 2 else ""

OLD = ("The public repository URL and its archival DOI are supplied in the "
       "camera-ready version; this submission is accompanied by the complete "
       "release directory and its SHA-256 manifest.")
NEW = ("Code, data and per-image records are available at " + url +
       (" (archived at Zenodo, DOI " + doi + ")" if doi else "") +
       "; the release includes a SHA-256 manifest.")

for rel in ["manuscript/tvcg.md", "manuscript/full.md",
            "README.md", "CITATION.cff"]:
    p = ROOT / rel
    t = p.read_text(encoding="utf-8")
    t2 = t.replace(OLD, NEW)
    t2 = t2.replace("__REPO_URL__", url).replace("__REPO_DOI__", doi or "pending")
    if t2 != t:
        p.write_text(t2, encoding="utf-8")
        print("updated", rel)

lines = []
for p in sorted(ROOT.rglob("*")):
    if not p.is_file() or p.name == "MANIFEST.sha256":
        continue
    if ".git" in p.parts or "__pycache__" in p.parts or p.suffix in {".pyc", ".pyo"}:
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    lines.append(h + "  " + p.relative_to(ROOT).as_posix())
(ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("MANIFEST.sha256 refreshed:", len(lines), "files")
print("Remember to rebuild manuscript/tvcg.pdf and manuscript/full.pdf from the updated .md files.")
