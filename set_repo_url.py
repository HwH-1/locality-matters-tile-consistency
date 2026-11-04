#!/usr/bin/env python3
"""Fill (or later update) the public repository URL and archival DOI.

Safe to run more than once: it replaces whatever URL/DOI sentence is currently
in the manuscripts, README and CITATION.cff.

Usage:
    python set_repo_url.py https://github.com/<user>/<repo>
    python set_repo_url.py https://github.com/<user>/<repo> 10.5281/zenodo.xxxx
"""
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if len(sys.argv) < 2:
    sys.exit("usage: python set_repo_url.py <repo-url> [doi]")

url = sys.argv[1].rstrip("/")
if url.endswith(".git"):
    url = url[:-4]
doi = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] not in ("", "pending") else ""

STATEMENT_RE = re.compile(
    r"(?:Code, data and per-image records are available at https://github\.com/"
    r"[^\s;)]*(?:\s*\(archived at Zenodo, DOI [^)]*\))?; the release includes a "
    r"SHA-256 manifest\.)"
    r"|(?:The public repository URL and its archival DOI are supplied in the "
    r"camera-ready version; this submission is accompanied by the complete "
    r"release directory and its SHA-256 manifest\.)"
)
NEW = ("Code, data and per-image records are available at " + url +
       (" (archived at Zenodo, DOI " + doi + ")" if doi else "") +
       "; the release includes a SHA-256 manifest.")

for rel in ["manuscript/tvcg.md", "manuscript/full.md"]:
    p = ROOT / rel
    t = p.read_text(encoding="utf-8")
    if STATEMENT_RE.search(t):
        t2 = STATEMENT_RE.sub(NEW, t)
    else:
        t2 = t.rstrip() + " " + NEW + "\n"
    if t2 != t:
        p.write_text(t2, encoding="utf-8")
        print("updated", rel)

p = ROOT / "README.md"
t = p.read_text(encoding="utf-8")
t2 = re.sub(r"\*\*Repository URL:\*\* .*", "**Repository URL:** " + url, t)
t2 = re.sub(r"\*\*Archival DOI:\*\* .*", "**Archival DOI:** " + (doi or "pending"), t2)
t2 = t2.replace("__REPO_URL__", url).replace("__REPO_DOI__", doi or "pending")
if t2 != t:
    p.write_text(t2, encoding="utf-8")
    print("updated README.md")

p = ROOT / "CITATION.cff"
t = p.read_text(encoding="utf-8")
t2 = re.sub(r"(?m)^url: .*$", 'url: "' + url + '"', t)
t2 = re.sub(r"(?m)^doi: .*$", 'doi: "' + (doi or "pending") + '"', t2)
if t2 != t:
    p.write_text(t2, encoding="utf-8")
    print("updated CITATION.cff")

lines = []
for f in sorted(ROOT.rglob("*")):
    if not f.is_file() or f.name == "MANIFEST.sha256":
        continue
    if ".git" in f.parts or "__pycache__" in f.parts or f.suffix in {".pyc", ".pyo"}:
        continue
    lines.append(hashlib.sha256(f.read_bytes()).hexdigest() + "  " + f.relative_to(ROOT).as_posix())
(ROOT / "MANIFEST.sha256").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
print("MANIFEST.sha256 refreshed:", len(lines), "files")
print("Public URL:", url, "| DOI:", doi or "pending")
print("Rebuild manuscript/tvcg.pdf and manuscript/full.pdf from the updated .md files.")
