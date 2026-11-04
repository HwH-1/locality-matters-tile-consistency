#!/usr/bin/env python3
"""Build an IEEEtran (TVCG-style) two-column submission from manuscript_tvcg.md.

The source uses raw LaTeX for floats and \\cite{key}; pandoc converts only the
prose.  References are numbered by BibTeX with IEEEtran.bst, figures are vector
PDF, and every float carries a real \\caption/\\label.

Paths are ASCII-only: LaTeX cannot handle the Chinese workspace directory name.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PAPER = Path(os.environ.get("TC_PAPER", "C:/Users/Jie/Desktop/论/tileconsistency/paper"))
FIGVEC = Path(os.environ.get("TC_FIGVEC", "C:/tc_work/figures"))
OUT = Path(os.environ.get("TC_IEEE_ROOT", "C:/tc_ieee"))
PANDOC = os.environ.get("TC_PANDOC", "pandoc")
SRC = PAPER / "manuscript_tvcg.md"

TITLE = ("Locality Matters: Tile-Consistent Content-Adaptive Anisotropic "
         "Kuwahara Filtering for Large-Canvas Abstraction")
KEYWORDS = ("image abstraction, anisotropic Kuwahara filter, tile consistency, "
            "out-of-core processing, reproducibility, evaluation protocol")

PRE = r"""\documentclass[journal]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{cite}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\hyphenation{Kuwa-hara}
\begin{document}
\title{__TITLE__}
\author{YingJie Wang%
\thanks{Y. Wang is with Handan College, Handan, Hebei, China.
E-mail: 717646407@qq.com. ORCID: 0009-0005-5898-0448.}%
}
\maketitle
\begin{abstract}
__ABSTRACT__
\end{abstract}
\begin{IEEEkeywords}
__KEYWORDS__
\end{IEEEkeywords}
"""

POST = r"""
\bibliographystyle{IEEEtran}
\bibliography{refs}
\end{document}
"""


def run(cmd, cwd=None, allow_fail=False):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if not allow_fail and r.returncode != 0:
        print("FAILED:", " ".join(str(c) for c in cmd))
        print((r.stdout or "")[-2500:])
        print((r.stderr or "")[-1500:])
        sys.exit(1)
    return r


def pandoc(text):
    p = subprocess.run([PANDOC, "-f", "markdown", "-t", "latex", "--wrap=none"],
                       input=text, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print("pandoc failed:", p.stderr[-1500:])
        sys.exit(1)
    return p.stdout


def clean(tex):
    # pandoc writes header anchors as \hypertarget{id}{% ... \section{..}\label{id}}
    lines = tex.split("\n")
    out, drop_next_brace = [], False
    for ln in lines:
        if ln.startswith(r"\hypertarget{"):
            drop_next_brace = True
            continue
        if drop_next_brace and (ln.startswith(r"\section") or ln.startswith(r"\subsection")):
            if ln.endswith("}"):
                ln = ln[:-1]
            drop_next_brace = False
        out.append(ln)
    tex = "\n".join(out)
    # pandoc emits \textasciitilde1 (no braces) before digits, \textasciitilde{}
    # elsewhere; both must become a non-breaking space, not a printed tilde.
    tex = re.sub(r"\\textasciitilde(?:\{\})?", "~", tex)
    # '\,' in prose loses its backslash, leaving ',%' instead of a thin space
    tex = tex.replace(r",\%", r"\,\%")
    # an en dash between letters is a compound word, not a numeric range
    tex = re.sub(r"(?<=[A-Za-z])\\textendash\{\}(?=[A-Za-z])", "-", tex)
    tex = re.sub(r"(?<=[A-Za-z])--(?=[A-Za-z])", "-", tex)
    return tex


def build_bib(text):
    """BibTeX mis-parses '%' comment lines containing '@cite'; strip them and
    spell the three non-ASCII author names with LaTeX accents for IEEEtran.bst."""
    kept = [ln for ln in text.splitlines() if not ln.lstrip().startswith("%")]
    t = "\n".join(kept)
    for raw, esc in [("\u00c1", r"{\'A}"), ("\u0103", r"{\u a}"),
                     ("\u00fa", r"{\'u}"), ("\u00e9", r"{\'e}"),
                     ("\u00f6", r'{\"o}'), ("\u00fc", r'{\"u}')]:
        t = t.replace(raw, esc)
    return t


def main():
    assert SRC.exists(), SRC
    md = SRC.read_text(encoding="utf-8")
    placeholder = "<!--N60-->" in md
    if placeholder:
        print("WARNING: n=60 placeholder still present; section VI-G is missing")
        md = md.replace("<!--N60-->", "")
    m = re.search(r"(?ms)^# Abstract\s*\n(.*?)(?=^# )", md)
    assert m, "no Abstract section"
    abstract_md = m.group(1).strip()
    body_md = md[m.end():]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    for p in FIGVEC.glob("*.pdf"):
        shutil.copy2(p, OUT / "figures" / p.name)
    (OUT / "refs.bib").write_text(build_bib((PAPER / "refs.bib").read_text(
        encoding="utf-8")), encoding="utf-8", newline="\n")

    abstract_tex = clean(pandoc(abstract_md)).strip()
    body_tex = clean(pandoc(body_md))
    full = (PRE.replace("__TITLE__", TITLE)
               .replace("__ABSTRACT__", abstract_tex)
               .replace("__KEYWORDS__", KEYWORDS) + body_tex + POST)
    (OUT / "main.tex").write_text(full, encoding="utf-8", newline="\n")
    print(f"main.tex: {len(full.splitlines())} lines, "
          f"{len(full.split())} tokens, {full.count(chr(92)+'cite{')} cite commands")

    # a stale .bbl from an earlier run can poison latex1 even after the bib is fixed
    for stale in ["main.aux", "main.bbl", "main.blg", "main.log", "main.pdf",
                  "main.out", "main.toc"]:
        (OUT / stale).unlink(missing_ok=True)

    log = ""
    for stage, cmd in [
        ("latex1", ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]),
        ("bibtex", ["bibtex", "main"]),
        ("latex2", ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]),
        ("latex3", ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]),
    ]:
        r = run(cmd, cwd=str(OUT), allow_fail=True)
        (OUT / f"{stage}.log").write_text((r.stdout or "") + "\n" + (r.stderr or ""),
                                          encoding="utf-8")
        log += (r.stdout or "")
        if r.returncode != 0:
            print(f"{stage} exited {r.returncode}")
            print((r.stdout or "")[-1800:])
            print((r.stderr or "")[-600:])

    texlog = (OUT / "latex3.log").read_text(encoding="utf-8", errors="replace")
    errs = [l for l in texlog.splitlines()
            if l.startswith("! ") or "Undefined control sequence" in l
            or "Missing character" in l or "Emergency stop" in l]
    warns = [l for l in texlog.splitlines() if "LaTeX Warning" in l]
    over = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", texlog)
    pdf = OUT / "main.pdf"
    pages = "?"
    if pdf.exists():
        info = run(["pdfinfo", str(pdf)], allow_fail=True).stdout
        mm = re.search(r"Pages:\s+(\d+)", info)
        if mm:
            pages = mm.group(1)
    print("PDF pages      :", pages)
    print("TeX errors     :", len(errs), errs[:5])
    print("overfull >20pt :", sum(1 for x in over if float(x) > 20), "of", len(over))
    print("warnings       :", len(warns), warns[:6])
    print("undefined cites:", texlog.count("Citation") , "| refs warnings:",
          sum(1 for w in warns if "Citation" in w or "Reference" in w))
    if errs:
        sys.exit(2)

    # publish the compiled package next to the source for the author
    dest = PAPER / "ieee"
    dest.mkdir(exist_ok=True)
    for f in ["main.tex", "main.pdf", "main.aux", "main.bbl", "refs.bib"]:
        if (OUT / f).exists():
            shutil.copy2(OUT / f, dest / f)
    (dest / "figures").mkdir(exist_ok=True)
    for p in (OUT / "figures").glob("*.pdf"):
        shutil.copy2(p, dest / "figures" / p.name)
    print("deliverable    :", dest)


if __name__ == "__main__":
    main()
