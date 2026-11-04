#!/usr/bin/env python3
"""Build the 10-page condensed manuscript (paper/manuscript_short.md).

Reuses the main pipeline's citation resolver (build.py) and LaTeX
post-processors (build_latex.py) so the short version typesets exactly like
the full one, but writes to paper/short/ and reads its own markdown.

Usage:  python build_short.py [ROOT]
"""
import os
import re
import subprocess
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import build as B          # citation resolution
import build_latex as L    # preamble + tex post-processing

PAPER = f"{ROOT}/paper"
OUT = f"{PAPER}/short"
SRC = f"{PAPER}/manuscript_short.md"
READY = f"{OUT}/manuscript_ready.md"

os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- citations
bib = B.parse_bib(f"{PAPER}/refs.bib")
md = open(SRC, encoding="utf-8").read()
order, missing = [], set()
txt = B.build_text(md, bib, order, missing)

refs = ["# References", ""]
for i, k in enumerate(order, 1):
    refs.append(f"{i}. " + B.fmt_entry(*bib[k]))
txt = txt.rstrip() + "\n\n" + "\n".join(refs) + "\n"
open(READY, "w", encoding="utf-8").write(txt)

print(f"citations resolved: {len(order)} | bib entries: {len(bib)}")
if missing:
    print("MISSING KEYS:", missing)
unused = set(bib) - set(order)
if unused:
    print("UNCITED (in bib, not in short version):", sorted(unused))

# ---------------------------------------------------------------- front matter
title, author, affil, keywords, abstract, body = L.split_front(txt)


def texify(s):
    r = subprocess.run(["pandoc", "-f", "markdown+east_asian_line_breaks", "-t", "latex"],
                       input=s, capture_output=True, text=True)
    if r.returncode:
        L.die("pandoc (front matter) failed: " + r.stderr[-400:])
    return " ".join(r.stdout.split("\n")).strip()


tmp = f"{OUT}/_body.md"
open(tmp, "w", encoding="utf-8").write(body)
r = subprocess.run(["pandoc", tmp, "-f", "markdown+east_asian_line_breaks",
                    "-t", "latex", "-o", f"{OUT}/_body.tex"],
                   capture_output=True, text=True)
if r.returncode:
    L.die("pandoc failed: " + r.stderr[-800:])
tex = open(f"{OUT}/_body.tex", encoding="utf-8").read()

tex, nfig = L.merge_figures(tex)
tex = L.add_breaks(tex)
# tex = L.keep_captions(tex)   # short version: tables are small; needspace costs a page
tex, ntab = L.size_tables(tex)

head = (L.PRE + "\\begin{document}\n\n"
        "{\\centering\n"
        f"{{\\LARGE\\bfseries {texify(title)}}}\\par\n"
        "\\vspace{1.2em}\n"
        f"{{\\large {texify(author)}}}\\par\n"
        "\\vspace{0.4em}\n"
        f"{{\\small {texify(affil)}}}\\par\n"
        "\\par\n}\n\n"
        "\\vspace{1.0em}\n"
        "\\begin{abstract}\n"
        f"{texify(abstract)}\n\n"
        "\\textbf{Keywords.} " + texify(keywords) + "\\par"
        "\\end{abstract}\n\n"
        "\\vspace{0.5em}\n")
full = head + tex + "\n\\end{document}\n"

for k, v in L.UNIQ.items():
    full = full.replace(k, v)
left = sorted({c for c in full if ord(c) > 127} - L.ACCENTED)
if left:
    L.die(f"unmapped non-ASCII characters: {[hex(ord(c)) for c in left]}")

lines = full.split("\n")
for i, ln in enumerate(lines):
    if "\\section{References}" in ln:
        lines.insert(i + 1, "\\begingroup\\small\\setlength{\\parskip}{0.8pt}")
        break
for i in range(len(lines) - 1, -1, -1):
    if lines[i].startswith("\\end{document}"):
        lines.insert(i, "\\endgroup")
        break
full = "\n".join(lines)
out_tex = f"{OUT}/manuscript.tex"
open(out_tex, "w", encoding="utf-8").write(full)
for junk in (tmp, f"{OUT}/_body.tex"):
    os.remove(junk)
print(f"figures merged : {nfig}")
print(f"longtables sized: {ntab}")

for _ in range(2):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-file-line-error",
                    "manuscript.tex"], cwd=OUT, capture_output=True, text=True)
log = open(f"{OUT}/manuscript.log", encoding="utf-8", errors="ignore").read()
err = [l for l in log.split("\n")
       if l.startswith("! ") or "Undefined control sequence" in l
       or "Missing character" in l or "Emergency stop" in l]
if err:
    L.die("TeX errors: %r" % err[:3])
pages = re.search(r"Output written on .*?\((\d+) pages", log)
print("PDF pages      :", pages.group(1) if pages else "?")
print("pdf size       :", os.path.getsize(f"{OUT}/manuscript.pdf"))
