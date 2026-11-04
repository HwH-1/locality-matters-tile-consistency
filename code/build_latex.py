#!/usr/bin/env python3
"""Typeset the manuscript with LaTeX -> PDF (single command).

Input : paper/build/manuscript_ready.md  (produced by build.py: citations already
        resolved to [n], the 8 figures already inserted with their captions)
Output: paper/latex/manuscript.tex, paper/latex/manuscript.pdf

The script is deliberately strict: every structural anchor it depends on is
looked up by exact string and the build aborts before writing anything if an
anchor is missing or ambiguous.  Nothing here invents content; it only lays out
what manuscript_ready.md already says.
"""
import os
import re
import subprocess
import sys

ROOT = os.environ.get("TC_ROOT", "/data/tileconsistency")
PAPER = f"{ROOT}/paper"
BUILD = f"{PAPER}/build"
LTX = f"{PAPER}/latex"
READY = f"{BUILD}/manuscript_ready.md"
FIGDIR = os.environ.get("TC_FIGDIR", "../../figures/")
MD_SRC = f"{PAPER}/manuscript.md"

PRE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathptmx}
\usepackage[a4paper,top=2.4cm,bottom=2.6cm,left=2.5cm,right=2.5cm,footskip=26pt]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{calc}
\usepackage{needspace}
\usepackage{array}
\usepackage{multirow}
\usepackage{caption}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}
% automatic numbering is suppressed: section numbers live in the source text
\setcounter{secnumdepth}{0}
\setlength{\LTcapwidth}{\textwidth}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\captionsetup{font=small,labelfont=bf,skip=6pt,justification=raggedright,singlelinecheck=false}
\setlist[enumerate]{leftmargin=*,itemsep=2pt,topsep=3pt}
\setlist[itemize]{leftmargin=*,itemsep=1pt,topsep=2pt}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\setlength{\emergencystretch}{2.5em}
\setlength{\parskip}{2pt}
\setlength{\parindent}{0pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize\itshape Tile-consistent content-adaptive abstraction filters}
\fancyhead[R]{\footnotesize\itshape YingJie Wang}
\renewcommand{\headrulewidth}{0.4pt}
\fancyfoot[C]{\footnotesize\thepage}
"""


def die(msg):
    print(f"ABORT: {msg}")
    sys.exit(1)


def split_front(md):
    """Return (title, author, affiliation, keywords, abstract, body)."""
    lines = md.split("\n")

    def one(pred, what):
        hits = [i for i, l in enumerate(lines) if pred(l)]
        if len(hits) != 1:
            die(f"{what}: expected 1 anchor, found {len(hits)}")
        return hits[0]

    if not lines[0].startswith("# ") or lines[0].startswith("# Abstract"):
        die("title is not the first line of manuscript_ready.md")
    it = 0
    ia = one(lambda l: l.strip() == "YingJie Wang\\*", "author line")
    iu = one(lambda l: l.startswith("\\*Handan College"), "affiliation line")
    ik = one(lambda l: l.startswith("**Keywords.**"), "keywords line")
    iab = one(lambda l: l.strip() == "# Abstract", "abstract heading")
    ibody = one(lambda l: l.strip() == "# 1 Introduction", "first section")
    if not (it < ia < iu < ik < iab < ibody):
        die("front matter headings are out of order")

    def para(i):
        j = i + 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        return lines[j]

    affil = lines[iu].lstrip("\\\\").lstrip("*").strip()
    kws = lines[ik].replace("**Keywords.**", "").strip()
    return (lines[it][2:].strip(), "YingJie Wang", affil, kws, para(iab),
            "\n".join(lines[ibody:]))


UNIQ = {
    "\u00d7": r"\ensuremath{\times}",
    "\u2192": r"\ensuremath{\rightarrow}",
    "\u2264": r"\ensuremath{\leq}",
    "\u2265": r"\ensuremath{\geq}",
    "\u00b1": r"\ensuremath{\pm}",
    "\u2212": r"\ensuremath{-}",
    "\u00a7": r"\S{}",
    "\u00b7": r"\textperiodcentered{}",
    "\u2013": "--",
    "\u2014": "---",
    "\u2018": "`",
    "\u2019": "'",
    "\u201c": "``",
    "\u201d": "''",
    "\u2026": r"\ldots{}",
    "\u00a0": " ",
}

# Accented Latin letters in author names are spelled out as LaTeX commands rather than
# left to inputenc, so the build fails loudly on an unknown character instead of eating it.
UNIQ.update({
    "\u00e1": r"\'{a}", "\u00e9": r"\'{e}", "\u00ed": r"\'{i}",
    "\u00f3": r"\'{o}", "\u00fa": r"\'{u}", "\u00f1": r"\~{n}",
    "\u00e7": r"\c{c}", "\u00c1": r"\'{A}", "\u00c9": r"\'{E}",
    "\u00cd": r"\'{I}", "\u00d3": r"\'{O}", "\u00da": r"\'{U}",
    "\u00d1": r"\~{N}", "\u00c7": r"\c{C}",
    "\u00e4": r"\"{a}", "\u00f6": r"\"{o}", "\u00fc": r"\"{u}",
    "\u00c4": r"\"{A}", "\u00d6": r"\"{O}", "\u00dc": r"\"{U}",
    "\u00df": r"\ss{}",
    "\u0103": r"\u{a}", "\u0102": r"\u{A}",        # a-breve, as deposited for Sălăgean
    "\u0219": r"\c{t}", "\u0218": r"\c{T}",        # Romanian s/t-comma, set cedilla-style
    "\u021b": r"\c{s}", "\u021a": r"\c{S}",
})
ACCENTED = set()

FIG_RE = re.compile(
    r"\\begin\{figure\}\s*\\centering\s*"
    r"\\includegraphics(?:\[[^\]]*\])?\{(?P<path>[^}]*)\}\s*"
    r"\\caption\{(?P<alt>[^}]*)\}\s*\\end\{figure\}[ \t]*\n[ \t]*\n"
    r"(?P<cap>\\emph\{(?:[^{}]|\{[^{}]*\})*\})",
    re.S)


def merge_figures(tex):
    n = [0]

    def sub(m):
        n[0] += 1
        cap = re.sub(r"\s+", " ", m.group("cap")[len("\\emph{"):-1]).strip()
        return ("\\begin{figure}[htbp]\n\\centering\n"
                "\\includegraphics[width=0.9\\textwidth]{" + FIGDIR + re.sub(r"^.*?/figures/", "", m.group("path")) + "}\n"
                "\\caption*{" + cap + "}\n\\end{figure}")

    out = FIG_RE.sub(sub, tex)
    if "\\begin{figure}" in out and n[0] == 0:
        die("figure blocks present but none matched the expected caption pattern")
    return out, n[0]


LT_RE = re.compile(r"\\begin\{longtable\}\[\]\{(?P<spec>@\{\}.*?@\{\})\}", re.S)


BREAK_IN = (("\\_", "\\_\\allowbreak{}"), ("=", "=\\allowbreak{}"),
            ("/", "/\\allowbreak{}"))


def _softbreak(s):
    for a, b in BREAK_IN:
        s = s.replace(a, b)
    return s


CELL_RE = re.compile(r"(\\begin\{minipage\}[^\n]*\n)(.*?)(\\end\{minipage\})", re.S)
TT_RE = re.compile(r"\\texttt\{((?:[^{}]|\{[^{}]*\})*)\}")
URL_RE = re.compile(r"https?://[^\s\\}]+")


def add_breaks(tex):
    r"""Table cells and file paths contain long unbreakable tokens (sColor=75,
    abstrabench/scripts/build\_benchmark.py).  Give TeX legal break points so
    they cannot run into the margin, and do the same for the DOI strings."""
    tex = CELL_RE.sub(lambda m: m.group(1) + _softbreak(m.group(2)) + m.group(3), tex)
    tex = TT_RE.sub(lambda m: "\\texttt{" + _softbreak(m.group(1)) + "}", tex)
    return URL_RE.sub(lambda m: re.sub(r"([/.])", r"\1\\allowbreak{}", m.group(0)), tex)


def keep_captions(tex):
    """Keep a table caption on the same page as its longtable."""
    return re.sub(r"(?m)^(\\textbf\{Table (?:[A-Z])?\d+\.\})",
                  r"\\needspace{9\\baselineskip}\n\1", tex)


def size_tables(tex):
    """Shrink wide tables so they stay inside the text block."""
    hits = list(LT_RE.finditer(tex))
    for m in reversed(hits):
        spec = m.group("spec")
        cols = len(re.findall(r"p\{", spec)) or len(re.findall(r"[lcr]", spec))
        size = r"\scriptsize" if cols >= 7 else (r"\footnotesize" if cols >= 5 else r"\small")
        i0 = m.start()
        pre = "\\begingroup" + size + "\n"
        tex = tex[:i0] + pre + tex[i0:]
        k = tex.find("\\end{longtable}", i0 + len(pre))
        if k < 0:
            die("unterminated longtable")
        k += len("\\end{longtable}")
        tex = tex[:k] + "\\endgroup" + tex[k:]
    return tex, len(hits)


def main():
    for f, what in ((READY, "manuscript_ready.md"), (MD_SRC, "manuscript.md")):
        if not os.path.exists(f):
            die(f"{what} missing - run build.py first")
    title, author, affil, keywords, abstract, body = split_front(open(READY, encoding="utf-8").read())

    os.makedirs(LTX, exist_ok=True)

    def texify(s):
        """Run one front-matter string through pandoc so markdown emphasis etc.
        becomes LaTeX, exactly as the body does."""
        r = subprocess.run(["pandoc", "-f", "markdown+east_asian_line_breaks", "-t", "latex"],
                           input=s, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode:
            die("pandoc (front matter) failed: " + r.stderr[-400:])
        return " ".join(r.stdout.split("\n")).strip()

    tmp = f"{LTX}/_body.md"
    open(tmp, "w", encoding="utf-8", newline="\n").write(body)
    r = subprocess.run(["pandoc", tmp, "-f", "markdown+east_asian_line_breaks",
                        "-t", "latex", "-o", f"{LTX}/_body.tex"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        die("pandoc failed: " + r.stderr[-800:])
    tex = open(f"{LTX}/_body.tex", encoding="utf-8").read()

    tex, nfig = merge_figures(tex)
    tex = add_breaks(tex)
    tex = keep_captions(tex)
    tex, ntab = size_tables(tex)

    head = (PRE + "\\begin{document}\n\n"
            "{\\centering\n"
            f"{{\\LARGE\\bfseries {texify(title)}}}\\par\n"
            "\\vspace{1.4em}\n"
            f"{{\\large {texify(author)}}}\\par\n"
            "\\vspace{0.5em}\n"
            f"{{\\small {texify(affil)}}}\\par\n"
            "\\par\n}\n\n"
            "\\vspace{1.2em}\n"
            "\\begin{abstract}\n"
            f"{texify(abstract)}\n\n"
            "\\textbf{Keywords.} " + texify(keywords) + "\\par"
            "\\end{abstract}\n\n"
            "\\vspace{0.6em}\n")
    full = head + tex + "\n\\end{document}\n"

    if "\\section{References}" in full:
        full = full.replace("\\section{References}",
                            "\\section{References}\\begingroup\\small\\setlength{\\parskip}{0.8pt}", 1)
        full = full.replace("\\end{document}", "\\endgroup\n\\end{document}", 1)

    for k, v in UNIQ.items():
        full = full.replace(k, v)
    left = sorted({c for c in full if ord(c) > 127} - ACCENTED)
    if left:
        die(f"unmapped non-ASCII characters: {[hex(ord(c)) for c in left]}")

    out_tex = f"{LTX}/manuscript.tex"
    open(out_tex, "w", encoding="utf-8", newline="\n").write(full)
    for junk in (tmp, f"{LTX}/_body.tex"):
        os.remove(junk)

    print(f"figures merged : {nfig}")
    print(f"longtables sized: {ntab}")
    print(f"wrote {out_tex} ({os.path.getsize(out_tex)} bytes)")

    log = []
    for i in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-file-line-error",
                            "manuscript.tex"], cwd=LTX, capture_output=True, text=True, encoding="utf-8", errors="replace")
        log.append(r.stdout)
        if "! " in r.stdout and "Emergency stop" in r.stdout:
            break
    open(f"{LTX}/manuscript.log", "w", encoding="utf-8").write(log[-1])

    pdf = f"{LTX}/manuscript.pdf"
    if not os.path.exists(pdf):
        die("no PDF produced - see paper/latex/manuscript.log")
    err = [l for l in log[-1].split("\n")
           if l.startswith("! ") or "Undefined control sequence" in l
           or "Missing character" in l or "Emergency stop" in l or "Misplaced" in l]
    over = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log[-1])
    warn = [l for l in log[-1].split("\n") if "LaTeX Warning" in l]
    if err:
        die("TeX reported %d error line(s); the PDF is not trustworthy: %r"
            % (len(err), err[:3]))
    try:
        info = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    except (OSError, IndexError):
        info = ""
    pages = re.search(r"Pages:\s+(\d+)", info)
    if not pages:
        pages = re.search(r"Output written on .*?\((\d+) pages", log[-1])
    print("PDF pages      :", pages.group(1) if pages else "?")
    print("TeX errors     :", len(err), err[:4])
    print("overfull >20pt :", sum(1 for x in over if float(x) > 20), "of", len(over))
    print("warnings       :", len(warn), warn[:4])
    print("pdf size       :", os.path.getsize(pdf))


if __name__ == "__main__":
    main()
