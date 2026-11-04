import re
import subprocess
import sys
from docx import Document

ROOT = "/data/tileconsistency"
md = open(f"{ROOT}/paper/manuscript.md").read()
ready = open(f"{ROOT}/paper/build/manuscript_ready.md").read()
fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("  " + detail if detail else ""))
    if not cond:
        fails.append(name)


# (a) 6.9 stays machine-generated
draft = open(f"{ROOT}/paper/sec_69_draft.md").read().strip()
i = md.find("## 6.9 ")
j = md.find("## 6.10", i)
j = md.find("\n## ", i + 5) if j < 0 else j
merged = md[i:j].rstrip()
head = merged.split("\n")
# the draft is the generated body; compare on the draft's own extent
check("6.9 merged text contains the generated draft verbatim", draft[:400] in merged
      and draft[-400:] in merged, f"len(merged)={len(merged)} len(draft)={len(draft)}")

# (b) tables
caps = [int(m.group(1)) for m in re.finditer(r"^\*\*Table (\d+)\.\*\*", md, re.M)]
check("table captions 1..14 in order", caps == list(range(1, 15)), str(caps))
uncited = [n for n in caps if len(re.findall(rf"Table {n}\b", md.replace(f"**Table {n}.**", ""))) == 0]
check("every table cited in prose", not uncited, str(uncited))

# (c) figures (captions live in build.py FIGS, text refs in manuscript.md)
figs = re.findall(r"Figure (\d+)\.", ready)
check("8 figure captions in ready file", len(set(figs)) == 8, str(sorted(set(figs))))
missing = [n for n in range(1, 9) if f"Figure {n}" not in md]
check("every figure referenced in manuscript prose", not missing, str(missing))

# (d) markers
check("no unresolved [@key] in ready file", "[@" not in ready)
check("no unresolved {{TOKEN}} in ready file", "{{" not in ready)
check("no bare cite keys in ready file", not re.search(r"(?<![\w@])@[a-z]{4,}", ready))

# (e) section cross-references
# H1 headings ("# 7 Discussion") are real sections too, and an inline-code span
# ("`out/broad_qa.md §0`") names a file, not a section of this paper — so match ^# and
# blank out code spans before collecting references.
heads = set()
for line in md.split("\n"):
    m = re.match(r"^#{1,4} (\d+(?:\.\d+)*)", line)
    if m:
        heads.add(m.group(1))
prose = re.sub(r"`[^`\n]*`", " ", md)
refs = set(re.findall(r"Section (\d+(?:\.\d+)*)", prose)) | set(re.findall(r"§(\d+(?:\.\d+)*)", prose))
dangling = sorted(r for r in refs if r not in heads and not any(h.startswith(r + ".") for h in heads))
check("every Section X.Y reference resolves", not dangling, str(dangling))

# (f) DOCX
d = Document(f"{ROOT}/paper/build/manuscript.docx")
ps = [p.text for p in d.paragraphs]
check("DOCX embeds 8 images", len([r for r in d.part.rels.values() if "image" in r.reltype]) == 8)
check("DOCX has 14 tables", len(d.tables) == 14)
absn = [t for t in ps if t.startswith("Content-adaptive abstraction filters pick")]
check("DOCX abstract is 248 words", absn and len(absn[0].split()) == 248,
      str(len(absn[0].split())) if absn else "absent")
# pandoc puts the list marker in numPr, not in paragraph text, so "^\d+\.\s" never matches.
# Count the non-empty paragraphs that follow the "References" heading instead, and require
# every one of them to carry a DOI/arXiv identifier.
ridx = [k for k, t in enumerate(ps) if t.strip() == "References"]
check("DOCX has a References heading", len(ridx) == 1, str(len(ridx)))
refps = [t.strip() for t in ps[ridx[0] + 1:] if t.strip()] if ridx else []
no_id = [t[:60] for t in refps if "doi.org" not in t and "arxiv.org" not in t]
check("DOCX prints 36 numbered references", len(refps) == 36, f"count={len(refps)}")
check("every printed reference carries a DOI/arXiv id", not no_id, str(no_id[:4]))

# (g) PDF
info = subprocess.run(["pdfinfo", f"{ROOT}/paper/latex/manuscript.pdf"],
                      capture_output=True, text=True).stdout
check("PDF is 34 pages", "Pages:           34" in info or re.search(r"Pages:\s+34", info))
log = open(f"{ROOT}/paper/latex/manuscript.log").read()
check("PDF build log free of undefined control sequences", "Undefined control sequence" not in log)
check("PDF build log free of Missing character", "Missing character" not in log)
imgs = subprocess.run(["pdfimages", "-list", f"{ROOT}/paper/latex/manuscript.pdf"],
                      capture_output=True, text=True).stdout
check("PDF embeds 8 images", len(re.findall(r"\bimage\b", imgs)) >= 16, f"rows={len(imgs.splitlines())-1}")
# the PDF is a second, independent render — a resolver bug can fix DOCX and leave the PDF stale
pdftxt = subprocess.run(["pdftotext", f"{ROOT}/paper/latex/manuscript.pdf", "-"],
                        capture_output=True, text=True).stdout
lit = re.findall(r"(?<![\w@])@[a-z]{4,}", pdftxt)
check("PDF text has no unresolved cite keys", not lit, str(sorted(set(lit))[:6]))

# (h) bib parser must not silently drop fields (it once deleted a DOI, printing an
# un-identifiable reference while reporting 0 errors)
sys.path.insert(0, ROOT)
import build as B

bib_txt = open(f"{ROOT}/paper/refs.bib").read()
bib_txt = "\n".join(l for l in bib_txt.split("\n") if not l.lstrip().startswith("%"))
parsed = B.parse_bib(f"{ROOT}/paper/refs.bib")
loss = []
for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", bib_txt, re.S):
    key = m.group(2).strip()
    raw = set(re.findall(r"^\s*(\w+)\s*=", m.group(3), re.M))
    got = set(parsed[key][1])
    if raw - got:
        loss.append((key, sorted(raw - got)))
check("no bib field silently dropped by the parser", not loss, str(loss[:4]))
noident = [k for k, (t, f) in parsed.items() if not (f.get("doi") or f.get("eprint") or f.get("url"))]
check("every bib entry has a doi / eprint / url", not noident, str(noident))

print()
print("WORD COUNT manuscript.md:", len(md.split()))
print("RESULT:", "ALL CHECKS PASS" if not fails else f"{len(fails)} FAILURES: {fails}")
sys.exit(1 if fails else 0)
