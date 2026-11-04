"""Build a submission-ready DOCX without pandoc-citeproc / LaTeX.

pandoc 2.9 in this sandbox has no --citeproc, so citations are resolved here:

  1. parse paper/refs.bib
  2. find every [@key] / @key in manuscript.md, in order of first appearance,
     and replace with numbered refs ([1], [1,3], "Author et al. [2]") -- the
     numbered style The Visual Computer uses
  3. emit a numbered References section with the verified metadata
  4. insert the generated figures inline at their sections (skip-if-missing)
  5. call pandoc to produce paper/build/manuscript.docx

Everything stays generated from manuscript.md + refs.bib: never edit the .docx
or build/manuscript_ready.md by hand.
"""
import os
import re
import subprocess

ROOT = os.environ.get("TC_ROOT", "/data/tileconsistency")
PAPER = f"{ROOT}/paper"
BUILD = f"{PAPER}/build"

# ---------------------------------------------------------------- bib parsing
def split_fields(body):
    """Brace-aware bib field parser: {Kuwahara} inside a value must not end it."""
    fields, i, n = {}, 0, len(body)
    while i < n:
        m = re.search(r"(\w+)\s*=\s*", body[i:])
        if not m:
            break
        name = m.group(1)
        j = i + m.end()
        if j < n and body[j] == "{":
            depth, k = 1, j + 1
            while k < n and depth:
                if body[k] == "{":
                    depth += 1
                elif body[k] == "}":
                    depth -= 1
                k += 1
            fields[name] = body[j + 1:k - 1].strip()
            i = k
        else:
            # Unbraced value (e.g. "month = jul,"). Stop at the comma/newline that precedes
            # the NEXT field name, and resume there — advancing past "name =" would drop that
            # field entirely (this silently deleted gastal2012amf's doi for a while).
            m2 = re.search(r",?\s*\n\s*\w+\s*=|\Z", body[j:])
            end = j + (m2.start() if m2 else len(body) - j)
            fields[name] = body[j:end].strip().rstrip(",")
            i = max(end, j)
        i += 1
    return fields


def parse_bib(path):
    txt = open(path, encoding="utf-8").read()
    # BibTeX comment lines can quote other people's entry headers (e.g.
    # "% ximgproc.bib:57-62 = @article{Kaiming15} ..."), which the regex below would
    # otherwise match and swallow the real entry that follows. Strip them first.
    txt = "\n".join(l for l in txt.split("\n") if not l.lstrip().startswith("%"))
    entries = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", txt, re.S):
        typ, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        entries[key] = (typ, split_fields(body))
    return entries


ACCENT = {'\\"o': "\u00f6", '\\"u': "\u00fc", '\\"a': "\u00e4", '\\"s': "\u00df",
          '\\"O': "\u00d6", '\\"U': "\u00dc", '\\&': '&', "''": "'"}


def plain(s):
    s = re.sub(r'\\"([a-zA-Z])', lambda m: ACCENT.get(m.group(0), m.group(1)), s)
    for k, v in ACCENT.items():
        s = s.replace(k, v)
    s = re.sub(r"\$_(\d+)\$", r"\1", s)          # L$_0$ -> L0 (no LaTeX in a DOCX ref list)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def fmt_authors(spec):
    """'Last, First and Last, First' -> 'Last, F. and Last, F.'"""
    out = []
    for a in [x.strip() for x in spec.split(" and ") if x.strip()]:
        if "," in a:
            last, first = [x.strip() for x in a.split(",", 1)]
            ini = ".".join(w[0] for w in first.split() if w) + "." if first else ""
            out.append(f"{last}, {ini}".strip().rstrip(","))
        else:
            out.append(a)
    if len(out) > 3:
        return ", ".join(out[:3]) + " et al."
    if len(out) > 1:
        return ", ".join(out[:-1]) + " and " + out[-1]
    return out[0]


def short_name(spec):
    auth = [a.strip() for a in spec.split(" and ") if a.strip()]
    last = auth[0].split(",")[0].strip()
    return last + (" et al." if len(auth) > 1 else "")


def fmt_entry(typ, f):
    parts = []
    if "author" in f:
        parts.append(fmt_authors(plain(f["author"])))
    title = plain(f.get("title", ""))
    parts.append(title + ".")
    if typ in ("article",):
        ct = plain(f.get("journal", ""))
        vol = f.get("volume", "")
        pg = f.get("pages", "")
        src = ct + (f", {vol}" if vol else "")
        src += (f"({f.get('number','')})" if f.get("number") else "")
        src += f": {pg.replace('--', '-')}" if pg else ""
        parts.append(src + f" ({f.get('year','')}).")
    elif typ == "misc":
        meta = plain(f.get("howpublished", "")) or plain(f.get("journal", ""))
        s = (meta + " " if meta else "") + f"({f.get('year','')})."
        parts.append(s)
    else:
        bt = plain(f.get("booktitle", f.get("series", "")))
        pub = plain(f.get("publisher", ""))
        pg = f.get("pages", "")
        s = "In: " + bt
        if pg:
            s += f", pp. {pg.replace('--','-')}"
        if pub:
            s += f". {pub}"
        s += f" ({f.get('year','')})."
        parts.append(s)
    extras = []
    if f.get("doi"):
        extras.append("https://doi.org/" + plain(f["doi"]))
    if f.get("eprint"):
        # arXiv-only preprint: give the reader the identifier and, when present, the link
        extras.append("arXiv:" + plain(f["eprint"]) + (f" {plain(f['url'])}" if f.get("url") else ""))
    elif f.get("url") and not f.get("doi"):
        extras.append(plain(f["url"]))
    if f.get("note"):
        extras.append("(" + plain(f["note"]) + ")")
    if extras:
        parts.append(" ".join(extras))
    return " ".join(parts)


# ---------------------------------------------------------------- citations
BRACKETED = re.compile(r"\[(@[^\]]+)\]")
INLINE = re.compile(r"(?<![\w@])@([\w:\-]+)")


def build_text(md, bib, order, missing):
    def rep_bracket(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        nums = []
        for k in keys:
            neg = k.startswith("-")
            kk = k.lstrip("-")
            if kk not in bib:
                missing.add(kk)
                continue
            if kk not in order:
                order.append(kk)
            nums.append(order.index(kk) + 1)
        if not nums:
            return ""
        return "[" + ", ".join(str(n) for n in sorted(nums)) + "]"

    def rep_inline(m):
        # An inline @key is a citation only if that key exists in refs.bib. Anything else
        # ("@0", "@48", an e-mail address) is left exactly as it is, so a typo shows up as a
        # literal @key in the output and is also reported below instead of deleting prose.
        k = m.group(1)
        if k not in bib:
            missing.add(k)
            return m.group(0)
        if k not in order:
            order.append(k)
        return f"{short_name(plain(bib[k][1].get('author','[author?]')))} [{order.index(k)+1}]"

    txt = BRACKETED.sub(rep_bracket, md)
    txt = INLINE.sub(rep_inline, txt)
    return txt


# ---------------------------------------------------------------- figures
FIGS = [
    ("## 6.1 ", "F1_tce_halo.png",
     "Figure 1. Tile-consistency error against overlap halo (median over the ten images, 4x4 grid, "
     "logarithmic axis). The two bounded-support operators fall to the exact floor, marked by open "
     "circles sitting on the “exact (0)” line; the two global-quantile configurations flatten at "
     "2.0e-3 (CA-AKF) and 1.3e-2 (SC-AKF). The dotted line is one 8-bit level."),
    ("## 6.2 ", "F2_bands.png",
     "Figure 2. Border-band decomposition at n=8, halo 16. The two global-quantile configurations are "
     "flat across the whole band range (1.75-2.00e-3 for CA-AKF, 8.1-9.7e-3 for SC-AKF), so their "
     "deviation sits in tile interiors rather than near the grid; the bounded-support operators are at "
     "5.2e-6 or zero beyond the first 2 px band. A decaying profile would have meant support truncation; "
     "a flat one means statistic non-locality."),
    ("## 6.5 ", "F4_metrics_blind.png",
     "Figure 3. Four of the six quality metrics cannot see the mismatch. Each panel plots one metric "
     "measured tile-wise against the same metric measured on the whole image, one point per image "
     "(CA-AKF, the class-B reference). Every point lies on the identity line: the mean absolute change "
     "is at most 0.0025 for SR, SSIM and FSR and 0.041 for ECE, over runs that differ on 12.6 % of pixels."),
    ("## 6.5 ", "F5_rank_flip.png",
     "Figure 4. Method ranking under whole-image versus tile-wise evaluation of the same ten images. "
     "One panel per metric, rank 1 = best; each thin line follows one method's rank between the two "
     "protocols, grey where the ranking is unchanged and red where it changes. Every red line is an "
     "inversion: 15 of the 40 image-metric cases, and none of them visible in a whole-image-only comparison."),
    ("## 6.7 ", "F6_canvas.png",
     "Figure 5. 2048x2048 public-domain canvas, tiled n=4 at halo 48. Columns: whole-canvas output "
     "crop, tiled output crop, and the deviation amplified x40 on a shared scale. The global-quantile "
     "configuration differs from whole-canvas processing even at 48 px of overlap; the local operator "
     "does not."),
    ("### 6.7.1", "F8_native.png",
     "Figure 6. Whole-canvas passes fail at 4096^2 and at native size, while every tile-wise pass "
     "completes; on the native crop the two bounded-support operators satisfy the one-level "
     "exactness gate at halo 32. (a) Peak RSS of whole-canvas versus tile-wise passes on four "
     "canvases: the assembled 2048^2 mosaic, the 4096^2 mosaic, a native 2048^2 crop of a "
     "4000x3853 painting, and that painting at full size. Two whole-canvas passes do not complete "
     "(kernel OOM at 4096^2; MemoryError at native size under a 5.5 GiB address-space cap). "
     "(b) The same halo ladder as Section 6.1 on the native crop: AKF is exact, LRS-AKF reaches "
     "1.4e-4 of a level, and the global-P95 operator plateaus at 4.6-5.0 % of pixels."),
    ("## 6.8 ", "F7_broad.png",
     "Figure 7. Breadth replication. Median fraction of pixels differing by more than one 8-bit level, "
     "as a function of the overlap halo, measured on 144 licence-audited images (n=4 tile grid). The "
     "local operator's curve reaches zero; the global-statistic operators plateau. AKF is drawn dashed "
     "with open markers because its curve coincides with LRS-AKF's."),
    ("## 6.9 ", "F9_cv.png",
     "Figure 8. The protocol applied to other people's code. (a) Median fraction of pixels differing "
     "by more than one 8-bit level versus overlap halo, for seven published algorithms as shipped in "
     "OpenCV 5.0.0 / opencv_contrib 5.0.0 on the same 144 images; dashed lines are our own operators "
     "at the Section 6.8 protocol. Bilateral filtering, guided filtering and non-local means converge "
     "to bit-exactness inside their pre-registered support bounds; L0 smoothing and Perona-Malik "
     "diffusion do not converge at any halo. (b) Determinism control: the fraction of images whose "
     "output changes when the same content is presented in a different memory state or to a second "
     "call. Six implementations are bit-identical; Perona-Malik diffusion is not, on every image."),
]


def insert_figs(txt):
    for anchor, fn, cap in FIGS:
        path = f"{ROOT}/figures/{fn}"
        if not os.path.exists(path):
            continue
        i = txt.find(anchor)
        if i < 0:
            continue
        block = f"\n![{cap.split('.')[0]}]({path})\n\n*{cap}*\n\n"
        txt = txt[:i] + block + txt[i:]
    return txt


def main():
    os.makedirs(BUILD, exist_ok=True)
    bib = parse_bib(f"{PAPER}/refs.bib")
    md = open(f"{PAPER}/manuscript.md", encoding="utf-8").read()
    order, missing = [], set()
    txt = build_text(md, bib, order, missing)

    refs = ["# References", ""]
    for i, k in enumerate(order, 1):
        refs.append(f"{i}. " + fmt_entry(*bib[k]))
    txt = txt.rstrip() + "\n\n" + "\n".join(refs) + "\n"

    txt = insert_figs(txt)
    out_md = f"{BUILD}/manuscript_ready.md"
    open(out_md, "w", encoding="utf-8", newline="\n").write(txt)

    unused = set(bib) - set(order)
    print("citations resolved:", len(order), "| bib entries:", len(bib))
    if missing:
        print("MISSING KEYS:", missing)
    if unused:
        print("UNCITED (in bib, not referenced):", unused)

    cmd = ["pandoc", out_md, "-f", "markdown+east_asian_line_breaks",
           "-t", "docx", "--toc", "--toc-depth=2", "-M", "toc-title=Contents",
           "-o", f"{BUILD}/manuscript.docx",
           "--resource-path", f"{ROOT}:{PAPER}"]
    print(" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-2000:], r.stderr[-2000:])
    print("exit", r.returncode, os.path.getsize(f"{BUILD}/manuscript.docx") if r.returncode == 0 else "")


if __name__ == "__main__":
    main()
