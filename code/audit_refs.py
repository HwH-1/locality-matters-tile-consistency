"""Re-read every reference's metadata from the registry that owns its identifier.

Motivation: one entry (huang2018tiling) carried four wrong author names and had no
identifier at all. Everything else was typed by hand or by a sub-agent at some point, so
the whole list — not just the new entries — gets machine-checked against Crossref (DOI)
or the arXiv API (eprint). Entries with neither are reported as unverifiable.

Never guesses: a mismatch is printed with both strings; nothing is rewritten here.
Usage: python3 audit_refs.py
"""
import json
import re
import sys
import urllib.request

sys.path.insert(0, "/data/tileconsistency")
import build as B  # noqa: E402

ROOT = "/data/tileconsistency"
bib = B.parse_bib(f"{ROOT}/paper/refs.bib")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ref-audit/1.0 (mailto:2462768325@qq.com)",
                                               "Accept": "application/json"})
    return urllib.request.urlopen(req, timeout=30).read()


def surname(a):
    return norm(a.split(",")[0]) if "," in a else norm(a.split()[-1])


def strip_markup(s):
    """Crossref puts JATS tags in titles (<i>L</i><sub>0</sub>); registries are the source
    of truth for the string but the markup is not part of the title."""
    return re.sub(r"<[^>]+>", "", s or "")


def cr_year(m, doi=""):
    """IEEE conference records often have issued = [[null]]; 'created' is when the DOI was
    registered, not when the paper appeared (ICCV-98's record was created in 2002). So fall
    back to a 4-digit year in the event name, then to the year inside the DOI slug itself
    (10.1109/AVSS.2005.1577245), and only then give up rather than trust 'created'."""
    for k in ("issued", "published", "published-print", "published-online"):
        dp = (m.get(k) or {}).get("date-parts") or [[None]]
        if dp[0][0]:
            return str(dp[0][0])
    ev = (m.get("event") or {}).get("name", "") or ""
    y4 = re.search(r"(19|20)\d{2}", ev)
    if y4:
        return y4.group(0)
    slug = re.search(r"10\.1109/[A-Za-z]+\.(19|20)\d{2}\.", doi)
    if slug:
        return slug.group(0).split("/")[1].split(".")[1]
    return ""


def compare(key, f, src_title, src_authors, src_year, src):
    """Return (ok, detail). src is a short description of the registry."""
    issues = []
    mine_t = norm(B.plain(f.get("title", "")))
    if norm(src_title) != mine_t:
        issues.append(f"title mine={f.get('title')!r} reg={src_title!r}")
    mine_a = [surname(x) for x in re.split(r"\s+and\s+", B.plain(f.get("author", ""))) if x]
    reg_a = [surname(x) for x in src_authors if x]
    if len(mine_a) != len(reg_a) or any(m != r for m, r in zip(mine_a, reg_a)):
        issues.append(f"authors mine={mine_a} reg={reg_a}")
    mine_y = str(f.get("year", "")).strip()
    if src_year and str(src_year).strip()[:4] != mine_y[:4]:
        issues.append(f"year mine={mine_y} reg={src_year}")
    return (not issues), "; ".join(issues), src


rows = []
for key in sorted(bib):
    typ, f = bib[key]
    doi, ep = f.get("doi"), f.get("eprint")
    if doi:
        # Zenodo/Figshare DOIs are registered with DataCite, not Crossref, so 10.5281 goes
        # straight to the DataCite API; anything else tries Crossref and falls back.
        m = None
        try:
            if doi.startswith("10.5281/") or doi.startswith("10.6084/"):
                a = json.loads(get("https://api.datacite.org/dois/" + doi))["data"]["attributes"]
                ok, det, src = compare(key, f, strip_markup(a["titles"][0]["title"]),
                                       [c.get("name", "") for c in a.get("creators", [])],
                                       a.get("publicationYear", ""), "datacite")
            else:
                m = json.loads(get("https://api.crossref.org/works/" + doi))["message"]
                ok, det, src = compare(key, f, strip_markup((m.get("title") or [""])[0]),
                                       [f"{x.get('family','')}, {x.get('given','')}" for x in m.get("author", [])],
                                       cr_year(m, doi), "crossref")
        except Exception as e:  # network or 404
            ok, det, src = None, f"{type(e).__name__}: {e}", "crossref/doi"
    elif ep:
        try:
            xml = get("https://export.arxiv.org/api/query?id_list=" + ep).decode()
            t = re.search(r"<entry>.*?<title>(.*?)</title>", xml, re.S)
            if not t:
                ok, det, src = False, "arXiv returned no entry for this id", "arxiv"
            else:
                title = " ".join(t.group(1).split())
                auths = re.findall(r"<name>(.*?)</name>", xml)
                pub = re.search(r"<published>(\d{4})", xml).group(1)
                ok, det, src = compare(key, f, title, auths, pub, "arxiv")
        except Exception as e:
            ok, det, src = None, f"{type(e).__name__}: {e}", "arxiv"
    else:
        ok, det, src = None, "no doi and no eprint — cannot be machine-verified", "none"
    rows.append((key, typ, src, ok, det))

n_ok = sum(1 for r in rows if r[3] is True)
n_bad = [r for r in rows if r[3] is False]
n_un = [r for r in rows if r[3] is None]
for key, typ, src, ok, det in rows:
    tag = "OK  " if ok is True else ("BAD " if ok is False else "UNV ")
    print(f"{tag} {key:<26} {typ:<12} {src:<9} {det[:150]}")
print(f"\nverified-consistent {n_ok} | mismatched {len(n_bad)} | unverifiable {len(n_un)} | total {len(rows)}")
sys.exit(1 if n_bad else 0)
