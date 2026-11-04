"""Merge the staged sections into paper/manuscript.md, once, with assertions.

Run AFTER the manuscript-fixing worker has finished (it owns manuscript.md until then).
Every edit is either applied and logged, or reported as not-found; nothing is silent.

  python3 merge_sections.py --dry-run
  python3 merge_sections.py
"""
import argparse
import re

M = "/data/tileconsistency/paper/manuscript.md"
D671 = "/data/tileconsistency/paper/sec_671_draft.md"
D69 = "/data/tileconsistency/paper/sec_69_draft.md"

log = []


def strip_comments(t):
    return re.sub(r"<!--.*?-->\n?", "", t, flags=re.S).strip() + "\n"


def need(cond, what):
    log.append(("OK  " if cond else "MISS") + "  " + what)
    return cond


def main(dry):
    md = open(M).read()
    orig_words = len(md.split())

    # ---- 1. Section 6.7.1, immediately before ## 6.8 ---------------------------
    body671_raw = open(D671).read()
    body671 = strip_comments(body671_raw.split("<!-- ===== REPLACEMENT")[0])
    i = md.find("## 6.8 Breadth")
    if need(i > 0, "anchor '## 6.8 Breadth'") and "### 6.7.1" not in md:
        md = md[:i] + body671 + "\n" + md[i:]
        log.append("     inserted 6.7.1 (%d words)" % len(body671.split()))
    elif "### 6.7.1" in md:
        log.append("     6.7.1 already present, skipped")

    # ---- 2. Section 6.9, immediately before # 7 --------------------------------
    body69 = open(D69).read().strip() + "\n"
    j = md.find("# 7 Threats to validity")
    if need(j > 0, "anchor '# 7 Threats to validity'") and "## 6.9 " not in md:
        md = md[:j] + body69 + "\n\n" + md[j:]
        log.append("     inserted 6.9 (%d words)" % len(body69.split()))
    elif "## 6.9 " in md:
        log.append("     6.9 already present, skipped")

    # ---- 3. Section 7, third limitation -> the native-resolution version --------
    # The replacement paragraph is kept in the 6.7.1 draft so that both live text
    # blocks have one source file.
    third_new = body671_raw.split("=====", 1)[1]
    third_new = third_new.split("-->", 1)[1].strip()
    k = md.find("# 7 Threats to validity")
    seg = md[k:]
    m = re.search(r"\nThird, \*[^*]*\*.*?\n\n", seg, re.S)
    if need(bool(m), "a 'Third, *...*' paragraph in Section 7"):
        before = seg[m.start():m.end()].strip()
        if third_new[:80] in md:
            log.append("     Section 7 third limitation already the native version, skipped")
        else:
            md = md[:k] + seg[:m.start()] + "\n" + third_new + "\n\n" + seg[m.end():]
            log.append("     replaced Section 7 third limitation\n       before: " + before[:120] + "…")

    # ---- 4. in-text figure renumbering (F7_broad becomes Figure 7) --------------
    # The availability paragraph is rewritten wholesale in step 5, so only the
    # body-text reference needs patching here.
    subs = [
        ("Figure 6 plots the halo ladder per stratum",
         "Figure 7 plots the halo ladder per stratum"),
        ("Figure 5 shows the same configuration visually",
         "Figure 5 shows the same configuration visually"),
    ]
    for a, b in subs:
        if a in md and a != b:
            md = md.replace(a, b, 1)
            log.append(f"OK    renumbered: {a[:48]!r}")
        else:
            log.append(f"skip  {a[:48]!r} (already correct or rewritten in step 5)")

    # ---- 5. availability: canonical script list with per-figure attribution ----
    canon = ("Implementation, sweep scripts and figure generators: `src/tc.py`, "
             "`src/metrics.py`, `src/structure.py`, `run_sweep.py`, `run_broad.py`, "
             "`run_rad.py`, `run_c.py`, `run_mem.py`, `run_giga.py`, `run_native.py`, "
             "`make_native.py`, `run_cv.py`, `probe_cv_det.py`, `probe_cv_bug.py`, "
             "`analyse.py` (Figures 1–4), `make_f6.py` (Figure 5), `make_f8.py` (Figure 6), "
             "`analyse_broad.py` (Figure 7), `analyse_cv.py` (Figure 8), `analyse_giga.py`, "
             "`analyse_theory.py` (Sections 3.1–3.2 tables), `build.py`, plus the two "
             "supervisor scripts (`broad_supervise.sh`, `native_supervise.sh`). `src/tc.py` "
             "imports the reference AKF pipeline that CA-AKF and SC-AKF are built on; it looks "
             "for that package on `sys.path` and honours the `CAAKF_REF` environment variable, "
             "and it raises an explanatory `ImportError` naming both if neither is set")
    a = re.search(r"Implementation, sweep scripts and figure generators:.*?supervisor scripts "
                  r"\(`broad_supervise\.sh`, `native_supervise\.sh`\)", md, re.S)
    if need(bool(a), "the availability script list"):
        if "run_cv.py" not in a.group(0):
            md = md[:a.start()] + canon + md[a.end():]
            log.append("     availability list rewritten (+run_cv/probe_cv/analyse_cv/make_f8)")
        else:
            log.append("     availability list already names run_cv.py, skipped")
    subs2 = [("The per-image records behind every number (`out/*.jsonl`"
              " and the `out/*.json` summaries)",
              "The per-image records behind every number (`out/*.jsonl`, including"
              " `out/native.jsonl`, `out/cv_all.jsonl`, `out/cv_det.jsonl` and"
              " `out/cv_bug.jsonl`, and the `out/*.json` summaries)")]
    for x, y in subs2:
        if x in md and "cv_all.jsonl" not in md:
            md = md.replace(x, y, 1)
            log.append("OK    availability: per-image records list extended")
        else:
            log.append("MISS  availability records sentence not matched")

    # ---- 6. front matter: thread the third-party result through abstract/§1/§7 --
    front = [
        # abstract
        ("The contribution is a design rule, an instrument, and an operator that satisfies the "
         "rule — not a better-tuned filter.",
         "The taxonomy is then tested on code we did not write: seven published algorithms "
         "shipped in OpenCV 5.0.0, each measured against a support bound registered before the "
         "ladder was run. Three bounded-support filters become bit-exact inside their bound on "
         "all 144 images; L0 smoothing is still wrong on 141/144 of them at a 48-pixel overlap, "
         "and the Perona–Malik implementation is not a function of pixel content at all (§6.9). "
         "The contribution is a design rule, an "
         "instrument, and an operator that satisfies the rule — not a better-tuned filter.",
         "tested on code we did not write"),
        # contribution 1
        ("so every exact halo we report in this paper is *measured*, not predicted (§6.8.3).",
         "so every exact halo we report in this paper is *measured*, not predicted (§6.8.3). The "
         "two classes are then identified in **other people's code**: seven published algorithms "
         "shipped in OpenCV 5.0.0, run under the same protocol with a support bound registered per "
         "method in advance, separate the way the taxonomy predicts — bilateral filtering, guided "
         "filtering and non-local means become bit-exact at or inside their bound on 144/144 "
         "images, adaptive-manifold and rolling-guidance filtering need more overlap than their "
         "parameter names (132/144 and 26/144 still wrong at halo 48), L0 smoothing is wrong on "
         "141/144 images at halo 48, and the Perona–Malik implementation returns different "
         "outputs for identical input on 40/40 images probed — a third failure mode the two-class "
         "taxonomy does not cover (§6.9).",
         "identified in **other people's code**"),
        # contribution 2
        ("**A measurement protocol, validated at two scales.**",
         "**A measurement protocol, validated at three scales.**",
         "validated at three scales"),
        ("the battery is run at $n=10$ and the central claim re-measured at $N=144$ images across "
         "three content types (§6.8).",
         "the battery is run at $n=10$, the central claim re-measured at $N=144$ images across "
         "three content types (§6.8), and both reproduced on a single native-resolution artwork "
         "where the whole-canvas reference cannot be executed at all (§6.7.1) and on seven "
         "third-party implementations (§6.9).",
         "and on seven third-party implementations"),
    ]
    for a, b, marker in front:
        if a in md and marker not in md:
            md = md.replace(a, b, 1)
            log.append(f"OK    front matter: {a[:44]!r}")
        else:
            log.append(f"skip  front matter: {a[:44]!r}")

    # ---- 7. Section 7: the version-pin threat, after the four limitations -------
    anchor7 = "and the analogue we do draw (data leakage across shards in learned pipelines, " \
              "Section 8) is labelled as an analogy rather than as evidence."
    add7 = ("\n\n*Third-party evidence is version-pinned and single-setting.* Section 6.9 measures "
            "OpenCV 5.0.0 / opencv_contrib 5.0.0 at one documented setting per method; it is a "
            "statement about that release, not about the algorithms in general, and the source "
            "files and line ranges are quoted so a reader can re-check them against the tag. The "
            "pre-registered support bounds are our reading of each published algorithm, not the "
            "library's claim, and one of the seven implementations does not match the algorithm "
            "it documents (its second conductivity is declared but unimplemented), which is "
            "reported in §6.9 rather than quietly absorbed.")
    if need(anchor7 in md, "the end of Section 7's four limitations"):
        if "version-pinned" not in md:
            md = md.replace(anchor7, anchor7 + add7, 1)
            log.append("     added version-pin threat to Section 7")

    # ---- 8. Section 7: the two paragraphs that now contradict Section 6.9 -------
    rewrites8 = [
        ("and we make no claim that any third-party implementation uses a global statistic; "
         "the audit of external codebases is left open.",
         "and Section 6.9 measures seven published algorithms in third-party code, where the "
         "taxonomy is confirmed on operators we did not write; what remains open is the audit "
         "of the adaptive-Kuwahara literature's own codebases, none of which publishes its "
         "threshold path.",
         "make no claim that any third-party"),
        ("Section 8 states the design rule for the wider class, but we have not audited "
         "third-party codebases for global statistics.",
         "Section 8 states the design rule for the wider class, and Section 6.9 audits seven "
         "published implementations of neighbouring operators at one pinned version and one "
         "documented setting each — a sample, not a census.",
         "we have not audited third-party codebases"),
    ]
    for old, new, marker in rewrites8:
        if old in md:
            md = md.replace(old, new, 1)
            log.append(f"OK    Section 7 aligned with 6.9: …{marker[:34]!r}")
        elif new in md:
            log.append(f"skip  Section 7 already aligned: {marker[:34]!r}")
        else:
            need(False, f"Section 7 sentence to align with 6.9: {marker[:34]!r}")

    print("\n".join(log))
    print(f"words: {orig_words} -> {len(md.split())} ({len(md.split()) - orig_words:+d})")
    if dry:
        print("DRY RUN — manuscript not written")
    else:
        open(M, "w").write(md)
        print("manuscript.md written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().dry_run)
