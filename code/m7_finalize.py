"""M7 pass 1: front matter, abstract, declarations, figure callouts, equation numbers.

Every edit is an exact-string replacement (no regex on the replacement side), so
LaTeX bodies like $p\\le5.6\\times10^{-19}$ cannot blow up the way re.sub's
template string does. The script aborts if any anchor is not found exactly once,
and only writes the file if all edits succeeded.
"""
import re
import sys

P = "/data/tileconsistency/paper/manuscript.md"
md = open(P).read()
edits = []


def rep(name, old, new, count=1):
    n = md.count(old)
    if n != count:
        print(f"ANCHOR FAIL [{name}]: found {n} expected {count}")
        sys.exit(1)
    edits.append((name, old, new))


# ---------------------------------------------------------------- 1. front matter
FRONT = """# Locality Matters: Tile-Consistent Content-Adaptive Anisotropic Kuwahara Filtering for Large-Canvas Abstraction

YingJie Wang\\*

\\*Handan College, Handan, Hebei, China. Corresponding author. E-mail: 717646407@qq.com. ORCID: 0009-0005-5898-0448

**Keywords.** non-photorealistic rendering · image abstraction · Kuwahara filter · anisotropic filtering · structure tensor · tiled and out-of-core processing · reproducibility

"""
rep("front-matter", "# Abstract\n\n", FRONT + "# Abstract\n\n")

# ---------------------------------------------------------------- 2. abstract
OLD_ABS = re.search(r"# Abstract\n\n(.*?)\n\n# 1 Introduction", md, re.S).group(1)
NEW_ABS = open("/tmp/abs_final.txt").read().strip()
rep("abstract", OLD_ABS, NEW_ABS)

# ---------------------------------------------------------------- 3. declarations
DECL = """# Declarations

**Funding.** The author received no specific funding, grant or other financial support for the research, authorship and publication of this work.

**Competing interests.** The author declares no competing interests.

**Ethics approval and consent to participate.** Not applicable: the study uses no human subjects, personal data or animal material. All stimuli are public-domain or CC0 images downloaded from the Metropolitan Museum of Art Open Access programme and from the Wikimedia Commons, plus synthetically generated phantoms; the per-file licence statements are in the released manifest (`abstrabench/manifest_met.csv`, `abstrabench/README.md`).

**Consent for publication.** Not applicable (no identifiable personal data or images of persons are reported).

**Author contributions.** Y. Wang conceived the study, derived the theoretical results, wrote and ran all experiment and analysis code, prepared the figures, and drafted and approved the manuscript.

**Acknowledgements.** None.

"""
rep("declarations", "# Data and code availability", DECL + "# Data and code availability")

# ---------------------------------------------------------------- 4. figure callouts
rep("fig1",
    "Table 1 gives TCE (median RMSE) as a function of the overlap halo for a $4\\times4$ grid.",
    "Table 1 gives TCE (median RMSE) as a function of the overlap halo for a $4\\times4$ grid, and "
    "Figure 1 plots the same ladder: the two local operators fall to machine exactness at their support "
    "width, the two global-quantile operators plateau.")
rep("fig2",
    "Border-band decomposition at $n=8$, halo 16 (mean $D$ per band, median over images):",
    "Border-band decomposition at $n=8$, halo 16, plotted in Figure 2 (mean $D$ per band, median over images):")
rep("fig3-4",
    "Across 4 metrics × 10 images, evaluating the *same* four filters tile-wise instead of whole-image "
    "inverts the method ranking in **15 of 40 cases**;",
    "Figure 3 shows that the established quality metrics are blind to the deviation that TCE measures, "
    "and Figure 4 plots the consequence: across 4 metrics × 10 images, evaluating the *same* four filters "
    "tile-wise instead of whole-image inverts the method ranking in **15 of 40 cases**;")
rep("fig6",
    "This subsection closes the gap on one artwork at its scanned resolution.",
    "This subsection closes the gap on one artwork at its scanned resolution; Figure 6 shows the memory "
    "comparison and the native-density ladder.")
rep("fig8",
    "opencv_contrib 5.0.0, on the same 144 benchmark images (1008\n"
    "$(\\text{image},\\text{method})$ records).",
    "opencv_contrib 5.0.0, on the same 144 benchmark images (1008\n"
    "$(\\text{image},\\text{method})$ records); Figure 8 shows the resulting ladder and the determinism control.")

# ---------------------------------------------------------------- 5. Section 6.4b
rep("sec64b", "(Section 6.4b)", "(Section 6.4)")

# ---------------------------------------------------------------- 6. equation numbers
rep("eq4",
    "$$\\tau = c\\,Q_p(v_0), \\qquad \\ell(x)=\\max\\{i: v_{r_i}(x)\\le\\tau\\},$$",
    "$$\\tau = c\\,Q_p(v_0), \\qquad \\ell(x)=\\max\\{i: v_{r_i}(x)\\le\\tau\\}, \\qquad (4)$$")
rep("eq5",
    "$$\\tau(x) = c\\cdot R(x), \\qquad R = \\mathcal B_{\\rho}\\!\\big[\\tilde v_0\\big],$$",
    "$$\\tau(x) = c\\cdot R(x), \\qquad R = \\mathcal B_{\\rho}\\!\\big[\\tilde v_0\\big], \\qquad (5)$$")

out = md
for name, old, new in edits:
    out = out.replace(old, new, 1)
open(P, "w").write(out)

a = re.search(r"# Abstract\n\n(.*?)\n\n# 1 Introduction", out, re.S).group(1)
print("edits applied:", len(edits))
print("abstract words:", len(a.split()))
print("total words:", len(out.split()))
print("figure callouts:", sorted(set(re.findall(r"Figure\s+(\d+)", out)), key=int))
print("headings:", re.findall(r"^# (.*)$", out, re.M))
