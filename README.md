# Locality Matters — public code and data release

Companion repository for *Locality Matters: Tile-Consistent Content-Adaptive
Anisotropic Kuwahara Filtering for Large-Canvas Abstraction* (YingJie Wang,
Handan College). The paper proves a correctness condition for tiled evaluation
of content-adaptive filters, supplies a measurement protocol (TCE), introduces
LRS-AKF, and shows that published quality numbers are not portable across
evaluation harnesses.

**Repository URL:** https://github.com/HwH-1/locality-matters-tile-consistency
**Archival DOI:** pending
**Release date:** 2026-11-04  |  **Licence:** MIT (code), per-source terms (images)

> After pushing, run `python set_repo_url.py https://github.com/<user>/<repo> [DOI]`
> to fill the two fields above, update both manuscripts and refresh the checksums.

## What is here

| path | contents |
|---|---|
| `manuscript/` | TVCG two-column PDF + LaTeX/BibTeX/Markdown source; 38-page full version PDF + source |
| `code/` | all experiment, analysis, figure and build scripts; `tc/` local machinery; `reference/` the MIT-licensed JLR operator |
| `data/records/` | unaggregated per-image JSONL behind every table and figure, plus derived JSON reports |
| `data/benchmark/` | 286-image manifest, per-object licence audit, acquisition scripts, native-image provenance |
| `data/images/` | 254 Met CC0 records, 10 skimage references, 22 synthetic phantoms, 2 native originals and crops |
| `figures/` | 8 vector PDF figures + the output-comparison gallery |
| `docs/` | provenance map, reproducibility notes |
| `environment/` | pinned requirements and the two-stack comparison |
| `MANIFEST.sha256` | checksum of every file in this archive |

Total size is about 160 MB. The largest single file is 31 MB, below GitHub's
100 MB per-file limit. If you fork and the image set grows, consider Git LFS.

## Start here

1. `docs/PROVENANCE.md` maps each paper table and figure to its records and script.
2. `code/README.md` gives the exact commands to re-derive every number.
3. `environment/ENVIRONMENT.md` explains why the primary tables and the
   23-image replication cohort are **not** pooled (the two stacks are not
   bit-exact across Linux and Windows).

## Known limitations, stated plainly

- The 1/255 quantisation threshold is a convention, not a perceptual study.
- The Section 4.5 mechanism probe has no released record; no numbers are claimed for it.
- The four distributed implementations audited in Section 6.10 are not redistributed;
  only their identifiers, line ranges and hashes appear in the paper.
- Timing and peak-RSS figures were recorded on a shared host and are load-dependent.

## Cite

See `CITATION.cff`. Until the journal version is available, please cite the
manuscript in `manuscript/` together with this repository and its archival DOI.
