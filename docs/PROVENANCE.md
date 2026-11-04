# Every number in the paper traces to a file in this release

| paper item | records | regenerating script |
|---|---|---|
| Table 4 / Table I (halo ladder), Figure 1 | `data/records/sweep.jsonl`, `sweep_final.jsonl` | `code/run_sweep.py`, `code/fix_figs.py` |
| Table 6 / Table III (quality with effect sizes and CIs) | `sweep.jsonl`, `sweep_final.jsonl` | `code/qual_stats.py` -> `data/records/quality_effects.json` |
| Table 7 / Table IV (harness portability) | `sweep.jsonl`, `sweep_final.jsonl` | `code/run_sweep.py` (whole and tiled columns) |
| Tables 13-14 / Figure 7 (144-image breadth) | `broad_all.jsonl` | `code/run_broad.py`, `code/analyse_broad.py` |
| Table 15 / Figure 4 (seven OpenCV algorithms) | `cv_all.jsonl`, `cv_det.jsonl`, `cv_bug.jsonl` | `code/run_cv.py`, `code/analyse_cv.py` |
| Table 17 / Table VI (independent replication, n = 23) | `sweep_replication.jsonl` | `code/analyse_replication.py` -> `data/records/replication_report.json` |
| Figures 1-8 (vector) | `figures/*.pdf` | `code/fix_figs.py`, `make_f6.py`, `make_f8.py`, `analyse_cv.py` |
| Output-comparison gallery | `figures/F10_gallery.pdf` | `code/make_gallery.py` |
| Section 4.5 mechanism probe | **no released record** | the paper therefore states no numbers for it |

`code/README.md` gives the exact commands. `MANIFEST.sha256` lets a reader
verify that the archive they downloaded is the one described here.
