# Two numeric stacks, and why they are not pooled

| | primary tables | independent replication cohort |
|---|---|---|
| OS | Linux (sandbox) | Windows 11 |
| Python | 3.12.13 | 3.13.14 |
| numpy / scipy | 2.5.3 / 1.18.1 | 2.5.3 / 1.18.1 |
| scikit-image | 0.26.0 | 0.26.0 |
| OpenCV | 5.0.0 + contrib | 5.0.0 + contrib |
| matplotlib | 3.11.2 | 3.11.2 |
| records | `data/records/sweep.jsonl`, `sweep_final.jsonl`, `broad_all.jsonl`, `cv_all.jsonl`, ... | `data/records/sweep_replication.jsonl` |

The operator source files and the image SHA-1 hashes are identical on both
stacks. Re-running the same image still gives a relative difference of 0.03 %
to 2 % in the `n8_h16` RMSE (the quantities being differenced are of order
1e-4), because of cross-platform floating-point behaviour. **The two stacks
are not bit-exact.** For that reason the replication cohort is reported as an
independent replication, never pooled with the primary quality table. See
Section 6.11 / Section VI-G of the manuscript.

Thread counts are set inside the scripts (`cv2.setNumThreads(1)` in
`run_cv.py` / `probe_cv_det.py`, 4 in `code/tc/tc.py`). Keep them.

For LaTeX output: pandoc 2.9.2.1 or newer, TeX Live / MiKTeX with IEEEtran,
booktabs, cite and calc. Build in an ASCII-only path; a path containing
non-ASCII characters triggers `unmapped non-ASCII` in pdfLaTeX.
