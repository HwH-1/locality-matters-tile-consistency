# Code map

All paths are resolved in `config.py`; run scripts from this directory.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r ../environment/requirements.txt
```

## Re-derive the published statistics (no GPU, seconds)

```bash
python qual_stats.py                 # Table 6 / Table III: effect sizes, bootstrap CIs
python analyse_replication.py        # Table 17 / Table VI: independent cohort, n=23
```

Both write their JSON reports next to the source records in `data/records/`.

## Re-run an experiment (minutes to hours)

```bash
export TC_SRC=$PWD/tc
export CAAKF_REF=$PWD/reference
python run_sweep.py   --glob "../data/images/met/*.jpg" --limit 10   --methods CA-AKF,SC-AKF,LRS-final,AKF   --out ../data/records/sweep_rerun.jsonl --workers 4
```

`CAAKF_REF` points at `code/reference/` because `tc/tc.py` imports the
operator as `from src import filters, methods`; the JLR sources therefore live
in `code/reference/src/`. Use an ASCII-only absolute path (cv2 cannot read
non-ASCII paths on Windows).

`run_replication.py` reproduces the 23-image cross-platform cohort; it writes
`sweep_replication_rerun.jsonl` so the released records are never overwritten.

## Rebuild the figures and the manuscripts

```bash
python make_gallery.py               # F10_gallery.pdf / .png
python fix_figs.py                   # F1, F2, F4, F5, F7
TC_ROOT=<ascii-path> python build.py          # long version, docx + ready.md
TC_ROOT=<ascii-path> python build_latex.py    # long version PDF
TC_PANDOC=pandoc python build_ieee.py         # TVCG two-column PDF
```

`build_ieee.py` writes to an ASCII-only build directory and copies the result
into `manuscript/`. See `environment/ENVIRONMENT.md` for the toolchain.

## Legacy default paths

Most experiment and figure scripts were written in the original Linux sandbox
and still contain the literal default `/data/tileconsistency`. They are kept
verbatim for provenance. Override them at run time instead of editing them:

```bash
export TC_ROOT=$PWD/..
export TC_FIGDIR=../figures/
python run_sweep.py --glob "../data/images/met/*.jpg" --out ../data/records/rerun.jsonl
```

Four scripts are fully path-adapted through `config.py` and need no overrides:
`qual_stats.py`, `analyse_replication.py`, `make_gallery.py`,
`run_replication.py`. Those four are the ones that regenerate the published
tables and the output-comparison gallery.

## What is deliberately absent

The 1.5 GB of intermediate `/tmp/*.npy` arrays are not shipped; every number
in the paper is already in `data/records/*.jsonl`. The Section 4.5 mechanism
probe has no released record and the paper states no numbers for it. The four
third-party source trees audited in Section 6.10 are not redistributed.
