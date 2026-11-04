"""Aggregate the third-party (OpenCV) panel into the Section 6.9 tables and figure.

Inputs (read-only):
  out/cv_all.jsonl   one record per (image, method): 8-rung halo ladder of TCE against
                     the whole-image reference, pre-registered support prediction,
                     grid-line visibility, GV-gate, per-image runtime
  out/cv_det.jsonl   determinism probes: same content in a fresh allocation, a
                     non-contiguous view, and a repeated call
  out/broad_all.jsonl the four operators of Section 6.8 on the SAME 144 images,
                     used here only as comparability references

Outputs:
  out/cv.md            tables for the manuscript (numbers copied verbatim from here)
  figures/F9_cv.png    (a) halo ladder, third-party vs ours; (b) determinism probes

Re-runnable; nothing is cached. Run it once the panel is complete (7 x 144 records).
"""
import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/data/tileconsistency"
OUT = f"{ROOT}/out"
FIG = f"{ROOT}/figures/F9_cv.png"
LADDER = [0, 2, 4, 8, 16, 24, 32, 48]          # rungs used by run_cv.py
BROAD_HALOS = [0, 16, 32]                        # rungs used by run_broad.py
LV = 255.0                                     # [0,1] -> 8-bit levels
CENS = 49                                      # sentinel: not exact anywhere in LADDER


def load(path, key=("img", "method")):
    """First-wins dedup on (img, method), the convention used across this project."""
    if not os.path.exists(path):
        return []
    seen, rows = set(), []
    for line in open(path):
        r = json.loads(line)
        k = tuple(r[f] for f in key)
        if k in seen:
            continue
        seen.add(k)
        rows.append(r)
    return rows


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def frac_at(rec, h):
    c = (rec.get("tce_halo") or {}).get(f"h{h}")
    return None if c is None else c["frac1"]


def gv_at(rec, h):
    if (rec.get("tce_halo") or {}).get(f"h{h}") is None:
        return None
    return (rec.get("gv") or {}).get("ratio")


# Records written before the bibliography was resolved carry provisional cite keys;
# this maps them onto the verified entries in paper/refs.bib.
CITE_MAP = {"perona1991scale": "perona1991diffusion", "tam2019amfilter": "gastal2012amf",
            "zhang2014rolling": "zhang2014rollingguidance", "xu2013l0": "xu2011l0smooth"}


# ------------------------------------------------------------------ panel table
def panel_table(cv):
    per = collections.defaultdict(list)
    for r in cv:
        per[r["method"]].append(r)
    rows = []
    for m, rs in sorted(per.items()):
        pred = rs[0]["pred_exact_halo"]
        he = [r["halo_exact"] for r in rs]
        ok = [h for h in he if h is not None]
        never = len(rs) - len(ok)
        # Censored median over ALL images: an image that never becomes exact in the
        # ladder contributes ">48" rather than being silently dropped from the median.
        cens = [h if h is not None else CENS for h in he]
        within = sum(1 for h in ok if pred is not None and h <= pred)
        f48 = med([frac_at(r, 48) for r in rs])
        f32 = med([frac_at(r, 32) for r in rs])
        f0 = med([frac_at(r, 0) for r in rs])
        maxd = med([(r["tce_halo"].get("h48") or {}).get("maxd") for r in rs])
        # GV gated exactly as Section 6.8.3: only images with >0.1% of pixels wrong
        g16 = [gv_at(r, 16) for r in rs
               if (frac_at(r, 16) or 0) > 1e-3]
        rows.append(dict(method=m, n=len(rs), pred=pred, med_exact=med(cens),
                         med_exact_conv=(med(ok) if ok else None),
                         exact_h0=sum(1 for h in he if h == 0),
                         within_pred=within, never=never,
                         f0=f0, f32=f32, f48=f48, maxd48=maxd,
                         gv16=med(g16), n_gv=len(g16),
                         t=med([r["t"] for r in rs]),
                         setting=rs[0]["setting"],
                         cite=CITE_MAP.get(rs[0]["cite"], rs[0]["cite"])))
    return rows


def broad_table(broad):
    per = collections.defaultdict(list)
    for r in broad:
        per[r["method"]].append(r)
    out = {}
    for m, rs in per.items():
        out[m] = dict(f0=med([frac_at(r, 0) for r in rs]),
                      f16=med([frac_at(r, 16) for r in rs]),
                      f32=med([frac_at(r, 32) for r in rs]),
                      exact32=sum(1 for r in rs if frac_at(r, 32) == 0.0),
                      n=len(rs))
    return out


# ------------------------------------------------------------- determinism table
def det_table(det):
    per = collections.defaultdict(list)
    for r in det:
        per[r["method"]].append(r)
    rows = []
    for m, rs in sorted(per.items()):
        def cnt(pfx):
            return sum(1 for r in rs if r[f"ndiff_{pfx}"] > 0)
        def mx(pfx, field):
            # maxd_* is a uint8 difference: the value is already in 8-bit levels
            vals = [r[f"{field}_{pfx}"] for r in rs if r[f"ndiff_{pfx}"] > 0]
            return (float(np.median(vals)), float(np.max(vals))) if vals else (0.0, 0.0)
        rows.append(dict(method=m, n=len(rs),
                         rep=cnt("repeat"), fc=cnt("fresh_copy"), nc=cnt("noncontig_view"),
                         rep_px=med([r["ndiff_repeat"] for r in rs if r["ndiff_repeat"] > 0]),
                         fc_px=med([r["ndiff_fresh_copy"] for r in rs
                                    if r["ndiff_fresh_copy"] > 0]),
                         nc_px=med([r["ndiff_noncontig_view"] for r in rs
                                    if r["ndiff_noncontig_view"] > 0]),
                         rep_lv=mx("repeat", "maxd"), fc_lv=mx("fresh_copy", "maxd"),
                         nc_lv=mx("noncontig_view", "maxd")))
    return rows


# ------------------------------------------------------------------------ report
def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n"
    s += "|" + "|".join("---" for _ in header) + "|\n"
    for r in rows:
        s += "| " + " | ".join(r) + " |\n"
    return s


def pct(x):
    return "n/a" if x is None else f"{x * 100:.3f} %"


def lv(x):
    return "n/a" if x is None else f"{x * LV:.2f}"


def write_md(path, rows, drows, brows, n_img):
    L = ["# Section 6.9 — third-party published implementations (generated by analyse_cv.py)", "",
         f"Panel records: {sum(r['n'] for r in rows)} over {n_img} images, "
         "4x4 tile grid at 384x384, ladder of halos 0/2/4/8/16/24/32/48.", ""]
    hdr = ["method (OpenCV call)", "refs.bib key", "setting", "n", "pre-registered exact halo",
           "median measured exact halo", "exact at halo 0", "measured <= predicted",
           "never exact in ladder", "median frac >1 lv @0", "@32", "@48",
           "worst pixel @48 (lv)", "GV @16 (gated)", "s/image"]
    body = []
    for r in rows:
        body.append([r["method"], "@" + r["cite"], r["setting"], str(r["n"]),
                     "none (global solve)" if r["pred"] is None else str(r["pred"]),
                     ">48 (censored)" if r["med_exact"] >= CENS else f"{r['med_exact']:.0f}",
                     str(r["exact_h0"]),
                     f"{r['within_pred']}/{r['n']}" if r["pred"] is not None else "n/a",
                     f"{r['never']}/{r['n']}",
                     pct(r["f0"]), pct(r["f32"]), pct(r["f48"]), lv(r["maxd48"]),
                     "n/a" if r["gv16"] is None else f"{r['gv16']:.2f} (n={r['n_gv']})",
                     f"{r['t']:.1f}"])
    L.append(md_table(hdr, body))
    L += ["", "## Determinism probes (same content, different memory history)", ""]
    dh = ["method", "images probed", "repeat call differs", "fresh allocation differs",
          "non-contiguous view differs", "median px differing (repeat)",
          "worst (lv) med/max", "worst (lv) med/max", "worst (lv) med/max"]
    db = [[r["method"], str(r["n"]), f"{r['rep']}/{r['n']}", f"{r['fc']}/{r['n']}",
           f"{r['nc']}/{r['n']}", "n/a" if r["rep_px"] is None else f"{r['rep_px']:.0f}",
           f"{r['rep_lv'][0]:.0f} / {r['rep_lv'][1]:.0f}",
           f"{r['fc_lv'][0]:.0f} / {r['fc_lv'][1]:.0f}",
           f"{r['nc_lv'][0]:.0f} / {r['nc_lv'][1]:.0f}"] for r in drows]
    L.append(md_table(dh, db))
    L += ["", f"## The same {n_img} images under our own operators (Section 6.8 protocol)", ""]
    bh = ["method", "n", "median frac >1 lv @0", "@16", "@32", "exactly 0 at halo 32"]
    bb = [[m, str(v["n"]), pct(v["f0"]), pct(v["f16"]), pct(v["f32"]),
           f"{v['exact32']}/{v['n']}"] for m, v in sorted(brows.items())]
    L.append(md_table(bh, bb))
    L += ["", "Note: the Section 6.8 ladder is 0/16/32; the third-party ladder adds "
          "2/4/8/24/48 so that a support bound can be bracketed rather than asserted.", ""]
    open(path, "w").write("\n".join(L))


# ------------------------------------------------------------------------ figure
COLOUR = {"bil-9": "tab:grey", "guided-16": "tab:purple", "nlm-7-21": "tab:cyan",
          "am-9": "tab:green", "rgf-15": "tab:pink", "pmdiff-20": "tab:red",
          "l0-0.02": "tab:olive", "CA-AKF": "tab:orange", "SC-AKF": "maroon",
          "LRS-final": "tab:blue", "AKF": "k"}
NAME = {"bil-9": "bilateral (ICCV'98)", "guided-16": "guided (ECCV'10)",
        "nlm-7-21": "NLM (CVPR'05)", "am-9": "adaptive manifolds (TOG'12)",
        "rgf-15": "rolling guidance (ECCV'14)", "pmdiff-20": "Perona-Malik (TPAMI'91)",
        "l0-0.02": "L0 gradient (TOG'11)", "CA-AKF": "CA-AKF (ref. impl., class B)",
        "SC-AKF": "SC-AKF (ref. impl., class B)", "LRS-final": "LRS-AKF (ours)",
        "AKF": "AKF (ref. impl., class A)"}


def figure(cv, broad, det, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.8))
    per = collections.defaultdict(list)
    for r in cv:
        per[r["method"]].append(r)
    for m, rs in sorted(per.items()):
        ys = [med([(frac_at(r, h) or 0.0) for r in rs]) * 100 for h in LADDER]
        ys = [max(y, 2e-4) for y in ys]
        ax1.plot(LADDER, ys, marker="o", ms=4.5, lw=1.5, color=COLOUR[m], label=NAME[m])
    bper = collections.defaultdict(list)
    for r in broad:
        bper[r["method"]].append(r)
    for m in ("AKF", "CA-AKF", "SC-AKF", "LRS-final"):
        rs = bper.get(m)
        if not rs:
            continue
        ys = [max(med([(frac_at(r, h) or 0.0) for r in rs]) * 100, 2e-4)
              for h in BROAD_HALOS]
        ax1.plot(BROAD_HALOS, ys, marker="s", ms=7, lw=2.6, ls="--",
                 color=COLOUR[m], label=NAME[m])
    ax1.axhline(1e-3 * 100, color="k", ls=":", lw=1)
    ax1.text(1, 0.13, "0.1 % of pixels", fontsize=7)
    ax1.text(48.5, 4e-3, "floor = exactly 0 %\n(bit-identical to the whole-image pass)",
             fontsize=6.6, ha="right", va="bottom")
    ax1.set_yscale("log")
    ax1.set_ylim(1.5e-4, 260)
    ax1.set_xlim(-1, 49)
    ax1.set_xlabel("overlap halo (px)")
    ax1.set_ylabel("median frac of pixels $>1$ 8-bit level (%)")
    ax1.set_title("(a) seven published algorithms, run as third-party code, on the same "
                  "144 images", fontsize=9)
    ax1.grid(alpha=.3, which="both")
    ax1.legend(fontsize=6.6, ncol=3, loc="upper center", bbox_to_anchor=(0.42, -0.13),
               framealpha=.95, handlelength=1.6, columnspacing=.9)

    # ---- (b) determinism probes -------------------------------------------------
    ms = [r["method"] for r in det]
    y = np.arange(len(ms))
    h = 0.26
    for i, (fld, lab) in enumerate((("rep", "same array, second call"),
                                    ("fc", "identical content, fresh allocation"),
                                    ("nc", "identical content, non-contiguous rows"))):
        frac = [r[fld] / max(r["n"], 1) for r in det]
        ax2.barh(y + (i - 1) * h, frac, height=h, label=lab,
                 color=["tab:grey", "tab:blue", "tab:green"][i])
    for i, r in enumerate(det):
        for j, fld in enumerate(("rep", "fc", "nc")):
            f = r[fld] / max(r["n"], 1)
            if f > 0:
                ax2.text(f + 0.015, y[i] + (j - 1) * h, f"{r[fld]}/{r['n']}",
                         va="center", fontsize=6.5)
            elif j == 1:
                ax2.text(0.008, y[i], f"0/{r['n']}", va="center",
                         fontsize=5.8, color="0.45")
    ax2.set_yticks(y)
    ax2.set_yticklabels([NAME.get(m, m) for m in ms], fontsize=7.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.22)
    ax2.set_xticks([0, .25, .5, .75, 1.0])
    ax2.set_xticklabels(["0", "25", "50", "75", "100 %"])
    ax2.set_xlabel("images whose output is NOT a function of pixel content alone")
    ax2.set_title("(b) a third failure mode: identical input, different output",
                  fontsize=9)
    ax2.legend(fontsize=7, loc="lower right")
    pm = det[[d["method"] for d in det].index("pmdiff-20")]
    clean = sum(1 for d in det if d["method"] != "pmdiff-20"
                and d["rep"] == d["fc"] == d["nc"] == 0)
    ax2.text(0.06, 3.0, f"the {clean} other published implementations return bit-identical\n"
             "output on all three probes; Perona-Malik diffusion does not,\n"
             f"on every image probed (median {int(pm['rep_px'] or 0)} px, "
             f"worst {pm['rep_lv'][1]:.0f} of 255 levels)", fontsize=6.8, va="center", ha="left")
    ax2.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print("wrote", path, os.path.getsize(path), "bytes")



# --------------------------------------------------------------- section emitter
PFX = {"bil-9": "BIL", "guided-16": "GUI", "nlm-7-21": "NLM", "pmdiff-20": "PM",
       "am-9": "AM", "rgf-15": "RGF", "l0-0.02": "L0"}
BROAD_PFX = {"AKF": "AKF", "LRS-final": "LRS", "SC-AKF": "SC", "CA-AKF": "CA"}


def _pct(x, dp=2):
    return "n/a" if x is None else f"{x * 100:.{dp}f} %"


def emit_section(rows, drows, brows, n_img, template, out):
    t = open(template).read()
    v = {"N_IMG": str(n_img), "N_REC": str(sum(r["n"] for r in rows))}
    n = max(r["n"] for r in rows)
    v["N"] = str(n)
    for r in rows:
        p = PFX[r["method"]]
        v[f"{p}_SET"] = r["setting"]
        v[f"{p}_PRED"] = "none (global solve)" if r["pred"] is None else f"{r['pred']} px"
        v[f"{p}_MED"] = ("not exact at 48" if r["med_exact"] >= CENS
                        else f"{r['med_exact']:.0f} px")
        v[f"{p}_WITHIN"] = "n/a" if r["pred"] is None else f"{r['within_pred']}/{r['n']}"
        v[f"{p}_NEVER"] = f"{r['never']}/{r['n']}"
        v[f"{p}_E0"] = str(r["exact_h0"])
        v[f"{p}_F0"] = _pct(r["f0"])
        v[f"{p}_F48"] = _pct(r["f48"], 3)
        v[f"{p}_MAX48"] = "n/a" if r["maxd48"] is None else f"{r['maxd48'] * LV:.1f} lv"
        v[f"{p}_GV"] = "n/a" if r["gv16"] is None else f"{r['gv16']:.2f}"
    for m, p in BROAD_PFX.items():
        b = brows.get(m, {})
        v[f"{p}_EX"] = str(b.get("exact32", "n/a"))
        v[f"{p}_N"] = str(b.get("n", "n/a"))
    d = {r["method"]: r for r in drows}
    pm = d.get("pmdiff-20", {})
    v["DET_N"] = str(pm.get("n", 0))
    v["DET_REP"] = str(pm.get("rep", 0))
    v["DET_FC"] = str(pm.get("fc", 0))
    v["DET_NC"] = str(pm.get("nc", 0))
    v["DET_PX"] = "n/a" if pm.get("rep_px") is None else f"{pm['rep_px']:,.0f}"
    for fld, tok in (("fc_px", "DET_FC_PX"), ("nc_px", "DET_NC_PX")):
        v[tok] = "n/a" if pm.get(fld) is None else f"{pm[fld]:,.0f}"
    v["DET_LV"] = f"{pm.get('rep_lv', (0, 0))[1]:.0f}"
    left = sorted(set(__import__("re").findall(r"\{\{([A-Z0-9_]+)\}\}", t)) - set(v))
    for k, val in v.items():
        t = t.replace("{{" + k + "}}", val)
    open(out, "w").write(t)
    print(f"wrote {out} ({len(t.split())} words); unresolved placeholders: "
          f"{left if left else 'none'}")


def main():
    cv = load(f"{OUT}/cv_all.jsonl")
    det = load(f"{OUT}/cv_det.jsonl")
    broad = load(f"{OUT}/broad_all.jsonl")
    if not cv:
        raise SystemExit("no cv_all records")
    rows = panel_table(cv)
    drows = det_table(det)
    brows = broad_table(broad)
    n_img = len({r["img"] for r in cv})
    write_md(f"{OUT}/cv.md", rows, drows, brows, n_img)
    print(f"out/cv.md written ({n_img} images, {len(cv)} panel records, "
          f"{len(det)} determinism records)")
    for r in rows:
        print(f"  {r['method']:11s} pred={str(r['pred']):>5s} med_exact={str(r['med_exact']):>6s} "
              f"med_conv={str(r['med_exact_conv']):>6s} "
              f"never={r['never']}/{r['n']} f48={None if r['f48'] is None else round(100*r['f48'],4)}%")
    figure(cv, broad, drows, FIG)
    emit_section(rows, drows, brows, n_img, f"{ROOT}/paper/sec_69_template.md",
                 f"{ROOT}/paper/sec_69_draft.md")


if __name__ == "__main__":
    main()
