"""One-at-a-time sensitivity of the JLR repair layer."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TEX = ROOT.parent / "02_manuscript" / "tables"
sys.path.insert(0, str(ROOT))
from src.data import load_all  # noqa: E402
from src.methods import METHODS, m_ours  # noqa: E402
from src.metrics import METRICS, compute_metrics  # noqa: E402
from src.repair import structure_repair  # noqa: E402
from src.structure import structure_tensor  # noqa: E402
from src.structure_benchmark import junction_phantom, line_phantom, junction_score, line_score  # noqa: E402

GRID = {
    "snr_min": [1.5, 2.0, 2.5, 3.0, 4.0],
    "line_percentile": [98.0, 98.5, 99.0, 99.5, 99.8],
    "w_max": [0.6, 0.7, 0.8, 0.9, 1.0],
    "junction_dilation": [1, 2, 3, 4],
    "line_dilation": [1, 2, 3],
    "mask_sigma": [0.5, 1.0, 1.5, 2.0, 2.5],
}
KEYS = ["line_percentile", "junction_dilation", "line_dilation", "mask_sigma", "w_max",
        "snr_min", "junction_min_distance", "junction_rel_threshold"]


def main() -> None:
    specs = {m.name: m for m in METHODS}
    ca = specs["ca_akf"]
    cfg0 = specs["jlr_cakf"].params
    images = load_all()
    bases = {im.name: m_ours(im.image, ca.params) for im in images}
    sts = {im.name: structure_tensor(im.image) for im in images}
    juncs = {k: junction_phantom(k) for k in ["axis", "diag", "T", "Y", "cross", "low"]}
    lines = {k: line_phantom(k) for k in ["v1", "d1", "curve", "dark", "w2", "low"]}
    jbases = {k: m_ours(v[0], ca.params) for k, v in juncs.items()}
    lbases = {k: m_ours(v[0], ca.params) for k, v in lines.items()}
    rows = []
    configs = [("default", "default", np.nan, {})]
    for key, values in GRID.items():
        for value in values:
            configs.append((key, key, value, {key: value}))
    for pname, key, value, override in configs:
        cfg = dict(cfg0)
        cfg.update(override)
        kwargs = {k: cfg[k] for k in KEYS}
        met_rows = []
        for im in images:
            out = structure_repair(im.image, bases[im.name], **kwargs)
            met_rows.append(compute_metrics(im.image, out, st_src=sts[im.name]))
        met = pd.DataFrame(met_rows)
        js, ls = [], []
        for k, (im, c, sp) in juncs.items():
            out = structure_repair(im, jbases[k], **kwargs)
            js.append(junction_score(out, c, sp))
        for k, (im, mk) in lines.items():
            out = structure_repair(im, lbases[k], **kwargs)
            ls.append(line_score(out, mk, im))
        row = {"parameter": pname, "value": value,
               "junction_score": float(np.median(js)), "line_score": float(np.median(ls))}
        for m in METRICS:
            row[m] = float(met[m].median())
        rows.append(row)
        print(f"{pname:18s} {value!s:>6s}  SR={row['sr']:.3f} SSIM={row['ssim']:.3f} "
              f"ESR={row['esr']:.3f} ECE={row['ece']:.2f} JC={row['junction_score']:.3f} "
              f"LR={row['line_score']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "sensitivity_jlr.csv", index=False, float_format="%.6f")

    lines_tex = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{One-at-a-time sensitivity of the repair layer (median over twelve images; structure scores over twelve phantoms). Each block varies one parameter with the others at their defaults; bold marks the default value in that block.}",
        r"\label{tab:sensitivity}", r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
        r"Parameter & Value & SR$\uparrow$ & ESR$\to1$ & ECE$\downarrow$ & JC$\uparrow$ & LR$\uparrow$ & FSR$\downarrow$ \\", r"\midrule",
    ]
    defaults = {"snr_min": 2.5, "line_percentile": 99.0, "w_max": 0.9,
                "junction_dilation": 3, "line_dilation": 2, "mask_sigma": 1.5}
    first = {"snr_min": True, "line_percentile": True, "w_max": True,
             "junction_dilation": True, "line_dilation": True, "mask_sigma": True}
    for key, values in GRID.items():
        for value in values:
            r = df[(df.parameter == key) & (df.value == value)].iloc[0]
            is_default = abs(value - defaults[key]) < 1e-9
            val = f"\\textbf{{{value:g}}}" if is_default else f"{value:g}"
            pname = key.replace("_", r"\_") if first[key] else ""
            first[key] = False
            lines_tex.append(
                f"{pname} & {val} & {r.sr:.3f} & {r.esr:.3f} & {r.ece:.2f} & "
                f"{r.junction_score:.3f} & {r.line_score:.3f} & {r.fsr:.3f} \\\\"
            )
        lines_tex.append(r"\midrule")
    lines_tex = lines_tex[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TEX / "table_sensitivity.tex").write_text("\n".join(lines_tex) + "\n")
    print("wrote sensitivity_jlr.csv and table_sensitivity.tex")


if __name__ == "__main__":
    main()
