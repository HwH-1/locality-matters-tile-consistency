"""Replication analysis for the central claim on a DISJOINT image set.

The primary analysis (10 images) is where the paired quality statistics live. This
script only re-tests the phenomenon claim -- class B plateaus, class A and LRS-AKF
go to zero -- on a separate block of images, so that the headline result is not an
artefact of the ten images the method was calibrated against.

Reads out/sweep_rep.jsonl (same schema as out/all.jsonl) and writes
out/replication.md with the numbers used in Section 6.8 of the manuscript.
"""
import json
import sys

import numpy as np
from scipy.stats import wilcoxon

OUT = "/data/tileconsistency/out"


def load(path):
    by = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        by.setdefault(r["method"], []).append(r)
    for m in by:
        by[m].sort(key=lambda r: r["img"])
    return by


def holm(raw):
    order = sorted(raw, key=lambda k: raw[k])
    n = len(order)
    prev, out = 0.0, {}
    for i, k in enumerate(order):
        adj = min(1.0, (n - i) * raw[k])
        adj = max(adj, prev)
        prev = adj
        out[k] = adj
    return out


def med(rows, cfg, key):
    return float(np.median([r["tce_n"][cfg][key] for r in rows if cfg in r.get("tce_n", {})]))


def main():
    rep = load(sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/sweep_rep.jsonl")
    imgs = sorted({r["img"] for m in rep.values() for r in m})
    lines = [f"# Replication on {len(imgs)} images disjoint from the primary set", ""]

    lines.append("| method | TCE @ n=8,h=0 | TCE @ n=8,h=16 | frac>1/255 @ h=16 | frac>1/255 @ h=32 (n=4) | worst pixel @ h=32 |")
    lines.append("|---|---|---|---|---|---|")
    for m in ["AKF", "CA-AKF", "SC-AKF", "LRS-final"]:
        if m not in rep:
            continue
        rows = rep[m]
        h32 = float(np.median([r["tce_halo"]["h32"]["frac1"] for r in rows]))
        d32 = float(np.max([r["tce_halo"]["h32"]["maxd"] for r in rows]))
        lines.append("| %-9s | %.3e | %.3e | %.4f | %.4f | %.3e |" % (
            m, med(rows, "n8_h0", "rmse"), med(rows, "n8_h16", "rmse"),
            med(rows, "n8_h16", "frac1"), h32, d32))
    lines.append("")

    if "CA-AKF" in rep and "LRS-final" in rep:
        a, b = rep["CA-AKF"], rep["LRS-final"]
        raws = {}
        for key in ["rmse", "frac1", "frac4", "meand", "maxd"]:
            x = np.array([r["tce_n"]["n8_h16"][key] for r in a])
            y = np.array([r["tce_n"]["n8_h16"][key] for r in b])
            _, p = wilcoxon(x, y)
            raws[key] = float(p)
            wins = int((y < x).sum())
            lines.append("- %s: %.3e -> %.3e (%.1fx), %d/%d images better, raw p=%.5f"
                         % (key, np.median(x), np.median(y),
                            np.median(x) / max(np.median(y), 1e-15), wins, len(x), p))
        adj = holm(raws)
        lines.append("")
        lines.append("Holm-adjusted within the consistency family: "
                     + ", ".join(f"{k}={adj[k]:.4f}" for k in sorted(adj, key=lambda z: adj[z])))
    txt = "\n".join(lines) + "\n"
    open(f"{OUT}/replication.md", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
