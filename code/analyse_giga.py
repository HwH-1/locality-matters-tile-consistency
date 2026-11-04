"""Aggregate the large-canvas runs into the numbers Section 6.7 quotes.

Sources: out/giga_<METHOD>.json (whole vs tile-wise on the 2048^2 mosaic, one
method per process) and out/mem.jsonl (one MODE per process, so peak RSS
separates the whole-canvas pass from the tile-wise pass).  Emits out/giga.md in
the same table style as out/tables.md so the manuscript text and the numbers
cannot drift apart.
"""
import glob
import json
import os

OUT = "/data/tileconsistency/out"
LEVEL = 1 / 255.0


def pct(x):
    return f"{100 * x:.2f} %"


def main():
    lines = ["# Large-canvas results (Section 6.7)", ""]

    mem = {}
    mp = f"{OUT}/mem.jsonl"
    if os.path.exists(mp):
        for line in open(mp):
            try:
                r = json.loads(line)
            except Exception:
                continue
            mem.setdefault(r["method"], []).append(r)

    methods = ["AKF", "CA-AKF", "SC-AKF", "LRS-final"]
    rows = []
    for m in methods:
        p = f"{OUT}/giga_{m}.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p))[m]
        rows.append((m, d))

    lines.append("## Whole-canvas cost, 2048x2048 mosaic (one method per process)")
    lines.append("")
    lines.append("| method | canvas array MB | whole pass s | whole pass peak RSS MB | ratio peak/array |")
    lines.append("|---|---|---|---|---|")
    for m, d in rows:
        w = d["whole"]
        arr = json.load(open(f"{OUT}/giga_{m}.json"))["array_mb"]
        lines.append(f"| {m} | {arr} | {w['s']} | {w.get('peak_mb', '')} | "
                     f"{w['peak_mb'] / arr:.1f}x |")
    lines.append("")

    lines.append("## Tile-wise vs whole-canvas deviation (TCE against the whole-canvas pass)")
    lines.append("")
    lines.append("| method | config | rmse | max\\|d\\| (levels) | >1 level | >4 levels | GV ratio |")
    lines.append("|---|---|---|---|---|---|---|")
    for m, d in rows:
        for key in sorted(k for k in d if k != "whole"):
            r = d[key]
            lines.append(
                f"| {m} | {key} "
                f"| {r['rmse']:.6f} | {r['maxd'] / LEVEL:.2f} | {pct(r['frac1'])} "
                f"| {pct(r['frac4'])} | {r['ratio']:.1f} |")
    lines.append("")

    if mem:
        lines.append("## Per-mode peak memory (one mode per process)")
        lines.append("")
        lines.append("| method | mode | wall s | peak RSS MB | filter s |")
        lines.append("|---|---|---|---|---|")
        for m in methods:
            for r in mem.get(m, []):
                lines.append(f"| {m} | {r['mode']} | {r.get('wall_s', '')} "
                             f"| {r['peak_mb']} | {r.get('filter_s', '')} |")
        lines.append("")

    for p in sorted(glob.glob(f"{OUT}/giga_4096*.json")):
        lines.append(f"## {os.path.basename(p)}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(json.load(open(p)), indent=1)[:2500])
        lines.append("```")
        lines.append("")

    txt = "\n".join(lines)
    with open(f"{OUT}/giga.md", "w") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
