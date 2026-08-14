#!/usr/bin/env python3

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter, FuncFormatter

P_SWEEP = [16, 32, 64, 80]          # must match make_table.py PS
P_FOCUS = 80                        # the ceiling column plotted in panel (a)
P_MEASURED = 80                     # largest partition count actually run


WINDOW_SERIES = ["W6", "W12", "W24", "W48"]

SIZE_MATCHED = ["W6", "sizematched_W12", "sizematched_W24",
                "sizematched_W48", "sizematched_Wfull"]

STUDY = "Wfull"

C_WIN, C_SM, C_STUDY = "#2b6cb0", "#c53030", "#276749"


def load(d):
    rows = {}
    for f in glob.glob(os.path.join(d, "sweep_*.json")):
        v = os.path.basename(f)[len("sweep_"):-len(".json")]
        j = json.load(open(f))
        ceil = {p: (s["ceiling"] * 100 if s["ceiling"] is not None else np.nan)
                for p, s in zip(P_SWEEP, j["sweep"])}
        rows[v] = dict(n=j["n_units"], N=j["n_records"], cv=j["cv"],
                       gini=j["gini"], pstar=j["p_star"],
                       patom=j["p_atom"], C=ceil)
    if not rows:
        sys.exit("no sweep_*.json in %s -- run the skewcast loop first" % d)
    return rows


def dedupe(rows):
    """Drop byte-identical aliases (W24==shape_W24_trim0, W6==sizematched_W6)."""
    seen, keep = {}, {}
    for v, r in sorted(rows.items()):
        sig = (r["n"], r["N"], r["pstar"], r["patom"])
        if sig in seen:
            continue
        seen[sig] = v
        keep[v] = r
    return keep


def tex(s):
    return s.replace("_", r"\_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="directory holding sweep_*.json")
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--usetex", action="store_true",
                    help="render text with LaTeX (needs a working latex)")
    a = ap.parse_args()

    rows = load(a.dir)
    uniq = dedupe(rows)
    print("loaded %d sweeps, %d distinct datasets" % (len(rows), len(uniq)))

    plt.rcParams.update({
        "font.size": 9, "font.family": "serif", "axes.grid": True,
        "grid.alpha": 0.25, "axes.axisbelow": True,
        "text.usetex": bool(a.usetex),
    })
    esc = tex if a.usetex else (lambda s: s)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.16, 3.5))

    # ---------------- panel (a): ceiling against record count -------------
    win = [v for v in WINDOW_SERIES if v in rows]
    sm = [v for v in SIZE_MATCHED if v in rows]

    xw = [rows[v]["N"] for v in win]
    yw = [rows[v]["C"][P_FOCUS] for v in win]
    axA.plot(xw, yw, "o-", color=C_WIN, lw=1.8, ms=6, zorder=3,
             label="window series: cohort fixed")

    xs = [rows[v]["N"] for v in sm]
    ys = [rows[v]["C"][P_FOCUS] for v in sm]
    axA.plot([np.mean(xs)] * 2, [min(ys), max(ys)], "-",
             color=C_SM, lw=1.4, alpha=0.55, zorder=2)
    axA.plot(xs, ys, "s", color=C_SM, ms=7, zorder=4,
             label="size-matched: $N$ fixed")

    if STUDY in rows:
        axA.plot(rows[STUDY]["N"], rows[STUDY]["C"][P_FOCUS], "*",
                 color=C_STUDY, ms=16, zorder=5, label="full-stay workload")

    fw = max(xw) / min(xw)
    ratio_w = max(yw) / min(yw)
    ratio_s = max(ys) / min(ys)
    axA.annotate(r"$%.1f\times$" % ratio_s,
                 xy=(np.mean(xs) * 1.15, np.sqrt(min(ys) * max(ys))),
                 color=C_SM, fontsize=10, fontweight="bold")
    axA.annotate(r"$%.1f\times$" % ratio_w,
                 xy=(np.sqrt(min(xw) * max(xw)) * 1.1, max(yw) * 1.18),
                 color=C_WIN, fontsize=10, fontweight="bold")
    axA.annotate(esc("sizematched_Wfull"), (np.mean(xs) * 1.15, max(ys) * 0.86),
                 fontsize=6.5, color="#555")

    axA.set_xscale("log")
    axA.set_yscale("log")
    # matplotlib labels log MINOR ticks when the range spans under a decade,
    # which overprints on this axis.  Label chosen decades only.
    axA.xaxis.set_minor_formatter(NullFormatter())
    axA.set_xticks([1e7, 3e7, 1e8])
    axA.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: {1e7: "$10^{7}$", 3e7: r"$3\times10^{7}$",
                      1e8: "$10^{8}$"}.get(v, "")))
    axA.set_yticks([1, 2, 5, 10, 20, 50])
    axA.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    axA.set_ylim(1.0, 80)
    axA.set_xlabel("records $N$")
    axA.set_ylabel(r"ceiling $C(%d)$   [\%%]" % P_FOCUS if a.usetex
                   else "ceiling C(%d)   [%%]" % P_FOCUS)
    axA.set_title("(a)  the advice does not follow data volume",
                  fontsize=9, loc="left")
    axA.legend(fontsize=6.5, loc="lower right", framealpha=0.96)
    print("  (a) volume varies %.2fx -> ceiling %.2fx ; "
          "volume fixed -> ceiling %.1fx" % (fw, ratio_w, ratio_s))

    # ---------------- panel (b): crossover per variant ---------------------
    o = sorted(uniq.items(), key=lambda kv: kv[1]["pstar"] or 0)
    names = [k for k, _ in o]
    ps = [v["pstar"] or 0 for _, v in o]
    cols = [C_STUDY if v["cv"] > 1 else C_WIN for _, v in o]

    axB.axvspan(0, P_MEASURED, color="#f6e05e", alpha=0.28, zorder=1)
    axB.barh(range(len(ps)), ps, color=cols, height=0.62, zorder=3)
    axB.axvline(P_MEASURED, color="#b7791f", ls="--", lw=1.2, zorder=4)
    axB.set_yticks(range(len(ps)))
    axB.set_yticklabels([esc(n) for n in names], fontsize=6.5)
    hi = max(ps)
    for i, p in enumerate(ps):
        axB.annotate(str(p), (p + hi * 0.02, i), va="center",
                     fontsize=6.5, color="#333")
    axB.annotate("measured here", xy=(P_MEASURED * 1.15, len(ps) - 0.45),
                 fontsize=6.5, color="#975a16")
    axB.set_xlim(0, hi * 1.18)
    axB.set_xlabel("crossover $p^{*}$")
    axB.set_title("(b)  where the advice changes, per structure",
                  fontsize=9, loc="left")
    axB.legend(handles=[Patch(color=C_STUDY, label="heavy-tailed (CV $>1$)"),
                        Patch(color=C_WIN, label="light-tailed (CV $<1$)")],
               fontsize=6.5, loc="center right", framealpha=0.96)

    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(a.out, "F8." + ext)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("  wrote", p)


if __name__ == "__main__":
    main()
