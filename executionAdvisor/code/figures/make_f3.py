#!/usr/bin/env python3

import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# efficiency residuals, percentage points (Table tab:efficiency, job 74347)
EFF_RESID = [4.90, 4.05, 5.66]
EFF_LABEL = ["$N{=}2$", "$N{=}4$", "$N{=}5$"]

NOISE_WITHIN = 2.39     # within-repetition IQR at T=16
NOISE_ACROSS = 10.0     # across-job, from 7.4% variance

C_EXACT, C_APPROX, C_NONE = "#276749", "#b7791f", "#a0aec0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", default="e0_validation.csv")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    v = pd.read_csv(a.validation)
    imb = np.abs(v["error_pct"].to_numpy())
    imb = imb[imb > 0]

    plt.rcParams.update({"font.size": 9, "font.family": "serif",
                         "axes.axisbelow": True})
    fig, ax = plt.subplots(figsize=(7.16, 2.9))

    # measurement resolution bands
    ax.add_patch(Rectangle((NOISE_WITHIN, -1.0), NOISE_ACROSS - NOISE_WITHIN,
                           3.7, color="#f6e05e", alpha=0.22, zorder=0))
    ax.axvspan(NOISE_ACROSS, 1e3, color="#fc8181", alpha=0.13, zorder=0)
    ax.axvline(NOISE_WITHIN, color="#b7791f", ls="--", lw=1.1, zorder=1)
    ax.axvline(NOISE_ACROSS, color="#c53030", ls="--", lw=1.1, zorder=1)

    # row 2: imbalance, reported
    ax.scatter(imb, np.full(imb.size, 2.0), s=54, marker="o",
               facecolors="white", edgecolors=C_EXACT, linewidths=1.7, zorder=4)
    ax.scatter([imb.max()], [2.0], s=54, marker="o", color=C_EXACT, zorder=5)
    ax.annotate("ten points, max $1.5\\times10^{-4}\\,\\%$",
                xy=(6e-4, 2.0), va="center", fontsize=7, color=C_EXACT)

    # row 1: efficiency, reported with a stated error
    ax.scatter(EFF_RESID, np.full(3, 1.0), s=64, marker="s",
               facecolors="white", edgecolors=C_APPROX, linewidths=1.7, zorder=4)
    ax.annotate("three node counts,\nresidual up to $5.7$ pp;\n"
                "sign explained, not modelled",
                xy=(6e-4, 1.0), va="center", fontsize=7, color=C_APPROX)

    # row 0: runtime, not attempted
    ax.annotate("no instrument: runtime, memory, node-hours and energy\n"
                "are outside what this model can support",
                xy=(6e-4, 0.0), va="center", fontsize=7, color="#4a5568")

    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["imbalance $I(p,\\mathrm{key})$\nand the two thresholds",
                        "parallel efficiency\n$I(16)/I(p)$",
                        "runtime, memory"], fontsize=7.5)
    for t, c in zip(ax.get_yticklabels(), [C_EXACT, C_APPROX, C_NONE]):
        t.set_color(c)
    ax.set_ylim(-0.95, 2.75)

    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1e3)
    ax.set_xlabel("magnitude of prediction error   [%]   (log scale)")
    ax.grid(axis="x", alpha=0.25, lw=0.5)

    ax.annotate("$2.39\\,\\%$\nwithin run", xy=(NOISE_WITHIN * 0.92, -0.62),
                fontsize=6.2, color="#975a16", ha="right", va="center")
    ax.annotate("$\\approx 10\\,\\%$\nacross jobs", xy=(NOISE_ACROSS * 1.35, -0.62),
                fontsize=6.2, color="#9b2c2c", ha="left", va="center")
    ax.annotate("below anything the cluster can resolve",
                xy=(1.4e-6, 2.56), fontsize=6.6, color=C_EXACT, va="center")

    ax.set_title("what the advisor reports, and where it stops",
                 fontsize=9, loc="left")
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(a.out, "F3." + ext)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("wrote", p)
    print("imbalance errors: n=%d  max=%.3g %%" % (imb.size, imb.max()))


if __name__ == "__main__":
    main()
