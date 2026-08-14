#!/usr/bin/env python3

import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P = "/projects/sb2ea/results/timing/"
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=P)
ap.add_argument("--out", default="figures")
a = ap.parse_args()

def load(f):
    d = pd.read_csv(os.path.join(a.dir, f))
    d = d[(d.warmup == 0) & (d.timed_region == "kernel")].copy()
    d["host"] = d.node.str.split(".").str[0]          # never trust `platform`
    d["mg"] = d.throughput_rec_s / 1e6 / d.achieved_ghz
    return d

BDW = load("e3g_c6_72897.csv")
SNB = load("e3g_c11_72949.csv")
BSMT = load("e3f_c6_72592.csv")

def med(d, **kw):
    m = d
    for k, v in kw.items():
        m = m[m[k] == v]
    return float(m.mg.median()) if len(m) else float("nan")

def d38(d):                      # single-socket efficiency, T=8 vs T=2
    t2 = med(d, partitioner="block", thread_placement="socket0", n_threads=2)
    t8 = med(d, partitioner="block", thread_placement="socket0", n_threads=8)
    return 100 * (t8 / 4) / t2

def part_vs_block(d, name):      # D41, at T=16 spread
    b = med(d, partitioner="block", thread_placement="spread",
            schedule="precomputed", n_threads=16)
    x = med(d, partitioner=name, thread_placement="spread",
            schedule="precomputed", n_threads=16)
    return 100 * (x / b - 1)

def dip14(d):                    # T=12 -> T=14, absolute throughput
    t12 = med(d, partitioner="block", thread_placement="spread",
              schedule="precomputed", n_threads=12)
    t14 = med(d, partitioner="block", thread_placement="spread",
              schedule="precomputed", n_threads=14)
    return 100 * (t14 / t12 - 1)

def smt_gain(d):                 # T=16 SMT vs T=8 one socket
    s = med(d, thread_placement="smt", n_threads=16)
    b = med(d, thread_placement="socket0", n_threads=8)
    return 100 * (s / b - 1)

ROWS = [
   
    ("single-socket efficiency\n(T=8, % of linear)",
     d38(BDW), d38(SNB), "%", True, True, "e3g pair"),
    ("contiguity: cyclic vs block\n(T=16)",
     part_vs_block(BDW, "cyclic"), part_vs_block(SNB, "cyclic"),
     "%", True, True, "e3g pair"),
    ("contiguity: greedy vs block\n(T=16)",
     part_vs_block(BDW, "greedy"), part_vs_block(SNB, "greedy"),
     "%", True, True, "e3g pair"),
    ("thread ordering\n(grouped vs alternating, T=16)",
     36.5, 27.9, "%", True, False, "D44: 72893 / 74020"),
    ("socket seam\n(2 threads, slowdown)",
     -11.5, -9.5, "%", True, False, "D47: 74370 / 74020"),
    ("SMT on one socket\n(T=16 vs 8 cores)",
     smt_gain(BSMT), smt_gain(SNB), "%", False, True,
     "e3f_c6_72592 / e3g_c11"),
    ("the T=14 dip\n(T=12 -> T=14)",
     dip14(BDW), dip14(SNB), "%", False, True, "e3g pair"),
    ("A8: nzbalanced vs block\n(T=16 grouped)",
     14.1, -5.9, "%", False, False, "A8: withdrawn, see REGISTER 5.5"),
]

R = pd.DataFrame(ROWS, columns=["label", "bdw", "snb", "unit", "transfers",
                                "derived", "source"])
R = pd.concat([R[R.transfers], R[~R.transfers]], ignore_index=True)
n_t = int(R.transfers.sum())

fig, ax = plt.subplots(figsize=(10.5, 7.4))
y = np.arange(len(R))[::-1]

for yi, (_, r) in zip(y, R.iterrows()):
    alpha = 1.0 if r.derived else .45
    ax.plot([r.bdw, r.snb], [yi, yi], color="#999", lw=1.3, zorder=1,
            alpha=alpha)
    ax.scatter(r.bdw, yi, s=130, marker="o", zorder=3,
               facecolor="#1f77b4" if r.derived else "none",
               edgecolor="#1f77b4", linewidths=1.8, alpha=alpha)
    ax.scatter(r.snb, yi, s=130, marker="s", zorder=3,
               facecolor="#d62728" if r.derived else "none",
               edgecolor="#d62728", linewidths=1.8, alpha=alpha)
    ratio = (max(abs(r.bdw), abs(r.snb)) /
             max(min(abs(r.bdw), abs(r.snb)), 1e-9))
    ax.annotate(f"{ratio:.1f}x" if ratio < 20 else "",
                ((r.bdw + r.snb) / 2, yi), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=7.5, color="#666")

ax.axhline(len(R) - n_t - .5, color="#333", lw=1.2, ls="--")
ax.axvline(0, color="#333", lw=.9, zorder=0)
ax.set_yticks(y); ax.set_yticklabels(R.label, fontsize=8.5)
ax.set_ylim(-0.8, len(R) - 0.3)
ax.set_xlabel("effect size  (%)   -- per achieved GHz")
ax.set_title("What transfers across four microarchitecture generations")
ax.grid(axis="x", alpha=.25, lw=.5)

xl = ax.get_xlim()
ax.annotate("SIGN TRANSFERS  (magnitude does not; ratios above each pair)",
            (xl[1], len(R) - n_t - .28),
            ha="right", va="bottom", fontsize=8.5, weight="bold",
            color="#2ca02c")
ax.annotate("SIGN INVERTS", (xl[1], len(R) - n_t - .72),
            ha="right", va="top", fontsize=8.5, weight="bold", color="#d62728")

for yi, (_, r) in zip(y, R.iterrows()):
    ax.annotate(r.source, (xl[0], yi - .34), textcoords="offset points",
                xytext=(4, 0), fontsize=6.0, color="#aaa", va="center")

ax.legend(handles=[
    Line2D([], [], marker="o", ls="none", mfc="#1f77b4", mec="#1f77b4",
           ms=10, label="Broadwell c6 (2016)"),
    Line2D([], [], marker="s", ls="none", mfc="#d62728", mec="#d62728",
           ms=10, label="Sandy Bridge c11 (2012)"),
    Line2D([], [], marker="o", ls="none", mfc="none", mec="#555", ms=10,
           label="open = quoted from REGISTER, not re-derived here"),
], frameon=False, fontsize=8.5, loc="upper center",
   bbox_to_anchor=(0.5, -0.10), ncol=3)

fig.tight_layout()
os.makedirs(a.out, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(a.out, f"fig6_transfer.{ext}"), dpi=200,
                bbox_inches="tight")

print(R.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print("\nhosts:", BDW.host.iloc[0], SNB.host.iloc[0], BSMT.host.iloc[0])
print("platform column says:", BDW.platform.iloc[0], SNB.platform.iloc[0],
      "  <-- both wrong for SNB; keyed off node instead")
print(f"\nwrote {a.out}/fig6_transfer.png|pdf")
