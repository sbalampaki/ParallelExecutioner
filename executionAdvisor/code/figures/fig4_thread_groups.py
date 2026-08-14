#!/usr/bin/env python3

import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--timings",
                default="/projects/sb2ea/results/timing/seam_c6_74370.csv")
ap.add_argument("--threads",
                default="/projects/sb2ea/results/timing/seam_c6_74370_threads.csv")
ap.add_argument("--out", default="figures")
a = ap.parse_args()

# ---- load ------------------------------------------------------------------
t = pd.read_csv(a.timings)
t = t[(t.warmup == 0) & (t.timed_region == "kernel")].copy()
t["seams"] = t.notes.str.extract(r"seams=(\d+)")[0].astype(int)
t["cpus_per_group"] = t.notes.str.extract(r"group=(\d+)")[0].astype(int)
t["n_groups"] = 16 // t.cpus_per_group
t["mrec_ghz"] = t.throughput_rec_s / 1e6 / t.achieved_ghz

th = pd.read_csv(a.threads)
key = t.set_index("run_id")[["seams", "n_groups"]].to_dict("index")
th["seams"] = th.run_id.map(lambda r: key.get(r, {}).get("seams"))
th = th[th.seams.notna()].copy()
th["seams"] = th.seams.astype(int)
th["mrec_s"] = th.n_records / th.secs / 1e6

# ---- panel (a): the ladder -------------------------------------------------
agg = (t.groupby(["seams", "n_groups"])
         .agg(mrec_ghz=("mrec_ghz", "median"),
              wimb=("work_imbalance", "median"),
              timb=("time_imbalance", "median"),
              n=("mrec_ghz", "size"))
         .reset_index().sort_values("n_groups"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

ax1.plot(agg.n_groups, agg.mrec_ghz, "o-", color="#1f77b4", lw=1.6, ms=8,
         zorder=3)
ax1.set_xscale("log", base=2)
ax1.set_xticks(agg.n_groups.tolist())
ax1.set_xticklabels([str(g) for g in agg.n_groups])
ax1.set_xlabel("contiguous same-socket thread groups   (= seams + 1)")
ax1.set_ylabel("throughput  (M records / GHz)")
ax1.set_title("(a) -8 to -10% per doubling, at fixed work_imbalance")
ax1.grid(alpha=.25, lw=.5)

base = float(agg.mrec_ghz.iloc[0])
for _, r in agg.iterrows():
    d = 100 * (r.mrec_ghz / base - 1)
    lbl = f"{r.mrec_ghz:.1f}" + ("" if d == 0 else f"\n{d:+.1f}%")
    ax1.annotate(lbl, (r.n_groups, r.mrec_ghz), textcoords="offset points",
                 xytext=(10, 6), fontsize=8.5, color="#1f77b4")
    ax1.annotate(f"imb {r.wimb:.4f}", (r.n_groups, r.mrec_ghz),
                 textcoords="offset points", xytext=(10, -16), fontsize=7.5,
                 color="#777")

span = 100 * (float(agg.mrec_ghz.iloc[0]) / float(agg.mrec_ghz.iloc[-1]) - 1)
ax1.annotate(f"1 seam vs {int(agg.seams.iloc[-1])} seams: +{span:.1f}%\n"
             "same CPUs, same partition, only the permutation differs",
             xy=(.97, .95), xycoords="axes fraction", ha="right", va="top",
             fontsize=8.5, color="#444")

# ---- panel (b): the distribution shift -------------------------------------
lo_s, hi_s = int(agg.seams.min()), int(agg.seams.max())
groups, labels, colors = [], [], []
for s, col in [(lo_s, "#2ca02c"), (hi_s, "#d62728")]:
    g = th[th.seams == s]
    groups.append(g.mrec_s.values)
    labels.append(f"{s} seam{'s' if s != 1 else ''}\n({16//(s+1)} CPUs/group)")
    colors.append(col)

parts = ax2.violinplot(groups, positions=[0, 1], widths=.75,
                       showmedians=False, showextrema=False)
for body, col in zip(parts["bodies"], colors):
    body.set_facecolor(col); body.set_alpha(.22); body.set_edgecolor(col)

rng = np.random.default_rng(0)
for i, (s, col) in enumerate([(lo_s, "#2ca02c"), (hi_s, "#d62728")]):
    g = th[th.seams == s]
    seam_adj = g.tid.isin([7, 8])
    jit = rng.uniform(-.10, .10, len(g))
    if s == lo_s:
        ax2.scatter(i + jit[seam_adj.values], g.mrec_s[seam_adj], s=64,
                facecolors="none", edgecolors="k", linewidths=1.4, zorder=4,
                label="tid 7, 8 (seam-adjacent)")
        ax2.scatter(i + jit[~seam_adj.values], g.mrec_s[~seam_adj], s=26,
                color=col, alpha=.65, zorder=3, edgecolors="none")
    else:
        ax2.scatter(i + jit, g.mrec_s, s=26, color=col, alpha=.65, zorder=3,
                    edgecolors="none")
    med = float(g.mrec_s.median())
    ax2.hlines(med, i-.34, i+.34, color=col, lw=2.4, zorder=5)
    ax2.annotate(f"median {med:.1f}", (i+.36, med), fontsize=8.5, color=col,
                 va="center")
                 
span_signed = 100*(agg.mrec_ghz.iloc[-1]/agg.mrec_ghz.iloc[0] - 1)
m_lo = float(th[th.seams == lo_s].mrec_s.median())
m_hi = float(th[th.seams == hi_s].mrec_s.median())
ax2.annotate("", xy=(1.5, m_hi), xytext=(1.5, m_lo),
             arrowprops=dict(arrowstyle="<->", color="#333", lw=1.2))
ax2.annotate(f"median thread {100*(m_hi/m_lo-1):+.1f}%\n"
             f"throughput  {span_signed:+.1f}%\n"
             "they track",
             (1.54, (m_lo+m_hi)/2), fontsize=9, va="center", color="#333")

ax2.set_xticks([0, 1]); ax2.set_xticklabels(labels)
ax2.set_xlim(-.6, 2.0)
ax2.set_ylabel("per-thread speed  (M records / s)")
ax2.set_title("(b) the whole distribution shifts, it does not straggle")
ax2.grid(axis="y", alpha=.25, lw=.5)
ax2.legend(frameon=False, fontsize=8.5, loc="lower left")

fig.tight_layout()
os.makedirs(a.out, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(a.out, f"fig4_thread_groups.{ext}"),
                dpi=200, bbox_inches="tight")

# ---- console check ---------------------------------------------------------
print("=== panel (a) ===")
print(agg.to_string(index=False))
print(f"\n1 seam vs {hi_s} seams: {span:+.1f}%")
print("\n=== panel (b): per-thread medians and spread ===")
for s in sorted(th.seams.unique()):
    g = th[th.seams == s]
    dev = 100 * (g.mrec_s / g.groupby("run_id").mrec_s.transform("median") - 1)
    sa = g.tid.isin([7, 8])
    print(f"  seams={s:2d}  median {g.mrec_s.median():7.2f}   "
          f"min-dev-from-own-median {dev.min():6.2f}%   "
          f"tid7/8 mean dev {dev[sa].mean():6.2f}%")
print(f"\nwrote {a.out}/fig4_thread_groups.png|pdf")
