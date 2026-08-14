#!/usr/bin/env python3

import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--timings",
                default="/projects/sb2ea/results/timing/e4ps_74398.csv")
ap.add_argument("--predictions",
                default="/projects/sb2ea/results/day6/e0_validation.csv")
ap.add_argument("--out", default="figures")
a = ap.parse_args()

t = pd.read_csv(a.timings)
t = t[t.warmup == 0].copy()
t["partitioner"] = t.notes.str.extract(r"partitioner=([^;]+)")[0]
t["p"] = t.notes.str.extract(r";p=(\d+)")[0].astype(float)
t = t[t.partitioner.notna()].copy()

bad = sorted(t.loc[t.bitidentical == "fail", "partitioner"].dropna().unique())
if bad:
    print(f"excluded (bit-identity): {bad}")
    t = t[~t.partitioner.isin(bad)]

t = t[t.timed_region == "kernel"].copy()
t["p"] = t.p.astype(int)
t["mrec_ghz"] = t.throughput_rec_s / 1e6 / t.achieved_ghz

e = pd.read_csv(a.predictions).rename(columns={"predicted_hier": "pred_imb"})
e = e[["partitioner", "p", "pred_imb"]]

g = (t.groupby(["p", "partitioner"])
       .agg(med=("mrec_ghz", "median"), lo=("mrec_ghz", "min"),
            hi=("mrec_ghz", "max"), n=("mrec_ghz", "size"))
       .reset_index().merge(e, on=["p", "partitioner"], how="left"))

PS = sorted(g.p.unique())
rows = []
for p in PS:
    sub = t[t.p == p]
    piv = sub.pivot_table(index="repetition", columns="partitioner",
                          values="mrec_ghz", aggfunc="median")
    ratio = 100 * (piv["contig_opt"] / piv["block"] - 1)
    pr = e[e.p == p].set_index("partitioner").pred_imb
    rows.append(dict(p=p,
                     measured=float(ratio.median()),
                     meas_lo=float(ratio.min()),
                     meas_hi=float(ratio.max()),
                     predicted=100*(pr["block"]/pr["contig_opt"] - 1),
                     n_pos=int((ratio > 0).sum()), n_rep=len(ratio)))
D = pd.DataFrame(rows)
D["realised"] = 100 * D.measured / D.predicted

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                               gridspec_kw={"width_ratios": [1, 1.25]})

# ---- panel (a): the dose ---------------------------------------------------
x = np.arange(len(D)); w = 0.34
ax1.bar(x - w/2, D.predicted, w, color="#d62728", alpha=.85,
        edgecolor="#8b1a1a", lw=.7, label="predicted from E0 imbalance")
ax1.bar(x + w/2, D.measured, w, color="#1f77b4",
        edgecolor="#0d3d61", lw=.7, label="measured",
        yerr=[D.measured - D.meas_lo, D.meas_hi - D.measured],
        capsize=5, error_kw=dict(ecolor="#333", lw=1.2))

for i, r in D.iterrows():
    ax1.annotate(f"+{r.predicted:.1f}%", (i - w/2, r.predicted),
                 textcoords="offset points", xytext=(0, 4), ha="center",
                 fontsize=9, color="#8b1a1a")
    ax1.annotate(f"+{r.measured:.1f}%", (i + w/2, r.measured),
                 textcoords="offset points", xytext=(0, 4), ha="center",
                 fontsize=9, weight="bold", color="#0d3d61")
    ax1.annotate(f"{r.realised:.0f}% of the available gain\n"
             f"positive in {int(r.n_pos)}/{int(r.n_rep)} paired reps",
             (i, -0.9), ha="center", va="top", fontsize=8.5, color="#444")

ax1.set_xticks(x)
ax1.set_xticklabels([f"p={int(r.p)}\n({int(r.p)//16} node"
                     f"{'s' if r.p//16 > 1 else ''})" for _, r in D.iterrows()])
ax1.set_ylabel("contig_opt vs block  (%)")
ax1.set_title("(a) the gain grows with the ceiling, recovering ~60% of it at both p")
ax1.set_ylim(-3.4, max(D.meas_hi.max(), D.predicted.max()) * 1.15)
ax1.axhline(0, color="#333", lw=.9)
ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
ax1.grid(axis="y", alpha=.25, lw=.5)

# ---- panel (b): what is and is not resolvable ------------------------------
order = (g[g.p == max(PS)].sort_values("med", ascending=False)
           .partitioner.tolist())
COL = {"contig_opt": "#d62728", "wbalanced": "#9467bd",
       "nzbalanced": "#2ca02c", "block": "#1f77b4"}
off = {PS[0]: -0.17, PS[-1]: 0.17}

for p in PS:
    sub = g[g.p == p].set_index("partitioner").loc[order].reset_index()
    norm = float(sub.loc[sub.partitioner == "block", "med"].iloc[0])
    y = np.arange(len(order)) + off[p]
    mk = "o" if p == PS[-1] else "s"
    for j, r in sub.iterrows():
        ax2.plot([100*r.lo/norm - 100, 100*r.hi/norm - 100], [y[j], y[j]],
                 color=COL[r.partitioner], lw=2.2, alpha=.45, zorder=2,
                 solid_capstyle="round")
        ax2.plot(100*r.med/norm - 100, y[j], mk, color=COL[r.partitioner],
                 ms=9, zorder=3, mec="#222", mew=.8)

ax2.axvline(0, color="#333", lw=1.0, zorder=1)
ax2.set_yticks(np.arange(len(order))); ax2.set_yticklabels(order)
ax2.invert_yaxis()
ax2.set_xlabel("per-cycle throughput vs block, same p  (%)")
ax2.set_title("(b) only contig_opt > block is resolvable")
ax2.grid(axis="x", alpha=.25, lw=.5)
ax2.plot([], [], "s", color="#777", label=f"p={PS[0]}  (bars: min-max of "
         f"{int(g[g.p==PS[0]].n.median())} reps)")
ax2.plot([], [], "o", color="#777", label=f"p={PS[-1]}")
ax2.legend(frameon=False, fontsize=8, loc="lower right")
ax2.annotate("contig_opt beats block in 5/5 paired repetitions at both p.\n"
             "Among the three leaders the ordering reverses within the job\n"
             "(nzbalanced leads at rep 1, trails by rep 5), so their ranking\n"
             "is not established -- that needs repetitions, not more nodes.",
             xy=(.02, .28), xycoords="axes fraction", fontsize=7.5,
             color="#666", va="bottom")

fig.tight_layout()
os.makedirs(a.out, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(a.out, f"fig5_partitioner_dose.{ext}"),
                dpi=200, bbox_inches="tight")

print("\n=== per-cell throughput (M rec/GHz) ===")
print(g.sort_values(["p", "med"], ascending=[True, False])
       .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\n=== the dose ===")
print(D.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print(f"\nwrote {a.out}/fig5_partitioner_dose.png|pdf")
