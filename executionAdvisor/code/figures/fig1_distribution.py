#!/usr/bin/env python3

import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--offsets", default="/projects/sb2ea/csr/Wfull/offsets.i64")
ap.add_argument("--summary",
                default="/projects/sb2ea/results/e0/distribution_summary.csv")
ap.add_argument("--out", default="figures")
a = ap.parse_args()

GINI_MIN, P99M_MIN = 0.4, 10.0

off = np.fromfile(a.offsets, dtype=np.int64)
r = np.diff(off).astype(np.float64)
n = r.size
s = np.sort(r)
c = np.cumsum(s)
gini = (n + 1 - 2 * (c.sum() / c[-1])) / n
p50, p90, p99 = np.percentile(r, [50, 90, 99])
cv = r.std(ddof=1) / r.mean()
top1 = s[int(.99 * n):].sum() / r.sum()

fig = plt.figure(figsize=(14.5, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1.15], wspace=.28)
ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))

pos = s[s > 0]
ccdf = 1.0 - np.arange(pos.size) / pos.size
ax1.loglog(pos, ccdf, color="#1f77b4", lw=1.8)
for v, lab, col in [(p50, "P50", "#888"), (p90, "P90", "#888"),
                    (p99, "P99", "#d62728"), (r.max(), "max", "#d62728")]:
    ax1.axvline(v, color=col, ls=":", lw=1.0, zorder=0)
    ax1.annotate(f"{lab}\n{v:,.0f}", (v, 1.3e-4), fontsize=7.5, color=col,
                 ha="center", rotation=0)
ax1.set_xlabel("records per stay")
ax1.set_ylabel("P(X > x)")
ax1.set_title("(a) records per stay, Wfull")
ax1.grid(alpha=.25, lw=.5, which="both")
ax1.set_ylim(1e-5, 1.4)
ax1.annotate(f"n = {n:,} stays\n{int(r.sum()):,} records\n"
             f"CV = {cv:.3f}\nP99/median = {p99/p50:.1f}",
             xy=(.03, .05), xycoords="axes fraction", fontsize=8.5,
             va="bottom", color="#333")

x = np.arange(1, n + 1) / n
ylo = c / c[-1]
ax2.plot([0, 1], [0, 1], ls="--", color="#888", lw=1.2, label="perfect balance")
ax2.plot(x, ylo, color="#d62728", lw=1.8, label="Wfull")
ax2.fill_between(x, ylo, x, color="#d62728", alpha=.13)
ax2.annotate(f"Gini = {gini:.3f}", (.42, .27), fontsize=11, color="#d62728",
             weight="bold")
ax2.plot([.99, .99], [ylo[int(.99*n)-1], 1.0], color="#333", lw=1.6)
ax2.annotate(f"top 1% of stays\nhold {100*top1:.1f}% of records",
             (.965, ylo[int(.99*n)-1]), ha="right", va="top", fontsize=8,
             color="#333")
ax2.set_xlabel("cumulative share of stays")
ax2.set_ylabel("cumulative share of records")
ax2.set_title("(b) the tail's share of the work")
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
ax2.grid(alpha=.25, lw=.5)

d = pd.read_csv(a.summary)
d = d[["window", "gini", "p99_over_median", "cv"]].copy()
d.loc[len(d)] = ["Wfull (recomputed)", gini, p99 / p50, cv]
d["pass"] = (d.gini > GINI_MIN) & (d.p99_over_median > P99M_MIN)
d = d.sort_values("gini")

ax3.axvspan(GINI_MIN, d.gini.max() * 1.15, color="#2ca02c", alpha=.07,
            zorder=0)
ax3.axvline(GINI_MIN, color="#2ca02c", lw=1.4, ls="--", zorder=1)
ax3.scatter(d.gini, d.p99_over_median, s=110, zorder=3,
            c=["#2ca02c" if p else "#bbb" for p in d["pass"]],
            edgecolors="#333", linewidths=.9)
ax3.axhline(P99M_MIN, color="#2ca02c", lw=1.4, ls="--", zorder=1)


LABEL = ("Wfull", "sizematched_Wfull", "shape_W24_trim0")
for k, (_, row) in enumerate(d.iterrows()):
    if not any(row.window.startswith(t) for t in LABEL):
        continue
    nm = (row.window.replace("sizematched_", "sm_").replace("shape_", "")
          .replace("_trim0", " (plan's primary)"))
    ax3.annotate(nm, (row.gini, row.p99_over_median),
                 textcoords="offset points",
                 xytext=(8, 5) if row["pass"] else (8, -12),
                 fontsize=8, weight="bold" if row["pass"] else "normal",
                 color="#222" if row["pass"] else "#666")
ax3.annotate(f"{int((~d['pass']).sum())} windowed variants,\nall failing "
             "(see the variants table)",
             (.97, .06), xycoords="axes fraction", ha="right", va="bottom",
             fontsize=7.5, color="#888")

ax3.set_yscale("log")
ax3.set_xlabel("Gini coefficient of work")
ax3.set_ylabel("P99 / median records per stay")
ax3.set_title("(c) plan.md's Day 3 gate across the sweep")
ax3.grid(alpha=.25, lw=.5, which="both")
ax3.annotate("GATE: Gini > 0.4 and P99/median > 10", (.03, .95),
             xycoords="axes fraction", fontsize=8.5, weight="bold",
             color="#2ca02c", va="top")

fig.tight_layout()
os.makedirs(a.out, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(a.out, f"fig1_distribution.{ext}"), dpi=200,
                bbox_inches="tight")

print(f"Wfull: n={n:,} records={int(r.sum()):,} cv={cv:.6f}")
print(f"  p50={p50:.0f} p90={p90:.0f} p99={p99:.0f} max={r.max():.0f}")
print(f"  p99/median={p99/p50:.3f} gini={gini:.6f} top1%={top1:.6f}")
print(f"  GATE: {'PASS' if gini > GINI_MIN and p99/p50 > P99M_MIN else 'FAIL'}")
print("\ngate across the sweep:")
print(d.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nwrote {a.out}/fig1_distribution.png|pdf")
