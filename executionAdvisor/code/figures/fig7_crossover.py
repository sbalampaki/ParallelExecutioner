#!/usr/bin/env python3

import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--curve",
                default="/home/sb2ea/loadimbalance/results/day12/pstar_curve.csv")
ap.add_argument("--offsets",
                default="/projects/sb2ea/csr/Wfull/offsets.i64")
ap.add_argument("--out", default="figures")
a = ap.parse_args()

# ---------------------------------------------------------------- inputs
c = pd.read_csv(a.curve)
n = np.diff(np.fromfile(a.offsets, dtype=np.int64))
N, MX = int(n.sum()), int(n.max())
P_ATOM = N / MX

# the measured configuration: eight threads per rank
line = c[c.n_threads == 8].sort_values("p").copy()
line["floor"] = line.p * MX / N

MEASURED = [16, 32, 64, 80, 96]
IQR, RUNG, SPAN = 2.39, 8.40, 32.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

# ------------------------------------------------------- (a) the crossover
near = line[line.p <= 4096]
# the oscillation is real -- divisibility of the stay count across first-stage
# cuts -- but it obscures the crossover at full opacity. Raw beneath, rolling
# median over it.
ax1.plot(near.p, near.ceiling_pct, color="#1f77b4", lw=0.6, alpha=.30,
         zorder=2)
w = max(5, len(near) // 40)
ax1.plot(near.p, near.ceiling_pct.rolling(w, center=True, min_periods=1).median(),
         color="#1f77b4", lw=1.9, zorder=3, label="ceiling (rolling median)")

for v, lbl, col, va in ((IQR, "within-repetition IQR  2.39%", "#999", "bottom"),
                        (RUNG, "one placement rung  8.40%", "#d62728", "bottom"),
                        (SPAN, "full placement span  32.0%", "#666", "bottom")):
    ax1.axhline(v, color=col, lw=1.0, ls="--", zorder=1)
    ax1.annotate(lbl, (near.p.min(), v), xytext=(3, 3),
                 textcoords="offset points", fontsize=7.5, color=col,
                 ha="left", va="bottom")

m = near[near.p.isin(MEASURED)]
ax1.scatter(m.p, m.ceiling_pct, s=55, zorder=6, color="#1f77b4",
            edgecolor="white", linewidths=1.2,
            label="measured partition counts")

cross = near[near.ceiling_pct >= RUNG]
if len(cross):
    p_star = int(cross.p.iloc[0])
    ax1.axvline(p_star, color="#d62728", lw=0.9, ls=":", zorder=2)
    ax1.annotate(f"$p^*={p_star}$", (p_star, RUNG), xytext=(6, 14),
                 textcoords="offset points", fontsize=9, color="#d62728",
                 weight="bold", ha="left")

ax1.axvspan(96, near.p.max(), color="#000", alpha=.045, zorder=0)
ax1.annotate("computed, not measured", (near.p.max(), ax1.get_ylim()[1]),
             xytext=(-4, -4), textcoords="offset points",
             fontsize=7.5, color="#888", ha="right", va="top")
ax1.set_xscale("log")
ax1.set_xlabel("partition count $p$   (8 threads per rank)")
ax1.set_ylabel("ceiling $C(p)$   (%)")
ax1.set_title("(a)  which knob repays attention", fontsize=10, loc="left")
ax1.grid(alpha=.25, lw=.5)
ax1.legend(frameon=False, fontsize=8, loc="lower right")

# ------------------------------------------------------- (b) the far field

far = line[line.p >= 256].copy()
far["r_bl"] = far.imb_block / far["floor"]
far["r_nz"] = far.imb_nzbalanced / far["floor"]

ax2.plot(far.p, far.r_bl, color="#d62728", lw=1.7, label="block")
ax2.plot(far.p, far.r_nz, color="#1f77b4", lw=1.7, label="nzbalanced")
ax2.axhline(1.0, color="#333", lw=1.3, ls="--",
            label=r"atomicity bound $p\,\max_i n_i/N$")

ax2.axvline(P_ATOM, color="#666", lw=0.9, ls=":", zorder=2)
ax2.annotate(f"$p_{{atom}}={P_ATOM:.0f}$", (P_ATOM, 1.0), xytext=(-4, 8), ha ="right",
             textcoords="offset points", fontsize=8.5, color="#666",
             va="bottom")

pk = line.loc[line.ceiling_pct.idxmax()]
pk_r = pk.imb_block / (pk.p * MX / N)
ax2.scatter([pk.p], [pk_r], s=48, color="#d62728", zorder=6,
            edgecolor="white", linewidths=1.1)
ax2.annotate(f"ceiling peaks {pk.ceiling_pct:.1f}%\nat $p={int(pk.p)}$",
             (pk.p, pk_r), xytext=(-10, 26), textcoords="offset points",
             fontsize=7.5, color="#d62728", ha="right")

ff = far[far.p >= 8192]
if len(ff):
    ax2.annotate("nzbalanced attains the bound\n(residual exactly zero)",
                 (ff.p.iloc[0], 1.0), xytext=(-6, 60),
                 textcoords="offset points", fontsize=7.5, color="#1f77b4",
                 ha="right",
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=.8))

ax2.set_xscale("log")
ax2.set_ylim(0.95, min(3.2, far.r_bl.max() * 1.12))
ax2.set_xlabel("partition count $p$")
ax2.set_ylabel(r"imbalance $\,/\,$ atomicity bound")
ax2.set_title("(b)  the atomicity floor", fontsize=10, loc="left")
ax2.grid(alpha=.25, lw=.5)
ax2.legend(frameon=False, fontsize=8, loc="upper right")

fig.tight_layout()
os.makedirs(a.out, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(a.out, f"fig7_crossover.{ext}"), dpi=200,
                bbox_inches="tight")

# ------------------------------------------------------- console check
print(f"N = {N:,}   max = {MX:,}   p_atom = {P_ATOM:.4f}")
print(f"rows on the T=8 line: {len(line):,}   of {len(c):,} total\n")

print("crossings along T=8:")
for thr, nm in ((IQR, "IQR 2.39%"), (RUNG, "one rung 8.40%"),
                (SPAN, "full span 32.0%")):
    x = line[line.ceiling_pct >= thr]
    print(f"  {nm:<18} first at p = {int(x.p.iloc[0]) if len(x) else '--'}")

print("\nmeasured anchors:")
for p in MEASURED:
    r = line[line.p == p]
    if len(r):
        r = r.iloc[0]
        print(f"  p={p:<4} block {r.imb_block:.6f}  loss {r.loss_block_pct:5.2f}%"
              f"  ceiling {r.ceiling_pct:5.2f}%")

print("\nfar field, nzbalanced against the exact bound:")
for p in (8192, 16384, 32768):
    r = line[line.p == p]
    if len(r):
        r = r.iloc[0]
        print(f"  p={p:<6} nz {r.imb_nzbalanced:.6f}  floor {r['floor']:.6f}"
              f"  slack {r.imb_nzbalanced - r['floor']:+.2e}"
              f"  ceiling {r.ceiling_pct:.2f}%")

print(f"\npeak ceiling {pk.ceiling_pct:.2f}% at p={int(pk.p)}  "
      f"(block {pk.imb_block:.4f}, nz {pk.imb_nzbalanced:.4f}, "
      f"loss at peak {(1-1/pk.imb_nzbalanced)*100:.2f}%)")
print(f"\nwrote {a.out}/fig7_crossover.pdf|png")
