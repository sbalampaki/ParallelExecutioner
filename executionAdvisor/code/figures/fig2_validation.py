#!/usr/bin/env python3

import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RHO = 5.31
THREADS_PER_RANK = 8
RANKS_PER_NODE = 2

# ---- partition rules (mirror partition.h) ----------------------------------
def cumsum(w): return np.concatenate([[0.0], np.cumsum(w)])

def linspace_cuts(w, p):
    n = w.size; step = np.float64(n)/np.float64(p)
    c = (np.arange(p, dtype=np.int64)*step).astype(np.int64)
    c = np.append(c, np.int64(n))
    for j in range(1, p+1):
        if c[j] < c[j-1]: c[j] = c[j-1]
    return c

def balanced_cuts(w, p):
    cs = cumsum(w); n = w.size; cuts = np.zeros(p+1, dtype=np.int64)
    for j in range(p+1):
        t = cs[n]*j/p
        idx = int(np.searchsorted(cs, t, side='left'))
        lo, hi = max(idx-1, 0), min(idx, n)
        cuts[j] = lo if abs(cs[lo]-t) <= abs(cs[hi]-t) else hi
    cuts[0] = 0; cuts[p] = n
    for j in range(1, p+1):
        if cuts[j] < cuts[j-1]: cuts[j] = cuts[j-1]
    cuts[p] = n
    return cuts

def opt_cuts(w, p):
    cs = cumsum(w); n = w.size
    def npieces(L):
        i = k = 0
        while i < n:
            j = int(np.searchsorted(cs, cs[i]+L, side='right')) - 1
            if j <= i: return 10**9
            k += 1; i = j
            if k > p: return 10**9
        return k
    lo, hi = float(w.max()), float(cs[-1])
    for _ in range(100):
        mid = 0.5*(lo+hi)
        if npieces(mid) <= p: hi = mid
        else: lo = mid
    cuts = [0]; i = 0
    while i < n and len(cuts)-1 < p:
        j = int(np.searchsorted(cs, cs[i]+hi, side='right')) - 1
        cuts.append(j); i = j
    while len(cuts)-1 < p: cuts.append(n)
    cuts[-1] = n
    return np.array(cuts, dtype=np.int64)

def hier(w, R, T, f):
    rb = f(w, R); out = [0]
    for r in range(R):
        lo, hi = int(rb[r]), int(rb[r+1])
        sub = f(w[lo:hi], T) + lo if hi > lo else np.full(T+1, lo)
        out.extend(int(x) for x in sub[1:])
    return np.array(out, dtype=np.int64)

ap = argparse.ArgumentParser()
ap.add_argument("--csr",    default="/projects/sb2ea/csr/Wfull")
ap.add_argument("--lookup", default="/projects/sb2ea/manifest/kernel_lookup.csv")
ap.add_argument("--validation",
                default="/projects/sb2ea/results/day6/e0_validation.csv")
ap.add_argument("--out", default="figures")
a = ap.parse_args()

off    = np.fromfile(os.path.join(a.csr, "offsets.i64"), dtype=np.int64)
itemid = np.fromfile(os.path.join(a.csr, "itemid.i32"),  dtype=np.int32)
rec    = np.diff(off).astype(np.float64)
lk = pd.read_csv(a.lookup)
K  = int(itemid.max())+1
keep = np.zeros(K, dtype=bool)
d, s = lk.dense_id.to_numpy(), lk.slot.to_numpy(); ok = d < K
keep[d[ok]] = (s[ok] >= 0)
kk = np.add.reduceat(keep[itemid].astype(np.int64), off[:-1])
kk[np.diff(off) == 0] = 0
work = rec + RHO*kk
CS = cumsum(rec)

def imb_rec(b):
    L = CS[b[1:]] - CS[b[:-1]]
    return L.max()/(CS[-1]/(len(b)-1))

v = pd.read_csv(a.validation)

SERIES = [("block",      linspace_cuts, rec,  "#1f77b4", "o"),
          ("nzbalanced", balanced_cuts, rec,  "#2ca02c", "s"),
          ("contig_opt", opt_cuts,      work, "#d62728", "^"),
          ("wbalanced",  balanced_cuts, work, "#9467bd", "D")]

# node counts the cluster supports; p = N * ranks_per_node * threads_per_rank
NODES = [1, 2, 3, 4, 5]
PGRID = [N * RANKS_PER_NODE * THREADS_PER_RANK for N in NODES]
RGRID = [N * RANKS_PER_NODE for N in NODES]

# ax2 (discrimination) is drawn LEFT and labelled (a);
# ax1 (prediction vs measurement) is drawn RIGHT and labelled (b).
fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(12.5, 4.8))

for nm, f, key, col, mk in SERIES:
    ys = [imb_rec(hier(key, R, THREADS_PER_RANK, f)) for R in RGRID]
    ax1.plot(PGRID, ys, "-", color=col, lw=1.3, alpha=.7, zorder=1,
             marker=".", ms=6)
    g = v[v.partitioner == nm]
    ax1.scatter(g.p, g.measured, s=72, facecolors="none", edgecolors=col,
                linewidths=1.9, marker=mk, zorder=3, label=nm)

ax1.set_xlabel("partitions $p$   (N nodes x 2 ranks x 8 threads)")
ax1.set_ylabel("work_imbalance  (max/mean records per thread)")
ax1.set_title("(b) computed from offsets alone, measured on 1-5 nodes")
ax1.set_xticks(PGRID)
ax1.legend(frameon=False, fontsize=9, loc="upper left")
ax1.grid(alpha=.25, lw=.5)

# secondary axis: the efficiency ceiling imbalance implies
ax1b = ax1.twinx()
lo_y, hi_y = ax1.get_ylim()
ax1b.set_ylim(100.0/hi_y, 100.0/lo_y)
ax1b.set_ylabel("efficiency cap = 100 / imbalance   (%)", fontsize=9)
ax1b.tick_params(labelsize=8)

# ---- panel (b): flat vs hierarchical where they differ ---------------------
v = v.assign(err_flat=(v.predicted_flat/v.measured - 1).abs(),
             err_hier=(v.predicted_hier/v.measured - 1).abs())
dis = v[(v.predicted_flat - v.predicted_hier).abs() > 1e-9].copy()
dis = dis.sort_values(["partitioner", "p"])
dis["lab"] = dis.partitioner + "\np=" + dis.p.astype(str)

x = np.arange(len(dis))
FLOOR = 3e-8          # below the 1e-6 precision the measurements are recorded at
ax2.bar(x-.19, dis.err_flat.clip(lower=FLOOR), .36, color="#c8c8c8",
        edgecolor="#666", lw=.6, label="one-stage (flat) model")
ax2.bar(x+.19, dis.err_hier.clip(lower=FLOOR), .36, color="#d62728",
        edgecolor="#8b1a1a", lw=.6,
        label="two-stage model (kernel_mpi.c:254-262)")
ax2.axhline(1e-6, ls=":", lw=1.0, color="#888", zorder=0)
ax2.set_yscale("log")
ax2.set_ylim(FLOOR, max(dis.err_flat.max()*4, 1e-2))
ax2.set_xticks(x); ax2.set_xticklabels(dis.lab, fontsize=8)
ax2.set_ylabel("|relative error| vs measured")
ax2.set_title(f"(a) the {len(dis)} points where the two models differ"
              f" - two-stage wins at all {len(dis)}")
ax2.legend(frameon=False, fontsize=9, loc="upper left")
ax2.grid(axis="y", alpha=.25, lw=.5)


fig.tight_layout()
os.makedirs(a.out, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(a.out, f"fig2_e0_validation.{ext}"),
                dpi=200, bbox_inches="tight")

print("discriminating points:", len(dis))
print(dis[["partitioner", "p", "predicted_flat", "predicted_hier", "measured",
           "err_flat", "err_hier"]].to_string(index=False))
print("\npanel (a) grid:")
for N, R, p in zip(NODES, RGRID, PGRID):
    print(f"   N={N}  ranks={R}  threads=8  p={p}")
print(f"\nwrote {a.out}/fig2_e0_validation.png|pdf")
