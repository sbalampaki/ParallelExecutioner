#!/usr/bin/env python3

import argparse, os, sys
import numpy as np
import pandas as pd

pd.set_option('display.width', 220)

ap = argparse.ArgumentParser()
ap.add_argument('--offsets', default='/projects/sb2ea/csr/Wfull/offsets.i64')
ap.add_argument('--curve',
                default='/home/sb2ea/loadimbalance/results/day12/pstar_curve.csv')
a = ap.parse_args()

n = np.diff(np.fromfile(a.offsets, dtype=np.int64))
N, mx = int(n.sum()), int(n.max())
p_atom = N / mx
print(f"N = {N:,}   max = {mx:,}   p_atom = {p_atom:.6f}\n")

c = pd.read_csv(a.curve)
print(f"{len(c):,} rows from {a.curve}")
print(f"columns: {list(c.columns)}\n")

c['floor'] = c.p * mx / N
c['slack_nz'] = c.imb_nzbalanced - c['floor']
c['slack_bl'] = c.imb_block - c['floor']

#  1. bound violations
print("=" * 72)
print("1.  Does ANY row violate the exact bound?")
print("=" * 72)
TOL = 1e-12
for col, s in (('imb_nzbalanced', 'slack_nz'), ('imb_block', 'slack_bl')):
    v = c[c[s] < -TOL]
    print(f"  {col:<16} {len(v):>5} of {len(c):,} rows below floor", end='')
    print(f"   (worst {v[s].min():.3e})" if len(v) else "   -- bound holds")
    if len(v):
        print(v.nsmallest(5, s)[['p', 'n_ranks', 'n_threads', col, 'floor', s]]
              .to_string(index=False))

#  2. the three rows
print("\n" + "=" * 72)
print("2.  The three far-field rows in Section X-E")
print("=" * 72)
PAPER = {8192: 6.5937, 16384: 13.1876, 32768: 26.3744}
line = c[(c.n_threads == 8)]
for p, claimed in PAPER.items():
    r = line[line.p == p]
    if r.empty:
        r = c[c.p == p]
        tag = " (T=8 absent; showing all rows at this p)"
    else:
        tag = ""
    print(f"\n  p = {p:,}   paper reports nzbalanced = {claimed}{tag}")
    print(f"     exact floor            = {p*mx/N:.9f}")
    if r.empty:
        print("     NOT IN CURVE")
        continue
    for _, x in r.iterrows():
        print(f"     curve  R={int(x.n_ranks):>5} T={int(x.n_threads):<3} "
              f"nz={x.imb_nzbalanced:.9f}  slack={x.slack_nz:+.3e}  "
              f"block={x.imb_block:.6f}  ceiling={x.ceiling_pct:.4f}%")
        d = abs(x.imb_nzbalanced - claimed)
        if d > 5e-4:
            print(f"        -> differs from the paper by {d:.2e}"
                  f"  ** TRANSCRIPTION ERROR **")

#  3. ceilings
print("\n" + "=" * 72)
print("3.  Far-field ceilings   (paper: 10.58 / 5.29 / 3.69 %)")
print("=" * 72)
for p, claimed in zip((8192, 16384, 32768), (10.58, 5.29, 3.69)):
    r = line[line.p == p]
    if r.empty:
        print(f"  p={p:<6} not on the T=8 line"); continue
    x = r.iloc[0]
    rec = (x.imb_block / x.imb_nzbalanced - 1) * 100
    print(f"  p={p:<6} stored {x.ceiling_pct:>8.4f}%   recomputed {rec:>8.4f}%"
          f"   paper {claimed:>6}%   {'OK' if abs(rec-claimed)<0.02 else '** DIFFERS **'}")

#  4. peak
print("\n" + "=" * 72)
print("4.  Peak ceiling  (paper: 91.18 % at p=1552; block 2.4760, nz 1.2951)")
print("=" * 72)
pk = line.loc[line.ceiling_pct.idxmax()] if len(line) else None
if pk is not None:
    print(f"  T=8 line peak: p={int(pk.p)}  ceiling={pk.ceiling_pct:.4f}%  "
          f"block={pk.imb_block:.4f}  nz={pk.imb_nzbalanced:.4f}")
    print(f"  loss at peak = {(1-1/pk.imb_nzbalanced)*100:.2f}%  (paper: 22.8%)")
r = line[line.p == 1552]
if not r.empty:
    x = r.iloc[0]
    print(f"  at p=1552:     ceiling={x.ceiling_pct:.4f}%  "
          f"block={x.imb_block:.4f}  nz={x.imb_nzbalanced:.4f}")

#  5. anchors
print("\n" + "=" * 72)
print("5.  Measured anchors  (Table: loss 4.71/6.82/9.81/11.46/9.89)")
print("=" * 72)
ANCH = {16: (1.049428, 4.71, 4.84), 32: (1.073144, 6.82, 6.98),
        64: (1.108724, 9.81, 10.22), 80: (1.129385, 11.46, 11.56),
        96: (1.109744, 9.89, 8.33)}
for p, (ib, loss, ceil) in ANCH.items():
    r = line[line.p == p]
    if r.empty:
        print(f"  p={p:<4} not on T=8 line"); continue
    x = r.iloc[0]
    print(f"  p={p:<4} block {x.imb_block:.6f} vs {ib:<9} "
          f"{'OK' if abs(x.imb_block-ib)<1e-5 else '** DIFFERS **':<15} "
          f"loss {x.loss_block_pct:.2f} vs {loss}   "
          f"ceiling {x.ceiling_pct:.2f} vs {ceil}")

print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)
nv = (c.slack_nz < -TOL).sum()
if nv == 0:
    print("The curve satisfies the exact bound everywhere.")
    print("Any figure in the paper below the floor is a TRANSCRIPTION ERROR.")
    print("Fix: copy the values from section 2 above, report floor and")
    print("achieved to six decimals, and add a slack column so 'on the")
    print("floor' is visible rather than asserted.")
else:
    print(f"{nv} rows violate an exact lower bound -> the curve has a BUG.")
    print("Check how imb_nzbalanced is normalised, and whether the")
    print("two-stage cut can split a stay. Then regenerate and re-derive")
    print("the ceilings, the peak, and Regime III's boundary.")
