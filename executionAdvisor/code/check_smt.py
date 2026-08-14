#!/usr/bin/env python3

import glob
import pandas as pd
pd.set_option('display.width', 240)
pd.set_option('display.max_rows', 300)

PAIR = [72897, 72949]
fs = (glob.glob('/projects/sb2ea/results/timing/*.csv') +
      glob.glob('/home/sb2ea/loadimbalance/results/timing/*.csv'))
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True, sort=False)

k = d[(d.timed_region == 'kernel') & (d.job_id.isin(PAIR)) &
      (d.warmup == 0)].copy()
k['mrec_ghz'] = k.throughput_rec_s / 1e6 / k.achieved_ghz
print(f"{len(k)} non-warmup kernel rows from {sorted(k.job_id.unique())}\n")
if k.empty:
    raise SystemExit("matched-pair jobs not found")

print("=== smt column values ===")
print(k.smt.astype(str).value_counts().to_string())

print("\n=== cells: platform x variant x smt x threads x placement ===")
g = (k.groupby(['platform', 'csr_variant', 'smt', 'n_threads',
                'thread_placement', 'partitioner', 'schedule'])['mrec_ghz']
       .agg(median='median', n='size').reset_index())
print(g.round(1).to_string(index=False))

print("\n" + "=" * 72)
print("SMT GAIN, computed per CSR variant")
print("=" * 72)
print("Comparing smt-on against smt-off at matched thread count,")
print("holding partitioner, schedule and placement fixed.\n")

base = k[(k.partitioner == 'block') & (k.schedule == 'precomputed')]
if base.empty:
    base = k
for plat in sorted(base.platform.astype(str).unique()):
    for var in sorted(base.csr_variant.astype(str).unique()):
        s = base[(base.platform == plat) & (base.csr_variant == var)]
        if s.empty:
            continue
        piv = s.pivot_table(index='n_threads', columns='smt',
                            values='mrec_ghz', aggfunc='median')
        cnt = s.pivot_table(index='n_threads', columns='smt',
                            values='mrec_ghz', aggfunc='size')
        if piv.shape[1] < 2:
            print(f"{plat:<14} {var:<7} only one smt level "
                  f"({list(piv.columns)}) -- no gain computable")
            continue
        a, b = piv.columns[0], piv.columns[1]
        piv['gain_%'] = (piv[b] / piv[a] - 1) * 100
        print(f"--- {plat} / {var} ---   (gain = {b} vs {a})")
        print(piv.round(2).to_string())
        print("  n per cell:")
        print(cnt.to_string())
        print()

print("=" * 72)
print("Paper: +29.8 (Broadwell) / +21.5 (Sandy Bridge).")
print("If the two variants give materially different gains, the row is")
print("blended and needs pinning like d38, part_vs_block and dip14.")
print("If only one variant has both smt levels, the row is already")
print("single-variant and nothing changes -- record which variant.")
