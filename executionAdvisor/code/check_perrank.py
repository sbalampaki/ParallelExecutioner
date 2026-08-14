#!/usr/bin/env python3

import glob
import pandas as pd
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 400)
pd.set_option('display.max_colwidth', 60)

fs = (glob.glob('/projects/sb2ea/results/timing/*.csv') +
      glob.glob('/home/sb2ea/loadimbalance/results/timing/*.csv'))
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True, sort=False)

k = d[(d.job_id == 73920) & (d.timed_region == 'kernel')].copy()
k['mrec'] = k.throughput_rec_s / 1e6 / k.achieved_ghz
k['arm'] = k.notes.astype(str).str.extract(r'per_rank_csr=(\d)')[0]
k['cores'] = k.n_ranks * k.n_threads
print(f"job 73920: {len(k)} kernel rows\n")

print("=" * 72)
print("1.  every column that varies")
print("=" * 72)
for c in k.columns:
    u = k[c].astype(str).unique()
    if 1 < len(u) <= 20 and c not in ('mrec', 'cores'):
        print(f"  {c:<22} {sorted(u, key=str)}")

print("\n" + "=" * 72)
print("2.  full notes strings (arm labels may live here)")
print("=" * 72)
print(k.notes.astype(str).value_counts().to_string())

print("\n" + "=" * 72)
print("3.  cells by arm and configuration")
print("=" * 72)
gc = [c for c in ('n_ranks', 'n_threads', 'arm', 'kernel_variant',
                  'stage_mode', 'row_order', 'thread_placement', 'warmup')
      if c in k.columns]
print(k.groupby(gc)['mrec'].agg(median='median', n='size').round(1).to_string())

print("\n" + "=" * 72)
print("4.  run order -- is the design a palindrome?")
print("=" * 72)
o = k[k.warmup == 0].sort_values('timestamp') if 'timestamp' in k.columns else k
cols = [c for c in ('timestamp', 'run_id', 'n_ranks', 'n_threads', 'arm',
                    'repetition', 'mrec') if c in o.columns]
print(o[cols].head(60).to_string(index=False))

print("\n" + "=" * 72)
print("5.  pooled-median estimator, per configuration")
print("=" * 72)
nw = k[k.warmup == 0]
for (r, t), g in nw.groupby(['n_ranks', 'n_threads']):
    a = g[g.arm == '1']['mrec']
    b = g[g.arm == '0']['mrec']
    if len(a) and len(b):
        print(f"  {int(r)}x{int(t)}: per-rank {a.median():7.1f} (n={len(a):>3})  "
              f"shared {b.median():7.1f} (n={len(b):>3})  "
              f"gain {100*(a.median()/b.median()-1):+6.2f}%")

print("\nVIII-G reports +1.50% pooled and +1.80% palindrome half-mean.")
print("If no configuration reproduces those, the estimator is restricted to a")
print("particular shared arm that section 1 above should reveal.")
