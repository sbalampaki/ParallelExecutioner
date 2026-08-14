#!/usr/bin/env python3
import glob
import pandas as pd
pd.set_option('display.width', 240)
pd.set_option('display.max_rows', 400)

JOBS = [72953, 73003, 73008, 73037, 73063, 73920]
fs = (glob.glob('/projects/sb2ea/results/timing/*.csv') +
      glob.glob('/home/sb2ea/loadimbalance/results/timing/*.csv'))
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True, sort=False)
k = d[(d.timed_region == 'kernel') & (d.job_id.isin(JOBS))].copy()
k['cores'] = k.n_ranks * k.n_threads
k['mrec_ghz'] = k.throughput_rec_s / 1e6 / k.achieved_ghz

print("=== columns that VARY within these jobs ===")
for c in k.columns:
    u = k[c].astype(str).unique()
    if 1 < len(u) <= 12 and c not in ('mrec_ghz', 'cores'):
        print(f"  {c:<20} {sorted(u)}")

print("\n=== does any single cell reach 810.8 ? ===")
near = k[(k.mrec_ghz > 800) & (k.mrec_ghz < 825)]
print(f"rows in (800, 825): {len(near)}")
if len(near):
    cols = [c for c in ('job_id','n_ranks','n_threads','n_nodes','csr_variant',
                        'kernel_variant','stage_mode','row_order','notes',
                        'mrec_ghz') if c in near.columns]
    print(near[cols].to_string(index=False))
print(f"\noverall max mrec_ghz in these jobs: {k.mrec_ghz.max():.1f}")
top = k.nlargest(8, 'mrec_ghz')
cols = [c for c in ('job_id','n_ranks','n_threads','n_nodes','csr_variant',
                    'kernel_variant','stage_mode','row_order','mrec_ghz') if c in top.columns]
print(top[cols].to_string(index=False))

print("\n=== sweep split by every varying descriptor ===")
desc = [c for c in ('job_id','kernel_variant','stage_mode','row_order','schedule',
                    'mempolicy','numa_balancing','partitioner','thread_placement')
        if c in k.columns and k[c].astype(str).nunique() > 1]
print(f"splitting on: {desc}\n")
s = k[k.cores == 16]
g = (s.groupby(desc + ['n_ranks','n_threads'])['mrec_ghz']
       .agg(median='median', n='size').reset_index())
print(g.round(1).to_string(index=False))

print("\n=== per-job winner ===")
for j, gj in s.groupby('job_id'):
    m = gj.groupby(['n_ranks','n_threads'])['mrec_ghz'].median().sort_values(ascending=False)
    best = m.index[0]
    print(f"  job {int(j)}: best {int(best[0])}x{int(best[1])} at {m.iloc[0]:.1f}   "
          f"spread {100*(m.iloc[0]-m.iloc[-1])/m.iloc[-1]:.1f}%")

print("\n=== notes column (may name the arm) ===")
if 'notes' in k.columns:
    print(k.notes.astype(str).value_counts().head(15).to_string())
