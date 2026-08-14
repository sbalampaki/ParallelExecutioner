#!/usr/bin/env python3
import glob, os
import pandas as pd

pd.set_option('display.width', 200)

JOBS = [72953, 73003, 73008, 73037, 73920, 73063]
fs = (glob.glob('/projects/sb2ea/results/timing/*.csv') +
      glob.glob('/home/sb2ea/loadimbalance/results/timing/*.csv'))
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True, sort=False)

k = d[(d.timed_region == 'kernel') & (d.job_id.isin(JOBS))].copy()
print(f"{len(k)} kernel rows from jobs {sorted(k.job_id.unique())}\n")
if k.empty:
    raise SystemExit("no rows; check JOBS list against your data")

k['cores'] = k.n_ranks * k.n_threads
k['mrec_ghz'] = k.throughput_rec_s / 1e6 / k.achieved_ghz

print("=== what configurations exist ===")
print(k.groupby(['job_id', 'csr_variant', 'n_nodes', 'n_ranks', 'n_threads',
                 'stage_mode' if 'stage_mode' in k.columns else 'paradigm'])
       .agg(rows=('mrec_ghz', 'size'), mrec=('mrec_ghz', 'median')).round(1).to_string())

print("\n=== the 16-core-per-node sweep ===")
s = k[(k.cores == 16) & (k.n_nodes == 1)]
if s.empty:
    s = k[k.cores == 16]
    print("(no n_nodes==1 rows; showing all node counts)")
g = (s.groupby(['n_ranks', 'n_threads', 'csr_variant'])['mrec_ghz']
       .agg(median='median', n='size').reset_index().sort_values('n_ranks'))
print(g.round(1).to_string(index=False))

print("\n=== single-socket 8-thread reference (for the ceiling) ===")
ref = k[(k.n_threads == 8) & (k.n_ranks == 1)]
if not ref.empty:
    print(ref.groupby(['job_id', 'csr_variant', 'thread_placement'])['mrec_ghz']
            .agg(['median', 'size']).round(1).to_string())
print("\nPaper: ceiling 871.2 = 2 x single-socket 8-thread; 8x2 = 810.8 = 93.1%")

print("\n=== LaTeX rows ===")
CEIL = 871.2
for _, r in g.iterrows():
    print(f"${int(r.n_ranks)} \\times {int(r.n_threads)}$ & "
          f"{r['median']:.1f} & {100*r['median']/CEIL:.1f} \\\\"
          f"   % {r.csr_variant}, n={int(r.n)}")
