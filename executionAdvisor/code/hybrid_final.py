#!/usr/bin/env python3
import glob
import pandas as pd
pd.set_option('display.width', 240)

JOBS = [73037, 73063, 73920]
fs = (glob.glob('/projects/sb2ea/results/timing/*.csv') +
      glob.glob('/home/sb2ea/loadimbalance/results/timing/*.csv'))
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True, sort=False)

k = d[(d.timed_region == 'kernel') & (d.job_id.isin(JOBS)) &
      (d.warmup == 0)].copy()
k['cores'] = k.n_ranks * k.n_threads
k['mrec_ghz'] = k.throughput_rec_s / 1e6 / k.achieved_ghz
k['arm'] = k.notes.astype(str).str.extract(r'per_rank_csr=(\d)')[0]
k = k[k.cores == 16]

print(f"{len(k)} non-warmup 16-core rows\n")

print("=== job 73920, per-rank arm (the paper's source) ===")
a = k[(k.job_id == 73920) & (k.arm == '1')]
g = (a.groupby(['n_ranks', 'n_threads'])['mrec_ghz']
       .agg(median='median', min='min', max='max', n='size')
       .reset_index().sort_values('n_ranks'))
print(g.round(1).to_string(index=False))
if len(g):
    b = g.loc[g['median'].idxmax()]
    sec = g.nlargest(2, 'median').iloc[-1]
    print(f"\n  best {int(b.n_ranks)}x{int(b.n_threads)} = {b['median']:.1f}"
          f"   second {int(sec.n_ranks)}x{int(sec.n_threads)} = {sec['median']:.1f}"
          f"   margin {100*(b['median']-sec['median'])/sec['median']:.2f}%")
    print(f"  spread across all configs: "
          f"{100*(g['median'].max()-g['median'].min())/g['median'].min():.1f}%")

print("\n=== job 73920, shared arm ===")
s0 = k[(k.job_id == 73920) & (k.arm == '0')]
print(s0.groupby(['n_ranks', 'n_threads'])['mrec_ghz']
        .agg(median='median', n='size').round(1).to_string())

print("\n=== all three jobs, per-rank arm, winner per job ===")
for j, gj in k[k.arm == '1'].groupby('job_id'):
    m = gj.groupby(['n_ranks', 'n_threads'])['mrec_ghz'].median().sort_values(ascending=False)
    if len(m) < 2:
        print(f"  job {int(j)}: only {len(m)} config(s)"); continue
    print(f"  job {int(j)}: best {int(m.index[0][0])}x{int(m.index[0][1])} "
          f"= {m.iloc[0]:.1f}, second {int(m.index[1][0])}x{int(m.index[1][1])} "
          f"= {m.iloc[1]:.1f}, margin {100*(m.iloc[0]-m.iloc[1])/m.iloc[1]:.2f}%")

print("\n=== ceiling check: single-socket 8-thread ===")
ref = d[(d.timed_region == 'kernel') & (d.warmup == 0) &
        (d.n_threads == 8) & (d.n_ranks == 1) &
        (d.thread_placement == 'socket0')].copy()
ref['mrec_ghz'] = ref.throughput_rec_s / 1e6 / ref.achieved_ghz
print(ref.groupby(['job_id', 'csr_variant'])['mrec_ghz']
        .agg(median='median', n='size').round(1).to_string())
print("\n  paper ceiling 871.2 implies single-socket 435.6")

print("\n=== LaTeX rows (per-rank arm, job 73920) ===")
CEIL = 871.2
for _, r in g.iterrows():
    print(f"${int(r.n_ranks)} \\times {int(r.n_threads)}$ & {r['median']:.1f} & "
          f"{100*r['median']/CEIL:.1f} \\\\   % n={int(r.n)}")
