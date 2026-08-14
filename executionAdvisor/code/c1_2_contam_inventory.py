#!/usr/bin/env python3

import sys, os, glob
import numpy as np
import pandas as pd

pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 300)

TREES = sys.argv[1:] or [
    '/projects/sb2ea/results/timing',
    '/home/sb2ea/loadimbalance/results/timing',
]

#  load
frames, skipped = [], []
for t in TREES:
    for f in sorted(glob.glob(os.path.join(t, '*.csv'))):
        try:
            d = pd.read_csv(f)
            if 'work_imbalance' not in d.columns:
                skipped.append((f, 'no work_imbalance')); continue
            d['__file'] = os.path.basename(f)
            d['__tree'] = t
            frames.append(d)
        except Exception as e:
            skipped.append((f, str(e)[:60]))

if not frames:
    sys.exit(f"No usable CSVs found under: {TREES}")

df = pd.concat(frames, ignore_index=True, sort=False)
print(f"loaded {len(frames)} files, {len(df):,} rows, from:")
for t in TREES:
    n = (df.__tree == t).sum()
    print(f"   {n:>7,} rows  {t}")
if skipped:
    print(f"\nskipped {len(skipped)} file(s):")
    for f, why in skipped[:10]:
        print(f"   {os.path.basename(f):<32} {why}")

k = df[df.timed_region == 'kernel'].copy()
k['contam'] = k.time_imbalance / k.work_imbalance
k['p'] = k.n_ranks * k.n_threads
# a rank straddles the socket boundary iff it holds more than 8 threads
k['seam_expected'] = k.n_threads > 8

print(f"\nkernel rows: {len(k):,}   job_ids: {sorted(k.job_id.unique())}")


#  A  head-to-head
def cell(job):
    return k[(k.job_id == job) & (k.partitioner == 'block') &
             (k.n_nodes == 5) & (k.p == 80)].sort_values('repetition')


print("\n" + "=" * 74)
print("A.  74347 vs 74398 — block, N=5, p=80, per repetition")
print("=" * 74)

cols = ['repetition', 'warmup', 'achieved_ghz', 'work_imbalance',
        'time_imbalance', 'contam', 'csr_variant', 'thread_placement',
        'schedule', 'n_ranks', 'n_threads']
for job in (74347, 74398):
    c = cell(job)
    print(f"\n--- job {job} ---   n={len(c)}")
    if c.empty:
        print("   NO ROWS. present for this job:")
        j = k[k.job_id == job]
        if j.empty:
            print("     job absent from the loaded trees entirely")
        else:
            print(j.groupby(['partitioner', 'n_nodes', 'n_ranks', 'n_threads'])
                   .size().to_string())
        continue
    print(c[[x for x in cols if x in c.columns]].to_string(index=False))
    if len(c) >= 3:
        rr = c.contam.corr(c.repetition)
        rg = c.contam.corr(c.achieved_ghz)
        print(f"   contam vs repetition r={rr:+.4f}   "
              f"contam vs ghz r={rg:+.4f}")
        print(f"   median all={c.contam.median():.4f}   "
              f"first={c.contam.iloc[0]:.4f}   last={c.contam.iloc[-1]:.4f}   "
              f"post-rep>=3={c[c.repetition >= 3].contam.median():.4f}")

a, b = cell(74347), cell(74398)
if not a.empty and not b.empty:
    print("\n--- verdict on the two shapes ---")
    for nm, c in (('74347', a), ('74398', b)):
        d = c.contam.iloc[0] - c.contam.iloc[-1]
        print(f"  {nm}: first-to-last change {d:+.4f}  "
              f"range {c.contam.min():.4f}-{c.contam.max():.4f}")
    print("  If BOTH decline and 74398 simply starts lower -> a settling")
    print("  effect with different initial conditions.")
    print("  If 74398 is FLAT near 1.00 -> the two jobs genuinely differ")
    print("  and no within-job correction reconciles them.")


print("\n" + "=" * 74)
print("B.  warmup column")
print("=" * 74)
if 'warmup' in k.columns:
    print(k.groupby(['warmup']).agg(
        rows=('contam', 'size'),
        reps=('repetition', lambda s: sorted(s.unique())[:8]),
    ).to_string())
    print("\nIf warmup is True only at repetition 0, the standing policy")
    print("probably already excludes it — state which policy the paper uses.")
else:
    print("no warmup column")

#  C  inventory
print("\n" + "=" * 74)
print("C.  per-cell inventory  (written to contam_inventory.csv)")
print("=" * 74)
gcols = [c for c in ['job_id', 'platform', 'csr_variant', 'partitioner',
                     'schedule', 'thread_placement', 'n_nodes', 'n_ranks',
                     'n_threads', 'p', 'seam_expected'] if c in k.columns]
inv = (k.groupby(gcols)['contam']
         .agg(median='median', min='min', max='max', n='size')
         .reset_index())
inv.to_csv('contam_inventory.csv', index=False)
print(f"{len(inv)} cells\n")
print(inv.sort_values('median').to_string(index=False))

#  D  the three questions
print("\n" + "=" * 74)
print("D.  the three C1.2 questions")
print("=" * 74)

print("\nQ1 — which cells give 1.06 to 1.16?  (VII-D's second range)")
band = inv[inv['median'].between(1.06, 1.16)]
if band.empty:
    print("   NONE. VII-D's '1.06 to 1.16 in every single-node cell'")
    print("   is not supported by any cell in these trees.")
else:
    print(band.to_string(index=False))
    for c in ['thread_placement', 'schedule', 'n_threads', 'csr_variant', 'job_id']:
        if c in band.columns:
            print(f"   {c}: {sorted(band[c].astype(str).unique())}")

print("\nQ2 — do ALL seam-expected cells (n_threads>8) fall in that band?")
se = inv[inv.seam_expected]
if se.empty:
    print("   no seam-expected cells found")
else:
    inb = se['median'].between(1.06, 1.16)
    print(f"   {inb.sum()} of {len(se)} in band; "
          f"median {se['median'].median():.4f}, "
          f"range {se['median'].min():.4f}-{se['median'].max():.4f}")
    if (~inb).any():
        print("   OUT OF BAND:")
        print(se[~inb].sort_values('median').to_string(index=False))

print("\nQ3 — distribution of rank-inside-socket cells (n_threads<=8)")
ns = inv[~inv.seam_expected]
if ns.empty:
    print("   none found")
else:
    print(f"   n={len(ns)}  median of medians {ns['median'].median():.4f}  "
          f"range {ns['median'].min():.4f}-{ns['median'].max():.4f}")
    print(f"   cells within 0.02 of 1.00: "
          f"{ns['median'].between(0.98, 1.02).sum()} of {len(ns)}")
    print(ns.sort_values('median').to_string(index=False))

print("\n" + "=" * 74)
print("For VII-D you need two numbers: the median across seam-expected")
print("cells, and the median across rank-inside-socket cells. The argument")
print("is the CONTRAST between those populations, not any single cell.")
print("=" * 74)
