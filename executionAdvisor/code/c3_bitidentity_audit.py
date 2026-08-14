#!/usr/bin/env python3

import sys, os, glob
import pandas as pd

pd.set_option('display.width', 220)
pd.set_option('display.max_rows', 200)

TREES = sys.argv[1:] or [
    '/projects/sb2ea/results/timing',
    '/home/sb2ea/loadimbalance/results/timing',
]

frames = []
for t in TREES:
    for f in sorted(glob.glob(os.path.join(t, '*.csv'))):
        try:
            d = pd.read_csv(f)
            if 'bitidentical' not in d.columns:
                continue
            d['__file'] = os.path.basename(f)
            frames.append(d)
        except Exception:
            pass
if not frames:
    sys.exit(f"no CSVs with a bitidentical column under {TREES}")

df = pd.concat(frames, ignore_index=True, sort=False)
print(f"{len(df):,} rows from {len(frames)} files\n")

#  1. values
print("=" * 72)
print("1.  What values does bitidentical actually take?")
print("=" * 72)
vc = df.bitidentical.astype(str).value_counts(dropna=False)
print(vc.to_string())
print("\nA sentinel ('skip', -1, NaN) means no verdict was recorded --")
print("either the region is exempt by design or the reference was absent.")

#  2. by region
print("\n" + "=" * 72)
print("2.  Coverage by timed_region and paradigm")
print("=" * 72)
d = df.copy()
d['verdict'] = d.bitidentical.astype(str)
idx = [c for c in ('paradigm', 'timed_region') if c in d.columns]
print(pd.crosstab([d[c] for c in idx], d.verdict).to_string())

#  3. MPI coverage
print("\n" + "=" * 72)
print("3.  MPI runs: does coverage depend on rank count?")
print("=" * 72)
mpi = d[d.get('paradigm', '').astype(str).str.contains('mpi', case=False, na=False)]
if mpi.empty:
    print("no rows with an mpi paradigm label")
else:
    e2e = mpi[mpi.timed_region == 'end_to_end']
    print("end_to_end rows only:\n")
    print(pd.crosstab([e2e.n_ranks, e2e.n_nodes], e2e.verdict).to_string())
    print("\nIf 'true' rows do not scale with n_ranks, one row is written per")
    print("run rather than per rank -- coverage is then a property of which")
    print("rank writes the row, not of how many were checked.")
    if 'node' in e2e.columns:
        print("\nrows per node (which hosts actually reported a verdict):")
        print(pd.crosstab(e2e.node, e2e.verdict).to_string())

#  4. failures
print("\n" + "=" * 72)
print("4.  Where are the FAILURES?  (the cyclic/greedy result)")
print("=" * 72)
fal = d[d.verdict.str.lower().isin(['false', '0', 'fail', 'mismatch'])]
print(f"{len(fal)} failing rows total")
if len(fal):
    gc = [c for c in ('job_id', 'paradigm', 'partitioner', 'timed_region',
                      'n_ranks', 'n_nodes', 'node', 'platform') if c in fal.columns]
    print(fal.groupby(gc).size().rename('rows').reset_index().to_string(index=False))
    print("\nPaper claims: cyclic and greedy fail in all 24 end_to_end rows each,")
    print("in job 74398, and pass in twelve OpenMP jobs.")
    for pname in ('cyclic', 'greedy'):
        sub = fal[fal.partitioner == pname] if 'partitioner' in fal.columns else fal.iloc[0:0]
        print(f"  {pname:<8} failing rows: {len(sub)}"
              + (f"   jobs {sorted(sub.job_id.unique())}" if len(sub) else ""))

#  5. pass side
print("\n" + "=" * 72)
print("5.  The pass side -- how many rows carry a real PASS?")
print("=" * 72)
ps = d[d.verdict.str.lower().isin(['true', '1', 'pass'])]
print(f"{len(ps):,} passing rows")
if 'partitioner' in d.columns:
    t = pd.crosstab(d.partitioner, d.verdict)
    print("\nby partitioner:\n" + t.to_string())

print("\n" + "=" * 72)
print("WHAT TO TAKE FROM THIS")
print("=" * 72)
print("A missing reference produces FALSE NEGATIVES, never false positives.")
print("Every failure above is a genuine detection and the cyclic/greedy")
print("result is unaffected. What is bounded is the PASS side: section 5's")
print("count is the number of rows actually verified, and the paper's")
print("'every parallel run' must be scoped to that.")
