#!/usr/bin/env python3
"""
C1.1 — Does the N=5 clock ramp explain the 1.0800 contamination at p=80?

Usage:
    python3 c1_1_contam_ramp.py                  # searches for the CSV
    python3 c1_1_contam_ramp.py path/to/file.csv # or point it at the file

Prints the per-repetition table, the contamination/clock correlation, and
the all-rep vs post-ramp medians.
"""
import sys, glob, os
import pandas as pd

pd.set_option('display.width', 200)

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    pats = ['**/e4mn_74347.csv', '**/*74347*.csv']
    hits = []
    for p in pats:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            break
    if not hits:
        sys.exit("Could not find e4mn_74347.csv under the current directory.\n"
                 "Run:  find / -name '*74347*.csv' 2>/dev/null\n"
                 "then: python3 c1_1_contam_ramp.py <that path>")
    if len(hits) > 1:
        print(f"[note] {len(hits)} candidates, using the first:")
        for h in hits:
            print("   ", h)
    path = hits[0]

print(f"file: {os.path.abspath(path)}\n")
df = pd.read_csv(path)
print(f"{len(df):,} rows, {len(df.columns)} columns")
print("columns:", list(df.columns), "\n")


def col(*names):
    """Return the first column present, else None."""
    for n in names:
        if n in df.columns:
            return n
    return None


c_part   = col('partitioner', 'part')
c_region = col('timed_region', 'region')
c_nodes  = col('n_nodes', 'nodes', 'N')
c_rep    = col('rep', 'repetition', 'iter', 'iteration', 'trial')
c_ghz    = col('achieved_ghz', 'ghz', 'clock_ghz', 'freq_ghz')
c_ti     = col('time_imbalance')
c_wi     = col('work_imbalance')

missing = [n for n, c in [('partitioner', c_part), ('timed_region', c_region),
                          ('n_nodes', c_nodes), ('time_imbalance', c_ti),
                          ('work_imbalance', c_wi)] if c is None]
if missing:
    sys.exit(f"Missing expected columns: {missing}\nAvailable: {list(df.columns)}")

m = (df[c_part] == 'block') & (df[c_region] == 'kernel') & (df[c_nodes] == 5)
a = df[m].copy()

print(f"filter: {c_part}=='block' & {c_region}=='kernel' & {c_nodes}==5")
print(f"  -> {len(a)} rows\n")

if a.empty:
    print("EMPTY. Values actually present:")
    for c in (c_part, c_region, c_nodes):
        print(f"  {c}: {sorted(df[c].dropna().unique())[:20]}")
    sys.exit(1)

a['contam'] = a[c_ti] / a[c_wi]

show = [c for c in (c_rep, c_ghz, c_wi, c_ti, 'contam') if c]
extra = [c for c in ('n_ranks', 'n_threads', 'csr_variant', 'placement') if c in a.columns]
out = a[show + extra]
if c_rep:
    out = out.sort_values(c_rep)
print(out.to_string(index=False), "\n")

if c_ghz is None:
    print("!! No achieved-clock column found — cannot run the correlation.")
    print("   Columns available:", list(df.columns))
else:
    n = a[['contam', c_ghz]].dropna().shape[0]
    if n < 3:
        print(f"!! Only {n} usable rows; correlation not meaningful.")
    else:
        r = a['contam'].corr(a[c_ghz])
        print(f"contam vs {c_ghz}:  r = {r:+.4f}   (n = {n})")

print(f"\nall-rep   median contam: {a['contam'].median():.4f}")
if c_rep:
    post = a[a[c_rep] >= 3]
    if len(post):
        print(f"post-ramp median contam: {post['contam'].median():.4f}  "
              f"({len(post)} of {len(a)} reps, {c_rep} >= 3)")
    else:
        print(f"post-ramp: no rows with {c_rep} >= 3 "
              f"(values present: {sorted(a[c_rep].unique())})")
else:
    print("post-ramp: no repetition column found — cannot split.")

print("\n" + "=" * 62)
print("DECISION RULE (CRITICAL_TASKS C1.1)")
print("=" * 62)
print("  r <= -0.7 AND post-ramp median ~ 1.00")
print("      -> clock ramp explains it. Recompute contamination on")
print("         post-ramp reps; rewrite VI-F as a ramp correction.")
print("  otherwise")
print("      -> ramp does NOT explain it. Keep the VI-F disclosure,")
print("         rebuild VII-D on population medians (C1.3), and soften")
print("         REGISTER section 3's seam-absence claim.")
