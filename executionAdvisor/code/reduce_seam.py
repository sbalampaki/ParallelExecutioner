#!/usr/bin/env python3

import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--timings", required=True)
ap.add_argument("--threads", required=True)
a = ap.parse_args()

t = pd.read_csv(a.timings); t = t[t.warmup == 0]
t["seams"] = t.notes.str.extract(r"seams=(\d+)")[0].astype(float)
if t.seams.isna().all():
    raise SystemExit("no seams= in notes -- was the job run with --note?")
t["seams"] = t.seams.astype(int)
t["mrec_ghz"] = t.throughput_rec_s / 1e6 / t.achieved_ghz

print("=== P22: throughput vs seam count ===")
g = (t.groupby("seams")
       .agg(mrec_ghz=("mrec_ghz", "median"), ghz=("achieved_ghz", "median"),
            work=("work_imbalance", "median"), time=("time_imbalance", "median"))
       .sort_index())
best = g.mrec_ghz.max()
g["vs_best_%"] = 100 * (g.mrec_ghz / best - 1)
print(g.to_string(float_format=lambda v: f"{v:.4f}"))
if 1 in g.index and 15 in g.index:
    d = 100 * (g.loc[1, "mrec_ghz"] / g.loc[15, "mrec_ghz"] - 1)
    print(f"\n  1 seam vs 15 seams: {d:+.1f}%   (D44 on Broadwell: +36.5%)")
    print("  monotone in between -> D44 is D47 counted more times")

print("\n=== P23: per-thread deficits at each seam count ===")
th = pd.read_csv(a.threads)
j = th.merge(t[["run_id", "seams"]], on="run_id", how="inner")
if j.empty:
    raise SystemExit("threads.csv did not join on run_id")
j["rate"] = j.n_records / j.secs
print(f"{'seams':>6} {'slow threads':>13} {'median deficit':>15} {'worst':>8}")
for s, gg in j.groupby("seams"):
    prof = gg.groupby("tid").rate.median()
    rel = 100 * (prof / prof.median() - 1)
    slow = rel[rel < -4]
    print(f"{s:>6} {len(slow):>13} "
          f"{(slow.median() if len(slow) else float('nan')):>14.1f}% "
          f"{rel.min():>7.1f}%")
    print(f"       tids below -4%: {sorted(slow.index.tolist())}")
print(f"\n  D47 predicts 2 x seams slow threads, capped at 16:"
      f" {[min(2*s,16) for s in sorted(j.seams.unique())]}")
print("  CONSTANT deficits across the sweep -> a max-of-threads model")
print("  cannot produce D44's 36.5%, so the mechanism is aggregate.")
print("  GROWING deficits -> seams interact and the cost is superlinear.")
