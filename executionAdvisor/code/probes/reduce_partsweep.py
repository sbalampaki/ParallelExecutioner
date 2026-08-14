#!/usr/bin/env python3

import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--timings", required=True)
ap.add_argument("--predictions", required=True)
ap.add_argument("--variant", default="Wfull")
ap.add_argument("--scored-on", default="records")
ap.add_argument("--drop-failed", action="store_true",
                help="exclude partitioners with any bitidentical=='fail' row "
                     "instead of aborting")
a = ap.parse_args()

t = pd.read_csv(a.timings)
t = t[t.warmup == 0]                      # keep BOTH timed regions
t["partitioner"] = t.notes.str.extract(r"partitioner=([^;]+)")[0]
t["p"] = t.notes.str.extract(r";p=(\d+)")[0].astype(float)
t = t[t.partitioner.notna()]
if t.empty:
    raise SystemExit("no exp=partsweep rows -- was --note passed?")
t["p"] = t.p.astype(int)
t["mrec_ghz"] = t.throughput_rec_s / 1e6 / t.achieved_ghz


_bad = t[t.bitidentical == "fail"]
if not _bad.empty:
    _badp = sorted(_bad.partitioner.dropna().unique())
    if a.drop_failed:
        _n0 = len(t)
        t = t[~t.partitioner.isin(_badp)]
        print(f"WARNING: excluded {_n0 - len(t)} rows from partitioners "
              f"failing bit-identity: {_badp}")
    else:
        raise AssertionError(
            "correctness failure in %d rows: %s. Rerun with --drop-failed."
            % (len(_bad), _badp))

t = t[t.timed_region == "kernel"]         # now narrow, after the gate
t["mrec_ghz"] = t.throughput_rec_s / 1e6 / t.achieved_ghz

# ---- predictions 
e = pd.read_csv(a.predictions)
if "predicted_hier" in e.columns:                 # e0_validation.csv
    e = e.rename(columns={"predicted_hier": "pred_imb",
                          "partitioner": "strategy"})[["p", "strategy",
                                                       "pred_imb"]]
    _src = "e0_validation (per-partitioner cut key, scored in records)"
elif "efficiency" in e.columns:                   # cv_work_partitions_extended
    e = e[(e.variant == a.variant) & (e.scored_on == a.scored_on)]
    e = e.assign(pred_imb=1.0 / e.efficiency)[["p", "strategy", "pred_imb"]]
    _src = f"cv_work_partitions_extended (scored_on={a.scored_on})"
    print("NOTE: predictions scored on a single key. `contig_opt` and "
          "`wbalanced` cut on work (kernel_mpi.c:256) but are measured in "
          "records, so their imb_err_% is not meaningful here. Use "
          "e0_validation.csv for those two.")
else:
    raise SystemExit(f"unrecognised predictions schema: {list(e.columns)}")
print(f"predictions: {_src}")

m = (t.groupby(["p", "partitioner"])
       .agg(meas_imb=("work_imbalance", "median"),
            mrec_ghz=("mrec_ghz", "median"),
            time_imb=("time_imbalance", "median")).reset_index()
       .merge(e, left_on=["p", "partitioner"], right_on=["p", "strategy"],
              how="left"))
m["imb_err_%"] = 100 * (m.meas_imb / m.pred_imb - 1)
m["contam"] = m.time_imb / m.meas_imb

for p, g in m.groupby("p"):
    g = g.sort_values("mrec_ghz", ascending=False)
    best = g.mrec_ghz.max()
    g = g.assign(**{"vs_best_%": 100 * (g.mrec_ghz / best - 1)})
    print(f"\n=== p = {p} ===")
    print(g[["partitioner", "pred_imb", "meas_imb", "imb_err_%",
             "mrec_ghz", "vs_best_%", "time_imb", "contam"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    _e = g["imb_err_%"].dropna()
    if len(_e):
        print(f"  P24: max |imbalance error| vs E0 = {_e.abs().max():.4f}% "
              f"over {len(_e)} partitioner(s)")
    _miss = g.loc[g.pred_imb.isna(), "partitioner"].tolist()
    if _miss:
        print(f"  (no prediction on file for: {', '.join(_miss)})")
    print("  contam = time_imb / meas_imb (schema.md S2): ~1.00 means thread "
          "time tracks the")
    print("  combinatorial partition exactly, i.e. no seam and no measurable "
          "contamination.")
    try:
        b = g.loc[g.partitioner == "block", "mrec_ghz"].iloc[0]
        c = g.loc[g.partitioner == "contig_opt", "mrec_ghz"].iloc[0]
        bi = g.loc[g.partitioner == "block", "pred_imb"].iloc[0]
        ci = g.loc[g.partitioner == "contig_opt", "pred_imb"].iloc[0]
        print(f"  contig_opt vs block: measured {100*(c/b-1):+.2f}%   "
              f"predicted from imbalance {100*(bi/ci-1):+.2f}%   "
              f"realised {100*(c/b-1)/(100*(bi/ci-1)):.0%}")
    except (IndexError, ZeroDivisionError):
        pass

print("\n  P26: the contig_opt-vs-block gap must be much larger at p=80")
print("  than at p=16. If it is not, the effect is not the tail.")
