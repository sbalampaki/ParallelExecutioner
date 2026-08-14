#!/usr/bin/env python3

import argparse
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timings", required=True)
    ap.add_argument("--threads", required=True)
    ap.add_argument("--job", type=int, default=None)
    a = ap.parse_args()

    t = pd.read_csv(a.timings)
    th = pd.read_csv(a.threads)
    if a.job is not None:
        t = t[t.job_id == a.job]
    t = t[(t.paradigm == "openmp") & (t.warmup == 0)]
    if t.empty:
        sys.exit("no openmp non-warmup rows -- check --job and --timings")

    # ---- 1. the scalar that needs no join 
    print("=== time_imbalance / work_imbalance by cell "
          "(contamination, D32 floor 1.03-1.05) ===")
    cells = (t.groupby(["n_threads", "timestamp"], as_index=False)
               .agg(work=("work_imbalance", "median"),
                    time=("time_imbalance", "median"),
                    mrec=("throughput_rec_s", "median"),
                    ghz=("achieved_ghz", "median"))
               .sort_values(["n_threads", "timestamp"]))
    cells["contam"] = cells.time / cells.work
    cells["mrec_ghz"] = cells.mrec / 1e6 / cells.ghz
    print(cells.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  Four Block A cells ran in this order: T16 intl, T16 algn,")
    print("  T8 intl, T8 algn. If the T8 algn row's time_imbalance exceeds")
    print("  the T8 intl row's, four threads were slowed by remote reads.\n")

    # ---- 2. the per-thread profile 
    j = th.merge(t[["run_id", "n_threads", "timestamp", "repetition"]],
                 on="run_id", how="inner")
    if j.empty:
        sys.exit("threads.csv did not join to timings.csv on run_id -- "
                 "check that run_id includes the pid (addendum 1 s10)")

    print("=== per-thread rate, by cell; tid<T/2 predicted LOCAL in the "
          "aligned tree ===")
    for (nt, ts), g in j.groupby(["n_threads", "timestamp"]):
        g = g.copy()
        g["rate"] = g.n_records / g.secs / 1e6           # M rec/s per thread
        prof = (g.groupby("tid")
                  .agg(cpu=("cpu", "first"), socket=("socket", "first"),
                       nrec=("n_records", "median"), rate=("rate", "median"))
                  .reset_index())
        half = nt // 2
        lo = prof[prof.tid < half].rate.median()
        hi = prof[prof.tid >= half].rate.median()
        print(f"\n  T={nt}  {ts}")
        print(prof.to_string(index=False,
                             float_format=lambda v: f"{v:.3f}"))
        if hi and lo:
            print(f"    median rate  tid<{half}: {lo:.3f}   "
                  f"tid>={half}: {hi:.3f}   ratio {lo / hi:.4f}")
            print(f"    => if this cell is the ALIGNED tree, the remote "
                  f"premium is {100 * (lo / hi - 1):+.2f}%")

    print("""
Reading it:

  * A ratio of ~1.00 in BOTH T=8 cells means remote streaming access is
    free for this kernel -- the record arrays are read sequentially and
    the prefetchers hide the extra latency. That confirms d7a's excluded
    row 1 ("remote reads 1%") by direct treatment rather than by a
    streaming-bandwidth proxy, and it means Block A's null is a real null.

  * A ratio well above 1.00 in exactly one T=8 cell, with a clean break at
    tid 4, is the remote premium, measured within one run.

  * If `secs` includes end-of-rep barrier wait, fast threads absorb the
    difference and every profile looks flat regardless. Check that against
    time_imbalance in part 1: a flat profile WITH time_imbalance at the
    1.03-1.05 floor is a real null; a flat profile with elevated
    time_imbalance means `secs` is barrier-contaminated and this test is
    inconclusive.
""")


if __name__ == "__main__":
    main()
