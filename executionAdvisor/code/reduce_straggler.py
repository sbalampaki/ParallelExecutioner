#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timings", required=True)
    ap.add_argument("--threads", required=True)
    a = ap.parse_args()

    t = pd.read_csv(a.timings)
    th = pd.read_csv(a.threads)
    t = t[t.warmup == 0]
    j = th.merge(t[["run_id", "n_threads", "schedule", "time_imbalance",
                    "work_imbalance", "throughput_rec_s"]],
                 on="run_id", how="inner")
    if j.empty:
        raise SystemExit("threads.csv did not join to timings.csv on run_id")
    j["rate"] = j.n_records / j.secs / 1e6           # M rec/s per thread

    p = j[j.schedule == "precomputed"]
    print("=== 1. PRECOMPUTED: per-thread rate by socket ===")
    print("   equal records per thread, so rate IS speed\n")
    print(f"{'T':>3} {'sock0 rate':>11} {'sock1 rate':>11} {'s1/s0':>7} "
          f"{'max/min tid':>12} {'time_imb':>9}")
    for T, g in p.groupby("n_threads"):
        s0 = g[g.socket == 0].rate.median()
        s1 = g[g.socket == 1].rate.median() if (g.socket == 1).any() else np.nan
        prof = g.groupby("tid").rate.median()
        print(f"{T:>3} {s0:11.3f} {s1:11.3f} "
              f"{(s1/s0 if s1 == s1 else np.nan):7.4f} "
              f"{prof.max()/prof.min():12.4f} {g.time_imbalance.median():9.4f}")

    print("\n=== 2. T=16 profile: precomputed rate vs dynamic record share ===")
    print("   under dynamic, records-per-thread is proportional to SPEED,")
    print("   measured without reference to any partition\n")
    pre = (p[p.n_threads == 16].groupby(["tid", "cpu", "socket"])
             .rate.median().reset_index())
    d = j[(j.schedule == "dynamic") & (j.n_threads == 16)]
    if not d.empty:
        dyn = (d.groupby("tid").n_records.median() /
               d.groupby("tid").n_records.median().mean())
        pre["dyn_share"] = pre.tid.map(dyn)
        pre["pre_norm"] = pre.rate / pre.rate.mean()
        print(pre.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        ok = pre.dyn_share.notna()
        if ok.sum() > 2:
            r = np.corrcoef(pre.pre_norm[ok], pre.dyn_share[ok])[0, 1]
            print(f"\n  corr(precomputed rate, dynamic record share) = {r:+.3f}")
            print("  Near +1 means the two independent views agree and the")
            print("  per-thread speed profile is real, not an artefact of")
            print("  either the partition or the schedule.")
    else:
        print(pre.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== 3. socket effect vs core effect, T=16 precomputed ===")
    g = p[p.n_threads == 16]
    s0 = g[g.socket == 0].groupby("tid").rate.median()
    s1 = g[g.socket == 1].groupby("tid").rate.median()
    print(f"  socket 0 threads (tid 0-7):  mean {s0.mean():.3f}  "
          f"spread {100*(s0.max()/s0.min()-1):.1f}%")
    print(f"  socket 1 threads (tid 8-15): mean {s1.mean():.3f}  "
          f"spread {100*(s1.max()/s1.min()-1):.1f}%")
    print(f"  between-socket gap: {100*(s1.mean()/s0.mean()-1):+.1f}%")
    print("  A large between-socket gap with small within-socket spread")
    print("  means the imbalance is a socket property. Large within-socket")
    print("  spread means it is per-core and the socket label is incidental.")

    print("\n=== 4. is the slowest thread the same one every time? ===")
    runs = g.groupby("run_id")
    slowest = runs.apply(lambda x: x.loc[x.rate.idxmin(), "tid"],
                         include_groups=False)
    print(f"  slowest tid across {len(slowest)} runs: "
          f"{slowest.value_counts().to_dict()}")
    print("  One tid dominating means a fixed straggler -- a core or a")
    print("  partition position. A scatter means it moves run to run, which")
    print("  a precomputed partition can never chase and dynamic always can.")

    if not d.empty:
        print("\n=== 5. what dynamic trades ===")
        for sch, s in (("precomputed", g), ("dynamic", d)):
            secs = s.groupby("tid").secs.median()
            recs = s.groupby("tid").n_records.median()
            print(f"  {sch:12s} secs  max/mean {secs.max()/secs.mean():.4f}"
                  f"   records max/mean {recs.max()/recs.mean():.4f}")
        print("  Dynamic unbalances records to balance time. The size of the")
        print("  record imbalance it chooses is a direct estimate of how")
        print("  unequal the threads actually are.")


if __name__ == "__main__":
    main()
