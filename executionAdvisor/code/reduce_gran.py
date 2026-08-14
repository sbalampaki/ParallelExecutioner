#!/usr/bin/env python3

import argparse
import sys

import pandas as pd

W24F = [2, 3, 4, 5, 6, 8, 10, 12, 16, 32]
SEQ = ([("W24", b, "precomputed") for b in W24F] +
       [("W24", b, "precomputed") for b in reversed(W24F)] +
       [("W24", 16, "precomputed"), ("W24", 16, "dynamic"),
        ("W24", 4, "precomputed"), ("W24", 4, "dynamic")] +
       [("Wfull", b, "precomputed") for b in [2, 4, 5, 8, 16, 32]] +
       [("Wfull", b, "precomputed") for b in [32, 16, 8, 5, 4, 2]])

PLF = {("W24", 2): .5086, ("W24", 3): .5074, ("W24", 4): .5140,
       ("W24", 5): .6365, ("W24", 6): .5005, ("W24", 8): .5255,
       ("W24", 10): .5041, ("W24", 12): .5005, ("W24", 16): .5005,
       ("W24", 32): .5005,
       ("Wfull", 2): .5012, ("Wfull", 4): .5045, ("Wfull", 5): .4709,
       ("Wfull", 8): .2515, ("Wfull", 16): .4732, ("Wfull", 32): .5359}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timings", required=True)
    a = ap.parse_args()

    t = pd.read_csv(a.timings)
    t["pid"] = t.run_id.str.split("-").str[2].astype(int)
    pids = sorted(t.pid.unique())
    print(f"runs found {len(pids)}, expected {len(SEQ)}")
    if len(pids) != len(SEQ):
        for p in pids:
            r = t[t.pid == p].iloc[0]
            print(f"    {r.csr_variant:6s} {r.schedule}")
        sys.exit("MISMATCH -- a run failed or the sbatch was edited")

    bad = []
    for i, p in enumerate(pids):
        r = t[t.pid == p].iloc[0]
        if r.csr_variant != SEQ[i][0] or r.schedule != SEQ[i][2]:
            bad.append((i, r.csr_variant, r.schedule, SEQ[i][0], SEQ[i][2]))
    if bad:
        print("LAUNCH ORDER DOES NOT MATCH. Not labelling.")
        for i, gv, gs, wv, ws in bad:
            print(f"    pos {i}: got {gv}/{gs}, expected {wv}/{ws}")
        sys.exit(1)
    print("  csr_variant and schedule agree with launch order everywhere\n")

    t["blk"] = t.pid.map(lambda p: SEQ[pids.index(p)][1])
    t["pos"] = t.pid.map(pids.index)
    t = t[t.warmup == 0]
    t["mrec"] = t.throughput_rec_s / 1e6
    t["mrec_ghz"] = t.mrec / t.achieved_ghz
   
    t["mean_thread"] = t.mrec * t.time_imbalance

    for V in ("W24", "Wfull"):
        s = t[(t.csr_variant == V) & (t.schedule == "precomputed") & (t.pos < 24)]
        if s.empty:
            continue
        g = (s.groupby("blk")
               .agg(mrec=("mrec", "median"),
                    work=("work_imbalance", "median"),
                    time=("time_imbalance", "median"),
                    mean_thread=("mean_thread", "median"),
                    ghz=("achieved_ghz", "median")))
        g["local"] = [PLF.get((V, b), float("nan")) for b in g.index]
        base = g.mean_thread.max()
        g["mean_vs_best_%"] = 100 * (g.mean_thread / base - 1)
        g["stragg_%"] = 100 * (1 / g.time - 1)
        print(f"=== {V} ladder ===")
        print(g.to_string(float_format=lambda v: f"{v:.4f}"))
        print()

    #  decompose the step 
    s = t[(t.csr_variant == "W24") & (t.schedule == "precomputed") & (t.pos < 24)]
    fast = s[s.blk <= 5]; slow = s[s.blk >= 6]
    if len(fast) and len(slow):
        f_m, s_m = fast.mrec.median(), slow.mrec.median()
        f_t, s_t = fast.mean_thread.median(), slow.mean_thread.median()
        f_i, s_i = fast.time_imbalance.median(), slow.time_imbalance.median()
        tot = 100 * (s_m / f_m - 1)
        mean = 100 * (s_t / f_t - 1)
        print("=== W24: decomposing the step between 5 and 6 MiB ===")
        print(f"  throughput          {f_m:8.1f} -> {s_m:8.1f}   {tot:+6.2f}%")
        print(f"  mean thread speed   {f_t:8.1f} -> {s_t:8.1f}   {mean:+6.2f}%")
        print(f"  time_imbalance      {f_i:8.4f} -> {s_i:8.4f}")
        print(f"  => straggling accounts for {tot-mean:+.2f} pp of {tot:.2f}%,")
        print(f"     mean thread slowness for {mean:+.2f} pp")
        print("  Broadwell's 32 MiB tree split 9.48% as 6.14 mean + 3.4 straggle.")

    p15 = t[t.pos.between(20, 23)]
    if len(p15):
        print("\n=== P15: dynamic vs precomputed ===")
        for b in (16, 4):
            c = p15[p15.blk == b]
            pre = c[c.schedule == "precomputed"].mrec.median()
            dyn = c[c.schedule == "dynamic"].mrec.median()
            ti_p = c[c.schedule == "precomputed"].time_imbalance.median()
            ti_d = c[c.schedule == "dynamic"].time_imbalance.median()
            print(f"  block {b:2d} MiB: precomputed {pre:7.1f} (time_imb {ti_p:.4f})"
                  f"   dynamic {dyn:7.1f} (time_imb {ti_d:.4f})"
                  f"   {100*(dyn/pre-1):+5.1f}%")


if __name__ == "__main__":
    main()
