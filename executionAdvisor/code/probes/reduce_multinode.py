#!/usr/bin/env python3

import argparse, re
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--timings", required=True)
a = ap.parse_args()

t = pd.read_csv(a.timings)
t = t[t.warmup == 0]
for k in ("exp", "variant", "nodes"):
    t[k] = t.notes.str.extract(rf"{k}=([^;]+)")[0]
if t.exp.isna().all():
    raise SystemExit("no exp= in notes -- was the job run with --note?")
t["nodes"] = t.nodes.astype(int)
t["mrec"] = t.throughput_rec_s / 1e6

for exp in ("strong", "weak"):
    s = t[t.exp == exp]
    if s.empty:
        continue
    print(f"\n=== {exp.upper()} SCALING ===")
    for reg in ("kernel", "end_to_end"):
        r = s[s.timed_region == reg]
        if r.empty:
            continue
        g = (r.groupby(["nodes", "variant"])
               .agg(mrec=("mrec", "median"), wall=("wall_time_s", "median"),
                    ghz=("achieved_ghz", "median"),
                    timb=("time_imbalance", "median"),
                    wimb=("work_imbalance", "median")).reset_index())
        g = g.sort_values("nodes")
        base = g.iloc[0]
        p = g.nodes / base.nodes
        g["speedup"] = g.mrec / base.mrec
        g["efficiency_%"] = 100 * g.speedup / p
        # Karp-Flatt: only meaningful where p > 1
        g["karp_flatt_e"] = [(1 / S - 1 / pp) / (1 - 1 / pp) if pp > 1 else float("nan")
                             for S, pp in zip(g.speedup, p)]
        print(f"\n  --- timed_region = {reg} ---")
        print(g.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        if reg == "end_to_end" and g.karp_flatt_e.notna().any():
            e = g.karp_flatt_e.dropna()
            trend = "RISING" if e.iloc[-1] > e.iloc[0] else "flat or falling"
            print(f"\n    Karp-Flatt e: {trend} across the sweep "
                  f"({e.iloc[0]:.4f} -> {e.iloc[-1]:.4f})")
            print("    Rising e means the serial fraction grows with p, which")
            print("    for this kernel is the 10 GbE gather, not the kernel.")

print("\n=== D47 check: the seam should be present and CONSTANT ===")
.
k = t[(t.timed_region == "kernel") & (t.exp == "strong")]
print(k.groupby("nodes")[["work_imbalance", "time_imbalance"]]
       .median().to_string(float_format=lambda v: f"{v:.4f}"))

EXPECTED_IMB = {1: 1.049428, 2: 1.073144, 4: 1.108724, 5: 1.129385}
_med = k.groupby("nodes").work_imbalance.median()
for _n, _want in EXPECTED_IMB.items():
    if _n in _med.index:
        _got = float(_med.loc[_n])
        assert abs(_got - _want) < 1e-5, (
            f"nodes={_n}: work_imbalance {_got:.6f}, expected {_want:.6f}. "
            "If the exp=='strong' filter regressed, nodes=1 reads ~1.0292 "
            "(weak sweep W24/W48 pooled in).")
print("\n  contamination = time_imbalance / work_imbalance (schema.md S2):")
_c = (_med2.time_imbalance / _med2.work_imbalance)   # your median frame's name
for _n, _v in _c.items():
    print(f"    nodes={_n}: {_v:.4f}")
print("  N=1 at ~1.000 means thread time tracks the combinatorial partition")
print("  exactly: 2 ranks x 8 threads puts each rank inside one socket, so")
print("  no partition cut crosses the socket boundary and D47's seam is absent.")
print("  N=4 below 1.000 means the slowest thread is not the fullest one --")
print("  that is network/MPI variance, not imbalance.")
