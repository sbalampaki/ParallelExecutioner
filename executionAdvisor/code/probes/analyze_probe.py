#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd

RUN_COLS = ["kind", "tag", "threads", "touch", "write_out", "rep", "warmup",
            "wall_s", "records", "records_per_s", "gb_s", "work_imb", "time_imb"]
THR_COLS = ["kind", "tag", "threads", "rep", "tid", "cpu", "socket",
            "stays", "records", "secs", "gb_s"]
NUMA_COLS = ["kind", "path", "file", "node", "pages", "total_pages", "frac", "mib"]


def read_probe(path):
  
    runs, thrs = [], []
    with open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) == len(RUN_COLS) and p[0] == "RUN" and p[1] != "tag":
                runs.append(p)
            elif len(p) == len(THR_COLS) and p[0] == "THR" and p[1] != "tag":
                thrs.append(p)

    R = pd.DataFrame(runs, columns=RUN_COLS)
    T = pd.DataFrame(thrs, columns=THR_COLS)
    for c in ["threads", "write_out", "rep", "warmup"]:
        R[c] = R[c].astype(int)
    for c in ["wall_s", "records", "records_per_s", "gb_s", "work_imb", "time_imb"]:
        R[c] = R[c].astype(float)
    for c in ["threads", "rep", "tid", "cpu", "socket", "stays", "records"]:
        T[c] = T[c].astype(int)
    for c in ["secs", "gb_s"]:
        T[c] = T[c].astype(float)

    return R[R.warmup == 0], T[T.rep > 0]


def read_numa(path):
   
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) == len(NUMA_COLS) and p[0] == "NUMA" and p[1] != "path":
                rows.append(p)
    N = pd.DataFrame(rows, columns=NUMA_COLS)
    if N.empty:
        return N
    for c in ["pages", "total_pages", "frac", "mib"]:
        N[c] = pd.to_numeric(N[c], errors="coerce")
    return N.dropna(subset=["frac"])


def med_iqr(s):
    q1, q3 = np.percentile(s, [25, 75])
    return float(np.median(s)), float(q3 - q1)


def summarise(R, tags):
    rows = []
    for tag in tags:
        sub = R[R.tag == tag]
        if sub.empty:
            continue
        rps_m, rps_i = med_iqr(sub.records_per_s)
        rows.append(dict(tag=tag, threads=int(sub.threads.iloc[0]),
                         touch=sub.touch.iloc[0],
                         mrec_s=rps_m / 1e6,
                         iqr_pct=100 * rps_i / rps_m if rps_m else np.nan,
                         gb_s=med_iqr(sub.gb_s)[0],
                         work_imb=med_iqr(sub.work_imb)[0],
                         time_imb=med_iqr(sub.time_imb)[0]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_csv")
    ap.add_argument("numa_csv", nargs="?")
    ap.add_argument("--kernel-rate", type=float, default=158e6)
    args = ap.parse_args()

    R, T = read_probe(args.probe_csv)

    print("=" * 74)
    print("TASK 1a - page placement (direct, move_pages)")
    print("=" * 74)
    if args.numa_csv:
        N = read_numa(args.numa_csv)
        N = N[N.node != "unknown"]
        if N.empty:
            print("  no usable NUMA rows")
        else:
            for path, g in N.groupby("path"):
                frac = {r.node: round(r.frac, 4) for r in g.itertuples()}
                worst = max(frac.values())
                verdict = ("ONE-NODE" if worst > 0.90 else
                           "SKEWED" if worst > 0.65 else "INTERLEAVED")
                print(f"  {path:<46} {frac}  -> {verdict}")

            print()
            N = N.copy()
            N["mode"] = np.where(N.path.str.contains("interleave"),
                                 "interleave", "serial")
            print("  byte-weighted share per staging mode:")
            for mode, g in N.groupby("mode"):
                tot = g.mib.sum()
                shares = {n: round(float(gg.mib.sum()) / float(tot), 3)
                          for n, gg in g.groupby("node")}
                txt = "  ".join(f"{n}={v:.3f}" for n, v in sorted(shares.items()))
                if len(shares) > 1:
                    # A node holding fraction f of the bytes serves fraction f
                    # of the traffic, so aggregate bandwidth is capped at
                    # B_node / f rather than n_nodes * B_node.
                    eff = 100 * (1.0 / len(shares)) / max(shares.values())
                    print(f"    {mode:<12} {txt}   "
                          f"aggregate-bandwidth ceiling ~{eff:.0f}% of ideal")
                else:
                    print(f"    {mode:<12} {txt}")
    print()

    print("=" * 74)
    print("TASK 1b - 8-thread placement contrast")
    print("=" * 74)
    t1 = summarise(R, ["t1_serialstage_socket0", "t1_serialstage_spread",
                       "t1_interstage_socket0", "t1_interstage_spread"])
    if not t1.empty:
        print(t1[["tag", "threads", "mrec_s", "gb_s", "iqr_pct",
                  "work_imb", "time_imb"]].to_string(index=False,
                                                     float_format="%.3f"))
        print()

        def rate(tag):
            r = t1[t1.tag == tag]
            return float(r.mrec_s.iloc[0]) if not r.empty else float("nan")

        for mode in ["serialstage", "interstage"]:
            s0, sp = rate(f"t1_{mode}_socket0"), rate(f"t1_{mode}_spread")
            if s0 == s0 and sp == sp:
                print(f"  {mode:<12} spread vs socket0: {100*(sp-s0)/s0:+6.1f}%")

        si, ss = rate("t1_interstage_spread"), rate("t1_serialstage_spread")
        if si == si and ss == ss:
            print(f"  {'staging':<12} interleave vs serial, 8 spread threads: "
                  f"{100*(si-ss)/ss:+6.1f}%")
        print()
        print("  READ AS:")
        print("    interleave >> serial at spread binding -> placement WAS the")
        print("      binding constraint; interleaved staging is mandatory for")
        print("      every timed run from here on, and becomes a CSV column.")
        print("    no difference -> pages were not the constraint; re-examine 4.3.")
    print()

    print("=" * 74)
    print("TASK 2 - empirical roof, 12 B/record, no aggregation")
    print("=" * 74)
    roof_tags = sorted([t for t in R.tag.unique() if t.startswith("t2_roof_all_")],
                       key=lambda s: int(s.rsplit("_", 1)[1]))
    t2 = summarise(R, roof_tags)
    if not t2.empty:
        t2 = t2.sort_values("threads").reset_index(drop=True)
        base = float(t2[t2.threads == 1].mrec_s.iloc[0])
        t2["speedup"] = t2.mrec_s / base
        t2["eff_pct"] = 100 * t2.speedup / t2.threads
        print(t2[["threads", "mrec_s", "gb_s", "speedup", "eff_pct",
                  "iqr_pct", "time_imb"]].to_string(index=False,
                                                    float_format="%.2f"))
        print()

        plateau = float(t2.mrec_s.max())
        hit = t2[t2.mrec_s >= 0.95 * plateau]
        knee = int(hit.threads.iloc[0]) if not hit.empty else None
        implied = plateau * 1e6 / args.kernel_rate
        print(f"  plateau             : {plateau:.1f} M rec/s "
              f"({plateau * 12e6 / 1e9:.2f} GB/s at 12 B/record)")
        print(f"  probe knee          : {knee} threads (95% of plateau)")
        print(f"  kernel single-core  : {args.kernel_rate/1e6:.1f} M rec/s")
        print(f"  IMPLIED KERNEL KNEE : {implied:.1f} cores   <-- P6")
        print()
        print("  READ AS:")
        print("    implied knee <= 12 -> 4.3 stands; a real saturation knee")
        print("                          sits inside 16 cores.")
        print("    implied knee >  16 -> no saturation inside the node; the")
        print("                          memory-ceiling framing in 4.3 is wrong")
        print("                          and thread scaling should be near-linear.")
    print()

    print("=" * 74)
    print("TASK 2b - 4 B vs 12 B per record (traffic vs arithmetic)")
    print("=" * 74)
    item_tags = sorted([t for t in R.tag.unique() if t.startswith("t2_roof_item_")],
                       key=lambda s: int(s.rsplit("_", 1)[1]))
    t2b = summarise(R, item_tags)
    if not t2b.empty and not t2.empty:
        print(t2b[["threads", "mrec_s", "gb_s"]].to_string(index=False,
                                                           float_format="%.2f"))
        one12 = float(t2[t2.threads == 1].mrec_s.iloc[0])
        one4 = float(t2b[t2b.threads == 1].mrec_s.iloc[0])
        print()
        print(f"  single core: 4 B/rec {one4:.1f} vs 12 B/rec {one12:.1f} M rec/s "
              f"(ratio {one4/one12:.2f}x)")
        print("  A ratio near 3x means the 0%-whitelist fixture's 6.3x advantage")
        print("  in D29 sec 3 is substantially a TRAFFIC effect, not purely a")
        print("  scatter-accumulate effect, and D29 sec 3.1 needs restating.")
    print()

    print("=" * 74)
    print("TASK 2c - output-tile write included")
    print("=" * 74)
    w_tags = sorted([t for t in R.tag.unique() if t.startswith("t2_write_")],
                    key=lambda s: int(s.rsplit("_", 1)[1]))
    t2c = summarise(R, w_tags)
    if not t2c.empty:
        print(t2c[["threads", "mrec_s", "gb_s", "iqr_pct"]].to_string(
            index=False, float_format="%.2f"))
    print()

    print("=" * 74)
    print("PER-SOCKET ASYMMETRY")
    print("=" * 74)
    for tag in sorted(T.tag.unique()):
        sub = T[T.tag == tag]
        g = sub.groupby("socket").secs.median()
        if len(g) == 2 and g.iloc[0] > 0:
            skew = 100 * (g.iloc[1] - g.iloc[0]) / g.iloc[0]
            flag = "  <-- REMOTE READS" if abs(skew) > 15 else ""
            print(f"  {tag:<26} s0 {g.iloc[0]:.4f}s  s1 {g.iloc[1]:.4f}s  "
                  f"({skew:+.1f}%){flag}")
    print()
    print("  Large positive skew means socket-1 threads are systematically")
    print("  slower, i.e. reading remote memory. Under correct interleaving")
    print("  the two sockets should agree to within run-to-run noise.")


if __name__ == "__main__":
    main()
