#!/usr/bin/env python3

import argparse
import sys

import pandas as pd

# (n_ranks, per_rank_csr, tree label) in launch order, from e4_blockb.sbatch
EXPECTED = [
    (1, 0, "shared_grp"), (1, 0, "shared_alt"),                  # omp anchors
    (2, 0, "shared_grp"), (2, 0, "shared_alt"), (2, 1, "per_rank"),
    (2, 1, "per_rank"),   (2, 0, "shared_alt"), (2, 0, "shared_grp"),
    (1, 0, "shared_grp"), (1, 0, "shared_alt"), (1, 1, "per_rank"),
    (2, 0, "shared_grp"), (2, 0, "shared_alt"), (2, 1, "per_rank"),
    (4, 0, "shared_grp"), (4, 0, "shared_alt"), (4, 1, "per_rank"),
    (8, 0, "shared_grp"), (8, 0, "shared_alt"), (8, 1, "per_rank"),
    (16, 0, "shared_grp"), (16, 0, "shared_alt"), (16, 1, "per_rank"),
]
ANCHOR = {"shared_grp": 688.6, "shared_alt": 741.2}   # job 73918, T=16 grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timings", required=True)
    a = ap.parse_args()

    t = pd.read_csv(a.timings)
    t = t[t.timed_region == "kernel"]          # MPI writes end_to_end too
    t["pid"] = t.run_id.str.split("-").str[2].astype(int)
    t["per_rank"] = (t.notes.str.extract(r"per_rank_csr=(\d)")[0]
                      .fillna("0").astype(int))

    pids = sorted(t.pid.unique())
    print(f"runs found: {len(pids)}   expected: {len(EXPECTED)}")
    if len(pids) != len(EXPECTED):
        print("  MISMATCH -- a run failed, or the sbatch was edited.")
        print("  Observed (n_ranks, per_rank, paradigm) in launch order:")
        for p in pids:
            r = t[t.pid == p].iloc[0]
            print(f"    {r.n_ranks:>2}x{r.n_threads:<2} per_rank={r.per_rank} "
                  f"{r.paradigm}")
        sys.exit(1)

    #  validate the guess against the two real columns 
    bad = []
    for i, p in enumerate(pids):
        r = t[t.pid == p].iloc[0]
        want_nr, want_pr, _ = EXPECTED[i]
        if int(r.n_ranks) != want_nr or int(r.per_rank) != want_pr:
            bad.append((i, int(r.n_ranks), int(r.per_rank), want_nr, want_pr))
    if bad:
        print("  LAUNCH ORDER DOES NOT MATCH THE SCRIPT. Not labelling.")
        for i, gr, gp, wr, wp in bad:
            print(f"    position {i}: got n_ranks={gr} per_rank={gp}, "
                  f"expected {wr}/{wp}")
        sys.exit(1)
    print("  n_ranks and per_rank_csr agree with launch order at every "
          "position -- tree labels are safe to use\n")

    lab = {p: EXPECTED[i] for i, p in enumerate(pids)}
    t["tree"] = t.pid.map(lambda p: lab[p][2])
    t["pos"] = t.pid.map(lambda p: pids.index(p))
    t = t[t.warmup == 0]
    t["mrec_ghz"] = t.throughput_rec_s / 1e6 / t.achieved_ghz

    #  anchors 
    print("=== OpenMP anchors vs job 73918 ===")
    for p in pids[:2]:
        s = t[t.pid == p]
        tree = lab[p][2]
        v = s.mrec_ghz.median()
        print(f"  {tree:11s} {v:7.1f} M rec/GHz   73918: {ANCHOR[tree]:.1f}"
              f"   {100*(v/ANCHOR[tree]-1):+6.2f}%")
    print("  Beyond a few percent and this node is not in 73918's state.\n")

    p12a = t[(t.n_ranks == 2) & (t.pos.between(2, 7))]
    print("=== P12a: 2x8, three arms, palindrome (positions 2-7) ===")
    g = p12a.groupby("tree").mrec_ghz.median()
    h = p12a.groupby(["tree", p12a.pos < 5]).mrec_ghz.median().unstack()
    h.columns = ["half2", "half1"]
    h["drift_%"] = 100 * (h.half2 / h.half1 - 1)
    print(h[["half1", "half2", "drift_%"]].to_string(
        float_format=lambda v: f"{v:.2f}"))
    for base in ("shared_grp", "shared_alt"):
        if base in g and "per_rank" in g:
            d = 100 * (g["per_rank"] / g[base] - 1)
            verdict = ("as predicted" if (base == "shared_grp" and 6 <= d <= 9)
                       or (base == "shared_alt" and d <= 2) else "OFF PREDICTION")
            print(f"  per_rank vs {base}: {d:+6.2f}%   "
                  f"(predicted {'6-9%' if base=='shared_grp' else '<=2%'}) "
                  f"-> {verdict}")
    print("  job 73037 reported +15.2% for this comparison.\n")

    print("=== P12b: ratio sweep by tree (positions 8-22) ===")
    sweep = t[t.pos >= 8]
    piv = (sweep.groupby(["n_ranks", "tree"]).mrec_ghz.median()
                .unstack().reindex([1, 2, 4, 8, 16]))
    piv.index = [f"{n}x{16//n}" for n in piv.index]
    print(piv.to_string(float_format=lambda v: f"{v:.1f}"))
    print()
    for tree in piv.columns:
        c = piv[tree].dropna()
        if len(c) < 2:
            continue
        spread = 100 * (c.max() / c.min() - 1)
        print(f"  {tree:11s} spans {spread:5.2f}%   best {c.idxmax()}   "
              f"worst {c.idxmin()}")
    print("\n  73063 measured +10.6% across this sweep, on shared_grp.")
    print("  P12b predicts shared_grp ~10% and shared_alt <5%.")
    print("  If shared_alt is flat, 'more ranks is better' was staging")
    print("  period, and 16x1 -- which has no OpenMP barrier at all -- is")
    print("  the end point of that argument rather than a process-model win.")


if __name__ == "__main__":
    main()
