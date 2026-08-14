#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", required=True,
                    help="cv_work_partitions.csv from Day 3 (E0)")
    ap.add_argument("--timings", required=True,
                    help="e4mn_*.csv from the multi-node sweep")
    ap.add_argument("--variant", default="Wfull")
    ap.add_argument("--partitioner", default="block")
    ap.add_argument("--scored-on", default="records",
                    help="long format only. schema.md defines work_imbalance "
                         "as max/mean RECORDS per thread, so 'records' is the "
                         "like-for-like comparison.")
    a = ap.parse_args()

    q = pd.read_csv(a.partitions)
    print("cv_work_partitions.csv columns:", list(q.columns))

    # E0 may have stored the prediction as imbalance, as efficiency, or as
    # max speedup. Find whichever is present rather than assuming.
    cols = {c.lower(): c for c in q.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None
    c_p   = pick("p", "n_partitions", "workers", "nworkers")
    c_var = pick("csr_variant", "variant")
    if c_p is None:
        raise SystemExit("no partition-count column found")

    q = q[q[c_var] == a.variant] if c_var else q
    if q.empty:
        raise SystemExit(f"no rows for variant={a.variant}")


    c_strat = pick("strategy", "partitioner")
    c_eff   = pick("efficiency", "eff")
    c_score = pick("scored_on")
    if c_strat and c_eff:                                   # long
        if c_score:
            q = q[q[c_score] == a.scored_on]
            if q.empty:
                raise SystemExit(f"no rows with scored_on={a.scored_on}")
        q = q[q[c_strat] == a.partitioner]
        if q.empty:
            raise SystemExit(f"no rows for strategy={a.partitioner}")
        v = q[c_eff].astype(float)
        if v.max() > 1.5:
            v = v / 100.0
        q = q.assign(pred_imb=1.0 / v)
        src = (f"long format, strategy='{a.partitioner}'"
               + (f", scored_on='{a.scored_on}'" if c_score else ""))
    elif a.partitioner in q.columns:                        # wide
        v = q[a.partitioner].astype(float)
        if v.max() > 1.5:
            v = v / 100.0
        q = q.assign(pred_imb=1.0 / v)
        src = f"wide format, column '{a.partitioner}' (efficiency -> 1/eff)"
    else:
        raise SystemExit(f"cannot find '{a.partitioner}'; columns are "
                         f"{list(q.columns)}")
    print(f"prediction taken from: {src}")
    extra = [c for c in ("scored_on", "rho") if c in q.columns]
    if extra:
        print("  E0 context: " + "  ".join(
            f"{c}={q[c].iloc[0]}" for c in extra))
    print()

    pred = q.groupby(c_p).pred_imb.median().sort_index()

    # Measured: 2 ranks x 8 threads per node, so p = 16 * n_nodes.
    t = pd.read_csv(a.timings)
    t = t[(t.warmup == 0) & (t.timed_region == "kernel")]
    t["nodes"] = t.notes.str.extract(r"nodes=(\d+)")[0].astype(float)
    t["exp"] = t.notes.str.extract(r"exp=([^;]+)")[0]
    t = t[(t.exp == "strong") & t.nodes.notna()]
    if t.empty:
        raise SystemExit("no strong-scaling rows with nodes= in notes")
    t["p"] = (t.nodes * t.n_threads * t.n_ranks / t.nodes).astype(int) \
        if "n_ranks" in t else (t.nodes * 16).astype(int)
    t["p"] = (t.nodes * 16).astype(int)      # 2 ranks x 8 threads per node
    meas = t.groupby("p").work_imbalance.median().sort_index()

    print("=== E0 predicted (Day 3) vs measured (Days 8-9) ===")
    print(f"{'p':>4} {'predicted':>10} {'measured':>10} {'error':>9} "
          f"{'pred eff':>9} {'meas eff':>9}  source")
    rows = []
    for p, m in meas.items():
        if p in pred.index:
            pv, how = pred.loc[p], "tabulated"
        else:
            xs = np.array(pred.index, float)
            pv, how = float(np.interp(p, xs, pred.values)), "INTERPOLATED"
        err = 100 * (m / pv - 1)
        rows.append((p, pv, m, err, how))
        print(f"{p:>4} {pv:>10.4f} {m:>10.4f} {err:>+8.2f}% "
              f"{100/pv:>8.1f}% {100/m:>8.1f}%  {how}")

    tab = [r for r in rows if r[4] == "tabulated"]
    if tab:
        e = np.array([r[3] for r in tab])
        print(f"\n  tabulated points only: n={len(tab)}, "
              f"mean error {e.mean():+.2f}%, max |error| {np.abs(e).max():.2f}%")
        print("  Within a few percent means a prediction made on Day 3 from")
        print("  offsets alone, with no kernel and no machine, holds against")
        print("  a five-node measurement taken nine days later.")
        print("  A systematic gap means block partitioning loses something")
        print("  the combinatorial model does not capture -- which is itself")
        print("  a result, since the model is exact for record counts.")

    print("\n=== flat vs hierarchical partitioning ===")
    print("  E0 models ONE flat p-way block cut. MPI partitions in two")
    print("  stages: stays across ranks, then across threads inside a rank.")
    print("  Two-stage cutting can balance better than one flat cut at the")
    print("  same p, so a measured imbalance BELOW the prediction is the")
    print("  expected direction, not a failure of the model.")
    print("  Cross-check: E0's p=16 block prediction against the SINGLE-NODE")
    print("  OpenMP T=16 measurement, which is a genuinely flat 16-way cut.")

    print("\n=== why this was untestable before the multi-node sweep ===")
    for p, pv, m, err, how in rows:
        band = "inside the 2.39% IQR" if abs(100 * (pv - 1)) < 2.39 else \
               "outside noise -- testable"
        print(f"  p={p:>3}: predicted imbalance {100*(pv-1):+5.2f}% -> {band}")


if __name__ == "__main__":
    main()
