#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd

CELLS = [
    ("T16 algn",     16, 1.0000), ("T16 intl_grp", 16, 0.5494),
    ("T16 intl_alt", 16, 0.5005), ("T16 anti",     16, 0.0000),
    ("T16 anti",     16, 0.0000), ("T16 intl_alt", 16, 0.5005),
    ("T16 intl_grp", 16, 0.5494), ("T16 algn",     16, 1.0000),
    ("T8  algn",      8, 0.5000), ("T8  anti",      8, 0.5000),
]
FLOOR_STREAM = 12 / 64          # 0.1875 lines/record, record stream alone
FLOOR_TOTAL  = 0.2030           # + out matrix + offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timings", required=True)
    ap.add_argument("--threads")
    a = ap.parse_args()

    t = pd.read_csv(a.timings)
    t["pid"] = t.run_id.str.split("-").str[2].astype(int)
    order = {p: i for i, p in enumerate(sorted(t.pid.unique()))}
    if len(order) != len(CELLS):
        raise SystemExit(f"expected {len(CELLS)} runs, found {len(order)}. "
                         "Did the preflight rows survive, or a cell fail?")
    t["cell"] = t.pid.map(lambda p: CELLS[order[p]][0])
    t["local_frac"] = t.pid.map(lambda p: CELLS[order[p]][2])
    t["pass"] = t.pid.map(lambda p: 1 + (order[p] >= 4))     # palindrome half

    t = t[t.warmup == 0]
    d = t.n_records * t.k_iterations
    t["l3_per_rec"]  = t.l3_tcm / d
    t["cyc_per_rec"] = t.tot_cyc / d
    t["ins_per_rec"] = t.tot_ins / d
    t["l3_miss_pct"] = 100 * t.l3_tcm / t.l3_tca
    t["mrec_ghz"]    = t.throughput_rec_s / 1e6 / t.achieved_ghz

    cols = ["l3_per_rec", "l3_miss_pct", "cyc_per_rec", "ins_per_rec",
            "achieved_ghz", "mrec_ghz", "time_imbalance", "work_imbalance"]

    print("=== per cell, median over reps 1-5, palindrome halves averaged ===")
    g = (t.groupby(["cell", "local_frac"])[cols].median()
           .reset_index().sort_values("local_frac"))
    g["l3_vs_stream"] = g.l3_per_rec / FLOOR_STREAM
    print(g.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    t16 = t[t.n_threads == 16]
    print("\n=== drift check: palindrome half 1 vs half 2 (T=16 only) ===")
    print("  Cells ran at positions 0,1,2,3 then mirrored 7,6,5,4. The time")
    print("  gap between a cell's two runs is therefore widest for the")
    print("  outermost pair. If drift is LINEAR the palindrome mean cancels")
    print("  it exactly, and |drift_%| should order algn > intl_grp >")
    print("  intl_alt > anti. If it does not, the drift is not linear and")
    print("  the palindrome mean is only a partial correction.")
    for metric in ["mrec_ghz", "l3_per_rec", "achieved_ghz",
                   "cyc_per_rec", "time_imbalance"]:
        h = t16.groupby(["cell", "pass"])[metric].median().unstack()
        h.columns = ["half1", "half2"]
        h["drift_%"] = 100 * (h.half2 / h.half1 - 1)
        print(f"\n  --- {metric} ---")
        print(h.to_string(float_format=lambda v: f"{v:.4f}"))

 
    print("\n=== rank stability of mrec_ghz within each half ===")
    r = t16.groupby(["cell", "pass"]).mrec_ghz.median().unstack()
    r.columns = ["half1", "half2"]
    print(r.to_string(float_format=lambda v: f"{v:.2f}"))
    for c in ["half1", "half2"]:
        o = r[c].sort_values(ascending=False)
        print(f"  {c}: " + "  >  ".join(f"{i.replace('T16 ','')} {v:.1f}"
                                        for i, v in o.items()))
    same = (r.half1.rank(ascending=False)
            == r.half2.rank(ascending=False)).all()
    spread = 100 * (r.max(axis=1).max() / r.min(axis=1).min() - 1)
    drift = (100 * (r.half2 / r.half1 - 1)).abs().max()
    print(f"\n  ranking identical in both halves: {same}")
    print(f"  between-cell spread {spread:.2f}%   max within-cell drift "
          f"{drift:.2f}%   ratio {spread/drift:.1f}x" if drift else "")
    print("  An effect is safe from drift when the ranking is stable AND the")
    print("  spread is several times the drift. Adjacent cells inside the")
    print("  noise floor stay unresolved either way.")

    print("\n=== P11: is l3_per_rec linear in page_local_frac? (T=16) ===")
    s = g[g.cell.str.startswith("T16")]
    x, y = s.local_frac.values, s.l3_per_rec.values
    A = np.vstack([x, np.ones_like(x)]).T
    (m, c), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - res[0] / ss if len(res) and ss > 0 else float("nan")
    print(f"  fit: l3_per_rec = {c:.4f} + {m:.4f} * local_frac   R^2={r2:.4f}")
    pred2pt = [0.1395, 0.1690, 0.1719, 0.1986]        # two-point extrapolation
    print("\n  local_frac   measured   2-pt predicted   residual")
    for lf, meas, p in zip(x, y, sorted(pred2pt)):
        print(f"    {lf:.4f}     {meas:.4f}      {p:.4f}       {meas-p:+.4f}")
    print(f"\n  mandatory record-stream floor: {FLOOR_STREAM:.4f}")
    below = s[s.l3_per_rec < FLOOR_STREAM]
    if len(below):
        print("  BELOW THE FLOOR (impossible for a complete line-fill count):")
        for _, r in below.iterrows():
            print(f"    {r.cell:14s} {r.l3_per_rec:.4f}  "
                  f"({100*(1-r.l3_per_rec/FLOOR_STREAM):.1f}% short)")
    else:
        print("  every cell at or above the floor -- the count is complete")

    if a.threads:
        th = pd.read_csv(a.threads)
        th["pid"] = th.run_id.str.split("-").str[2].astype(int)
        th["seq"] = th.run_id.str.split("-").str[3].astype(int)
        th = th[(th.seq > 0) & th.pid.isin([p for p in order
                                            if order[p] >= 8])]
        th["cell"] = th.pid.map(lambda p: CELLS[order[p]][0])
        th["rate"] = th.n_records / th.secs
        print("\n=== T=8 socket0, per-thread, PAIRED by tid ===")
        print("  algn puts tid0-3 on local memory, anti puts tid4-7 there:")
        print("  mirrored exposure on the SAME cores. Comparing halves within")
        print("  one cell measures core position, not locality -- the pattern")
        print("  is U-shaped and identical in both cells. Pairing by tid")
        print("  cancels it, which is the design that worked on job 73063.")
        piv = th.groupby(["cell", "tid"]).rate.median().unstack() / 1e6
        print(piv.to_string(float_format=lambda v: f"{v:.3f}"))
        try:
            a = piv.loc[[i for i in piv.index if "algn" in i][0]]
            b = piv.loc[[i for i in piv.index if "anti" in i][0]]
        except IndexError:
            return
        half = len(a) // 2
        d = (a / b - 1) * 100
        print("\n   tid   algn/anti     algn exposure")
        for i, v in d.items():
            print(f"    {i}    {v:+7.3f}%    "
                  f"{'LOCAL  (anti remote)' if i < half else 'REMOTE (anti local)'}")
        lo, hi = d[d.index < half], d[d.index >= half]
        pen = (lo.mean() - hi.mean()) / 2
        print(f"\n  tid<{half}  (local in algn):  {lo.mean():+.3f}%   "
              f"{(lo > 0).sum()}/{len(lo)} positive")
        print(f"  tid>={half} (remote in algn): {hi.mean():+.3f}%   "
              f"{(hi < 0).sum()}/{len(hi)} negative")
        print(f"  implied 0 -> 100% remote penalty: {pen:.2f}%")
        print(f"  job 73063 paired estimate: 1.24%, 95% CI [0.79, 1.70] -> "
              f"{'CONSISTENT' if 0.79 <= pen <= 1.70 else 'OUTSIDE THE CI'}")
        k = (lo > 0).sum() + (hi < 0).sum()
        print(f"  sign test: {k}/{len(d)} threads move in the predicted "
              f"direction")


if __name__ == "__main__":
    main()
