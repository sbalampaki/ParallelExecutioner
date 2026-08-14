#!/usr/bin/env python3


import argparse
import sys
import time

import numpy as np

# thresholds, all from measurement 
THRESHOLDS = {
    "noise_floor_D36": 0.0239,       # within-repetition IQR at T=16
    "one_seam_rung_D44": 0.0840,     # smallest measured seam-doubling rung
    "full_placement_D44": 0.3200,    # grouped vs alternating, same 16 CPUs
}

MAX_MEASURABLE_P = 96                # 6 el9 nodes x 16 physical cores
FAR_FIELD = (8192, 16384, 32768)     # GPU-scale, for the future-work remark


def ref_block_bounds(m, k):
    return [(i * m) // k for i in range(k + 1)]


def ref_nz_nearest(counts, k):
    n = len(counts)
    if k >= n:
        return list(range(n + 1)) + [n] * (k - n)
    csum = np.concatenate([[0], np.cumsum(counts)])
    total = csum[-1]
    cuts = [0]
    for i in range(1, k):
        t = total * i / k
        j = int(np.searchsorted(csum, t, side="left"))
        if j > 0 and abs(csum[j - 1] - t) <= abs(csum[min(j, n)] - t):
            j -= 1
        cuts.append(max(cuts[-1] + 1, min(j, n - (k - i))))
    return cuts + [n]


def ref_imbalance(counts, R, T, kind):
    n = len(counts)
    loads = []
    if kind == "block":
        rb = ref_block_bounds(n, R)
        for r in range(R):
            lo, hi = rb[r], rb[r + 1]
            tb = ref_block_bounds(hi - lo, T)
            for t in range(T):
                loads.append(counts[lo + tb[t]: lo + tb[t + 1]].sum())
    else:
        rc = ref_nz_nearest(counts, R)
        for r in range(R):
            sub = counts[rc[r]: rc[r + 1]]
            tc = ref_nz_nearest(sub, T) if len(sub) else [0] * (T + 1)
            for t in range(T):
                loads.append(sub[tc[t]: tc[t + 1]].sum())
    loads = np.asarray(loads, dtype=np.float64)
    return float(loads.max() / (counts.sum() / (R * T)))


def block_loads(csum, n, R, T):
    """Two-stage floor-division index cut. Returns the p partition loads."""
    rb = (np.arange(R + 1, dtype=np.int64) * n) // R
    m = rb[1:] - rb[:-1]                                  # (R,)
    tt = np.arange(T + 1, dtype=np.int64)                 # (T+1,)
    b = rb[:-1][:, None] + (tt[None, :] * m[:, None]) // T   # (R, T+1)
    return (csum[b[:, 1:]] - csum[b[:, :-1]]).ravel()


def _nearest_cuts(csum, lo, hi, k):
    """
    Static-target nearest-boundary cuts of [lo, hi) into k groups, for many
    segments at once.  lo, hi are int arrays of shape (S,).  Returns an
    (S, k+1) array of GLOBAL indices.

    Reproduces ref_nz_nearest exactly, including its two clamps:
      (a) min(j, n_seg - (k - i))   -- leave room for the trailing groups
      (b) max(prev + 1, j)          -- strictly increasing
    Clamp (b) is a running maximum, applied here as a cumulative max on
    u_i = j_i - i, which is algebraically identical.
    """
    S = len(lo)
    nseg = (hi - lo).astype(np.int64)
    out = np.empty((S, k + 1), dtype=np.int64)
    out[:, 0] = lo
    out[:, k] = hi
    if k == 1:
        return out

    base = csum[lo].astype(np.float64)
    total = (csum[hi] - csum[lo]).astype(np.float64)
    i = np.arange(1, k, dtype=np.int64)                    # (k-1,)

    t = base[:, None] + total[:, None] * i[None, :] / k    # (S, k-1)
    j = np.searchsorted(csum, t.ravel(), side="left").reshape(S, k - 1)
    j = np.clip(j - lo[:, None], 0, nseg[:, None])         # -> segment-local

    # nearest boundary
    jm = np.maximum(j - 1, 0)
    lhs = np.abs(csum[lo[:, None] + jm].astype(np.float64) - t)
    rhs = np.abs(csum[lo[:, None] + np.minimum(j, nseg[:, None])]
                 .astype(np.float64) - t)
    j = np.where((j > 0) & (lhs <= rhs), j - 1, j)

    # clamp (a), then clamp (b)
    j = np.minimum(j, nseg[:, None] - (k - i)[None, :])
    u = np.concatenate([np.zeros((S, 1), dtype=np.int64), j - i[None, :]],
                       axis=1)
    j = np.maximum.accumulate(u, axis=1)[:, 1:] + i[None, :]


    j = np.minimum(j, nseg[:, None])

    out[:, 1:k] = lo[:, None] + j
    return out


def nz_loads(csum, n, R, T):
    """Two-stage nearest-boundary record-balanced cut."""
    rc = _nearest_cuts(csum, np.array([0]), np.array([n]), R)[0]     # (R+1,)
    lo, hi = rc[:-1], rc[1:]
    tc = _nearest_cuts(csum, lo, hi, T)                              # (R, T+1)
    return (csum[tc[:, 1:]] - csum[tc[:, :-1]]).ravel()


def imbalance(csum, n, R, T, kind):
    loads = block_loads(csum, n, R, T) if kind == "block" \
        else nz_loads(csum, n, R, T)
    return float(loads.max() / (csum[n] / (R * T)))


def selftest():
    rng = np.random.default_rng(20260726)
    print("--- selftest: vectorised vs scalar reference ---")
    worst = 0.0
    for trial in range(6):
        n = int(rng.integers(400, 1200))
        counts = (rng.pareto(1.1, n) * 20 + 1).astype(np.int64)
        csum = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        for R in (1, 2, 3, 5, 8, 13):
            for T in (1, 2, 4, 8, 16):
                if R * T > n // 2:
                    continue
                for kind in ("block", "nz"):
                    a = imbalance(csum, n, R, T, kind)
                    b = ref_imbalance(counts, R, T, kind)
                    worst = max(worst, abs(a - b))
    print(f"worst |vectorised - scalar| = {worst:.3e}")
    if worst > 1e-12:
        print("SELFTEST FAILED"); sys.exit(1)
    print("SELFTEST PASSED\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets")
    ap.add_argument("--validation")
    ap.add_argument("--out", default="pstar_curve.csv")
    ap.add_argument("--pmax", type=int, default=4096)
    ap.add_argument("--tmax", type=int, default=16,
                    help="max threads per rank; 16 physical cores per node")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return
    if not (args.offsets and args.validation):
        ap.error("--offsets and --validation are required")

    counts = np.diff(np.fromfile(args.offsets, dtype=np.int64)).astype(np.int64)
    n = len(counts)
    csum = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    print(f"offsets   : {args.offsets}")
    print(f"n_stays   : {n:,}   n_records : {int(csum[n]):,}   "
          f"CV : {counts.std(ddof=0)/counts.mean():.4f}   "
          f"min records/stay : {int(counts.min())}")

    rows = []
    with open(args.validation) as fh:
        hdr = fh.readline().strip().split(",")
        for line in fh:
            f = dict(zip(hdr, line.strip().split(",")))
            rows.append({"partitioner": f["partitioner"], "p": int(f["p"]),
                         "R": int(f["n_ranks"]), "T": int(f["n_threads"]),
                         "hier": float(f["predicted_hier"])})

    # ---- gate: block AND nzbalanced, both hard 
    print("\n--- gate ---")
    worst = 0.0
    for r in rows:
        if r["partitioner"] not in ("block", "nzbalanced"):
            continue
        kind = "block" if r["partitioner"] == "block" else "nz"
        d = abs(imbalance(csum, n, r["R"], r["T"], kind) - r["hier"])
        worst = max(worst, d)
        print(f"  {r['partitioner']:<11} p={r['p']:<4} "
              f"expected {r['hier']:.12f}   |resid| {d:.2e}")
    print(f"worst {worst:.3e}")
    if worst >= 1e-9:
        print("\nFAILED. No curve emitted.")
        sys.exit(1)
    print("PASSED -- both cut rules reproduced. The curve is this same "
          "computation at new p.\n")

    # ---- the curve 
    t0 = time.time()
    out, plist = [], list(range(2, args.pmax + 1)) + list(FAR_FIELD)
    for idx, p in enumerate(plist):
        for T in range(1, min(args.tmax, p) + 1):
            if p % T:
                continue
            R = p // T
            if R > n or T > n:
                continue
            ib = imbalance(csum, n, R, T, "block")
            iz = imbalance(csum, n, R, T, "nz")
            out.append({
                "p": p, "n_ranks": R, "n_threads": T,
                "imb_block": ib, "imb_nzbalanced": iz,
                "loss_block_pct": 100.0 * (1.0 - 1.0 / ib),
                "ceiling_pct": 100.0 * (ib / iz - 1.0),
                "measurable": int(p <= MAX_MEASURABLE_P and T <= 16),
            })
        if idx % 512 == 0 and idx:
            print(f"  ... p={p}  ({time.time()-t0:.0f} s)")

    with open(args.out, "w") as fh:
        cols = list(out[0])
        fh.write(",".join(cols) + "\n")
        for r in out:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"wrote {len(out):,} rows -> {args.out}   ({time.time()-t0:.0f} s)")

    line = sorted([r for r in out if r["n_threads"] == 8], key=lambda r: r["p"])
    print("\n--- crossings, T=8 per rank (the measured configuration) ---")
    print("  ceiling = % gain permitted by the block -> nzbalanced "
          "imbalance difference\n")
    for name, thr in THRESHOLDS.items():
        hit = next((r for r in line if r["ceiling_pct"] >= 100 * thr), None)
        if hit:
            flag = "" if hit["measurable"] else "   [beyond available hardware]"
            print(f"  {name:<20} {100*thr:5.2f}%   first crossed at "
                  f"p = {hit['p']:<6} (R={hit['n_ranks']}, T={hit['n_threads']})"
                  f"{flag}")
        else:
            print(f"  {name:<20} {100*thr:5.2f}%   not crossed by "
                  f"p = {args.pmax}")

    print("\n  measured anchors:")
    for p in (16, 32, 64, 80, 96):
        r = next((x for x in line if x["p"] == p), None)
        if r:
            print(f"    p={p:<4} block {r['imb_block']:.6f}   "
                  f"loss {r['loss_block_pct']:5.2f}%   "
                  f"ceiling {r['ceiling_pct']:5.2f}%")

    print("\n  far field (GPU scale, arithmetic only):")
    for p in FAR_FIELD:
        r = next((x for x in out if x["p"] == p and x["n_threads"] == 8), None)
        if r:
            print(f"    p={p:<6} block {r['imb_block']:.4f}   "
                  f"loss {r['loss_block_pct']:5.2f}%   "
                  f"ceiling {r['ceiling_pct']:6.2f}%")

    print("\n--- factorisation spread at fixed p, T<=16 (open item 3) ---")
    for p in (16, 32, 64, 80, 96, 128, 256, 512, 1024):
        cand = [r for r in out if r["p"] == p]
        if len(cand) < 2:
            continue
        lo_r = min(cand, key=lambda r: r["imb_block"])
        hi_r = max(cand, key=lambda r: r["imb_block"])
        print(f"  p={p:<5} {lo_r['imb_block']:.6f} (R={lo_r['n_ranks']},"
              f"T={lo_r['n_threads']}) .. {hi_r['imb_block']:.6f} "
              f"(R={hi_r['n_ranks']},T={hi_r['n_threads']})   spread "
              f"{100*(hi_r['imb_block']/lo_r['imb_block']-1):.2f}%")

    print("\nArithmetic on offsets. No kernel, no machine, no timing, "
          "no fitted parameter.")


if __name__ == "__main__":
    main()
