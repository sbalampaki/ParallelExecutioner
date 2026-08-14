#!/usr/bin/env python3


import argparse
import json
import os
import sys

import numpy as np

__version__ = "1.0"


PLACEMENT_RUNG = 0.084      # one doubling of same-socket thread groups (8.4%)
PLACEMENT_SPAN = 0.320      # grouped vs fully alternating (32.0%)
NOISE_FLOOR = 0.0239        # within-repetition spread (2.39%)
DEFAULT_THREADS_PER_RANK = 8
DEFAULT_CORES_PER_NODE = 16
TRANSFERRED_PSTAR = 56      # study cohort; used only when offsets unavailable


def block_imbalance(csum, n, R, T):
    """Index-balanced (block) cut, two stage."""
    total = int(csum[n])
    rb = (np.arange(R + 1, dtype=np.int64) * n) // R
    m = rb[1:] - rb[:-1]
    tt = np.arange(T + 1, dtype=np.int64)
    b = rb[:-1][:, None] + (tt[None, :] * m[:, None]) // T
    loads = (csum[b[:, 1:]] - csum[b[:, :-1]]).ravel()
    return float(loads.max() / (total / (R * T)))


def _nearest_cuts(csum, lo, hi, k):
 
    lo = np.asarray(lo, dtype=np.int64)
    hi = np.asarray(hi, dtype=np.int64)
    nseg = hi - lo
    out = np.empty((len(lo), k + 1), dtype=np.int64)
    out[:, 0] = lo
    out[:, k] = hi
    if k <= 1:
        return out

    base = csum[lo]
    tot = csum[hi] - base
    i = np.arange(1, k, dtype=np.int64)

    # interior targets, one row per segment
    t = base[:, None] + tot[:, None] * i[None, :] / k
    j = np.searchsorted(csum, t) - lo[:, None]
    j = np.clip(j, 0, nseg[:, None])

    # snap to whichever neighbouring boundary is closer
    jm = np.maximum(j - 1, 0)
    left = np.abs(csum[lo[:, None] + jm] - t)
    right = np.abs(csum[lo[:, None] + np.minimum(j, nseg[:, None])] - t)
    j = np.where((j > 0) & (left <= right), j - 1, j)

    # feasibility: leave room for the remaining k-i cuts
    j = np.minimum(j, nseg[:, None] - (k - i)[None, :])
    j = np.maximum(j, 0)

    # monotonicity: cuts must be strictly increasing where segments allow
    u = np.concatenate([np.zeros((len(lo), 1), dtype=np.int64),
                        j - i[None, :]], axis=1)
    j = np.minimum(np.maximum.accumulate(u, axis=1)[:, 1:] + i[None, :],
                   nseg[:, None])

    # degenerate segments (nseg <= k) collapse to min(i, nseg)
    tiny = nseg <= k
    if np.any(tiny):
        j[tiny] = np.minimum(i[None, :], nseg[tiny][:, None])

    out[:, 1:k] = lo[:, None] + j
    return out


def balanced_imbalance(csum, n, R, T):
    """Record-balanced (nzbalanced) cut, two stage."""
    total = int(csum[n])
    rc = _nearest_cuts(csum, np.array([0]), np.array([n]), R)[0]
    tc = _nearest_cuts(csum, rc[:-1], rc[1:], T)
    loads = (csum[tc[:, 1:]] - csum[tc[:, :-1]]).ravel().astype(np.float64)
    return float(loads.max() / (total / (R * T)))


def ceiling(csum, n, R, T):
    """C(p): fractional gain a distribution-aware cut permits over block."""
    return block_imbalance(csum, n, R, T) / balanced_imbalance(csum, n, R, T) - 1.0



def find_pstar(csum, n, T=DEFAULT_THREADS_PER_RANK, pmax=8192,
               rung=PLACEMENT_RUNG):
    """Smallest p = R*T at which C(p) first exceeds one placement rung."""
    for R in range(1, pmax // T + 1):
        p = R * T
        if p > n // 2:
            break
        if ceiling(csum, n, R, T) >= rung:
            return p
    return None


def analyse(counts, threads_per_rank=DEFAULT_THREADS_PER_RANK):
    """Everything computable from the row-length array alone."""
    counts = np.asarray(counts, dtype=np.int64)
    n = len(counts)
    csum = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    N = int(csum[n])
    mx = int(counts.max())

    srt = np.sort(counts)
    rank = np.arange(1, n + 1, dtype=np.float64)
    gini = float(2.0 * np.sum(rank * srt) / (n * srt.sum()) - (n + 1) / n)

    p50 = float(np.percentile(counts, 50))
    p99 = float(np.percentile(counts, 99))

    return {
        "n_units": n,
        "n_records": N,
        "max_unit": mx,
        "mean_unit": N / n,
        "median_unit": p50,
        "p99_unit": p99,
        "p99_over_median": p99 / p50 if p50 else float("inf"),
        "cv": float(counts.std(ddof=0) / counts.mean()),
        "gini": gini,
        "max_over_mean": mx / (N / n),
        "p_atom": N / mx,
        "p_star": find_pstar(csum, n, threads_per_rank),
        "_csum": csum,
        "_n": n,
    }


REGIMES = {
    "I": ("placement-limited",
          "Fix thread placement before touching the partitioner.",
          ["Bind threads so consecutive workers share a socket "
           "(OMP_PROC_BIND=close, OMP_PLACES=cores).",
           "Read CPU-to-socket mapping from sysfs; do not assume the "
           "numbering scheme.",
           "Partitioner choice is worth less than the placement decision "
           "at this scale."]),
    "II": ("partition-limited",
           "The partitioner is now the lever, and its value grows with p.",
           ["Use a contiguous, work-balanced partitioner.",
            "Avoid non-contiguous assignments under a two-stage "
            "(rank-then-thread) decomposition.",
            "Choose the node count so the unit count divides evenly across "
            "first-stage cuts.",
            "Keep thread placement fixed -- its cost has not gone away, it "
            "has merely been overtaken."]),
    "III": ("atomicity-limited",
            "No assignment of whole units can help at this scale.",
            ["The mean partition is smaller than the largest indivisible "
             "unit; imbalance now grows linearly in p.",
             "Only sub-unit decomposition escapes the floor (split a unit "
             "across workers and reduce).",
             "Adding workers past this point buys less than proportional "
             "throughput regardless of partitioner."]),
}


def classify(p, p_star, p_atom):
    if p_star is not None and p < p_star:
        return "I"
    if p < p_atom:
        return "II"
    return "III"


def advise(res, p, threads_per_rank=DEFAULT_THREADS_PER_RANK):
    """Regime + bounded gain for a given partition count."""
    p_star, p_atom = res["p_star"], res["p_atom"]
    reg = classify(p, p_star, p_atom)

    R = max(1, p // threads_per_rank)
    T = threads_per_rank if p >= threads_per_rank else p
    c = None
    if R * T <= res["_n"] // 2:
        c = ceiling(res["_csum"], res["_n"], R, T)

    return {
        "p": p,
        "regime": reg,
        "regime_name": REGIMES[reg][0],
        "headline": REGIMES[reg][1],
        "actions": REGIMES[reg][2],
        "ceiling": c,
        "ceiling_vs_rung": (c / PLACEMENT_RUNG) if c is not None else None,
        "resolvable": (c is not None and c > NOISE_FLOOR),
        "p_over_p_atom": p / p_atom,
    }


BAR = "=" * 72


def print_structure(res, label=None, computed_pstar=True):
    print(BAR)
    print("WORKLOAD STRUCTURE" + (f"  --  {label}" if label else ""))
    print(BAR)
    print(f"  units (rows)            {res['n_units']:>16,}")
    print(f"  records (nonzeros)      {res['n_records']:>16,}")
    print(f"  largest unit            {res['max_unit']:>16,}")
    print(f"  mean per unit           {res['mean_unit']:>16,.0f}")
    print(f"  median per unit         {res['median_unit']:>16,.0f}")
    print()
    print(f"  coefficient of variation{res['cv']:>16.3f}"
          f"   {'heavy tail' if res['cv'] > 1 else 'roughly uniform'}")
    print(f"  Gini                    {res['gini']:>16.3f}")
    print(f"  P99 / median            {res['p99_over_median']:>16.2f}")
    print(f"  max / mean              {res['max_over_mean']:>16.1f}x")
    print()
    print("  THRESHOLDS")
    print(f"    p_atom = N / max(n_i) {res['p_atom']:>16,.0f}"
          "   atomicity floor")
    if res["p_star"] is not None:
        tag = "computed from this data" if computed_pstar else "transferred"
        print(f"    p*                    {res['p_star']:>16,}"
              f"   crossover ({tag})")
    else:
        print(f"    p*                    {'not reached':>16}"
              "   below one placement rung throughout")
    print()


def print_advice(a, cores_per_node=DEFAULT_CORES_PER_NODE):
    print(BAR)
    print(f"ADVICE AT p = {a['p']:,}")
    print(BAR)
    nodes = a["p"] / cores_per_node
    print(f"  allocation              {a['p']:,} workers "
          f"(~{nodes:.1f} nodes at {cores_per_node} cores/node)")
    print(f"  regime                  {a['regime']} -- {a['regime_name']}")
    if a["ceiling"] is not None:
        print(f"  partitioner ceiling     {a['ceiling'] * 100:.2f}%"
              f"   ({a['ceiling_vs_rung']:.2f} placement rungs)")
        if not a["resolvable"]:
            print(f"  {'':<23} below the {NOISE_FLOOR * 100:.2f}% noise "
                  "floor -- not worth pursuing")
    print(f"  placement span          {PLACEMENT_SPAN * 100:.1f}%"
          "   (per-node, does not grow with p)")
    print()
    print(f"  >> {a['headline']}")
    for i, act in enumerate(a["actions"], 1):
        print(f"     {i}. {act}")
    print()


def print_sweep(res, ps, cores_per_node=DEFAULT_CORES_PER_NODE):
    print(BAR)
    print("ALLOCATION SWEEP")
    print(BAR)
    print(f"  {'p':>8} {'nodes':>7} {'regime':<20} {'ceiling':>9} "
          f"{'rungs':>7} {'p/p_atom':>9}")
    print("  " + "-" * 66)
    for p in ps:
        a = advise(res, p)
        c = f"{a['ceiling'] * 100:.2f}%" if a["ceiling"] is not None else "--"
        r = f"{a['ceiling_vs_rung']:.2f}" if a["ceiling"] is not None else "--"
        print(f"  {p:>8,} {p / cores_per_node:>7.1f} "
              f"{a['regime'] + ' ' + a['regime_name']:<20} {c:>9} {r:>7} "
              f"{a['p_over_p_atom']:>9.3f}")
    print()


def print_scope():
    print(BAR)
    print("SCOPE")
    print(BAR)
    print("  Reported here (arithmetic on the offset array, no execution):")
    print("    - which optimisation repays attention at a given worker count")
    print("    - the ceiling on what any contiguous partitioner can recover")
    print("    - the worker count past which whole-unit assignment cannot help")
    print()
    print("  NOT reported (no timing model):")
    print("    - runtime, memory, node-hours, energy")
    print("    - how many nodes to request to meet a deadline")
    print()
    print("  Thresholds are machine-independent. The placement magnitudes")
    print("  above are measured on one platform: expect the sign to transfer")
    print("  to other server-class hardware, not the magnitude.")
    print()



def _label_for(path):
    """Short display name.

    Handles both layouts: csr/Wfull/offsets.i64 -> 'Wfull', and
    demo_Wfull.i64 -> 'demo_Wfull'. Falls back to the parent directory only
    when the filename itself carries no information.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.lower().startswith("offset"):
        parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
        return parent or stem
    return stem


def counts_from_offsets(path, dtype=np.int64):
    off = np.fromfile(path, dtype=dtype)
    if off.size < 2:
        sys.exit(f"{path}: offset array too short ({off.size} entries)")
    counts = np.diff(off)
    if np.any(counts < 0):
        sys.exit(f"{path}: offsets are not monotonically non-decreasing")
    if counts.sum() <= 0:
        sys.exit(f"{path}: empty workload")
    return counts.astype(np.int64)


def synthetic_counts(n, N, mx, seed=0):
    """A lognormal cohort matching (n, N, max) -- for summary-stats mode.

    p_atom and the distributional summaries are exact from the three given
    numbers; p* depends on the full shape and is NOT computed here.
    """
    rng = np.random.default_rng(seed)
    x = rng.lognormal(0.0, 1.0, n)
    x = x / x.sum() * N
    x = np.maximum(np.round(x), 1).astype(np.int64)
    x[int(np.argmax(x))] = mx
    return x



def _scalar_balanced(counts, R, T):
    """Plain-Python transcription of the canonical nearest-boundary cut.

    Deliberately written as an element-by-element loop so it exercises the
    same rule as the vectorised path by a different route.
    """
    n = len(counts)
    csum = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = int(csum[n])

    def cuts(lo, hi, k):
        nseg = hi - lo
        if k <= 1:
            return [lo, hi]
        if nseg <= k:
            # fewer units than cuts: one unit each, remainder empty
            return [lo] + [lo + min(i, nseg) for i in range(1, k)] + [hi]
        base = int(csum[lo])
        tot = int(csum[hi]) - base

        raw = []
        for i in range(1, k):
            t = base + tot * i / k
            j = int(np.searchsorted(csum, t)) - lo
            j = max(0, min(j, nseg))
            # snap to the nearer of the two neighbouring boundaries
            jm = max(j - 1, 0)
            if j > 0 and abs(int(csum[lo + jm]) - t) <= \
                    abs(int(csum[lo + min(j, nseg)]) - t):
                j -= 1
            # leave room for the remaining k-i cuts
            j = min(j, nseg - (k - i))
            raw.append(j)

        # monotone via running maximum of the offset (j - i)
        out, run = [], 0
        for idx, j in enumerate(raw):
            i = idx + 1
            run = max(run, j - i)
            out.append(min(run + i, nseg))

        return [lo] + [lo + j for j in out] + [hi]

    rc = cuts(0, n, R)
    loads = []
    for r in range(R):
        tc = cuts(int(rc[r]), int(rc[r + 1]), T)
        for a, b in zip(tc[:-1], tc[1:]):
            loads.append(int(csum[b]) - int(csum[a]))
    return float(max(loads) / (total / (R * T)))


def selftest():
    rng = np.random.default_rng(20260812)
    print("selftest: vectorised two-stage cut vs scalar reference")
    worst = 0.0
    checks = 0
    for _ in range(6):
        n = int(rng.integers(400, 1200))
        counts = (rng.pareto(1.1, n) * 20 + 1).astype(np.int64)
        csum = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        for R in (1, 2, 3, 5, 8, 13):
            for T in (1, 2, 4, 8):
                if R * T > n // 2:
                    continue
                a = balanced_imbalance(csum, n, R, T)
                b = _scalar_balanced(counts, R, T)
                worst = max(worst, abs(a - b))
                checks += 1
    print(f"  {checks} configurations checked")
    print(f"  worst |vectorised - scalar| = {worst:.3e}")

    # invariants
    counts = np.array([100, 100, 100, 100], dtype=np.int64)
    r = analyse(counts)
    assert abs(r["gini"]) < 1e-12, "uniform cohort must have Gini 0"
    assert abs(r["p_atom"] - 4.0) < 1e-9, "p_atom wrong on uniform cohort"
    counts = np.array([1, 1, 1, 97], dtype=np.int64)
    r = analyse(counts)
    assert r["p_atom"] < 1.05, "p_atom wrong on dominated cohort"
    print("  invariants: Gini(uniform)=0, p_atom exact  [ok]")

    if worst > 1e-9:
        print("  SELFTEST FAILED")
        return 1
    print("  SELFTEST PASSED")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Pre-execution parallelisation advice from a row-offset "
                    "array. Reports which optimisation matters; does not "
                    "predict runtime or memory.")
    src = ap.add_argument_group("input")
    src.add_argument("--offsets", help="binary row-offset array (int64)")
    src.add_argument("--batch", nargs="+", metavar="OFFSETS",
                     help="compare several offset arrays")
    src.add_argument("--n-records", type=int)
    src.add_argument("--max-stay", type=int)
    src.add_argument("--n-stays", type=int)
    src.add_argument("--dtype", default="int64",
                     choices=["int32", "int64", "uint32", "uint64"])

    cfg = ap.add_argument_group("configuration")
    cfg.add_argument("--cores", type=int, help="worker count to advise on")
    cfg.add_argument("--sweep", help="comma-separated worker counts")
    cfg.add_argument("--threads-per-rank", type=int,
                     default=DEFAULT_THREADS_PER_RANK)
    cfg.add_argument("--cores-per-node", type=int,
                     default=DEFAULT_CORES_PER_NODE)

    out = ap.add_argument_group("output")
    out.add_argument("--json", metavar="PATH", help="write results as JSON")
    out.add_argument("--quiet", action="store_true")
    out.add_argument("--selftest", action="store_true")
    out.add_argument("--version", action="version",
                     version=f"skewcast {__version__}")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    dt = getattr(np, args.dtype)

    if args.batch:
        rows, payload = [], []
        for path in args.batch:
            c = counts_from_offsets(path, dt)
            r = analyse(c, args.threads_per_rank)
            label = _label_for(path)
            rows.append((label, r))
            payload.append({k: v for k, v in r.items()
                            if not k.startswith("_")} | {"label": label,
                                                         "source": path})
        if not args.quiet:
            print(BAR)
            print("VARIANT COMPARISON")
            print(BAR)
            print(f"  {'variant':<20} {'units':>9} {'records':>14} "
                  f"{'CV':>6} {'Gini':>6} {'p*':>7} {'p_atom':>10}")
            print("  " + "-" * 76)
            for label, r in rows:
                ps = f"{r['p_star']:,}" if r["p_star"] else "none"
                print(f"  {label:<20} {r['n_units']:>9,} "
                      f"{r['n_records']:>14,} {r['cv']:>6.3f} "
                      f"{r['gini']:>6.3f} {ps:>7} {r['p_atom']:>10,.0f}")
            print()
            print("  'p* = none' means the partitioner ceiling never reaches")
            print("  one placement rung: placement dominates at every scale")
            print("  reachable on this workload.")
            print()
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(payload, fh, indent=2)
            print(f"  wrote {args.json}")
        return

    computed = True
    if args.offsets:
        counts = counts_from_offsets(args.offsets, dt)
        label = args.offsets
    elif args.n_records and args.max_stay and args.n_stays:
        counts = synthetic_counts(args.n_stays, args.n_records, args.max_stay)
        label = "summary statistics"
        computed = False
    else:
        ap.error("need --offsets, --batch, or all of "
                 "--n-records --max-stay --n-stays")

    res = analyse(counts, args.threads_per_rank)

    if not computed:
        # shape-dependent quantities are not trustworthy from 3 numbers
        res["p_star"] = TRANSFERRED_PSTAR

    if not args.quiet:
        print_structure(res, label, computed_pstar=computed)
        if not computed:
            print("  NOTE: p_atom above is exact from the supplied totals, but")
            print("  p* and the distributional summaries depend on the full")
            print("  shape. p* is TRANSFERRED from the study cohort; supply")
            print("  --offsets to compute it for your own data.")
            print()

    if args.sweep:
        ps = [int(x) for x in args.sweep.split(",")]
        print_sweep(res, ps, args.cores_per_node)

    if args.cores:
        a = advise(res, args.cores, args.threads_per_rank)
        print_advice(a, args.cores_per_node)

    if not args.quiet:
        print_scope()

    if args.json:
        payload = {k: v for k, v in res.items() if not k.startswith("_")}
        payload["source"] = label
        payload["pstar_computed"] = computed
        if args.cores:
            payload["advice"] = advise(res, args.cores, args.threads_per_rank)
        if args.sweep:
            payload["sweep"] = [advise(res, int(x))
                                for x in args.sweep.split(",")]
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
