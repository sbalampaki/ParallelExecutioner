#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Reuse the validated implementations rather than duplicating them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_work import (RHO_REF, load_keep, per_stay_counts, cv, eff,  # noqa: E402
                     loads_from_cuts)

P_LIST = [8, 16, 96]
FLAG_LO, FLAG_HI = 5.0, 95.0      # percentile band outside which we flag


def expected_max_gaussian(p, seed=0):
 
    try:
        from scipy.special import ndtr
        x = np.linspace(-12.0, 12.0, 400_001)
        F = ndtr(x) ** p
        pos, neg = x >= 0, x < 0
        return float(np.trapezoid(1.0 - F[pos], x[pos])
                     - np.trapezoid(F[neg], x[neg]))
    except Exception:
        rng = np.random.default_rng(seed)
        tot, n_batch, batch = 0.0, 400, 500
        for _ in range(n_batch):
            tot += rng.standard_normal((batch, p)).max(axis=1).mean()
        return tot / n_batch


#  partitions
def block_loads(cost, p):
    cuts = np.linspace(0, cost.size, p + 1).astype(np.int64)
    return loads_from_cuts(cost, cuts)


def cyclic_loads(cost, p):
   
    q = cost.size // p
    out = cost[:q * p].reshape(q, p).sum(axis=0)
    tail = cost.size - q * p
    if tail:
        out[:tail] += cost[q * p:]
    return out


def nzbal_loads(cost, key, p):
    cs = np.concatenate(([0.0], np.cumsum(key)))
    targets = np.linspace(0.0, cs[-1], p + 1)
    idx = np.searchsorted(cs, targets)
    lo = np.clip(idx - 1, 0, cost.size)
    hi = np.clip(idx, 0, cost.size)
    pick = np.where(np.abs(cs[lo] - targets) <= np.abs(cs[hi] - targets), lo, hi)
    pick[0], pick[-1] = 0, cost.size
    cuts = np.maximum.accumulate(pick).astype(np.int64)
    cuts[-1] = cost.size
    return loads_from_cuts(cost, cuts)


STRATEGIES = ("block", "cyclic", "nzbalanced")


def efficiencies(cost, key, p):
    return {
        "block": eff(block_loads(cost, p)),
        "cyclic": eff(cyclic_loads(cost, p)),
        "nzbalanced": eff(nzbal_loads(cost, key, p)),
    }


#  main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csr", action="append", required=True)
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--lookup-sha", default=None)
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--cost", choices=["records", "work"], default="records",
                    help="records reproduces A5/E0; work uses D34's cost model")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    keep = load_keep(args.lookup, args.lookup_sha)

    emax = {p: expected_max_gaussian(p) for p in P_LIST}
    print("\nE[max of p standard normals] -- the term A5 approximates:")
    print(f"  {'p':>4} {'sqrt(2 ln p)':>13} {'true E[max]':>12} {'ratio':>7}")
    for p in P_LIST:
        a = np.sqrt(2.0 * np.log(p))
        print(f"  {p:4d} {a:13.4f} {emax[p]:12.4f} {a/emax[p]:7.4f}")
    print("  ratio > 1 means A5 systematically OVER-predicts imbalance,")
    print("  proportionally to CV/sqrt(m).")

    rows, flags = [], []

    for csr_dir in args.csr:
        name = os.path.basename(csr_dir.rstrip("/"))
        print(f"\n{'=' * 78}\n{name}   (cost = {args.cost}, B = {args.nboot})\n"
              f"{'=' * 78}")

        n, k = per_stay_counts(csr_dir, keep)
        nf = n.astype(np.float64)
        cost = nf if args.cost == "records" else nf + RHO_REF * k.astype(np.float64)
        cv_cost = cv(cost)

        # Canonical draw: the file order IS the permutation D8 froze.
        canon = {p: efficiencies(cost, nf, p) for p in P_LIST}

        # Bootstrap over random permutations.
        rng = np.random.default_rng(args.seed)
        boot = {p: {s: np.empty(args.nboot) for s in STRATEGIES} for p in P_LIST}
        idx = np.arange(n.size)
        for b in range(args.nboot):
            rng.shuffle(idx)
            c_perm, k_perm = cost[idx], nf[idx]
            for p in P_LIST:
                e = efficiencies(c_perm, k_perm, p)
                for s in STRATEGIES:
                    boot[p][s][b] = e[s]

        for p in P_LIST:
            if p > n.size:
                continue
            m = n.size / p
            base = cv_cost / np.sqrt(m)
            eff_asym = 1.0 / (1.0 + base * np.sqrt(2.0 * np.log(p)))
            eff_fin = 1.0 / (1.0 + base * emax[p])

            bb = boot[p]["block"]
            bmean, bsd = float(bb.mean()), float(bb.std(ddof=1))
            lo, hi = np.percentile(bb, [2.5, 97.5])
            pct = float((bb <= canon[p]["block"]).mean() * 100.0)

            print(f"\n  p = {p}   (m = {m:.1f} stays/worker, "
                  f"CV = {cv_cost:.4f})")
            print(f"    A5 asymptotic        {100*eff_asym:6.2f}%")
            print(f"    A5 finite-p          {100*eff_fin:6.2f}%   "
                  f"({100*(eff_fin-eff_asym):+5.2f} pp  <- finite-p bias)")
            print(f"    bootstrap mean       {100*bmean:6.2f}%   "
                  f"({100*(bmean-eff_fin):+5.2f} pp  <- non-normality)")
            print(f"    bootstrap 95% CI     [{100*lo:.2f}, {100*hi:.2f}]  "
                  f"sd {100*bsd:.2f} pp")
            print(f"    canonical draw       {100*canon[p]['block']:6.2f}%   "
                  f"({100*(canon[p]['block']-bmean):+5.2f} pp  <- the seed)"
                  f"   percentile {pct:.1f}")

            if pct < FLAG_LO or pct > FLAG_HI:
                flags.append((name, p, pct, canon[p]["block"], bmean))
                print(f"    ** FLAG: canonical order is at the {pct:.1f}th "
                      f"percentile of the permutation distribution")

            # block vs cyclic: identical in distribution, so their gap on one
            # draw is a direct read on single-draw noise.
            cb, cc = canon[p]["block"], canon[p]["cyclic"]
            mb = boot[p]["block"].mean()
            mc = boot[p]["cyclic"].mean()
            print(f"    block vs cyclic      canonical {100*cb:.2f} / "
                  f"{100*cc:.2f} = {100*(cb-cc):+.2f} pp   "
                  f"bootstrap means {100*mb:.2f} / {100*mc:.2f} = "
                  f"{100*(mb-mc):+.2f} pp")

            print(f"    seed sensitivity (sd across permutations):")
            for s in STRATEGIES:
                print(f"      {s:<12} {100*boot[p][s].std(ddof=1):5.2f} pp")

            for s in STRATEGIES:
                arr = boot[p][s]
                rows.append(dict(
                    variant=name, p=p, m=m, cost=args.cost, strategy=s,
                    cv_cost=cv_cost, n_boot=args.nboot,
                    canonical_eff=canon[p][s],
                    boot_mean=float(arr.mean()), boot_sd=float(arr.std(ddof=1)),
                    boot_p2_5=float(np.percentile(arr, 2.5)),
                    boot_p97_5=float(np.percentile(arr, 97.5)),
                    canonical_percentile=float((arr <= canon[p][s]).mean() * 100),
                    a5_asymptotic_eff=eff_asym if s == "block" else np.nan,
                    a5_finite_p_eff=eff_fin if s == "block" else np.nan,
                ))

    print(f"\n{'=' * 78}\nCALIBRATION VERDICT for the canonical order (D8)\n{'=' * 78}")
    if not flags:
        print(f"  No configuration puts the canonical draw outside the "
              f"{FLAG_LO:.0f}-{FLAG_HI:.0f}th percentile band.")
        print("  The frozen order is unremarkable. Record the percentiles and")
        print("  proceed -- measured imbalance is not an artefact of the seed.")
    else:
        print("  Canonical order is in the tail for:")
        for name, p, pct, c, b in flags:
            print(f"    {name:<22} p={p:<4} percentile {pct:5.1f}   "
                  f"canonical {100*c:.2f}% vs bootstrap mean {100*b:.2f}%")
        print("\n  Reseeding is free NOW and impossible after the sweep. If you")
        print("  reseed: do it ONCE, blind, and record the original percentile.")
        print("  Do not search over seeds -- that is the failure D30 prevents.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
