#!/usr/bin/env python3

import argparse
import hashlib
import heapq
import os
import sys

import numpy as np
import pandas as pd

RHO_SWEEP = [3.0, 4.0, 5.31, 6.5, 8.0]
RHO_REF = 5.31
P_LIST = [8, 16, 96]


def load_keep(lookup_path, sha_path=None):
    if sha_path and os.path.exists(sha_path):
        h = hashlib.sha256(open(lookup_path, "rb").read()).hexdigest()
        want = open(sha_path).read().split()[0]
        if h != want:
            sys.exit(f"FATAL: kernel_lookup.csv sha256 {h[:12]} != manifest "
                     f"{want[:12]}.")
        print(f"lookup sha256 verified: {h[:12]}")

    lk = pd.read_csv(lookup_path)
    for c in ("dense_id", "slot"):
        if c not in lk.columns:
            sys.exit(f"lookup needs {c}; got {list(lk.columns)}")
    K = int(lk.dense_id.max()) + 1
    keep = np.zeros(K, dtype=np.uint8)
    keep[lk.dense_id.to_numpy()] = (lk.slot.to_numpy() >= 0).astype(np.uint8)
    print(f"lookup: K={K}, whitelisted dense ids={int(keep.sum())}, "
          f"slots={lk.loc[lk.slot >= 0, 'slot'].nunique()}")
    return keep


def per_stay_counts(csr_dir, keep, target_records=4_000_000):
    off = np.fromfile(os.path.join(csr_dir, "offsets.i64"), dtype=np.int64)
    item = np.memmap(os.path.join(csr_dir, "itemid.i32"), dtype=np.int32, mode="r")

    n_stays = off.size - 1
    n_records = int(off[-1])
    if item.size != n_records:
        sys.exit(f"{csr_dir}: itemid has {item.size}, offsets say {n_records}")

    n = np.diff(off).astype(np.int64)
    k = np.zeros(n_stays, dtype=np.int64)
    chunk = max(1, int(target_records / max(1.0, n_records / max(1, n_stays))))

    for start in range(0, n_stays, chunk):
        end = min(start + chunk, n_stays)
        lo, hi = int(off[start]), int(off[end])
        if hi == lo:
            continue
        ids = np.asarray(item[lo:hi])
        if ids.max(initial=0) >= keep.size:
            sys.exit(f"{csr_dir}: dense_id {int(ids.max())} >= K={keep.size}")
        seg = keep[ids]
        cs = np.empty(seg.size + 1, dtype=np.int64)
        cs[0] = 0
        np.cumsum(seg, out=cs[1:])
        loc = off[start:end + 1] - lo
        k[start:end] = cs[loc[1:]] - cs[loc[:-1]]
        del ids, seg, cs

    return n, k


#  stats
def cv(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.mean()
    return float(x.std(ddof=0) / m) if m > 0 else float("nan")


def a5_efficiency(cv_val, m, p):
    imb = (cv_val / np.sqrt(m)) * np.sqrt(2.0 * np.log(p))
    return 1.0 / (1.0 + imb), imb


#  partitions
def eff(loads):
    loads = np.asarray(loads, dtype=np.float64)
    mx = loads.max()
    return float(loads.mean() / mx) if mx > 0 else float("nan")


def loads_from_cuts(cost, cuts):
    cs = np.concatenate(([0.0], np.cumsum(cost)))
    return cs[cuts[1:]] - cs[cuts[:-1]]


def part_block(cost, p):
    cuts = np.linspace(0, cost.size, p + 1).astype(np.int64)
    return loads_from_cuts(cost, cuts)


def part_cyclic(cost, p):
    return np.array([cost[j::p].sum() for j in range(p)])


def part_contig_balanced(cost, key, p):
    """Contiguous blocks with equal cumulative KEY.

    Picks the NEARER of the two candidate cut points. searchsorted alone
    always takes the boundary before the target, which with a heavy tail
    dumps a straddling giant stay wholesale into the next block and makes
    the result depend on rounding luck.
    """
    cs = np.concatenate(([0.0], np.cumsum(key.astype(np.float64))))
    targets = np.linspace(0.0, cs[-1], p + 1)
    idx = np.searchsorted(cs, targets)
    lo = np.clip(idx - 1, 0, cost.size)
    hi = np.clip(idx, 0, cost.size)
    pick = np.where(np.abs(cs[lo] - targets) <= np.abs(cs[hi] - targets), lo, hi)
    pick[0], pick[-1] = 0, cost.size
    cuts = np.maximum.accumulate(pick).astype(np.int64)
    cuts[-1] = cost.size
    return loads_from_cuts(cost, cuts)


def part_contig_optimal(cost, p):
    """Optimal contiguous partition: binary search on the max block load.

    This is the ceiling for any contiguous strategy, which is the right
    reference for nzbalanced. greedy/LPT is not, because it is
    non-contiguous and gives up the cache and prefetch behaviour that
    A8 promoted nzbalanced for preserving.
    """
    cost = np.asarray(cost, dtype=np.float64)

    def feasible(cap):
        blocks, cur = 1, 0.0
        for x in cost:
            if x > cap:
                return False
            if cur + x > cap:
                blocks += 1
                cur = x
                if blocks > p:
                    return False
            else:
                cur += x
        return True

    lo, hi = max(cost.max(), cost.sum() / p), cost.sum()
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return np.array([hi] + [cost.sum() / p] * (p - 1))


def part_greedy(cost, p):
    heap = [(0.0, j) for j in range(p)]
    heapq.heapify(heap)
    for x in np.sort(cost)[::-1]:
        load, j = heapq.heappop(heap)
        heapq.heappush(heap, (load + float(x), j))
    return np.array([load for load, _ in heap])


def partition_row(cost, n, w, p):
    """All strategies scored against `cost`. Cuts for nzbal use records,
    cuts for wbal use work, regardless of which cost is being scored."""
    return dict(
        block=eff(part_block(cost, p)),
        cyclic=eff(part_cyclic(cost, p)),
        nzbalanced=eff(part_contig_balanced(cost, n.astype(np.float64), p)),
        wbalanced=eff(part_contig_balanced(cost, w, p)),
        contig_opt=float(cost.mean() * cost.size / p / part_contig_optimal(cost, p)[0]),
        greedy=eff(part_greedy(cost, p)),
    )


#  main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csr", action="append", required=True)
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--lookup-sha", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--p", type=int, default=96)
    args = ap.parse_args()

    keep = load_keep(args.lookup, args.lookup_sha)
    rows, part_rows = [], []

    for csr_dir in args.csr:
        name = os.path.basename(csr_dir.rstrip("/"))
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")

        n, k = per_stay_counts(csr_dir, keep)
        nz = n > 0
        h = np.zeros(n.size, dtype=np.float64)
        h[nz] = k[nz] / n[nz]

        cv_n, cv_k = cv(n), cv(k)
        hit = float(k.sum()) / float(n.sum())
        c_nk = float(np.corrcoef(n[nz], k[nz])[0, 1]) if nz.sum() > 1 else np.nan
        c_nh = float(np.corrcoef(n[nz], h[nz])[0, 1]) if nz.sum() > 1 else np.nan

        print(f"  stays={n.size:,}  records={int(n.sum()):,}  "
              f"median={int(np.median(n))}  max={int(n.max())}  "
              f"max/median={n.max()/max(1,np.median(n)):.1f}")
        print(f"  pooled hit rate={hit:.4f}  mean h={h[nz].mean():.4f}  "
              f"sd h={h[nz].std():.4f}")
        print(f"  CV(records)={cv_n:.4f}  CV(whitelisted)={cv_k:.4f}  "
              f"corr(n,k)={c_nk:+.4f}  corr(n,h)={c_nh:+.4f}")

        print(f"\n  CV(work) sensitivity to rho = b/a:")
        print(f"    {'rho':>6} {'CV(w)':>9} {'CV(w)/CV(n)':>13}")
        cvw_ref = None
        for rho in RHO_SWEEP:
            wv = n.astype(np.float64) + rho * k.astype(np.float64)
            c = cv(wv)
            if abs(rho - RHO_REF) < 1e-9:
                cvw_ref = c
            print(f"    {rho:6.2f} {c:9.4f} {c/cv_n:13.4f}"
                  f"{'  <- D29' if abs(rho-RHO_REF) < 1e-9 else ''}")

        m = n.size / args.p
        eff_n, imb_n = a5_efficiency(cv_n, m, args.p)
        eff_w, imb_w = a5_efficiency(cvw_ref, m, args.p)
        print(f"\n  A5 at p={args.p}  (m = {m:.1f} stays/worker)")
        print(f"    on CV(records): imbalance {100*imb_n:5.2f}%  "
              f"efficiency {100*eff_n:5.1f}%   <- P1 as written")
        print(f"    on CV(work)   : imbalance {100*imb_w:5.2f}%  "
              f"efficiency {100*eff_w:5.1f}%   <- corrected")
        print(f"    shift         : {100*(eff_w-eff_n):+5.2f} pp")

        rows.append(dict(variant=name, n_stays=n.size, n_records=int(n.sum()),
                         median_n=int(np.median(n)), max_n=int(n.max()),
                         hit_rate=hit, cv_records=cv_n, cv_whitelisted=cv_k,
                         cv_work=cvw_ref, cv_ratio=cvw_ref / cv_n,
                         mean_h=float(h[nz].mean()), sd_h=float(h[nz].std()),
                         corr_n_k=c_nk, corr_n_h=c_nh, p=args.p, m=m,
                         eff_p1_records=eff_n, eff_p1_work=eff_w))

        w = n.astype(np.float64) + RHO_REF * k.astype(np.float64)
        nf = n.astype(np.float64)
        for label, cost in (("records", nf), ("work", w)):
            print(f"\n  Partition efficiency, scored on {label.upper()}:")
            print(f"    {'p':>4} {'block':>8} {'cyclic':>8} {'nzbal':>8} "
                  f"{'wbal':>8} {'contigopt':>10} {'greedy':>8}")
            for p in P_LIST:
                if p > n.size:
                    continue
                r = partition_row(cost, n, w, p)
                print(f"    {p:4d} {100*r['block']:7.2f}% {100*r['cyclic']:7.2f}% "
                      f"{100*r['nzbalanced']:7.2f}% {100*r['wbalanced']:7.2f}% "
                      f"{100*r['contig_opt']:9.2f}% {100*r['greedy']:7.2f}%")
                part_rows.append(dict(variant=name, p=p, scored_on=label,
                                      rho=RHO_REF, **r))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        pb = args.out.replace(".csv", "_partitions.csv")
        pd.DataFrame(part_rows).to_csv(pb, index=False)
        print(f"\nwrote {args.out}\nwrote {pb}")


if __name__ == "__main__":
    main()
