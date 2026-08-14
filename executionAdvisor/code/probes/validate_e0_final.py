#!/usr/bin/env python3

import numpy as np, pandas as pd, argparse, os

RHO = 5.31

def cumsum(w): return np.concatenate([[0.0], np.cumsum(w)])

def linspace_cuts(w, p):                      # pt__linspace_cuts
    n = w.size; step = np.float64(n)/np.float64(p)
    c = (np.arange(p, dtype=np.int64)*step).astype(np.int64)
    c = np.append(c, np.int64(n))
    for j in range(1, p+1):
        if c[j] < c[j-1]: c[j] = c[j-1]
    return c

def balanced_cuts(w, p):                      # pt__balanced_cuts
    cs = cumsum(w); n = w.size; cuts = np.zeros(p+1, dtype=np.int64)
    for j in range(p+1):
        t = cs[n]*j/p
        idx = int(np.searchsorted(cs, t, side='left'))
        lo, hi = max(idx-1, 0), min(idx, n)
        cuts[j] = lo if abs(cs[lo]-t) <= abs(cs[hi]-t) else hi
    cuts[0] = 0; cuts[p] = n
    for j in range(1, p+1):
        if cuts[j] < cuts[j-1]: cuts[j] = cuts[j-1]
    cuts[p] = n
    return cuts

def opt_cuts(w, p):                           # binary search on max block load
    cs = cumsum(w); n = w.size
    def npieces(L):
        i = k = 0
        while i < n:
            j = int(np.searchsorted(cs, cs[i]+L, side='right')) - 1
            if j <= i: return 10**9
            k += 1; i = j
            if k > p: return 10**9
        return k
    lo, hi = float(w.max()), float(cs[-1])
    for _ in range(100):
        mid = 0.5*(lo+hi)
        if npieces(mid) <= p: hi = mid
        else: lo = mid
    cuts = [0]; i = 0
    while i < n and len(cuts)-1 < p:
        j = int(np.searchsorted(cs, cs[i]+hi, side='right')) - 1
        cuts.append(j); i = j
    while len(cuts)-1 < p: cuts.append(n)
    cuts[-1] = n
    return np.array(cuts, dtype=np.int64)

def hier(w, R, T, f):
    """Two-level cut: ranks first, then threads within a rank
    (kernel_mpi.c:254-262)."""
    rb = f(w, R); out = [0]
    for r in range(R):
        lo, hi = int(rb[r]), int(rb[r+1])
        sub = f(w[lo:hi], T) + lo if hi > lo else np.full(T+1, lo)
        out.extend(int(x) for x in sub[1:])
    return np.array(out, dtype=np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csr',    default='/projects/sb2ea/csr/Wfull')
    ap.add_argument('--lookup', default='/projects/sb2ea/manifest/kernel_lookup.csv')
    ap.add_argument('--out',    default='/projects/sb2ea/results/day6/e0_validation.csv')
    a = ap.parse_args()

    off    = np.fromfile(os.path.join(a.csr, 'offsets.i64'), dtype=np.int64)
    itemid = np.fromfile(os.path.join(a.csr, 'itemid.i32'),  dtype=np.int32)
    rec    = np.diff(off).astype(np.float64)

    lk = pd.read_csv(a.lookup)
    K  = int(itemid.max())+1
    keep = np.zeros(K, dtype=bool)
    d, s = lk.dense_id.to_numpy(), lk.slot.to_numpy(); ok = d < K
    keep[d[ok]] = (s[ok] >= 0)
    kk = np.add.reduceat(keep[itemid].astype(np.int64), off[:-1])
    kk[np.diff(off) == 0] = 0
    work = rec + RHO*kk

    print(f'n_stays={rec.size}  n_records={int(rec.sum())}')
    print(f'hit_rate={keep[itemid].mean():.6f}   (cv_work.csv: 0.365815)')
    print(f'corr(n,k)={np.corrcoef(rec,kk)[0,1]:.6f}  (cv_work.csv: 0.972386)')

    CS = cumsum(rec)
    def imb_rec(b):
        L = CS[b[1:]] - CS[b[:-1]]
        return L.max()/(CS[-1]/(len(b)-1))

    # (partitioner, cut rule, cut key, [(p, R, T, measured)])
    CASES = [
      ('block', linspace_cuts, 'index', [(16,2,8,1.049428), (32,4,8,1.073144),
                                     (64,8,8,1.108724), (80,10,8,1.129385)]),
      ('nzbalanced', balanced_cuts, 'records', [(16,2,8,1.000955), (80,10,8,1.012361)]),
      ('contig_opt', opt_cuts,      'work',    [(16,2,8,1.009385), (80,10,8,1.024051)]),
      ('wbalanced',  balanced_cuts, 'work',    [(16,2,8,1.009385), (80,10,8,1.030682)]),
    ]
    rows = []
    print(f"\n{'partitioner':<12}{'key':<9}{'p':>4}{'flat':>11}{'hier':>11}"
          f"{'measured':>11}{'err %':>9}")
    for nm, f, keyname, pts in CASES:
        key = rec if keyname in ('records','index') else work
        for p, R, T, m in pts:
            fl, hi = imb_rec(f(key, p)), imb_rec(hier(key, R, T, f))
            err = 100*(hi/m - 1)
            print(f'{nm:<12}{keyname:<9}{p:>4}{fl:>11.6f}{hi:>11.6f}'
                  f'{m:>11.6f}{err:>8.3f}%')
            rows.append(dict(partitioner=nm, cut_key=keyname, p=p,
                             n_ranks=R, n_threads=T,
                             predicted_flat=fl, predicted_hier=hi,
                             measured=m, error_pct=err))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f'\nwrote {a.out}   max |err| = {df.error_pct.abs().max():.4f}%')
    assert df.error_pct.abs().max() < 0.01, 'E0 validation regressed'

if __name__ == '__main__':
    main()
