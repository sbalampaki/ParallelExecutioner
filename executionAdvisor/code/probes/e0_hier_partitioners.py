#!/usr/bin/env python3
"""Flat vs hierarchical for contig_opt and nzbalanced. Replicates
pt__balanced_cuts and the binary-search optimal contiguous partition."""
import numpy as np, sys

off = np.fromfile(sys.argv[1] if len(sys.argv)>1 else
                  '/projects/sb2ea/csr/Wfull/offsets.i64', dtype=np.int64)
rec = np.diff(off).astype(np.float64)
N   = rec.size
RHO = 5.31

def cumsum(w): return np.concatenate([[0.0], np.cumsum(w)])

def balanced_cuts(w, p):                       # pt__balanced_cuts
    cs = cumsum(w); n = w.size; cuts = np.zeros(p+1, dtype=np.int64)
    for j in range(p+1):
        t   = cs[n] * j / p
        idx = int(np.searchsorted(cs, t, side='left'))
        lo, hi = max(idx-1,0), min(idx,n)
        cuts[j] = lo if abs(cs[lo]-t) <= abs(cs[hi]-t) else hi
    cuts[0] = 0; cuts[p] = n
    for j in range(1,p+1):
        if cuts[j] < cuts[j-1]: cuts[j] = cuts[j-1]
    cuts[p] = n
    return cuts

def opt_cuts(w, p):                            # binary search on max load
    cs = cumsum(w); n = w.size
    def npieces(L):
        i = k = 0
        while i < n:
            j = int(np.searchsorted(cs, cs[i] + L, side='right')) - 1
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
        j = int(np.searchsorted(cs, cs[i] + hi, side='right')) - 1
        cuts.append(j); i = j
    while len(cuts)-1 < p: cuts.append(n)
    cuts[-1] = n
    return np.array(cuts, dtype=np.int64)

def hier2(w, R, T, f_rank, f_thread):
    rb = f_rank(w, R); out = [0]
    for r in range(R):
        lo, hi = int(rb[r]), int(rb[r+1])
        sub = f_thread(w[lo:hi], T) + lo if hi > lo else np.full(T+1, lo)
        out.extend(int(x) for x in sub[1:])
    return np.array(out, dtype=np.int64)

def imb(w, b):
    cs = cumsum(w); L = cs[b[1:]] - cs[b[:-1]]
    return L.max() / (cs[-1]/(len(b)-1))

MEAS = {'contig_opt': 1.0241, 'nzbalanced': 1.0124}
for score, w in [('records', rec), ('work', rec + RHO)]:
    print(f'\n=== scored_on = {score} ===')
    print(f"{'partitioner':<12}{'p':>5}{'flat':>11}{'hier':>11}{'measured':>11}{'hier err':>10}")
    for nm, f in [('contig_opt', opt_cuts), ('nzbalanced', balanced_cuts)]:
        for p, R, T in [(16,2,8), (80,10,8)]:
            fl = imb(w, f(w, p)); hi = imb(w, hier(w, R, T, f))
            m  = MEAS[nm] if p == 80 else float('nan')
            e  = 100*(hi/m - 1) if p == 80 else float('nan')
            print(f'{nm:<12}{p:>5}{fl:>11.6f}{hi:>11.6f}{m:>11.4f}{e:>9.3f}%')
