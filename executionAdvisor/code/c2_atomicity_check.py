#!/usr/bin/env python3

import argparse, glob, os, sys
import numpy as np
import pandas as pd

pd.set_option('display.width', 220)

ap = argparse.ArgumentParser()
ap.add_argument('--offsets', default='/projects/sb2ea/csr/Wfull/offsets.i64')
ap.add_argument('--curve',   default=None)
a = ap.parse_args()

#    floor from offsets
print("=" * 70)
print("C2.1  cohort constants, recomputed from the offsets file")
print("=" * 70)
if not os.path.exists(a.offsets):
    hits = glob.glob('/projects/**/offsets.i64', recursive=True) + \
           glob.glob(os.path.expanduser('~') + '/**/offsets.i64', recursive=True)
    if not hits:
        sys.exit(f"offsets not found at {a.offsets}; pass --offsets PATH")
    print(f"[note] {a.offsets} missing; using {hits[0]}")
    a.offsets = hits[0]

off = np.fromfile(a.offsets, dtype=np.int64)
n = np.diff(off)
N, mx, ns = int(n.sum()), int(n.max()), len(n)
print(f"file      : {a.offsets}")
print(f"n_stays   : {ns:,}          paper: 67,218      {'OK' if ns==67218 else '*** DIFFERS ***'}")
print(f"N         : {N:,}     paper: 156,865,224 {'OK' if N==156865224 else '*** DIFFERS ***'}")
print(f"max(n_i)  : {mx:,}         paper: 126,261     {'OK' if mx==126261 else '*** DIFFERS ***'}")
print(f"p_atom    : {N/mx:.6f}    paper: 1242.4")

FAR = [8192, 16384, 32768]
PAPER = {8192: (6.5938, 6.5937), 16384: (13.1875, 13.1876), 32768: (26.3750, 26.3744)}
print(f"\n{'p':>7} {'exact floor':>16} {'paper floor':>12} {'paper nz':>12} {'nz - floor':>13}")
for p in FAR:
    fl = p * mx / N
    pf, pn = PAPER[p]
    print(f"{p:>7} {fl:>16.9f} {pf:>12} {pn:>12} {pn-fl:>+13.2e}"
          + ("   *** BELOW ***" if pn < fl else ""))

#    curve at full precision
print("\n" + "=" * 70)
print("C2.2  what the curve actually computed")
print("=" * 70)
curve = a.curve
if curve is None:
    hits = sorted(glob.glob('**/pstar_curve.csv', recursive=True)) + \
           sorted(glob.glob('/projects/**/pstar_curve.csv', recursive=True)) + \
           sorted(glob.glob(os.path.expanduser('~') + '/**/pstar_curve.csv', recursive=True))
    if not hits:
        print("pstar_curve.csv not found. Either pass --curve PATH, or")
        print("regenerate:\n"
              f"  python3 pstar_curve.py --offsets {a.offsets} \\\n"
              "      --validation e0_validation.csv --out pstar_curve.csv")
        sys.exit(0)
    curve = hits[0]
print(f"file: {curve}")
c = pd.read_csv(curve)
print(f"{len(c):,} rows; columns: {list(c.columns)}\n")

pcol = 'p' if 'p' in c.columns else None
icol = next((x for x in ('imbalance', 'imb', 'I') if x in c.columns), None)
if pcol is None or icol is None:
    sys.exit(f"could not find p / imbalance columns in {list(c.columns)}")

far = c[c[pcol].isin(FAR)].copy()
if far.empty:
    print(f"No rows at p in {FAR}. p values present (tail): "
          f"{sorted(c[pcol].unique())[-12:]}")
    sys.exit(0)

far['floor'] = far[pcol] * mx / N
far['slack'] = far[icol] - far['floor']
show = [x for x in (pcol, 'n_ranks', 'n_threads', 'partitioner', icol, 'floor', 'slack')
        if x in far.columns]
with pd.option_context('display.float_format', lambda v: f'{v:.9f}'):
    print(far[show].to_string(index=False))

viol = far[far['slack'] < 0]
print(f"\nrows below the floor: {len(viol)} of {len(far)}")


print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
if ns != 67218 or N != 156865224 or mx != 126261:
    print("(a) COHORT CONSTANTS DIFFER from the paper. This is the cause.")
    print("    Correct N/max/p_atom in Sections V-A, X-E and regime_calculator.py,")
    print("    then re-derive every far-field number.")
elif len(viol) == 0:
    print("(b) The curve satisfies the bound at full precision.")
    print("    26.3744 in the paper is a TRANSCRIPTION ERROR.")
    print("    Fix: report floor and achieved to six decimals with a slack")
    print("    column, so 'on the floor' is visible rather than asserted.")
else:
    print("(c) The curve genuinely violates an exact lower bound -> BUG.")
    print("    Check whether `imbalance` is normalised by N/p or by something")
    print("    else, and whether the two-stage cut uses the same max(n_i).")
    print("    Then regenerate and re-derive the far-field ceilings")
    print("    (10.58 / 5.29 / 3.69 %), the p=1552 peak, and Regime III's")
    print("    boundary at p_atom.")
