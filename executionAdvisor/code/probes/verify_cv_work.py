#!/usr/bin/env python3

import sys

import numpy as np
import pandas as pd

# variant -> (stays, records, cv_records)
REGISTER = {
    "W6":                 (67218,  15_313_613,  0.38),
    "W12":                (None,   26_006_481,  0.34),
    "W24":                (67218,  43_983_911,  0.33),
    "W48":                (None,   69_544_493,  0.38),
    "Wfull":              (67218,  156_865_224, 1.54),
    "sizematched_W6":     (67218,  15_313_613,  0.38),
    "sizematched_W12":    (39634,  15_313_887,  0.34),
    "sizematched_W24":    (23431,  15_313_737,  0.33),
    "sizematched_W48":    (14832,  15_314_138,  0.39),
    "sizematched_Wfull":  (6676,   15_315_189,  1.52),
    "shape_W24_trim0":    (67218,  None,        None),
    "shape_W24_trim1":    (66542,  None,        None),
    "shape_W24_trim5":    (63839,  None,        None),
    "shape_W24_trim10":   (60474,  None,        None),
}

SHAPE = {
    "shape_W24_trim0":  (609, 6643),
    "shape_W24_trim1":  (607, 1324),
    "shape_W24_trim5":  (597, 1060),
    "shape_W24_trim10": (584,  939),
}

A8_WFULL_P96 = {"block": 86.51 / 96, "cyclic": 80.63 / 96,
                "nzbalanced": 93.66 / 96, "greedy": 95.99 / 96}

CV_TOL = 0.01      # register CVs are quoted to 2 dp
EFF_TOL = 0.015    # 1.5 pp; A8 used a different cut rule


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    return bool(ok)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    S = pd.read_csv(sys.argv[1])
    P = pd.read_csv(sys.argv[2]) if len(sys.argv) > 2 else None

    print("=" * 78)
    print("COVERAGE")
    print("=" * 78)
    got = set(S.variant)
    want = set(REGISTER)
    print(f"  variants in output : {len(got)}")
    missing = sorted(want - got)
    if missing:
        print(f"  ABSENT from output : {', '.join(missing)}")
        print("    -> the sbatch skips directories that do not exist. If these")
        print("       variants were meant to exist, the CSR build is incomplete")
        print("       and A5/P1 cannot be restated for the size-matched family.")
    extra = sorted(got - want)
    if extra:
        print(f"  not in register    : {', '.join(extra)}")
    print()

    ok = True

    print("=" * 78)
    print("REPRODUCTION vs E0 (independent code path)")
    print("=" * 78)
    for _, r in S.iterrows():
        if r.variant not in REGISTER:
            continue
        stays, records, cvr = REGISTER[r.variant]
        if stays is not None:
            ok &= check(f"{r.variant} n_stays", int(r.n_stays) == stays,
                        f"{int(r.n_stays)} vs {stays}")
        if records is not None:
            ok &= check(f"{r.variant} n_records", int(r.n_records) == records,
                        f"{int(r.n_records):,} vs {records:,}")
        if cvr is not None:
            d = abs(r.cv_records - cvr)
            ok &= check(f"{r.variant} CV(records)", d <= CV_TOL,
                        f"{r.cv_records:.4f} vs {cvr:.2f} (d={d:.4f})")
    print()

    print("=" * 78)
    print("SHAPE FAMILY vs A6")
    print("=" * 78)
    for _, r in S.iterrows():
        if r.variant not in SHAPE:
            continue
        med, mx = SHAPE[r.variant]
        ok &= check(f"{r.variant} median", int(r.median_n) == med,
                    f"{int(r.median_n)} vs {med}")
        ok &= check(f"{r.variant} max", int(r.max_n) == mx,
                    f"{int(r.max_n)} vs {mx}")
    print()

    if P is not None:
        print("=" * 78)
        print("PARTITIONERS vs A8 (Wfull, p=96, scored on RECORDS)")
        print("=" * 78)
        sel = P[(P.variant == "Wfull") & (P.p == 96) & (P.scored_on == "records")]
        if sel.empty:
            print("  no Wfull p=96 records-scored row -- rerun with the patched"
                  " cv_work.py, which emits both scorings")
        else:
            row = sel.iloc[0]
            for kname, want_eff in A8_WFULL_P96.items():
                d = abs(row[kname] - want_eff)
                ok &= check(f"Wfull p=96 {kname}", d <= EFF_TOL,
                            f"{100*row[kname]:.2f}% vs A8 {100*want_eff:.2f}% "
                            f"(d={100*d:.2f} pp)")
        print()

        print("=" * 78)
        print("A8 AUDIT — what the records-based score cost")
        print("=" * 78)
        for v in sorted(P.variant.unique()):
            for p in sorted(P[P.variant == v].p.unique()):
                a = P[(P.variant == v) & (P.p == p) & (P.scored_on == "records")]
                b = P[(P.variant == v) & (P.p == p) & (P.scored_on == "work")]
                if a.empty or b.empty:
                    continue
                a, b = a.iloc[0], b.iloc[0]
                print(f"  {v:<20} p={p:<4} nzbal on records {100*a.nzbalanced:6.2f}%"
                      f"  on work {100*b.nzbalanced:6.2f}%"
                      f"  ({100*(b.nzbalanced-a.nzbalanced):+5.2f} pp)"
                      f"   wbal on work {100*b.wbalanced:6.2f}%"
                      f"   contig ceiling {100*b.contig_opt:6.2f}%")
        print()
        print("  READ AS:")
        print("    wbal >= nzbal on work, and both close to contig ceiling")
        print("      -> A8 survives; record-balancing is a good proxy for work.")
        print("    wbal - nzbal large -> nzbalanced balances the wrong quantity;")
        print("      carry a work-balanced partitioner as a separate treatment.")
        print("    contig ceiling - wbal large -> the cut rule is leaving")
        print("      efficiency on the table, not the strategy.")
        print()

    print("=" * 78)
    print("CV CORRECTION SUMMARY")
    print("=" * 78)
    cols = ["variant", "cv_records", "cv_work", "cv_ratio", "corr_n_h",
            "sd_h", "eff_p1_records", "eff_p1_work"]
    print(S[cols].to_string(index=False, float_format="%.4f"))
    print()
    if (S.cv_ratio > 1.05).any():
        print("  Some variants INFLATE CV -> P1 is optimistic there. Restate.")
    elif (S.cv_ratio < 0.95).all():
        print("  All variants COMPRESS CV -> P1 is uniformly conservative.")
        print("  Direction is safe, but the magnitude should be recorded so")
        print("  Day 12 does not fit a systematic bias as model error.")
    print()
    print("=" * 78)
    print("OVERALL: " + ("PASS" if ok else "FAIL — see above"))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
