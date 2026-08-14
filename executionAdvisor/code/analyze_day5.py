#!/usr/bin/env python3

import csv
import glob
import sys

import numpy as np

P = "/projects/sb2ea"


def summarize_sweep(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print(f"\n=== tile sweep: {path} ===\n  empty, skipped")
        return
    print(f"\n=== tile sweep: {path} ===")


    has_rep = "rep" in rows[0]
    if not has_rep:
        print("  NOTE: legacy single-shot schema, no `rep` column.")
        print("        No repetitions -> no IQR, and the flatness test")
        print("        below cannot be run. Superseded by the D29 sweep.")

    by = {}
    for r in rows:
        if has_rep and int(r["rep"]) == 0:   # platform.md sec 8: discard run 1
            continue
        by.setdefault((r["window"], int(r["tile"])), []).append(
            float(r["records_per_s"]))

    for w in sorted({k[0] for k in by}, reverse=True):
        tiles = sorted(t for (ww, t) in by if ww == w)
        print(f"\n  {w}   (F=160 -> tile buffer = tile x 1280 B)")
        print(f"  {'tile':>7} {'buf MB':>8} {'median':>9} {'IQR':>7} "
              f"{'IQR%':>6}  {'n':>2}")
        meds, within = [], []
        for t in tiles:
            v = np.array(by[(w, t)])
            med = float(np.median(v))
            q1, q3 = np.percentile(v, [25, 75])
            meds.append(med)
            within.append(q3 - q1)
            print(f"  {t:>7} {t*1280/1e6:>8.1f} {med:>9.1f} "
                  f"{q3-q1:>7.1f} {100*(q3-q1)/med:>5.1f}% {len(v):>3}")

        meds = np.array(meds)
        rng = meds.max() - meds.min()
        b1, b3 = np.percentile(meds, [25, 75])
        between_iqr = b3 - b1
        typ_within = float(np.median(within))

        print(f"\n    between-tile range  : {rng:6.1f} M rec/s "
              f"({100*rng/meds.mean():.1f}% of mean)")
        print(f"    between-tile IQR    : {between_iqr:6.1f} M rec/s")
        if not has_rep:
            print("    within-tile IQR     :    n/a (single-shot schema)")
            continue
        print(f"    within-tile IQR     : {typ_within:6.1f} M rec/s "
              f"(median across tiles)")

        rho = float(np.corrcoef(np.log2(tiles), meds)[0, 1])
        ratio = between_iqr / typ_within if typ_within > 0 else float("inf")
        flat = ratio < 2.0
        print(f"    IQR ratio (between/within) : {ratio:5.2f}  "
              f"(< 2.0 = flat)")
        print(f"    rho(log2 tile, median)     : {rho:+.2f}  "
              f"(diagnostic only)")

        if flat:
            print("    -> FLAT within noise. P2 unobservable, not "
                  "falsified (D29 sec 4).")
        else:
            best = tiles[int(np.argmax(meds))]
            worst = tiles[int(np.argmin(meds))]
            print(f"    -> STRUCTURE: best tile={best}, worst tile={worst}, "
                  f"rho={rho:+.2f}")
            print("       If throughput falls as tile grows AND the knee is "
                  "near C~15,600")
            print("       (20 MB L3, Broadwell), that is P2 and D29 sec 4 "
                  "needs revising.")


def check_features(window="W24"):
    print(f"\n=== feature matrix cross-check: {window} ===")
    try:
        spec = {r["variable"]: (int(r["slot"]), float(r["cov"]))
                for r in csv.DictReader(open(f"{P}/manifest/feature_spec.csv"))}
    except FileNotFoundError:
        print("  feature_spec.csv not found; skipping")
        return
    F = len(spec) * 8
    X = np.fromfile(f"{P}/features/{window}_serial.f64",
                    dtype="<f8").reshape(-1, F)
    print(f"  shape {X.shape}   (expect 67218 x 160)")

    print(f"\n  {'variable':<15}{'slot':>5}{'spec cov':>10}"
          f"{'kernel':>10}{'delta':>9}")
    worst = 0.0
    for v, (s, cov) in sorted(spec.items(), key=lambda kv: kv[1][0]):
        obs = float((X[:, s * 8] > 0).mean())
        worst = max(worst, abs(obs - cov))
        flag = "  <-- CHECK" if abs(obs - cov) > 0.02 else ""
        print(f"  {v:<15}{s:>5}{cov:>10.4f}{obs:>10.4f}"
              f"{obs-cov:>9.4f}{flag}")
    print(f"\n  max |delta| : {worst:.4f}   "
          f"({'OK' if worst < 0.02 else 'DIVERGED'})")


    cnt = X[:, 0::8]
    bad_mean = int((~np.isnan(X[:, 1::8]) & (cnt == 0)).sum())
    bad_cnt = int(np.isnan(cnt).sum())
    n1 = (cnt == 1)
    bad_std = int((n1 & (X[:, 4::8] != 0.0)).sum())
    bad_slope = int((n1 & ~np.isnan(X[:, 7::8])).sum())
    print(f"\n  D23 sentinel checks")
    print(f"    count==0 with non-NaN mean   : {bad_mean}  (must be 0)")
    print(f"    NaN in a count column        : {bad_cnt}  (must be 0)")
    print(f"    count==1 with std != 0       : {bad_std}  (must be 0)")
    print(f"    count==1 with non-NaN slope  : {bad_slope}  (must be 0)")

    fin = np.isfinite(X)
    print(f"\n  finite cells : {int(fin.sum()):,} / {X.size:,} "
          f"({100*fin.mean():.1f}%)")
    print(f"  NaN cells    : {int(np.isnan(X).sum()):,}")
    if np.isinf(X).any():
        print(f"  !! INF cells : {int(np.isinf(X).sum()):,}  <-- investigate")


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(
        glob.glob(f"{P}/results/kernel/tile_sweep_*.csv"))
    for p in paths:
        summarize_sweep(p)
    for w in ("W24", "Wfull"):
        try:
            check_features(w)
        except FileNotFoundError as e:
            print(f"\n  skipped {w}: {e}")
