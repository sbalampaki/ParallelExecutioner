#!/usr/bin/env python3

import argparse
import sys

import numpy as np
import pandas as pd

STRATEGIES = ["block", "cyclic", "nzbalanced"]
XCHECK_TOL = 1e-9        # same algorithm, so only float-assoc noise is allowed
FLAG_LO, FLAG_HI = 5.0, 95.0


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    return bool(ok)


def cross_check(boot, parts, cost_label):
    """Check 1: canonical_eff vs cv_work_partitions.csv."""
    p = parts[parts.scored_on == cost_label]
    if p.empty:
        print(f"  no scored_on=='{cost_label}' rows in the partitions file")
        return True, 0
    long = p.melt(id_vars=["variant", "p"], value_vars=STRATEGIES,
                  var_name="strategy", value_name="cv_work_eff")
    m = boot.merge(long, on=["variant", "p", "strategy"], how="inner")
    if m.empty:
        print("  no overlapping (variant, p, strategy) rows")
        return False, 0
    m["absdiff"] = (m.canonical_eff - m.cv_work_eff).abs()
    bad = m[m.absdiff > XCHECK_TOL]
    ok = bad.empty
    print(f"  {len(m)} comparisons, max |diff| = {m.absdiff.max():.2e}")
    if not ok:
        print("  MISMATCHES:")
        for _, r in bad.head(12).iterrows():
            print(f"    {r.variant:<22} p={int(r.p):<4} {r.strategy:<11} "
                  f"boot {r.canonical_eff:.6f} vs cv_work {r.cv_work_eff:.6f}")
    return ok, len(m)


def machinery(boot):
    """Check 2: block and cyclic bootstrap means must converge."""
    b = boot[boot.strategy == "block"].set_index(["variant", "p"])
    c = boot[boot.strategy == "cyclic"].set_index(["variant", "p"])
    j = b[["boot_mean", "boot_sd", "n_boot", "canonical_eff"]].join(
        c[["boot_mean", "boot_sd", "canonical_eff"]],
        lsuffix="_b", rsuffix="_c", how="inner")
    j["mean_gap_pp"] = 100 * (j.boot_mean_b - j.boot_mean_c)
    j["canon_gap_pp"] = 100 * (j.canonical_eff_b - j.canonical_eff_c)
    # Paired over the same permutations, so the SE of the difference is
    # bounded above by the single-strategy SE.
    j["se_pp"] = 100 * j.boot_sd_b / np.sqrt(j.n_boot)
    j["ratio"] = j.mean_gap_pp.abs() / j.se_pp.replace(0, np.nan)
    ok = bool((j.mean_gap_pp.abs() < 0.5).all())
    print(f"  bootstrap-mean gap |block - cyclic|: "
          f"max {j.mean_gap_pp.abs().max():.3f} pp "
          f"(should be ~0; they are the same distribution)")
    print(f"  canonical-draw gap |block - cyclic|: "
          f"max {j.canon_gap_pp.abs().max():.3f} pp "
          f"(this is single-draw NOISE, not a strategy effect)")
    worst = j.reindex(j.canon_gap_pp.abs().sort_values(ascending=False).index)
    print("\n  largest canonical gaps -- each is pure noise:")
    print(f"    {'variant':<22} {'p':>4} {'canon gap':>10} {'boot gap':>10}")
    for (v, p), r in worst.head(6).iterrows():
        print(f"    {v:<22} {int(p):>4} {r.canon_gap_pp:>9.2f}p "
              f"{r.mean_gap_pp:>9.2f}p")
    return ok, j


def decomposition(boot):
    """Check 3: which A5 form predicts the bootstrap mean better?"""
    b = boot[(boot.strategy == "block") & boot.a5_asymptotic_eff.notna()].copy()
    if b.empty:
        print("  no block rows with A5 columns")
        return True
    b["err_asym_pp"] = 100 * (b.boot_mean - b.a5_asymptotic_eff)
    b["err_fin_pp"] = 100 * (b.boot_mean - b.a5_finite_p_eff)
    b["seed_pp"] = 100 * (b.canonical_eff - b.boot_mean)

    rms_a = float(np.sqrt((b.err_asym_pp ** 2).mean()))
    rms_f = float(np.sqrt((b.err_fin_pp ** 2).mean()))
    print(f"  RMS error against bootstrap mean:")
    print(f"    A5 asymptotic  {rms_a:6.3f} pp")
    print(f"    A5 finite-p    {rms_f:6.3f} pp")
    better = rms_f < rms_a
    print(f"    -> finite-p correction {'HELPS' if better else 'DOES NOT HELP'}"
          f" ({100*(1-rms_f/rms_a):+.0f}% RMS)")

    print(f"\n  decomposition by m (stays per worker), block, sorted by m:")
    print(f"    {'variant':<22} {'p':>4} {'m':>8} {'CV':>7} "
          f"{'finite-p':>9} {'non-norm':>9} {'seed':>8} {'pctile':>7}")
    for _, r in b.sort_values("m").iterrows():
        fin = 100 * (r.a5_finite_p_eff - r.a5_asymptotic_eff)
        print(f"    {r.variant:<22} {int(r.p):>4} {r.m:>8.1f} {r.cv_cost:>7.3f} "
              f"{fin:>+8.2f}p {r.err_fin_pp:>+8.2f}p {r.seed_pp:>+7.2f}p "
              f"{r.canonical_percentile:>7.1f}")
    print("\n    finite-p  = A5 asymptotic -> A5 finite-p   (always positive)")
    print("    non-norm  = A5 finite-p   -> bootstrap mean (skew of block sums)")
    print("    seed      = bootstrap mean -> canonical draw (the frozen order)")
    return True


def calibration(boot, label):
    """Check 4: is the canonical order an outlier?"""
    b = boot[boot.strategy == "block"]
    flagged = b[(b.canonical_percentile < FLAG_LO)
                | (b.canonical_percentile > FLAG_HI)]
    print(f"  block, {len(b)} configurations, percentile band "
          f"[{FLAG_LO:.0f}, {FLAG_HI:.0f}]")
    print(f"  median percentile {b.canonical_percentile.median():.1f}, "
          f"range [{b.canonical_percentile.min():.1f}, "
          f"{b.canonical_percentile.max():.1f}]")
    if flagged.empty:
        print(f"  -> canonical order is UNREMARKABLE for {label}. Record the")
        print("     percentiles; measured imbalance is not a seed artefact.")
        return True
    print("  -> canonical order is in the tail for:")
    for _, r in flagged.iterrows():
        print(f"     {r.variant:<22} p={int(r.p):<4} "
              f"percentile {r.canonical_percentile:5.1f}  "
              f"canonical {100*r.canonical_eff:.2f}% vs "
              f"boot mean {100*r.boot_mean:.2f}%")
    return False


def seed_sensitivity(boot, label):
    print(f"  sd of efficiency across permutations ({label}), pp:")
    t = (boot.groupby(["p", "strategy"]).boot_sd.agg(["median", "max"]) * 100)
    print(t.to_string(float_format="%.2f"))
    blk = boot[(boot.strategy == "block")].boot_sd.median() * 100
    nzb = boot[(boot.strategy == "nzbalanced")].boot_sd.median() * 100
    print(f"\n  median: block {blk:.2f} pp, nzbalanced {nzb:.2f} pp")
    print("  Compare against the straggler-ratio floor of 1.03-1.05 (D32).")
    print("  If block's spread is of the same order as the effects Days 6-7")
    print("  measure, single-seed partition comparisons are not resolvable")
    print("  and the permutation distribution must be reported instead.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot-records", required=True)
    ap.add_argument("--boot-work", default=None)
    ap.add_argument("--partitions", required=True)
    a = ap.parse_args()

    parts = pd.read_csv(a.partitions)
    ok = True

    for path, cost_label in ((a.boot_records, "records"),
                             (a.boot_work, "work")):
        if not path:
            continue
        boot = pd.read_csv(path)
        print("\n" + "=" * 78)
        print(f"FILE: {path}   (cost = {cost_label})")
        print("=" * 78)
        print(f"  {boot.variant.nunique()} variants, "
              f"{sorted(boot.p.unique())} worker counts, "
              f"B = {int(boot.n_boot.iloc[0])}")

        print("\n--- 1. CROSS-CHECK vs cv_work_partitions.csv " + "-" * 32)
        c_ok, n = cross_check(boot, parts, cost_label)
        ok &= check(f"canonical draws reproduce cv_work ({n} comparisons)", c_ok)

        print("\n--- 2. MACHINERY: block vs cyclic must converge " + "-" * 30)
        m_ok, _ = machinery(boot)
        ok &= check("bootstrap means agree within 0.5 pp", m_ok)

        print("\n--- 3. D36 DECOMPOSITION " + "-" * 52)
        decomposition(boot)

        print("\n--- 4. CALIBRATION of the canonical order (D8) " + "-" * 31)
        cal = calibration(boot, cost_label)
        if not cal:
            print("\n     Reseeding is free now and impossible after the sweep.")
            print("     If you reseed: ONCE, blind, and record the original")
            print("     percentile. Do not search over seeds.")

        print("\n--- SEED SENSITIVITY " + "-" * 56)
        seed_sensitivity(boot, cost_label)

    print("\n" + "=" * 78)
    print("OVERALL: " + ("PASS" if ok else "FAIL — see above"))
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
