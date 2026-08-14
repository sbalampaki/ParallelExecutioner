#!/usr/bin/env python3

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             mean_absolute_error, r2_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

P = "/projects/sb2ea"
SEED = 20260726                     # PERMUTATION_SEED, reused for determinism
AGGS = ["count", "mean", "min", "max", "std", "first", "last", "slope"]

# D28 pre-registered bands. Written before the model was fit.
GATE = {"auroc": (0.82, 0.85), "auprc": (0.35, 0.45),
        "auroc_stop": (0.80, 0.87)}


def log(msg=""):
    print(msg, flush=True)


def load(window, split_name):
    spec = pd.read_csv(f"{P}/manifest/feature_spec.csv").sort_values("slot")
    names = [f"{v}_{a}" for v in spec["variable"] for a in AGGS]
    F = len(names)

    X = np.fromfile(f"{P}/features/{window}_serial.f64",
                    dtype="<f8").reshape(-1, F)
    co = pd.read_parquet(f"{P}/cohort/cohort.parquet").sort_values("row_index")
    if len(co) != len(X):
        sys.exit(f"FATAL: cohort has {len(co)} rows, matrix has {len(X)}")

    log(f"features   : {X.shape}  from {window}_serial.f64")
    log(f"cohort     : {len(co)} stays, {co.subject_id.nunique()} subjects")

    idx = {s: (co["split"].values == s) for s in ("train", "val", "test")}
    for s, m in idx.items():
        log(f"  {s:<6} {m.sum():>6} stays  "
            f"{co.loc[m, 'subject_id'].nunique():>6} subjects  "
            f"mortality {co.loc[m, 'hospital_expire_flag'].mean()*100:.2f}%")
    return X, co, names, idx["train"], idx[split_name]


def prepare(X, tr, ev):
    """Impute with TRAIN medians, standardise on TRAIN. Never on eval."""
    med = np.nanmedian(X[tr], axis=0)
    med = np.where(np.isnan(med), 0.0, med)        # column all-NaN in train
    def fill(A):
        A = A.copy()
        bad = np.isnan(A)
        A[bad] = np.take(med, np.where(bad)[1])
        return A
    Xtr, Xev = fill(X[tr]), fill(X[ev])
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xev), med


def boot_ci(y, p, subj, n_boot=2000, seed=SEED):
    """Bootstrap over SUBJECTS, carrying all of each subject's stays."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(subj)
    rows = {s: np.where(subj == s)[0] for s in uniq}
    au, ap = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        ix = np.concatenate([rows[s] for s in pick])
        yy = y[ix]
        if yy.min() == yy.max():          # degenerate resample
            continue
        au.append(roc_auc_score(yy, p[ix]))
        ap.append(average_precision_score(yy, p[ix]))
    f = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return f(au), f(ap), len(au)


def calibration(y, p, bins=10):
    edge = np.quantile(p, np.linspace(0, 1, bins + 1))
    edge[0], edge[-1] = -np.inf, np.inf
    out = []
    for i in range(bins):
        m = (p > edge[i]) & (p <= edge[i + 1])
        if m.sum():
            out.append({"bin": i, "n": int(m.sum()),
                        "pred": float(p[m].mean()),
                        "obs": float(y[m].mean())})
    return out


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--window", default="W24")
    ap_.add_argument("--use-test", action="store_true",
                     help="Day 12 only. Do not use to debug the pipeline.")
    ap_.add_argument("--n-boot", type=int, default=2000)
    a = ap_.parse_args()

    split = "test" if a.use_test else "val"
    if a.use_test:
        log("!! EVALUATING ON TEST. This is a Day-12 action. If you are "
            "debugging, stop.\n")

 
    if a.window not in ("W6", "W12", "W24"):
        sys.exit(f"FATAL: {a.window} is performance-only (D3/A1). "
                 f"Clinical metrics are valid only for W <= 24 h.")

    log(f"=== Day 5 ML validation -- {a.window}, evaluating on {split} ===")
    log(f"sklearn {sklearn.__version__}   numpy {np.__version__}   "
        f"seed {SEED}\n")

    X, co, names, tr, ev = load(a.window, split)
    Xtr, Xev, med = prepare(X, tr, ev)

    y = co["hospital_expire_flag"].values.astype(int)
    ytr, yev = y[tr], y[ev]
    subj_ev = co["subject_id"].values[ev]
    prev = float(yev.mean())

    log("\n--- mortality: l2 logistic regression ---")
    res = {}
    for tag, cw in (("plain", None), ("balanced", "balanced")):
        m = LogisticRegression(max_iter=5000, C=1.0, class_weight=cw,
                               random_state=SEED)
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xev)[:, 1]
        r = {"auroc": float(roc_auc_score(yev, p)),
             "auprc": float(average_precision_score(yev, p)),
             "brier": float(brier_score_loss(yev, p))}
        res[tag] = (r, m, p)
        log(f"  {tag:<9} AUROC {r['auroc']:.4f}   AUPRC {r['auprc']:.4f}   "
            f"Brier {r['brier']:.4f}")


    r, model, p = res["plain"]
    (au_lo, au_hi), (ap_lo, ap_hi), nb = boot_ci(yev, p, subj_ev, a.n_boot)

    log(f"\n  prevalence (AUPRC no-skill floor) : {prev:.4f}")
    log(f"  AUROC {r['auroc']:.4f}  95% CI [{au_lo:.4f}, {au_hi:.4f}]")
    log(f"  AUPRC {r['auprc']:.4f}  95% CI [{ap_lo:.4f}, {ap_hi:.4f}]"
        f"   lift over no-skill {r['auprc']/prev:.2f}x")
    log(f"  ({nb} subject-level bootstrap resamples, "
        f"{len(np.unique(subj_ev))} subjects)")


    log("\n--- D28 gate (pre-registered, day4_decisions.md) ---")
    g_au = GATE["auroc"][0] <= r["auroc"] <= GATE["auroc"][1]
    g_ap = GATE["auprc"][0] <= r["auprc"] <= GATE["auprc"][1]
    stop = not (GATE["auroc_stop"][0] <= r["auroc"] <= GATE["auroc_stop"][1])
    log(f"  AUROC in [{GATE['auroc'][0]}, {GATE['auroc'][1]}] : "
        f"{'PASS' if g_au else 'outside band'}")
    log(f"  AUPRC in [{GATE['auprc'][0]}, {GATE['auprc'][1]}] : "
        f"{'PASS' if g_ap else 'outside band'}")
    if stop:
        log("  *** STOP: outside 0.80-0.87. Debug the PIPELINE, not the "
            "model.")
        log("      Above 0.87 for plain LR on 24 h aggregates is a leakage")
        log("      signal -- check no W48/Wfull matrix was substituted "
            "(A1/D3).")
    elif g_au and g_ap:
        log("  -> GATE PASSED. Proceed to Day 6.")
    else:
        log("  -> Marginal. Proceed, but record the deviation and its "
            "likely cause.")

    #  LOS 
    log("\n--- length of stay: ridge on log1p, reported on days ---")
    los = co["log1p_los_hosp"].values
    rg = Ridge(alpha=1.0, random_state=SEED).fit(Xtr, los[tr])
    pred_d = np.expm1(rg.predict(Xev))
    true_d = np.expm1(los[ev])
    surv = yev == 0
    los_res = {}
    for tag, m in (("all", np.ones_like(surv, bool)), ("survivors", surv)):
        los_res[tag] = {"n": int(m.sum()),
                        "mae_days": float(mean_absolute_error(true_d[m], pred_d[m])),
                        "r2_log1p": float(r2_score(los[ev][m], rg.predict(Xev)[m]))}
        log(f"  {tag:<10} n={m.sum():>6}  MAE {los_res[tag]['mae_days']:6.2f} d"
            f"   R2(log1p) {los_res[tag]['r2_log1p']:.4f}")
    log("  (decedents kept but reported separately -- D4 competing risk)")

    #  artifacts 
    out = f"{P}/results/day5"
    os.makedirs(out, exist_ok=True)

    pd.DataFrame({"feature": names, "coef": model.coef_[0]}) \
      .assign(abs_coef=lambda d: d.coef.abs()) \
      .sort_values("abs_coef", ascending=False) \
      .to_csv(f"{out}/coefficients_mortality_{a.window}.csv", index=False)

    pd.DataFrame(calibration(yev, p)).to_csv(
        f"{out}/calibration_{split}_{a.window}.csv", index=False)

    json.dump({
        "generated": datetime.now(timezone.utc).isoformat(),
        "window": a.window, "eval_split": split, "seed": SEED,
        "sklearn": sklearn.__version__, "numpy": np.__version__,
        "n_features": len(names), "prevalence": prev,
        "mortality": {"plain": res["plain"][0], "balanced": res["balanced"][0],
                      "auroc_ci": [au_lo, au_hi], "auprc_ci": [ap_lo, ap_hi],
                      "n_boot": nb},
        "gate": {"bands": GATE, "auroc_pass": bool(g_au),
                 "auprc_pass": bool(g_ap), "stop": bool(stop)},
        "los": los_res,
        "intercept": float(model.intercept_[0]),
    }, open(f"{out}/metrics_{a.window}_{split}.json", "w"), indent=2)

    log(f"\nwrote {out}/metrics_{a.window}_{split}.json")
    log(f"      {out}/coefficients_mortality_{a.window}.csv")
    log(f"      {out}/calibration_{split}_{a.window}.csv")
    log("\nCoefficients are the D27 reference: parallel and GPU variants "
        "are diffed\nagainst this file for max absolute and relative "
        "deviation.")

    log("\n--- top 12 coefficients ---")
    top = pd.read_csv(f"{out}/coefficients_mortality_{a.window}.csv").head(12)
    for _, row in top.iterrows():
        log(f"  {row['feature']:<24} {row['coef']:+.4f}")


if __name__ == "__main__":
    main()
