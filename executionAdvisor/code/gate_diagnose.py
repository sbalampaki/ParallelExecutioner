#!/usr/bin/env python3

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

P = "/projects/sb2ea"
SEED = 20260726
AGGS = ["count", "mean", "min", "max", "std", "first", "last", "slope"]
GCS = {"gcs_eye", "gcs_motor", "gcs_verbal"}
LABS = {"bun", "creatinine", "hemoglobin", "lactate", "platelets", "wbc",
        "bicarbonate", "chloride", "sodium"}


def load(window):
    spec = pd.read_csv(f"{P}/manifest/feature_spec.csv").sort_values("slot")
    variables = list(spec["variable"])
    names = [f"{v}_{a}" for v in variables for a in AGGS]
    X = np.fromfile(f"{P}/features/{window}_serial.f64",
                    dtype="<f8").reshape(-1, len(names))
    co = pd.read_parquet(f"{P}/cohort/cohort.parquet").sort_values("row_index")
    tr = co["split"].values == "train"
    ev = co["split"].values == "val"
    return X, co, names, variables, tr, ev


def fit_eval(X, y, tr, ev, cols):
    Xs = X[:, cols]
    med = np.nanmedian(Xs[tr], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    def fill(A):
        A = A.copy(); bad = np.isnan(A)
        A[bad] = np.take(med, np.where(bad)[1]); return A
    a, b = fill(Xs[tr]), fill(Xs[ev])
    sc = StandardScaler().fit(a)
    m = LogisticRegression(max_iter=5000, random_state=SEED)
    m.fit(sc.transform(a), y[tr])
    p = m.predict_proba(sc.transform(b))[:, 1]
    return roc_auc_score(y[ev], p), average_precision_score(y[ev], p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", default=["W24"])
    a = ap.parse_args()

    # ---- window sweep: the discriminating test 
    print("=== window sweep (all 160 features, val) ===")
    print("  leakage  -> flat across W;  richer features -> rises with W\n")
    print(f"  {'window':<8}{'AUROC':>9}{'AUPRC':>9}{'d AUROC':>10}")
    prev_au = None
    for w in a.windows:
        try:
            X, co, names, variables, tr, ev = load(w)
        except FileNotFoundError:
            print(f"  {w:<8}   (matrix not built -- see note below)")
            continue
        y = co["hospital_expire_flag"].values.astype(int)
        au, apr = fit_eval(X, y, tr, ev, np.arange(X.shape[1]))
        d = "" if prev_au is None else f"{au-prev_au:+9.4f}"
        print(f"  {w:<8}{au:>9.4f}{apr:>9.4f}{d:>10}")
        prev_au = au

    # ---- ablations on the primary window 
    w = a.windows[-1]
    X, co, names, variables, tr, ev = load(w)
    y = co["hospital_expire_flag"].values.astype(int)
    nm = np.array(names)
    var_of = np.array([n.rsplit("_", 1)[0] for n in names])
    agg_of = np.array([n.rsplit("_", 1)[1] for n in names])
    allc = np.arange(len(names))

    groups = {
        "all 160":              allc,
        "drop count cols":      np.where(agg_of != "count")[0],
        "drop GCS":             np.where(~np.isin(var_of, list(GCS)))[0],
        "drop GCS + count":     np.where(~np.isin(var_of, list(GCS))
                                         & (agg_of != "count"))[0],
        "labs only":            np.where(np.isin(var_of, list(LABS)))[0],
        "labs only, no count":  np.where(np.isin(var_of, list(LABS))
                                         & (agg_of != "count"))[0],
        "GCS only":             np.where(np.isin(var_of, list(GCS)))[0],
        "count cols only":      np.where(agg_of == "count")[0],
    }

    print(f"\n=== ablations on {w} (val) ===")
    print("  'labs only, no count' is the closest match to the comparator")
    print("  that anchored D28 (11 analytes, 8 aggregates, no vitals/GCS,")
    print("  AUROC 0.823 / AUPRC 0.376 at 10.7% prevalence).\n")
    print(f"  {'subset':<22}{'ncol':>6}{'AUROC':>9}{'AUPRC':>9}")
    base = None
    for tag, cols in groups.items():
        au, apr = fit_eval(X, y, tr, ev, cols)
        if base is None:
            base = au
        print(f"  {tag:<22}{len(cols):>6}{au:>9.4f}{apr:>9.4f}")


    print("\n=== negative control: shuffled TRAIN labels ===")
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(5):
        ysh = y.copy()
        ysh[tr] = rng.permutation(ysh[tr])
        draws.append(fit_eval(X, ysh, tr, ev, allc)[0])
    mean_au = float(np.mean(draws))
    ok = abs(mean_au - 0.5) < 0.03
    print(f"  draws  : {', '.join(f'{d:.4f}' for d in draws)}")
    print(f"  mean   : {mean_au:.4f}   "
          f"{'OK (0.500 expected)' if ok else '*** BROKEN JOIN ***'}")
    if not ok:
        print("  A model trained on shuffled labels must not discriminate.")
        print("  If it does, the feature/label alignment is wrong and every")
        print("  number above is meaningless. Check that cohort.parquet is")
        print("  sorted by row_index and that n_stays matches the matrix.")

    # ---- single-feature scan 
    print("\n=== strongest single features (val AUROC) ===")
    print("  A single feature above ~0.80 alone would be a leak, not a")
    print("  predictor. Clinical single-feature AUROCs top out near 0.70.\n")
    scores = []
    for j in range(len(names)):
        v = X[:, j]
        m = ~np.isnan(v)
        mv = m & ev
        if mv.sum() < 500 or len(np.unique(y[mv])) < 2:
            continue
        s = roc_auc_score(y[mv], v[mv])
        scores.append((max(s, 1 - s), nm[j]))
    scores.sort(reverse=True)
    for s, n in scores[:10]:
        flag = "   <-- SUSPICIOUS" if s > 0.80 else ""
        print(f"  {n:<24}{s:.4f}{flag}")

    print("\nNote: W6/W12 matrices are built with")
    print("  $PROJ/bin/kernel_serial --csr $PROJ/csr/W6 "
          "--lookup $PROJ/manifest/kernel_lookup.csv \\")
    print("      --out $PROJ/features/W6_serial.f64")


if __name__ == "__main__":
    main()
