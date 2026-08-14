#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

PARQUET = Path("/projects/sb2ea/parquet")
MANIFEST = Path("/projects/sb2ea/manifest")
WORK = Path("/projects/sb2ea/work")

MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY_LIMIT", "24GB")
THREADS = int(os.environ.get("DUCKDB_THREADS", "8"))

COVERAGE_THRESHOLD = 99.0
ICUSTAYS_EXPECTED = 94_458


def p(name: str) -> str:
    """Parquet path as a SQL string literal source."""
    return f"read_parquet('{PARQUET / (name + '.parquet')}')"


def connect() -> duckdb.DuckDBPyConnection:
    WORK.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads TO {THREADS}")
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory = '{WORK}'")
    return con


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}", flush=True)



def c1_icustays_count(con) -> dict:
    n = con.execute(f"SELECT count(*) FROM {p('icustays')}").fetchone()[0]
    ok = n == ICUSTAYS_EXPECTED
    print(f"C1  icustays rows = {n:,} (expected {ICUSTAYS_EXPECTED:,})  ->  "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return {"check": "icustays_count", "value": n,
            "expected": ICUSTAYS_EXPECTED, "pass": ok}


def c2_heart_rate_coverage(con) -> dict:
    items = con.execute(
        f"""
        SELECT itemid, label
        FROM {p('d_items')}
        WHERE linksto = 'chartevents' AND lower(label) = 'heart rate'
        ORDER BY itemid
        """
    ).fetchall()

    if not items:
        print("C2  no itemid with label 'Heart Rate' in d_items  ->  FAIL", flush=True)
        return {"check": "heart_rate_coverage", "pass": False,
                "note": "no matching itemid"}

    ids = [int(i) for i, _ in items]
    print(f"C2  heart-rate itemid(s): {', '.join(f'{i} ({l})' for i, l in items)}",
          flush=True)

    n_stays, n_with = con.execute(
        f"""
        WITH with_hr AS (
            SELECT DISTINCT stay_id
            FROM {p('chartevents')}
            WHERE itemid IN ({','.join(str(i) for i in ids)})
        )
        SELECT
            (SELECT count(*) FROM {p('icustays')}),
            (SELECT count(*) FROM {p('icustays')} s
             WHERE EXISTS (SELECT 1 FROM with_hr w WHERE w.stay_id = s.stay_id))
        """
    ).fetchone()

    pct = 100.0 * n_with / n_stays if n_stays else 0.0
    ok = pct >= COVERAGE_THRESHOLD
    print(f"C2  stays with >=1 heart rate: {n_with:,} / {n_stays:,} = {pct:.3f}%  "
          f"(threshold {COVERAGE_THRESHOLD}%)  ->  {'PASS' if ok else 'FAIL'}",
          flush=True)
    return {"check": "heart_rate_coverage", "itemids": ids, "n_stays": n_stays,
            "n_with": n_with, "pct": round(pct, 4), "pass": ok}


def c3_diagnoses_coverage(con) -> dict:
    n_hadm, n_with = con.execute(
        f"""
        WITH icu_hadm AS (
            SELECT DISTINCT hadm_id FROM {p('icustays')} WHERE hadm_id IS NOT NULL
        ),
        dx_hadm AS (
            SELECT DISTINCT hadm_id FROM {p('diagnoses_icd')} WHERE hadm_id IS NOT NULL
        )
        SELECT
            (SELECT count(*) FROM icu_hadm),
            (SELECT count(*) FROM icu_hadm i
             WHERE EXISTS (SELECT 1 FROM dx_hadm d WHERE d.hadm_id = i.hadm_id))
        """
    ).fetchone()

    pct = 100.0 * n_with / n_hadm if n_hadm else 0.0
    ok = pct >= COVERAGE_THRESHOLD
    print(f"C3  ICU hadm_id billed in diagnoses_icd: {n_with:,} / {n_hadm:,} = "
          f"{pct:.3f}%  (threshold {COVERAGE_THRESHOLD}%)  ->  "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return {"check": "diagnoses_coverage", "n_hadm": n_hadm, "n_with": n_with,
            "pct": round(pct, 4), "pass": ok}



def d1_null_valuenum_by_param_type(con) -> dict:
    rows = con.execute(
        f"""
        SELECT
            coalesce(di.param_type, '(not in d_items)') AS param_type,
            count(*)                                       AS n_records,
            count(*) FILTER (WHERE ce.valuenum IS NULL)    AS n_null,
            round(100.0 * count(*) FILTER (WHERE ce.valuenum IS NULL)
                  / count(*), 2)                           AS pct_null_within,
            round(100.0 * count(*) FILTER (WHERE ce.valuenum IS NULL)
                  / sum(count(*) FILTER (WHERE ce.valuenum IS NULL)) OVER (), 2)
                                                           AS pct_of_all_nulls
        FROM {p('chartevents')} ce
        LEFT JOIN {p('d_items')} di USING (itemid)
        GROUP BY 1
        ORDER BY n_null DESC
        """
    ).fetchall()

    print(f"\nD1  chartevents NULL valuenum by param_type")
    print(f"    {'param_type':<24} {'records':>14} {'nulls':>14} "
          f"{'%null':>8} {'%of nulls':>10}")
    for pt, n, nn, pw, pa in rows:
        print(f"    {pt:<24} {n:>14,} {nn:>14,} {pw:>8.2f} {pa:>10.2f}")

    top = con.execute(
        f"""
        SELECT ce.itemid, coalesce(di.label, '?') AS label,
               coalesce(di.param_type, '?') AS param_type,
               count(*) FILTER (WHERE ce.valuenum IS NULL) AS n_null
        FROM {p('chartevents')} ce
        LEFT JOIN {p('d_items')} di USING (itemid)
        GROUP BY 1, 2, 3
        HAVING count(*) FILTER (WHERE ce.valuenum IS NULL) > 0
        ORDER BY n_null DESC
        LIMIT 15
        """
    ).fetchall()

    print(f"\n    top 15 itemids by NULL valuenum count")
    print(f"    {'itemid':>8}  {'label':<44} {'param_type':<18} {'nulls':>13}")
    for iid, lab, pt, nn in top:
        print(f"    {iid:>8}  {lab[:44]:<44} {pt[:18]:<18} {nn:>13,}")

    return {
        "by_param_type": [
            {"param_type": r[0], "n_records": r[1], "n_null": r[2],
             "pct_null_within": r[3], "pct_of_all_nulls": r[4]} for r in rows
        ],
        "top_itemids": [
            {"itemid": r[0], "label": r[1], "param_type": r[2], "n_null": r[3]}
            for r in top
        ],
    }


def d2_usable_projection(con) -> dict:
    ce_total, ce_usable = con.execute(
        f"""
        SELECT count(*), count(*) FILTER (WHERE valuenum IS NOT NULL)
        FROM {p('chartevents')}
        """
    ).fetchone()

    le_total, le_usable = con.execute(
        f"""
        SELECT count(*),
               count(*) FILTER (WHERE valuenum IS NOT NULL AND hadm_id IS NOT NULL)
        FROM {p('labevents')}
        """
    ).fetchone()

    total = ce_total + le_total
    usable = ce_usable + le_usable
    bytes_12 = usable * 12

    print(f"\nD2  usable-record projection (before cohort and window filters)")
    print(f"    chartevents  {ce_usable:>13,} / {ce_total:>13,} "
          f"= {100.0*ce_usable/ce_total:6.2f}%")
    print(f"    labevents    {le_usable:>13,} / {le_total:>13,} "
          f"= {100.0*le_usable/le_total:6.2f}%")
    print(f"    combined     {usable:>13,} / {total:>13,} "
          f"= {100.0*usable/total:6.2f}%")
    print(f"    at 12 B/record -> {bytes_12/2**30:.2f} GiB, all stays, full window")

    return {
        "chartevents": {"total": ce_total, "usable": ce_usable},
        "labevents": {"total": le_total, "usable": le_usable},
        "combined": {"total": total, "usable": usable,
                     "gib_at_12B": round(bytes_12 / 2**30, 3)},
    }



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-heavy", action="store_true",
                    help="run only checks that do not scan chartevents")
    args = ap.parse_args()

    missing = [t for t in ("icustays", "d_items", "diagnoses_icd",
                           "chartevents", "labevents")
               if not (PARQUET / f"{t}.parquet").is_file()]
    if missing:
        print(f"missing parquet: {missing}", file=sys.stderr)
        return 2

    MANIFEST.mkdir(parents=True, exist_ok=True)
    con = connect()
    print(f"duckdb {duckdb.__version__} | threads {THREADS} | mem {MEMORY_LIMIT}")

    banner("Pass / fail")
    checks = [c1_icustays_count(con), c3_diagnoses_coverage(con)]
    if not args.skip_heavy:
        checks.insert(1, c2_heart_rate_coverage(con))

    diagnostics = {}
    if not args.skip_heavy:
        banner("Diagnostics")
        diagnostics["null_valuenum_by_param_type"] = d1_null_valuenum_by_param_type(con)
        diagnostics["usable_projection"] = d2_usable_projection(con)

    failed = [c["check"] for c in checks if not c["pass"]]

    banner("Summary")
    for c in checks:
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['check']}")
    print(f"\n  {len(checks) - len(failed)}/{len(checks)} pass/fail checks passed")

    out = MANIFEST / "validation.json"
    out.write_text(json.dumps({
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duckdb_version": duckdb.__version__,
        "coverage_threshold_pct": COVERAGE_THRESHOLD,
        "checks": checks,
        "diagnostics": diagnostics,
    }, indent=2) + "\n")
    print(f"  written -> {out}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
