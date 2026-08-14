#!/usr/bin/env python3
"""
Cohort construction, labels, and the frozen patient-level split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

PARQUET = Path("/projects/sb2ea/parquet")
COHORT = Path("/projects/sb2ea/cohort")
MANIFEST = Path("/projects/sb2ea/manifest")
WORK = Path("/projects/sb2ea/work")

MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY_LIMIT", "24GB")
THREADS = int(os.environ.get("DUCKDB_THREADS", "8"))


MIN_AGE = 18
MIN_ICU_LOS_DAYS = 1.0                       # 24 h
SPLIT_SALT = "mimiciv-3.1-loadimbalance-v1"
SPLIT_BOUNDS = {"train": (0, 70), "val": (70, 85), "test": (85, 100)}
PERMUTATION_SEED = 20260726                  # D8 canonical row order
PREVALENCE_TOLERANCE_PP = 1.0                # split balance assertion


def p(name: str) -> str:
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



BASE_SQL = f"""
CREATE OR REPLACE TEMP VIEW icu_ranked AS
SELECT
    s.subject_id,
    s.hadm_id,
    s.stay_id,
    s.intime,
    s.outtime,
    s.los                          AS los_icu_days,
    s.first_careunit,
    row_number() OVER (
        PARTITION BY s.hadm_id ORDER BY s.intime, s.stay_id
    )                              AS stay_rank
FROM {p('icustays')} s
WHERE s.hadm_id IS NOT NULL;

CREATE OR REPLACE TEMP VIEW joined AS
SELECT
    r.*,
    pt.gender,
    pt.anchor_age,
    pt.anchor_year,
    pt.dod,
    a.admittime,
    a.dischtime,
    a.deathtime,
    a.hospital_expire_flag,
    -- Age at admission, not anchor_age. See module docstring.
    pt.anchor_age + (extract(year FROM a.admittime) - pt.anchor_year)
                                   AS admission_age,
    epoch(a.dischtime - a.admittime) / 86400.0
                                   AS los_hosp_days
FROM icu_ranked r
JOIN {p('patients')}   pt USING (subject_id)
JOIN {p('admissions')} a  USING (subject_id, hadm_id);
"""

COHORT_SQL = f"""
CREATE OR REPLACE TEMP VIEW cohort AS
SELECT
    subject_id, hadm_id, stay_id,
    intime, outtime, los_icu_days, first_careunit,
    gender, admission_age,
    admittime, dischtime, deathtime,
    hospital_expire_flag,
    los_hosp_days,
    ln(1.0 + los_hosp_days)        AS log1p_los_hosp
FROM joined
WHERE stay_rank = 1
  AND admission_age >= {MIN_AGE}
  AND los_icu_days  >= {MIN_ICU_LOS_DAYS}
  -- A handful of admissions (6 in v3.1) have dischtime < admittime, an
  -- internally inconsistent record. Excluded rather than label-nulled so
  -- that mortality and LOS share one cohort and no downstream stage needs
  -- NULL handling. Recorded in the funnel.
  AND los_hosp_days > 0;
"""


def funnel(con) -> list[dict]:
    steps = [
        ("all ICU stays",             f"SELECT count(*) FROM {p('icustays')}"),
        ("hadm_id not null",          "SELECT count(*) FROM icu_ranked"),
        ("first stay per hadm_id",    "SELECT count(*) FROM joined WHERE stay_rank = 1"),
        (f"age >= {MIN_AGE}",
         f"SELECT count(*) FROM joined WHERE stay_rank = 1 "
         f"AND admission_age >= {MIN_AGE}"),
        (f"ICU LOS >= {MIN_ICU_LOS_DAYS * 24:.0f} h",
         f"SELECT count(*) FROM joined WHERE stay_rank = 1 "
         f"AND admission_age >= {MIN_AGE} "
         f"AND los_icu_days >= {MIN_ICU_LOS_DAYS}"),
        ("hosp LOS > 0",              "SELECT count(*) FROM cohort"),
    ]
    rows, prev = [], None
    print(f"  {'step':<28} {'stays':>10} {'removed':>10}")
    for label, sql in steps:
        n = con.execute(sql).fetchone()[0]
        removed = "" if prev is None else f"{prev - n:,}"
        print(f"  {label:<28} {n:>10,} {removed:>10}")
        rows.append({"step": label, "n": int(n),
                     "removed": None if prev is None else int(prev - n)})
        prev = n
    return rows


def bucket_of(subject_id: int) -> int:
    """Deterministic 0..99 bucket. Standard SHA-256, no library-version risk."""
    digest = hashlib.sha256(f"{SPLIT_SALT}|{subject_id}".encode()).hexdigest()
    return int(digest[:16], 16) % 100


def split_of(bucket: int) -> str:
    for name, (lo, hi) in SPLIT_BOUNDS.items():
        if lo <= bucket < hi:
            return name
    raise ValueError(bucket)


def assign_splits(con) -> None:
    subjects = [r[0] for r in
                con.execute("SELECT DISTINCT subject_id FROM cohort "
                            "ORDER BY subject_id").fetchall()]
    rows = [(int(s), bucket_of(s), split_of(bucket_of(s))) for s in subjects]
    con.execute("CREATE OR REPLACE TEMP TABLE splits "
                "(subject_id INTEGER, bucket SMALLINT, split VARCHAR)")
    con.executemany("INSERT INTO splits VALUES (?, ?, ?)", rows)
    print(f"  {len(rows):,} unique subjects hashed")


def assert_split_integrity(con) -> dict:
    ok = True

    # 1. No subject appears in more than one split.
    dupes = con.execute(
        "SELECT count(*) FROM (SELECT subject_id FROM splits "
        "GROUP BY subject_id HAVING count(DISTINCT split) > 1)"
    ).fetchone()[0]
    print(f"  subjects in >1 split: {dupes}  ->  {'PASS' if dupes == 0 else 'FAIL'}")
    ok &= dupes == 0

    # 2. Prevalence balance across splits.
    pooled = con.execute(
        "SELECT avg(hospital_expire_flag)::DOUBLE FROM cohort"
    ).fetchone()[0]
    per_split = con.execute(
        """
        SELECT s.split,
               count(*)                                  AS n_stays,
               count(DISTINCT c.subject_id)              AS n_subjects,
               avg(c.hospital_expire_flag)::DOUBLE       AS prevalence,
               median(c.los_hosp_days)::DOUBLE           AS median_los_hosp
        FROM cohort c JOIN splits s USING (subject_id)
        GROUP BY 1 ORDER BY 1
        """
    ).fetchall()

    print(f"\n  {'split':<8} {'stays':>9} {'subjects':>10} {'mortality':>11} "
          f"{'delta pp':>9} {'med LOS d':>10}")
    detail = []
    for name, n, ns, prev, med in per_split:
        delta = 100.0 * (prev - pooled)
        flag = abs(delta) <= PREVALENCE_TOLERANCE_PP
        ok &= flag
        print(f"  {name:<8} {n:>9,} {ns:>10,} {100*prev:>10.2f}% "
              f"{delta:>+9.2f} {med:>10.2f}{'' if flag else '   FAIL'}")
        detail.append({"split": name, "n_stays": int(n), "n_subjects": int(ns),
                       "prevalence": round(prev, 6),
                       "delta_pp": round(delta, 4),
                       "median_los_hosp_days": round(med, 3)})
    print(f"  {'pooled':<8} {'':>9} {'':>10} {100*pooled:>10.2f}%")

    # 3. Cohort-level sanity.
    bad = con.execute(
        f"""
        SELECT
          count(*) FILTER (WHERE los_icu_days < {MIN_ICU_LOS_DAYS}) AS short_icu,
          count(*) FILTER (WHERE admission_age < {MIN_AGE})         AS underage,
          count(*) FILTER (WHERE intime >= outtime)                 AS bad_times,
          count(*) FILTER (WHERE hospital_expire_flag IS NULL)      AS null_death,
          count(*) FILTER (WHERE los_hosp_days IS NULL)             AS null_los,
          count(*) FILTER (WHERE los_hosp_days < 0)                 AS neg_los,
          count(*) - count(DISTINCT stay_id)                        AS dup_stay
        FROM cohort
        """
    ).fetchone()
    labels = ["short_icu", "underage", "bad_times", "null_death",
              "null_los", "neg_los", "dup_stay"]
    print()
    for lab, v in zip(labels, bad):
        print(f"  {lab:<12} {v:>8,}  ->  {'PASS' if v == 0 else 'FAIL'}")
        ok &= v == 0

    return {"pass": bool(ok), "pooled_prevalence": round(pooled, 6),
            "per_split": detail,
            "sanity": {k: int(v) for k, v in zip(labels, bad)}}



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="write nothing")
    args = ap.parse_args()

    con = connect()
    print(f"duckdb {duckdb.__version__} | threads {THREADS} | mem {MEMORY_LIMIT}")

    con.execute(BASE_SQL)
    con.execute(COHORT_SQL)

    banner("Cohort funnel (D2)")
    fun = funnel(con)

    banner("Split (D5)")
    print(f"  salt   {SPLIT_SALT!r}")
    print(f"  hash   sha256, first 16 hex digits mod 100")
    print(f"  bounds {SPLIT_BOUNDS}")
    assign_splits(con)

    banner("Assertions")
    integrity = assert_split_integrity(con)

    banner("Cohort characteristics")
    stats = con.execute(
        """
        SELECT count(*)                              AS n_stays,
               count(DISTINCT subject_id)            AS n_subjects,
               avg(admission_age)::DOUBLE            AS mean_age,
               avg(CASE WHEN gender='F' THEN 1 ELSE 0 END)::DOUBLE AS frac_female,
               avg(hospital_expire_flag)::DOUBLE     AS mortality,
               median(los_icu_days)::DOUBLE          AS med_los_icu,
               median(los_hosp_days)::DOUBLE         AS med_los_hosp,
               quantile_cont(los_icu_days, 0.9)::DOUBLE AS p90_los_icu
        FROM cohort
        """
    ).fetchone()
    keys = ["n_stays", "n_subjects", "mean_age", "frac_female", "mortality",
            "median_los_icu_days", "median_los_hosp_days", "p90_los_icu_days"]
    summary = dict(zip(keys, stats))
    for k, v in summary.items():
        print(f"  {k:<24} {v:,.4f}" if isinstance(v, float) else f"  {k:<24} {v:,}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0 if integrity["pass"] else 1

    COHORT.mkdir(parents=True, exist_ok=True)
    MANIFEST.mkdir(parents=True, exist_ok=True)

    # D8: canonical row order is a seeded random permutation, saved so any
    # ordering can be reconstructed. Sorting by size would make static block
    # partitioning near-optimal and erase the effect under study.
    con.execute(f"SELECT setseed({(PERMUTATION_SEED % 10**6) / 10**6})")
    con.execute(
        f"""
        COPY (
            SELECT c.*, s.split, s.bucket AS split_bucket,
                   row_number() OVER (ORDER BY random()) - 1 AS row_index
            FROM cohort c JOIN splits s USING (subject_id)
        ) TO '{COHORT / 'cohort.parquet'}' (FORMAT PARQUET, COMPRESSION 'zstd')
        """
    )
    con.execute(
        f"COPY (SELECT * FROM splits ORDER BY subject_id) "
        f"TO '{COHORT / 'splits.parquet'}' (FORMAT PARQUET, COMPRESSION 'zstd')"
    )

    n_written = con.execute(
        f"SELECT count(*) FROM read_parquet('{COHORT / 'cohort.parquet'}')"
    ).fetchone()[0]
    assert n_written == summary["n_stays"], (n_written, summary["n_stays"])

    out = MANIFEST / "cohort_summary.json"
    out.write_text(json.dumps({
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duckdb_version": duckdb.__version__,
        "mimic_version": "3.1",
        "parameters": {
            "min_age": MIN_AGE,
            "min_icu_los_days": MIN_ICU_LOS_DAYS,
            "split_salt": SPLIT_SALT,
            "split_hash": "sha256(salt|subject_id)[:16] mod 100",
            "split_bounds": SPLIT_BOUNDS,
            "permutation_seed": PERMUTATION_SEED,
        },
        "funnel": fun,
        "summary": {k: (round(v, 6) if isinstance(v, float) else int(v))
                    for k, v in summary.items()},
        "split_integrity": integrity,
    }, indent=2) + "\n")

    print(f"\n  cohort.parquet  {n_written:,} rows -> {COHORT}")
    print(f"  summary         -> {out}")
    return 0 if integrity["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
