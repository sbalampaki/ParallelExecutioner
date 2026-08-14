#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

SRC = Path("/projects/sb2ea/datasets/mimiciv-3.1")
DST = Path("/projects/sb2ea/parquet")
WORK = Path("/projects/sb2ea/work")
MANIFEST = Path("/projects/sb2ea/manifest")

MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY_LIMIT", "48GB")
THREADS = int(os.environ.get("DUCKDB_THREADS", "16"))

SCHEMAS: dict[str, dict] = {
    "d_labitems": {
        "path": "hosp/d_labitems.csv.gz",
        "columns": {
            "itemid": "INTEGER",
            "label": "VARCHAR",
            "fluid": "VARCHAR",
            "category": "VARCHAR",
        },
    },
    "d_items": {
        "path": "icu/d_items.csv.gz",
        "columns": {
            "itemid": "INTEGER",
            "label": "VARCHAR",
            "abbreviation": "VARCHAR",
            "linksto": "VARCHAR",
            "category": "VARCHAR",
            "unitname": "VARCHAR",
            "param_type": "VARCHAR",
            "lownormalvalue": "DOUBLE",
            "highnormalvalue": "DOUBLE",
        },
    },
    "patients": {
        "path": "hosp/patients.csv.gz",
        "columns": {
            "subject_id": "INTEGER",
            "gender": "VARCHAR",
            "anchor_age": "INTEGER",
            "anchor_year": "INTEGER",
            "anchor_year_group": "VARCHAR",
            "dod": "DATE",
        },
    },
    "icustays": {
        "path": "icu/icustays.csv.gz",
        "columns": {
            "subject_id": "INTEGER",
            "hadm_id": "INTEGER",
            "stay_id": "INTEGER",
            "first_careunit": "VARCHAR",
            "last_careunit": "VARCHAR",
            "intime": "TIMESTAMP",
            "outtime": "TIMESTAMP",
            "los": "DOUBLE",
        },
    },
    "admissions": {
        "path": "hosp/admissions.csv.gz",
        "columns": {
            "subject_id": "INTEGER",
            "hadm_id": "INTEGER",
            "admittime": "TIMESTAMP",
            "dischtime": "TIMESTAMP",
            "deathtime": "TIMESTAMP",
            "admission_type": "VARCHAR",
            "admit_provider_id": "VARCHAR",
            "admission_location": "VARCHAR",
            "discharge_location": "VARCHAR",
            "insurance": "VARCHAR",
            "language": "VARCHAR",
            "marital_status": "VARCHAR",
            "race": "VARCHAR",
            "edregtime": "TIMESTAMP",
            "edouttime": "TIMESTAMP",
            "hospital_expire_flag": "SMALLINT",
        },
    },
    "diagnoses_icd": {
        "path": "hosp/diagnoses_icd.csv.gz",
        "columns": {
            "subject_id": "INTEGER",
            "hadm_id": "INTEGER",
            "seq_num": "INTEGER",
            "icd_code": "VARCHAR",
            "icd_version": "SMALLINT",
        },
    },
    "labevents": {
        "path": "hosp/labevents.csv.gz",
        "columns": {
            "labevent_id": "BIGINT",
            "subject_id": "INTEGER",
            "hadm_id": "INTEGER",          # nullable: outpatient labs
            "specimen_id": "BIGINT",
            "itemid": "INTEGER",
            "order_provider_id": "VARCHAR",
            "charttime": "TIMESTAMP",
            "storetime": "TIMESTAMP",
            "value": "VARCHAR",            # redacted to ___ in places; unusable
            "valuenum": "DOUBLE",          # the only numeric source
            "valueuom": "VARCHAR",
            "ref_range_lower": "DOUBLE",
            "ref_range_upper": "DOUBLE",
            "flag": "VARCHAR",
            "priority": "VARCHAR",
            "comments": "VARCHAR",         # quoted, embedded commas/newlines
        },
    },
    "chartevents": {
        "path": "icu/chartevents.csv.gz",
        "columns": {
            "subject_id": "INTEGER",
            "hadm_id": "INTEGER",
            "stay_id": "INTEGER",
            "caregiver_id": "INTEGER",
            "charttime": "TIMESTAMP",
            "storetime": "TIMESTAMP",
            "itemid": "INTEGER",
            "value": "VARCHAR",
            "valuenum": "DOUBLE",
            "valueuom": "VARCHAR",
            "warning": "SMALLINT",         # CIS out-of-range flag; see D11
        },
    },
}

PROFILE_COLUMNS = {
    "labevents": ["hadm_id", "valuenum"],
    "chartevents": ["stay_id", "valuenum"],
}


def connect() -> duckdb.DuckDBPyConnection:
    WORK.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads TO {THREADS}")
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory = '{WORK}'")
    # Streaming write; without this DuckDB buffers to preserve row order and
    # can balloon memory on the large tables.
    con.execute("SET preserve_insertion_order = false")
    return con


def coldef(columns: dict[str, str]) -> str:
    return "{" + ", ".join(f"'{k}': '{v}'" for k, v in columns.items()) + "}"


def convert_one(con, name: str, spec: dict) -> dict:
    src = SRC / spec["path"]
    dst = DST / f"{name}.parquet"
    if not src.is_file():
        raise FileNotFoundError(src)

    print(f"[{name}] reading {src}", flush=True)
    t0 = time.time()

    con.execute(
        f"""
        COPY (
            SELECT * FROM read_csv(
                '{src}',
                columns = {coldef(spec["columns"])},
                header = true,
                compression = 'gzip',
                delim = ',',
                quote = '"',
                escape = '"'
            )
        ) TO '{dst}' (FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE 1000000)
        """
    )

    elapsed = time.time() - t0

    # Parquet stores the row count in its footer, so this does not scan.
    rows = con.execute(f"SELECT count(*) FROM read_parquet('{dst}')").fetchone()[0]

    rec = {
        "rows": int(rows),
        "src_bytes": src.stat().st_size,
        "dst_bytes": dst.stat().st_size,
        "seconds": round(elapsed, 1),
    }
    print(
        f"[{name}] {rec['rows']:,} rows | "
        f"{rec['src_bytes']/2**30:.2f} GiB gz -> {rec['dst_bytes']/2**30:.2f} GiB parquet | "
        f"{rec['seconds']:.0f}s",
        flush=True,
    )
    return rec


def profile_one(con, name: str, columns: list[str]) -> dict:
    dst = DST / f"{name}.parquet"
    if not dst.is_file():
        return {}
    exprs = ", ".join(f"count(*) FILTER (WHERE {c} IS NULL) AS null_{c}" for c in columns)
    row = con.execute(
        f"SELECT count(*) AS total, {exprs} FROM read_parquet('{dst}')"
    ).fetchone()
    names = ["total"] + [f"null_{c}" for c in columns]
    out = {k: int(v) for k, v in zip(names, row)}
    frac = {
        f"pct_{k}": round(100.0 * v / out["total"], 3)
        for k, v in out.items()
        if k != "total" and out["total"]
    }
    out.update(frac)
    print(f"[{name}] profile {out}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated table names")
    ap.add_argument("--skip", help="comma-separated table names")
    ap.add_argument("--profile", action="store_true", help="null counts after conversion")
    ap.add_argument("--expected", help="JSON of {table: rowcount} to assert against")
    args = ap.parse_args()

    tables = list(SCHEMAS)
    if args.only:
        wanted = [t.strip() for t in args.only.split(",")]
        unknown = set(wanted) - set(SCHEMAS)
        if unknown:
            print(f"unknown table(s): {sorted(unknown)}", file=sys.stderr)
            return 2
        tables = [t for t in tables if t in wanted]
    if args.skip:
        drop = {t.strip() for t in args.skip.split(",")}
        tables = [t for t in tables if t not in drop]

    DST.mkdir(parents=True, exist_ok=True)
    MANIFEST.mkdir(parents=True, exist_ok=True)

    con = connect()
    print(f"duckdb {duckdb.__version__} | threads {THREADS} | mem {MEMORY_LIMIT}", flush=True)

    results: dict[str, dict] = {}
    counts_path = MANIFEST / "rowcounts.json"
    if counts_path.is_file():
        results = json.loads(counts_path.read_text()).get("tables", {})

    for name in tables:
        results[name] = convert_one(con, name, SCHEMAS[name])
        payload = {
            "mimic_version": "3.1",
            "duckdb_version": duckdb.__version__,
            "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tables": results,
        }
        counts_path.write_text(json.dumps(payload, indent=2) + "\n")

    if args.profile:
        prof = {}
        pp = MANIFEST / "dropcounts.json"
        if pp.is_file():
            prof = json.loads(pp.read_text()).get("tables", {})
        for name, cols in PROFILE_COLUMNS.items():
            if name in tables:
                prof[name] = profile_one(con, name, cols)
        pp.write_text(
            json.dumps(
                {
                    "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "tables": prof,
                },
                indent=2,
            )
            + "\n"
        )

    rc = 0
    if args.expected:
        expected = json.loads(Path(args.expected).read_text())
        print("\n-- D1 stage 3 --")
        for name in sorted(results):
            want = expected.get(name)
            got = results[name]["rows"]
            if want is None:
                print(f"  {name:<16} {got:>13,}   (no reference)")
            elif int(want) == got:
                print(f"  {name:<16} {got:>13,}   OK")
            else:
                print(f"  {name:<16} {got:>13,}   MISMATCH, expected {int(want):,}")
                rc = 1

    print(f"\nrowcounts -> {counts_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
