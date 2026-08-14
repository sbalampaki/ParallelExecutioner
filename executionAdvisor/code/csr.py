#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np

PARQUET = Path("/projects/sb2ea/parquet")
COHORT = Path("/projects/sb2ea/cohort")
CSR = Path("/projects/sb2ea/csr")
MANIFEST = Path("/projects/sb2ea/manifest")
WORK = Path("/projects/sb2ea/work")

MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY_LIMIT", "32GB")
THREADS = int(os.environ.get("DUCKDB_THREADS", "8"))

WINDOWS: dict[str, float | None] = {
    "6": 6.0, "12": 12.0, "24": 24.0, "48": 48.0, "full": None,
}
BATCH_ROWS = 1_000_000
SUBSET_SEED = 20260726          
SHAPE_TRIM_PCT = [0, 1, 5, 10]  


def p(name: str) -> str:
    return f"read_parquet('{PARQUET / (name + '.parquet')}')"


def connect() -> duckdb.DuckDBPyConnection:
    WORK.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads TO {THREADS}")
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory = '{WORK}'")
    con.execute("SET preserve_insertion_order = false")
    return con


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}", flush=True)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)



def build_events(con) -> dict:
    """Cohort-restricted, non-null events with dense itemids, in memory."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cohort AS
        SELECT stay_id, hadm_id, intime, outtime,
               row_index::INTEGER              AS row_index,
               epoch(outtime - intime)/3600.0  AS los_icu_hours
        FROM read_parquet('{COHORT / 'cohort.parquet'}');
    """)
    n_stays = con.execute("SELECT count(*) FROM cohort").fetchone()[0]

    lo, hi, n_distinct = con.execute(
        "SELECT min(row_index), max(row_index), count(DISTINCT row_index) FROM cohort"
    ).fetchone()
    assert (lo, hi, n_distinct) == (0, n_stays - 1, n_stays), \
        f"row_index is not a dense permutation: {lo}..{hi}, {n_distinct} distinct"

    print(f"  cohort stays              {n_stays:,}")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ev_chart AS
        SELECT c.row_index, ce.itemid,
               epoch(ce.charttime - c.intime)/3600.0 AS t,
               ce.valuenum                           AS value
        FROM {p('chartevents')} ce
        JOIN cohort c USING (stay_id)
        WHERE ce.valuenum IS NOT NULL
          AND ce.charttime >= c.intime
          AND ce.charttime <= c.outtime;
    """)
    n_chart = con.execute("SELECT count(*) FROM ev_chart").fetchone()[0]
    print(f"  chartevents in window     {n_chart:,}")

 
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ev_lab AS
        SELECT c.row_index, le.itemid,
               epoch(le.charttime - c.intime)/3600.0 AS t,
               le.valuenum                           AS value
        FROM {p('labevents')} le
        JOIN cohort c USING (hadm_id)
        WHERE le.valuenum IS NOT NULL
          AND le.charttime >= c.intime
          AND le.charttime <= c.outtime;
    """)
    n_lab = con.execute("SELECT count(*) FROM ev_lab").fetchone()[0]
    print(f"  labevents in window       {n_lab:,}")

    con.execute("""
        CREATE OR REPLACE TEMP TABLE dict AS
        WITH ci AS (SELECT DISTINCT itemid FROM ev_chart),
             li AS (SELECT DISTINCT itemid FROM ev_lab),
             kc AS (SELECT count(*) AS k FROM ci)
        SELECT itemid, 'chart' AS source,
               (row_number() OVER (ORDER BY itemid) - 1)::INTEGER AS dense_id
        FROM ci
        UNION ALL
        SELECT itemid, 'lab' AS source,
               ((SELECT k FROM kc) + row_number() OVER (ORDER BY itemid) - 1)::INTEGER
        FROM li;
    """)
    k_chart, k_total = con.execute(
        "SELECT count(*) FILTER (WHERE source='chart'), count(*) FROM dict"
    ).fetchone()
    print(f"  distinct itemids          {k_total:,} "
          f"(chart [0,{k_chart}), lab [{k_chart},{k_total}))")

    con.execute("""
        CREATE OR REPLACE TEMP TABLE events AS
        SELECT e.row_index, d.dense_id AS itemid, e.t, e.value
        FROM (
            SELECT row_index, itemid, t, value, 'chart' AS source FROM ev_chart
            UNION ALL
            SELECT row_index, itemid, t, value, 'lab'   AS source FROM ev_lab
        ) e
        JOIN dict d ON d.itemid = e.itemid AND d.source = e.source;
    """)
    con.execute("DROP TABLE ev_chart; DROP TABLE ev_lab;")

    n_records, t_min, t_max, v_min, v_max, n_bad = con.execute("""
        SELECT count(*), min(t), max(t), min(value), max(value),
               count(*) FILTER (WHERE t < 0 OR NOT isfinite(value))
        FROM events
    """).fetchone()
    print(f"  total records             {n_records:,}")
    print(f"  t range                   [{t_min:.4f}, {t_max:.2f}] h")
    print(f"  value range               [{v_min:g}, {v_max:g}]")
    assert n_bad == 0, f"{n_bad} records with t<0 or non-finite value"

    f32_max_err = float(np.spacing(np.float32(t_max)))
    print(f"  float32 resolution at t_max: {f32_max_err*3600:.4f} s "
          f"(charting resolution is 60 s)")

    CSR.mkdir(parents=True, exist_ok=True)
    dict_path = CSR / "itemid_dict.parquet"
    con.execute(f"""
        COPY (
            SELECT d.dense_id, d.source, d.itemid,
                   coalesce(di.label, dl.label)         AS label,
                   di.param_type,
                   coalesce(di.category, dl.category)   AS category,
                   count(e.t)                           AS n_records
            FROM dict d
            LEFT JOIN {p('d_items')}     di ON di.itemid = d.itemid AND d.source='chart'
            LEFT JOIN {p('d_labitems')}  dl ON dl.itemid = d.itemid AND d.source='lab'
            LEFT JOIN events e ON e.itemid = d.dense_id
            GROUP BY 1,2,3,4,5,6
            ORDER BY 1
        ) TO '{dict_path}' (FORMAT PARQUET, COMPRESSION 'zstd')
    """)
    print(f"  dictionary -> {dict_path}")

    return {"n_stays": int(n_stays), "n_chart": int(n_chart), "n_lab": int(n_lab),
            "n_records": int(n_records), "k_chart": int(k_chart),
            "k_total": int(k_total), "t_max_hours": round(float(t_max), 4),
            "value_min": float(v_min), "value_max": float(v_max),
            "itemid_dict_sha256": sha256_file(dict_path)}


def write_csr(con, out_dir: Path, n_stays: int, where: str,
              stay_table: str | None = None) -> dict:
    """
    Write one CSR directory.

    `where` filters `events`. `stay_table` optionally names a temp table with
    (row_index, dense_row) restricting and re-indexing the stay set for D10
    subsets; when None, cohort.row_index is used directly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if stay_table:
        idx_expr = f"s.dense_row"
        src = f"events e JOIN {stay_table} s USING (row_index)"
        all_rows = f"SELECT dense_row AS row_index FROM {stay_table}"
    else:
        idx_expr = "e.row_index"
        src = "events e"
        all_rows = "SELECT row_index FROM cohort"

    counts = con.execute(f"""
        WITH per AS (
            SELECT {idx_expr} AS ri, count(*) AS n
            FROM {src} WHERE {where} GROUP BY 1
        )
        SELECT a.row_index, coalesce(per.n, 0) AS n
        FROM ({all_rows}) a LEFT JOIN per ON per.ri = a.row_index
        ORDER BY a.row_index
    """).fetchnumpy()

    n = counts["n"].astype(np.int64)
    assert len(n) == n_stays, (len(n), n_stays)
    offsets = np.zeros(n_stays + 1, dtype=np.int64)
    np.cumsum(n, out=offsets[1:])
    n_records = int(offsets[-1])

    offsets.tofile(out_dir / "offsets.i64")


    stream = con.sql(f"""
        SELECT e.itemid, e.t, e.value
        FROM {src} WHERE {where}
        ORDER BY {idx_expr}, e.t, e.itemid, e.value
    """)

    written = 0
    with (out_dir / "itemid.i32").open("wb") as fi, \
         (out_dir / "t.f32").open("wb") as ft, \
         (out_dir / "value.f32").open("wb") as fv:
        reader = stream.fetch_record_batch(BATCH_ROWS)
        for batch in reader:
            fi.write(batch.column("itemid").to_numpy(zero_copy_only=False)
                     .astype(np.int32, copy=False).tobytes())
            ft.write(batch.column("t").to_numpy(zero_copy_only=False)
                     .astype(np.float32).tobytes())
            fv.write(batch.column("value").to_numpy(zero_copy_only=False)
                     .astype(np.float32).tobytes())
            written += batch.num_rows

    assert written == n_records, f"stream wrote {written}, offsets say {n_records}"
    for name, itemsize, count in (("itemid.i32", 4, n_records),
                                  ("t.f32", 4, n_records),
                                  ("value.f32", 4, n_records),
                                  ("offsets.i64", 8, n_stays + 1)):
        got = (out_dir / name).stat().st_size
        assert got == itemsize * count, (name, got, itemsize * count)

    nz = int((n > 0).sum())
    print(f"    {out_dir.name:<22} {n_stays:>8,} stays  {n_records:>13,} records  "
          f"{4*3*n_records/2**30:6.3f} GiB  empty={n_stays-nz}")

    return {"n_stays": n_stays, "n_records": n_records,
            "n_empty_stays": n_stays - nz,
            "bytes": int(8 * (n_stays + 1) + 12 * n_records),
            "sha256": {f: sha256_file(out_dir / f) for f in
                       ("offsets.i64", "itemid.i32", "t.f32", "value.f32")}}


def write_meta(out_dir: Path, base: dict, extra: dict) -> None:
    (out_dir / "meta.json").write_text(json.dumps({**base, **extra}, indent=2) + "\n")


def verify_prefix(full_dir: Path, win_dir: Path, W: float, n_sample: int) -> bool:
    off_f = np.fromfile(full_dir / "offsets.i64", dtype=np.int64)
    off_w = np.fromfile(win_dir / "offsets.i64", dtype=np.int64)
    if len(off_f) != len(off_w):
        print(f"    {win_dir.name}: offset length mismatch")
        return False

    n_stays = len(off_f) - 1
    arrays = {}
    for d, tag in ((full_dir, "f"), (win_dir, "w")):
        for nm, dt in (("itemid.i32", np.int32), ("t.f32", np.float32),
                       ("value.f32", np.float32)):
            arrays[(tag, nm)] = np.memmap(d / nm, dtype=dt, mode="r")

    rng = np.random.default_rng(SUBSET_SEED)
    sample = rng.choice(n_stays, size=min(n_sample, n_stays), replace=False)

    for i in sample:
        a, b = off_f[i], off_f[i + 1]
        seg_t = arrays[("f", "t.f32")][a:b]
        if not np.all(np.diff(seg_t) >= 0):
            print(f"    stay {i}: t not sorted in full window")
            return False
        k = int(np.searchsorted(seg_t, np.float32(W), side="left"))
        c, d_ = off_w[i], off_w[i + 1]
        if d_ - c != k:
            print(f"    stay {i}: window count {d_-c} != prefix length {k}")
            return False
        for nm in ("itemid.i32", "t.f32", "value.f32"):
            if not np.array_equal(arrays[("f", nm)][a:a + k],
                                  arrays[("w", nm)][c:d_]):
                print(f"    stay {i}: {nm} differs from full-window prefix")
                return False

    print(f"    {win_dir.name:<22} prefix oracle OK on {len(sample):,} sampled stays")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=",".join(WINDOWS),
                    help="comma-separated subset of " + ",".join(WINDOWS))
    ap.add_argument("--verify", action="store_true", help="sampled prefix oracle")
    ap.add_argument("--verify-sample", type=int, default=2000)
    ap.add_argument("--families", action="store_true", help="D10 variants")
    args = ap.parse_args()

    wanted = [w.strip() for w in args.windows.split(",")]
    if set(wanted) - set(WINDOWS):
        print(f"unknown window(s): {sorted(set(wanted) - set(WINDOWS))}",
              file=sys.stderr)
        return 2

    if not (COHORT / "cohort.parquet").is_file():
        print("cohort.parquet missing -- run cohort.py first", file=sys.stderr)
        return 2

    con = connect()
    print(f"duckdb {duckdb.__version__} | numpy {np.__version__} | "
          f"threads {THREADS} | mem {MEMORY_LIMIT}")

    banner("Extraction (D6 / D7)")
    info = build_events(con)
    n_stays = info["n_stays"]

    base = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mimic_version": "3.1",
        "duckdb_version": duckdb.__version__,
        "row_order": "random_permutation",
        "row_order_source": "cohort.row_index",
        "k_chart": info["k_chart"],
        "k_total": info["k_total"],
        "itemid_dict_sha256": info["itemid_dict_sha256"],
        "layout": {"offsets": "int64[n_stays+1]", "itemid": "int32[n_records]",
                   "t": "float32[n_records] hours since intime",
                   "value": "float32[n_records] from valuenum"},
    }

    banner("Windows (D9)")
    results = {}
    for w in wanted:
        W = WINDOWS[w]
        where = "TRUE" if W is None else f"e.t < {W}"
        d = CSR / f"W{w}"
        results[w] = write_csr(con, d, n_stays, where)
        write_meta(d, base, {"window": w, "window_hours": W, **results[w]})

    if args.verify:
        banner("Prefix oracle")
        if "full" not in wanted:
            print("    skipped: W=full not built in this run")
        else:
            ok = all(verify_prefix(CSR / "Wfull", CSR / f"W{w}", WINDOWS[w],
                                   args.verify_sample)
                     for w in wanted if WINDOWS[w] is not None)
            if not ok:
                return 1

    if args.families:
        banner("D10 size-matched (vary tail index at fixed record count)")
        target = min(r["n_records"] for r in results.values())
        print(f"    target record count {target:,} (smallest window)")
        for w in wanted:
            W = WINDOWS[w]
            where = "TRUE" if W is None else f"e.t < {W}"
            # Take stays in permutation order until the target is reached, so
            # the subset stays a prefix of the canonical ordering.
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE sub AS
                WITH per AS (
                    SELECT c.row_index, count(e.t) AS n
                    FROM cohort c LEFT JOIN events e
                      ON e.row_index = c.row_index AND {where.replace('e.t', 'e.t')}
                    GROUP BY 1
                ), cum AS (
                    SELECT row_index, n,
                           sum(n) OVER (ORDER BY row_index
                                        ROWS UNBOUNDED PRECEDING) AS csum
                    FROM per
                )
                SELECT row_index,
                       (row_number() OVER (ORDER BY row_index) - 1)::INTEGER AS dense_row
                FROM cum WHERE csum - n < {target};
            """)
            m = con.execute("SELECT count(*) FROM sub").fetchone()[0]
            d = CSR / f"sizematched_W{w}"
            r = write_csr(con, d, m, where, stay_table="sub")
            write_meta(d, base, {"family": "size_matched", "window": w,
                                 "window_hours": W, "target_records": int(target),
                                 **r})

        banner("D10 shape-varied (vary tail index at fixed window)")
        for k in SHAPE_TRIM_PCT:
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE sub AS
                WITH per AS (
                    SELECT c.row_index, count(e.t) AS n
                    FROM cohort c LEFT JOIN events e
                      ON e.row_index = c.row_index AND e.t < 24.0
                    GROUP BY 1
                ), ranked AS (
                    SELECT row_index, n,
                           percent_rank() OVER (ORDER BY n DESC) AS pr
                    FROM per
                )
                SELECT row_index,
                       (row_number() OVER (ORDER BY row_index) - 1)::INTEGER AS dense_row
                FROM ranked WHERE pr >= {k / 100.0};
            """)
            m = con.execute("SELECT count(*) FROM sub").fetchone()[0]
            d = CSR / f"shape_W24_trim{k}"
            r = write_csr(con, d, m, "e.t < 24.0", stay_table="sub")
            write_meta(d, base, {"family": "shape_varied", "window": "24",
                                 "window_hours": 24.0, "trim_top_pct": k, **r})

    banner("Summary")
    total = sum(r["bytes"] for r in results.values())
    for w in wanted:
        r = results[w]
        print(f"  W{w:<6} {r['n_records']:>13,} records  "
              f"{r['bytes']/2**30:6.3f} GiB")
    print(f"  {'total':<7} {'':>13} {total/2**30:6.3f} GiB")

    out = MANIFEST / "csr_summary.json"
    out.write_text(json.dumps({**base, "extraction": info,
                               "windows": results}, indent=2) + "\n")
    print(f"\n  summary -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
