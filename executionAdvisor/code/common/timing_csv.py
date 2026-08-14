#!/usr/bin/env python3

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass, field, asdict, fields

SCHEMA_VERSION = 2

SUPPORTED_VERSIONS = {1, 2}

TIMING_COLUMNS = [
    "schema_version", "run_id", "job_id", "node", "git_sha", "timestamp",
    "csr_variant", "window_W", "n_stays", "n_records",
    "paradigm", "platform", "gpu", "kernel_variant", "precision",
    "partitioner", "schedule", "chunk_size", "row_order",
    "n_ranks", "n_threads", "smt", "thread_placement", "n_nodes",
    "stage_mode", "numa_balancing", "mempolicy", "whitelist_frac",
    "k_iterations", "timed_region", "repetition", "warmup",
    "wall_time_s", "throughput_rec_s",
    "work_imbalance", "time_imbalance",
    "n_records_max_thread", "n_records_mean_thread",
    "bitidentical", "l3_tcm", "l3_tca", "tot_cyc", "tot_ins",
    "achieved_ghz", "notes",
]

THREAD_COLUMNS = ["schema_version", "run_id", "tid", "cpu", "socket",
                  "n_stays", "n_records", "secs"]

PARTITIONER = {"none", "block", "cyclic", "block_cyclic", "nzbalanced",
               "wbalanced", "contig_opt", "greedy"}
SCHEDULE = {"precomputed", "static", "static_chunked", "dynamic", "guided",
            "stealing"}
ROW_ORDER = {"canonical", "sorted_desc", "sorted_asc"}
STAGE_MODE = {"serial", "interleave", "mbind"}
MEMPOLICY = {"default", "interleave", "bind"}
TIMED_REGION = {"kernel", "end_to_end"}
BITIDENTICAL = {"pass", "fail", "skip"}
THREAD_PLACEMENT = {"socket0", "socket1", "spread", "smt", "unknown"}

CHUNKED = {"static_chunked", "dynamic", "guided", "stealing"}


@dataclass
class TimingRow:
    csr_variant: str                      # required; window_W is NOT a key
    wall_time_s: float
    schema_version: int = SCHEMA_VERSION
    run_id: str = ""
    job_id: int = -1
    node: str = ""
    git_sha: str = ""
    timestamp: str = ""
    window_W: str = ""
    n_stays: int = 0
    n_records: int = 0
    paradigm: str = "serial"
    platform: str = "broadwell"
    gpu: str = "none"
    kernel_variant: str = "shifted"
    precision: str = "fp64"
    partitioner: str = "block"
    schedule: str = "static"
    chunk_size: int = -1
    row_order: str = "canonical"
    n_ranks: int = 1
    n_threads: int = 1
    smt: int = 0
    thread_placement: str = "unknown"
    n_nodes: int = 1
    stage_mode: str = "interleave"        
    numa_balancing: int = -1
    mempolicy: str = "default"
    whitelist_frac: float = 1.0
    k_iterations: int = 1
    timed_region: str = "kernel"
    repetition: int = 0
    warmup: int = 1
    throughput_rec_s: float = 0.0
    work_imbalance: float = 0.0
    time_imbalance: float = 0.0
    n_records_max_thread: int = 0
    n_records_mean_thread: float = 0.0
    bitidentical: str = "skip"
    l3_tcm: int = -1
    l3_tca: int = -1
    tot_cyc: int = -1
    tot_ins: int = -1
    achieved_ghz: float = -1.0
    notes: str = ""


def validate(row) -> list:
    """Mirror of timing_row_validate() in timing_csv.h. Returns a list of
    problems; empty means valid."""
    g = (lambda k: row[k]) if isinstance(row, dict) else (lambda k: getattr(row, k))
    e = []
    if int(g("schema_version")) not in SUPPORTED_VERSIONS:
        e.append(f"schema_version {g('schema_version')} not in {SUPPORTED_VERSIONS}")
    if not str(g("csr_variant")):
        e.append("csr_variant is required (window_W is not a key)")
    for name, allowed in (("partitioner", PARTITIONER), ("schedule", SCHEDULE),
                          ("row_order", ROW_ORDER), ("stage_mode", STAGE_MODE),
                          ("mempolicy", MEMPOLICY),
                          ("timed_region", TIMED_REGION),
                          ("bitidentical", BITIDENTICAL),
                          ("thread_placement", THREAD_PLACEMENT)):
        if name == "thread_placement" and int(g("schema_version")) < 2:
            continue
        if str(g(name)) not in allowed:
            e.append(f"bad {name} '{g(name)}'")

    sched, chunk = str(g("schedule")), int(g("chunk_size"))
    if sched in CHUNKED and chunk < 1:
        e.append(f"schedule '{sched}' requires chunk_size >= 1")
    if sched not in CHUNKED and chunk != -1:
        e.append(f"schedule '{sched}' must have chunk_size = -1")

  
    part = str(g("partitioner"))
    if part == "none" and sched not in {"dynamic", "guided", "stealing"}:
        e.append(f"partitioner 'none' with schedule '{sched}': record the "
                 f"explicit partitioner instead")
    if sched == "precomputed" and part == "none":
        e.append("schedule 'precomputed' requires a partitioner")

    if not 0.0 <= float(g("whitelist_frac")) <= 1.0:
        e.append("whitelist_frac outside [0,1]")
    if (int(g("repetition")) == 0) != (int(g("warmup")) == 1):
        e.append("warmup must be 1 iff repetition == 0")
    if int(g("n_threads")) < 1:
        e.append("n_threads < 1")
    if (int(g("n_threads")) > 16 and not int(g("smt"))
            and str(g("platform")) == "broadwell"):
        e.append("n_threads > 16 on broadwell requires smt=1")
    if int(g("k_iterations")) < 1:
        e.append("k_iterations < 1")
    if float(g("wall_time_s")) <= 0:
        e.append("wall_time_s <= 0")
    return e


def open_timings(path):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=TIMING_COLUMNS, extrasaction="raise")
    if new:
        w.writeheader()
    return fh, w


def write_row(writer, row: TimingRow):
    problems = validate(row)
    if problems:
        raise ValueError("invalid timing row: " + "; ".join(problems))
    d = asdict(row)
    d["notes"] = str(d["notes"]).replace(",", ";").replace("\n", " ")
    writer.writerow({k: d[k] for k in TIMING_COLUMNS})


def load(path, drop_warmup=True):
    import pandas as pd
    df = pd.read_csv(path)
    missing = [c for c in TIMING_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in TIMING_COLUMNS]
    if missing or extra:
        raise ValueError(f"schema mismatch: missing={missing} extra={extra}")
    records = df.to_dict("records")
    bad = [(i, problems) for i, r in enumerate(records)
           if (problems := validate(r))]
    if bad:
        print(f"WARNING: {len(bad)} invalid rows; first: {bad[0]}",
              file=sys.stderr)
    return df[df.warmup == 0] if drop_warmup else df



def header_from_c(path):
    """Extract TIMING_CSV_HEADER from timing_csv.h by concatenating its
    continued string literals."""
    src = open(path).read()
    m = re.search(r"#define\s+TIMING_CSV_HEADER\s+((?:.*?\\\n)*.*)", src)
    if not m:
        raise ValueError("TIMING_CSV_HEADER not found in " + path)
    return "".join(re.findall(r'"([^"]*)"', m.group(1)))


def check_drift(header_path):
    c_cols = header_from_c(header_path).split(",")
    ok = True
    if c_cols != TIMING_COLUMNS:
        ok = False
        print("SCHEMA DRIFT between timing_csv.h and timing_csv.py\n")
        for i, (a, b) in enumerate(
                zip(c_cols + [""] * len(TIMING_COLUMNS),
                    TIMING_COLUMNS + [""] * len(c_cols))):
            if a != b:
                print(f"  col {i:>2}:  C='{a}'   py='{b}'")
        print(f"\n  C has {len(c_cols)} columns, Python has {len(TIMING_COLUMNS)}")
    else:
        print(f"schema v{SCHEMA_VERSION}: C and Python agree on "
              f"{len(c_cols)} columns")

    m = re.search(r"#define\s+TIMING_SCHEMA_VERSION\s+(\d+)", open(header_path).read())
    if m and int(m.group(1)) != SCHEMA_VERSION:
        ok = False
        print(f"  VERSION DRIFT: C={m.group(1)} py={SCHEMA_VERSION}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", metavar="TIMING_CSV_H",
                    help="compare column list against the C header")
    ap.add_argument("--print-header", action="store_true")
    a = ap.parse_args()
    if a.print_header:
        print(",".join(TIMING_COLUMNS))
        sys.exit(0)
    if a.check:
        sys.exit(check_drift(a.check))
    ap.print_help()
