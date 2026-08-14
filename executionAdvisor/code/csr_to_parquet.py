#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJ = "/projects/sb2ea"

SCHEMA = pa.schema([
    ("row_index", pa.int32()),
    ("dense_id",  pa.int32()),
    ("t",         pa.float32()),
    ("value",     pa.float32()),
])


def sha256(path, buf=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(buf):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="Wfull")
    ap.add_argument("--csr-root", default=f"{PROJ}/csr")
    ap.add_argument("--out", default=f"{PROJ}/work/events_full.parquet")
    ap.add_argument("--chunk-records", type=int, default=8_000_000)
    ap.add_argument("--verify-sha", action="store_true",
                    help="check binaries against meta.json (slow, ~2 GiB read)")
    args = ap.parse_args()

    d = os.path.join(args.csr_root, args.window)
    f_off = os.path.join(d, "offsets.i64")
    f_itm = os.path.join(d, "itemid.i32")
    f_t   = os.path.join(d, "t.f32")
    f_val = os.path.join(d, "value.f32")
    for f in (f_off, f_itm, f_t, f_val):
        if not os.path.exists(f):
            sys.exit(f"FATAL: missing {f}")

    # ---- offsets define the geometry 
    offsets = np.fromfile(f_off, dtype="<i8")
    n_stays = offsets.size - 1
    n_records = int(offsets[-1])

    # ---- cross-check against meta.json 
    meta_path = os.path.join(d, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        for key, got in (("n_stays", n_stays), ("n_records", n_records)):
            want = meta.get(key)
            if want is not None and int(want) != got:
                sys.exit(f"FATAL: {key} mismatch -- meta.json {want}, binaries {got}")
        print(f"meta.json  : n_stays={n_stays} n_records={n_records} OK")
    else:
        print(f"meta.json  : absent; n_stays={n_stays} n_records={n_records}")

    # ---- file sizes must agree with n_records 
    for f, width, name in ((f_itm, 4, "itemid"), (f_t, 4, "t"), (f_val, 4, "value")):
        expect = n_records * width
        actual = os.path.getsize(f)
        if actual != expect:
            sys.exit(f"FATAL: {name} is {actual} B, expected {expect} B "
                     f"({n_records} records x {width})")
    print("sizes      : all four binaries consistent")

    if args.verify_sha and meta:
        for name, f in (("offsets.i64", f_off), ("itemid.i32", f_itm),
                        ("t.f32", f_t), ("value.f32", f_val)):
            want = (meta.get("sha256") or {}).get(name)
            if want:
                got = sha256(f)
                status = "OK" if got == want else "MISMATCH"
                print(f"sha256 {name:<12} {status}")
                if status == "MISMATCH":
                    sys.exit("FATAL: CSR binary does not match its manifest")

    # ---- mmap, never load whole arrays 
    itemid = np.memmap(f_itm, dtype="<i4", mode="r")
    t      = np.memmap(f_t,   dtype="<f4", mode="r")
    value  = np.memmap(f_val, dtype="<f4", mode="r")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = pq.ParquetWriter(args.out, SCHEMA, compression="zstd")

    # ---- walk stays in blocks bounded by record count 
    written = 0
    n_neg_t = 0
    n_bad_id = 0
    id_min, id_max = np.iinfo(np.int32).max, np.iinfo(np.int32).min
    s0 = 0
    blocks = 0

    while s0 < n_stays:
        target = offsets[s0] + args.chunk_records
        s1 = int(np.searchsorted(offsets, target, side="left"))
        s1 = max(s1, s0 + 1)          # always make progress on a huge stay
        s1 = min(s1, n_stays)

        r0, r1 = int(offsets[s0]), int(offsets[s1])
        if r1 > r0:
            counts = np.diff(offsets[s0:s1 + 1]).astype(np.int64)
            rows = np.repeat(
                np.arange(s0, s1, dtype=np.int32), counts
            )
            ids  = np.asarray(itemid[r0:r1], dtype=np.int32)
            ts   = np.asarray(t[r0:r1],      dtype=np.float32)
            vals = np.asarray(value[r0:r1],  dtype=np.float32)

            n_neg_t += int((ts < 0).sum())
            n_bad_id += int((ids < 0).sum())
            if ids.size:
                id_min = min(id_min, int(ids.min()))
                id_max = max(id_max, int(ids.max()))

            writer.write_table(pa.table(
                {"row_index": rows, "dense_id": ids, "t": ts, "value": vals},
                schema=SCHEMA,
            ))
            written += r1 - r0
            blocks += 1

        s0 = s1
        if blocks % 5 == 0:
            pct = 100.0 * written / max(n_records, 1)
            print(f"  {written:>12,} / {n_records:,} records ({pct:5.1f}%)",
                  flush=True)

    writer.close()

    #  report 
    print()
    print(f"written    : {written:,} records in {blocks} row groups")
    if written != n_records:
        sys.exit(f"FATAL: wrote {written}, expected {n_records}")
    print(f"dense_id   : range [{id_min}, {id_max}]")
    print(f"negative t : {n_neg_t}   (must be 0 -- window predicate, D6)")
    print(f"negative id: {n_bad_id}  (must be 0)")
    if n_neg_t or n_bad_id:
        sys.exit("FATAL: invariant violated")

    size = os.path.getsize(args.out)
    print(f"output     : {args.out}  ({size / 2**30:.2f} GiB)")
    print("OK")


if __name__ == "__main__":
    main()
