#!/usr/bin/env python3

import argparse
import difflib
import os
import re
import sys


def cut_between(src, start_marker, end_marker, label, results):
    """Delete start_marker .. end_marker inclusive. Both must appear once."""
    i = src.find(start_marker)
    if i < 0:
        results.append((label, False, f"start anchor not found: {start_marker[:60]!r}"))
        return src
    j = src.find(end_marker, i)
    if j < 0:
        results.append((label, False, f"end anchor not found: {end_marker[:60]!r}"))
        return src
    j += len(end_marker)
    results.append((label, True, f"removed {src[i:j].count(chr(10)) + 1} lines"))
    return src[:i] + src[j:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    src = open(a.path).read()
    orig = src
    results = []

    # -- 1. constants now live in kernel_core.h 
    m = re.search(r"#define N_AGG\s+8\s*\n#define MAX_DENSE\s+65536\s*\n", src)
    if m:
        src = src[:m.start()] + src[m.end():]
        results.append(("N_AGG / MAX_DENSE defines", True, "removed"))
    else:
        results.append(("N_AGG / MAX_DENSE defines", False, "not found"))

    # -- 2. lookup_t struct 
    src = cut_between(src,
                      "/* ---------------------------------------------------------------------\n"
                      " * Lookup table (D26).",
                      "} lookup_t;\n",
                      "lookup_t struct", results)

    # -- 3. load_lookup() 
    src = cut_between(src,
                      "/* kernel_lookup.csv columns:",
                      "    return L;\n}\n",
                      "load_lookup()", results)

    # -- 4. acc_t struct 
    src = cut_between(src,
                      "/* ---------------------------------------------------------------------\n"
                      " * Per-stay accumulator,",
                      "} acc_t;\n",
                      "acc_t struct", results)

    # -- 5. include the shared core 
    anchor = "#include <unistd.h>\n"
    if anchor in src and '#include "kernel_core.h"' not in src:
        src = src.replace(anchor, anchor + '\n#include "kernel_core.h"\n', 1)
        results.append(('#include "kernel_core.h"', True, "added"))
    else:
        results.append(('#include "kernel_core.h"',
                        '#include "kernel_core.h"' in src,
                        "already present" if '#include "kernel_core.h"' in src
                        else "anchor <unistd.h> not found"))

    # -- 6. replace the per-stay body with a kernel_stay() call 
    start = src.find("            for (int32_t s = 0; s < S; s++) {\n"
                     "                acc[s].n = 0;")
    end_marker = ("                    o[7] = (a->n >= 2 && t_ss > 0.0) ? cov / t_ss : NAN;\n"
                  "                }\n"
                  "            }\n")
    end = src.find(end_marker, start) if start >= 0 else -1
    if start >= 0 and end >= 0:
        end += len(end_marker)
        n_lines = src[start:end].count("\n")
        src = (src[:start]
               + "            kernel_stay(offsets, itemid, t, value, i, L, acc,\n"
                 "                        tile + (i - base) * F);\n"
               + src[end:])
        results.append(("per-stay body -> kernel_stay()", True,
                        f"replaced {n_lines} lines with 2"))
    else:
        results.append(("per-stay body -> kernel_stay()", False,
                        "anchors not found -- has the file already been refactored?"))

    # -- 7. S is used only for F once the body is gone; fold it in 
    pair = ("    const int32_t S = L->n_slots;\n"
            "    const int64_t F = (int64_t)S * N_AGG;\n")
    if pair in src:
        src = src.replace(pair,
                          "    const int64_t F = (int64_t)L->n_slots * N_AGG;\n", 1)
        results.append(("fold `S` into F", True,
                        "S was used only by the extracted body"))
    elif "const int32_t S = L->n_slots;" in src:
        results.append(("fold `S` into F", False,
                        "S declaration found but not in the expected pair"))

    # -- report 
    print(f"{'step':<34} {'ok':<5} detail")
    print("-" * 74)
    ok = True
    for label, good, detail in results:
        ok &= bool(good)
        print(f"{label:<34} {'yes' if good else 'NO':<5} {detail}")
    print()

    if not ok:
        print("REFUSING to write: at least one anchor did not match.")
        print("The file is not in the state this script expects. Do not")
        print("hand-edit around it -- send me the file and I will re-anchor.")
        return 1

    diff = list(difflib.unified_diff(orig.splitlines(True), src.splitlines(True),
                                     fromfile=a.path, tofile=a.path + " (refactored)"))
    print("".join(diff))

    if not a.write:
        print("\nDRY RUN. Re-run with --write to apply.")
        return 0

    bak = a.path + ".preextract.bak"
    if not os.path.exists(bak):
        open(bak, "w").write(orig)
        print(f"\nbackup: {bak}")
    open(a.path, "w").write(src)
    print(f"written: {a.path}")
    print("\nNEXT, in order:")
    print("  1. snapshot the OLD binary's output BEFORE rebuilding, if you")
    print("     have not already -- this step is irreversible:")
    print("       /projects/sb2ea/bin/kernel_serial --csr /projects/sb2ea/csr/W24 \\")
    print("         --lookup /projects/sb2ea/manifest/kernel_lookup.csv \\")
    print("         --out /tmp/ref_preextract.f64")
    print("  2. rebuild, then produce /tmp/ref_postextract.f64 the same way")
    print("  3. cmp /tmp/ref_preextract.f64 /tmp/ref_postextract.f64")
    print("     Byte comparison is valid: NaN != NaN is a COMPARISON issue,")
    print("     not a bit issue, and every NaN here comes from the same")
    print("     NAN macro.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
