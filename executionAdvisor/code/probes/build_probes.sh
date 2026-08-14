#!/usr/bin/env bash

set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-/projects/sb2ea/bin}"
mkdir -p "$OUT"

echo "gcc: $(which gcc)"
gcc --version | head -1


CFLAGS_STREAM="-O3 -std=c11 -g -march=haswell -mtune=broadwell -fopenmp"

gcc -O2 -std=c11 -g            -o "$OUT/numa_pages" "$SRC/numa_pages.c"
gcc $CFLAGS_STREAM             -o "$OUT/csr_stream" "$SRC/csr_stream.c"
gcc -O2 -std=c11 -g -fopenmp   -o "$OUT/stage_csr"  "$SRC/stage_csr.c"

echo
ls -l "$OUT/numa_pages" "$OUT/csr_stream" "$OUT/stage_csr"
echo "OK"
