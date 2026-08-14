#!/usr/bin/env bash

set -euo pipefail
export USER=sb2ea

REPO=/home/sb2ea/loadimbalance
PROJ=/projects/sb2ea
COMMON="$REPO/code/common"
OUT="$PROJ/bin/snb"
mkdir -p "$OUT"

python3 "$COMMON/timing_csv.py" --check "$COMMON/timing_csv.h" || {
    echo "FATAL: timing schema drift" >&2; exit 1; }

conda deactivate 2>/dev/null || true
if [ -z "${MODULEPATH:-}" ] && [ -f /etc/profile.d/lmod.sh ]; then
    . /etc/profile.d/lmod.sh
fi
module load gnu12 2>/dev/null || { echo "FATAL: gnu12 not loadable" >&2; exit 1; }
echo "gcc: $(which gcc)  $(gcc --version | head -n1)"


FLAGS="-O3 -std=c11 -g -march=sandybridge -mtune=sandybridge -Wall -Wextra"
echo "flags: $FLAGS"
echo

PAPI_CFLAGS=""; PAPI_LDFLAGS=""
if module load papi 2>/dev/null; then
    PAPI_CFLAGS="-DUSE_PAPI ${PAPI_INC:+-I$PAPI_INC}"
    PAPI_LDFLAGS="${PAPI_LIB:+-L$PAPI_LIB -Wl,-rpath,$PAPI_LIB} -lpapi"
    echo "PAPI: enabled"
else
    echo "PAPI: not found -- counters will be -1" >&2
fi

gcc $FLAGS -I "$COMMON" -c "$COMMON/kernel_core.c" -o "$OUT/kernel_core.o"
sha256sum "$OUT/kernel_core.o" | tee "$OUT/kernel_core.o.sha256"
echo "  (expected to DIFFER from the haswell object -- different ISA)"
gcc --version | head -n1 > "$OUT/kernel_core.o.cc"
echo

gcc $FLAGS -I "$COMMON" "$REPO/code/kernel_serial.c" \
    "$OUT/kernel_core.o" -o "$OUT/kernel_serial" -lm
echo "built: $OUT/kernel_serial"

gcc $FLAGS -fopenmp $PAPI_CFLAGS -I "$COMMON" "$REPO/code/kernel_omp.c" \
    "$OUT/kernel_core.o" -o "$OUT/kernel_omp" -lm $PAPI_LDFLAGS
echo "built: $OUT/kernel_omp"

gcc -O2 -std=c11 -g -o "$OUT/numa_pages" "$REPO/code/probes/numa_pages.c"
gcc -O2 -std=c11 -g -fopenmp -o "$OUT/stage_csr" "$REPO/code/probes/stage_csr.c"
gcc -O3 -std=c11 -g -march=sandybridge -fopenmp \
    -o "$OUT/csr_stream" "$REPO/code/probes/csr_stream.c"
echo "built: numa_pages, stage_csr, csr_stream"
echo
echo "run with:  --export=ALL,BIN_OVERRIDE=$OUT"
