#!/usr/bin/env bash



set -euo pipefail
export USER=sb2ea

REPO=/home/sb2ea/loadimbalance
PROJ=/projects/sb2ea
COMMON="$REPO/code/common"
OUT="$PROJ/bin/el8"
EXPECT=38066a02e9d683cd7e12cb4e84885f2757d7aeabe5cf1933b818a1ac57ff6706
mkdir -p "$OUT"

python3 "$COMMON/timing_csv.py" --check "$COMMON/timing_csv.h" || {
    echo "FATAL: timing schema drift" >&2; exit 1; }

conda deactivate 2>/dev/null || true
if [ -z "${MODULEPATH:-}" ] && [ -f /etc/profile.d/lmod.sh ]; then
    . /etc/profile.d/lmod.sh
fi
module load gnu12 2>/dev/null || { echo "FATAL: gnu12 not loadable" >&2; exit 1; }
echo "gcc: $(which gcc)  $(gcc --version | head -n1)"

FLAGS="-O3 -std=c11 -g -march=haswell -mtune=broadwell -Wall -Wextra"
echo "flags: $FLAGS   (link adds -static)"
echo

gcc $FLAGS -I "$COMMON" -c "$COMMON/kernel_core.c" -o "$OUT/kernel_core.o"
GOT=$(sha256sum "$OUT/kernel_core.o" | awk '{print $1}')
echo "$GOT  kernel_core.o" | tee "$OUT/kernel_core.o.sha256"
if [ "$GOT" = "$EXPECT" ]; then
    echo "  MATCHES the Broadwell object -- cross-platform bit-identity is"
    echo "  expected between c20 and c6-c8, and is worth asserting."
else
    echo "  DIFFERS from the Broadwell object. Cross-platform bit-identity" >&2
    echo "  is then not expected; investigate before comparing matrices." >&2
fi
echo

SYSROOT=/opt/ohpc/pub/apps/miniconda/pkgs/sysroot_linux-64-2.17-h4a8ded7_17/x86_64-conda-linux-gnu/sysroot/usr/lib64


LINK="-static -static-libgcc -L$SYSROOT"

gcc $FLAGS $LINK -I "$COMMON" "$REPO/code/kernel_serial.c" \
    "$OUT/kernel_core.o" -o "$OUT/kernel_serial" -lm
echo "built: $OUT/kernel_serial"

gcc $FLAGS $LINK -fopenmp -I "$COMMON" "$REPO/code/kernel_omp.c" \
    "$OUT/kernel_core.o" -o "$OUT/kernel_omp" -lm
echo "built: $OUT/kernel_omp  (no PAPI)"

gcc -O2 -std=c11 -g $LINK -o "$OUT/numa_pages" "$REPO/code/probes/numa_pages.c"
gcc -O2 -std=c11 -g $LINK -fopenmp -o "$OUT/stage_csr" "$REPO/code/probes/stage_csr.c"
gcc -O3 -std=c11 -g -march=haswell $LINK -fopenmp \
    -o "$OUT/csr_stream" "$REPO/code/probes/csr_stream.c"
echo "built: numa_pages, stage_csr, csr_stream"
echo

echo "linkage check (should say 'not a dynamic executable'):"
for b in kernel_serial kernel_omp stage_csr numa_pages; do
    printf '  %-14s ' "$b"
    ldd "$OUT/$b" 2>&1 | head -1
done
echo
echo "run with:  --export=ALL,BIN_OVERRIDE=$OUT"
