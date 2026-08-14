#!/usr/bin/env bash
sh ~/loadimbalance/code/build_kernel.sh
#
set -euo pipefail
export USER=sb2ea

REPO=/home/sb2ea/loadimbalance
PROJ=/projects/sb2ea
COMMON="$REPO/code/common"
mkdir -p "$PROJ/bin"


python3 "$COMMON/timing_csv.py" --check "$COMMON/timing_csv.h" || {
    echo "FATAL: timing schema drift -- fix before building" >&2
    exit 1
}

# conda shadows the OHPC toolchain 
conda deactivate 2>/dev/null || true


if [ -z "${MODULEPATH:-}" ] && [ -f /etc/profile.d/lmod.sh ]; then
    # shellcheck disable=SC1091
    . /etc/profile.d/lmod.sh
fi
if ! module load gnu12 2>/dev/null; then
    echo "FATAL: could not load module gnu12." >&2
    echo "  Builds must be pinned to one compiler. Do not fall back to" >&2
    echo "  /usr/bin/gcc -- kernel_core.o would change under every driver." >&2
    exit 1
fi

echo "gcc: $(which gcc)  $(gcc --version | head -n1)"
case "$(gcc --version | head -n1)" in
    *12.4.0*) ;;
    *) echo "WARNING: expected gcc 12.4.0; comparability with existing" >&2
       echo "         timing runs is not guaranteed." >&2 ;;
esac


FLAGS="-O3 -std=c11 -g -march=haswell -mtune=broadwell -Wall -Wextra"
echo "flags: $FLAGS"


PAPI_CFLAGS=""; PAPI_LDFLAGS=""
if module load papi 2>/dev/null; then
    PAPI_CFLAGS="-DUSE_PAPI ${PAPI_INC:+-I$PAPI_INC}"
    PAPI_LDFLAGS="${PAPI_LIB:+-L$PAPI_LIB -Wl,-rpath,$PAPI_LIB} -lpapi"
    echo "PAPI: enabled  (inc=${PAPI_INC:-default} lib=${PAPI_LIB:-default})"
elif [ -f /usr/include/papi.h ]; then
    PAPI_CFLAGS="-DUSE_PAPI"; PAPI_LDFLAGS="-lpapi"
    echo "PAPI: enabled (system papi.h)"
else
    echo "PAPI: NOT FOUND -- l3_tcm/l3_tca/tot_cyc/tot_ins will be -1" >&2
fi
echo


gcc $FLAGS -I "$COMMON" -c "$COMMON/kernel_core.c" -o "$PROJ/bin/kernel_core.o"
# --- kernel_core.o identity, all ISA levels 
NEW_SHA=$(sha256sum "$PROJ/bin/kernel_core.o" | awk '{print $1}')
echo "$NEW_SHA  kernel_core.o" | tee "$PROJ/bin/kernel_core.o.sha256"

if [ -x "$REPO/code/verify_core_objects.sh" ]; then
    "$REPO/code/verify_core_objects.sh" || {
        echo "WARNING: kernel_core.o identity check FAILED" >&2
        echo "  Bit-identity with earlier runs must be re-established, or the" >&2
        echo "  expected values updated deliberately:" >&2
        echo "    $REPO/code/verify_core_objects.sh --update" >&2
    }
else
    echo "WARNING: verify_core_objects.sh missing; hashes unchecked" >&2
fi    

echo "$NEW_SHA  kernel_core.o" | tee "$PROJ/bin/kernel_core.o.sha256"

gcc --version | head -n1 > "$PROJ/bin/kernel_core.o.cc"
echo "built: $PROJ/bin/kernel_core.o"
echo

# --- serial reference 
gcc $FLAGS -I "$COMMON" "$REPO/code/kernel_serial.c" \
    "$PROJ/bin/kernel_core.o" -o "$PROJ/bin/kernel_serial" -lm
echo "built: $PROJ/bin/kernel_serial"
echo

# --- OpenMP driver 

gcc $FLAGS -fopenmp $PAPI_CFLAGS -I "$COMMON" "$REPO/code/kernel_omp.c" \
    "$PROJ/bin/kernel_core.o" -o "$PROJ/bin/kernel_omp" -lm $PAPI_LDFLAGS
echo "built: $PROJ/bin/kernel_omp"
echo

# --- Pthreads driver with work stealing 

gcc $FLAGS -fopenmp $PAPI_CFLAGS -I "$COMMON" "$REPO/code/kernel_pthreads.c" \
    "$PROJ/bin/kernel_core.o" -o "$PROJ/bin/kernel_pthreads" \
    -lm -lpthread $PAPI_LDFLAGS
echo "built: $PROJ/bin/kernel_pthreads"
echo

# --- hybrid MPI+OpenMP driver 

MPI_CC=""
for M in openmpi4 openmpi3 mpich impi openmpi; do
    if module load "$M" 2>/dev/null && command -v mpicc >/dev/null 2>&1; then
        MPI_CC=$(command -v mpicc); echo "MPI: $M -> $MPI_CC"; break
    fi
done
if [ -n "$MPI_CC" ]; then

    MPI_LIB="$(dirname "$(dirname "$MPI_CC")")/lib"

    RPATH="-Wl,--disable-new-dtags -Wl,-rpath,$MPI_LIB"

    HW=/opt/ohpc/pub/libs/hwloc/lib/libhwloc.so.15
    if [ ! -e "$HW" ]; then
        HW=$(find /opt/ohpc/pub -maxdepth 6 -name 'libhwloc.so.15*' \
                  -print -quit 2>/dev/null || true)
    fi
 
    if [ -n "$HW" ]; then
        RPATH="$RPATH -Wl,-rpath,$(dirname "$HW")"
    else
        echo "  (libhwloc not found under /opt/ohpc; rpath omits it)" >&2
    fi
    echo "MPI rpath: $RPATH"
    "$MPI_CC" $FLAGS -fopenmp -I "$COMMON" "$REPO/code/kernel_mpi.c" \
        "$PROJ/bin/kernel_core.o" -o "$PROJ/bin/kernel_mpi" -lm $RPATH
    echo "built: $PROJ/bin/kernel_mpi"
  
    if env -u LD_LIBRARY_PATH ldd "$PROJ/bin/kernel_mpi" 2>&1 | grep -q 'not found'; then
        echo "WARNING: kernel_mpi does not resolve without LD_LIBRARY_PATH:" >&2
        env -u LD_LIBRARY_PATH ldd "$PROJ/bin/kernel_mpi" 2>&1 | grep 'not found' >&2
        echo "  The rpath is incomplete; batch jobs will fail to launch it." >&2
    else
        echo "  resolves cleanly without LD_LIBRARY_PATH (rpath is complete)"
    fi
    readelf -d "$PROJ/bin/kernel_mpi" 2>/dev/null \
        | grep -E 'RPATH|RUNPATH' | sed 's/^/  /' || true
else
    echo "MPI: no module found (tried openmpi4/openmpi3/mpich/impi/openmpi)" >&2
    echo "     kernel_mpi NOT built; Days 8-9 need one of these." >&2
fi
echo


if [ -f "$REPO/code/probes/stage_aligned.c" ]; then
    gcc -O2 -std=c11 -g -fopenmp -o "$PROJ/bin/stage_aligned" \
        "$REPO/code/probes/stage_aligned.c"
    echo "built: $PROJ/bin/stage_aligned"
    echo
fi

gcc $FLAGS -ffast-math -I "$COMMON" -c "$COMMON/kernel_core.c" \
    -o "$PROJ/bin/kernel_core_fastmath.o"
gcc $FLAGS -ffast-math -I "$COMMON" "$REPO/code/kernel_serial.c" \
    "$PROJ/bin/kernel_core_fastmath.o" \
    -o "$PROJ/bin/kernel_serial_fastmath" -lm
echo "built: $PROJ/bin/kernel_serial_fastmath  (labelled variant, not the oracle)"
