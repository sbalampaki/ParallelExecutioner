#!/bin/bash
#SBATCH --job-name=e4_hybrid
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH --time=1:15:00
#SBATCH --output=/projects/sb2ea/logs/e4hyb_%j.out
#SBATCH --error=/projects/sb2ea/logs/e4hyb_%j.err


set -euo pipefail
export USER=sb2ea


MPI_PREFIX=/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.6
HWLOC_LIB=/opt/ohpc/pub/libs/hwloc/lib

MPIRUN="$MPI_PREFIX/bin/mpirun"
[ -x "$MPIRUN" ] || { echo "FATAL: $MPIRUN not executable" >&2; exit 1; }
export LD_LIBRARY_PATH="$MPI_PREFIX/lib:$HWLOC_LIB:${LD_LIBRARY_PATH:-}"
export PATH="$MPI_PREFIX/bin:$PATH"
echo "mpirun:   $MPIRUN"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Resolve everything before spending 30 minutes finding out we could not.
for b in "$MPIRUN" /projects/sb2ea/bin/kernel_mpi; do
    miss=$(ldd "$b" 2>/dev/null | grep 'not found' || true)
    if [ -n "$miss" ]; then
        echo "FATAL: unresolved libraries in $b" >&2
        echo "$miss" >&2
        echo "  If this is libmpi/libhwloc, rebuild: build_kernel.sh now bakes" >&2
        echo "  -rpath into kernel_mpi so it does not need a module at runtime." >&2
        exit 1
    fi
done

if ! "$MPIRUN" -np 2 --bind-to none -x LD_LIBRARY_PATH hostname >/dev/null 2>&1; then
    echo "FATAL: mpirun cannot launch 2 ranks here." >&2
    "$MPIRUN" -np 2 --bind-to none -x LD_LIBRARY_PATH hostname 2>&1 | head -10 >&2
    exit 1
fi
echo "mpirun launches 2 ranks OK"

BIN=/projects/sb2ea/bin
PROJ=/projects/sb2ea
LOOKUP=$PROJ/manifest/kernel_lookup.csv
OUT=$PROJ/results/timing
NODE=$(hostname -s)
GIT_SHA="${GIT_SHA:-unknown}"
TMP=/tmp/sb2ea_e4
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP" "$OUT"
TIM=$OUT/e4_${NODE}_${SLURM_JOB_ID}.csv
THR=$OUT/e4_${NODE}_${SLURM_JOB_ID}_threads.csv

echo "=== node $NODE  job $SLURM_JOB_ID  git $GIT_SHA  $(date -Is)"

declare -A SEEN
S0=(); S1=()
while IFS=, read -r cpu core sock; do
    [ -z "${cpu:-}" ] && continue
    key="$sock:$core"; [ -n "${SEEN[$key]:-}" ] && continue; SEEN[$key]=1
    if [ "$sock" = "0" ]; then S0+=("$cpu"); else S1+=("$cpu"); fi
done < <(lscpu -p=CPU,CORE,SOCKET | grep -v '^#')
NC0=${#S0[@]}; NC1=${#S1[@]}
echo "cores: socket0=$NC0 socket1=$NC1"
echo


echo "########## STAGING ##########"
OMP_NUM_THREADS=$((NC0+NC1)) OMP_PROC_BIND=close \
OMP_PLACES="$(printf '{%s},' "${S0[@]}" "${S1[@]}" | sed 's/,$//')" \
    "$BIN/stage_csr" --src "$PROJ/csr/W24" --dst "$TMP/W24" --mode interleave
OMP_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES="{${S0[0]}}" \
    "$BIN/stage_csr" --src "$PROJ/csr/W24" --dst "$TMP/W24_r0" --mode interleave
OMP_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES="{${S1[0]}}" \
    "$BIN/stage_csr" --src "$PROJ/csr/W24" --dst "$TMP/W24_r1" --mode interleave
for P in W24 W24_r0 W24_r1; do
    printf '  %-8s ' "$P"
    "$BIN/numa_pages" "$TMP/$P/value.f32" | tail -n +2 \
        | awk -F, '{printf "%s=%.4f  ", $4, $7} END {print ""}'
done
echo "  W24_r0 must be ~1.00 node0 and W24_r1 ~1.00 node1, or the"
echo "  per-rank test is not testing placement."
"$BIN/kernel_serial" --csr "$TMP/W24" --lookup "$LOOKUP" \
    --out "$TMP/oracle.f64" >/dev/null 2>&1
echo "  oracle generated"
echo

# grouped cpu list for one socket's first n cores
grp_list() { local -n A=$1; local n=$2 o="" i
    for ((i=0;i<n;i++)); do o+="${A[$i]},"; done; echo "${o%,}"; }

# --- reference: pure OpenMP, grouped, 16 threads 
echo "########## REFERENCE: OpenMP grouped 16 ##########"
PL="$(printf '{%s},' "${S0[@]}" "${S1[@]}" | sed 's/,$//')"
printf '  %-34s ' "openmp 16 grouped"
OMP_PROC_BIND=close OMP_PLACES="$PL" \
"$BIN/kernel_omp" --csr "$TMP/W24" --lookup "$LOOKUP" --variant W24 \
    --window 24 --threads 16 --schedule precomputed --partitioner block \
    --tile 0 --repeat 64 --reps 6 --warmup-s 2 --platform broadwell \
    --reference "$TMP/oracle.f64" --timings "$TIM" --threads-csv "$THR" \
    --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" 2>&1 \
  | awk '/^rep [1-9]/ {s+=$6; g+=$13; n++}
         END {printf "%8.1f M rec/s  %.2f GHz  %7.1f M rec/GHz\n", s/n, g/n, (s/n)/(g/n)}'
echo

# --- hybrid ratios 
# rank cpu groups, grouped order within each rank, ranks filling socket 0
# before socket 1 so that 2x8 is socket-pure.
ranks_for() {   # $1 = n_ranks  -> "cpus;cpus;..." with 16/n per rank
    local nr=$1 per=$((16 / $1)) o="" r i idx
    for ((r=0;r<nr;r++)); do
        local g=""
        for ((i=0;i<per;i++)); do
            idx=$((r * per + i))
            if (( idx < NC0 )); then g+="${S0[$idx]},"
            else                     g+="${S1[$((idx-NC0))]},"; fi
        done
        o+="${g%,};"
    done
    echo "${o%;}"
}

run_mpi() {   # $1 label  $2 nranks  $3 nthreads  $4 csr  $5 extra
    local log="$TMP/run.log"
    printf '  %-34s ' "$1"

    env -u OMP_PLACES OMP_PROC_BIND=false OMP_NUM_THREADS="$3" \
      "$MPIRUN" -np "$2" --bind-to none -x LD_LIBRARY_PATH \
        -x OMP_PROC_BIND -x OMP_NUM_THREADS \
      "$BIN/kernel_mpi" --csr "$4" --lookup "$LOOKUP" --variant W24 \
        --window 24 --threads "$3" --rank-cpus "$(ranks_for "$2")" \
        --partitioner block --tile 0 --repeat 64 --reps 6 --warmup-s 2 \
        --platform broadwell --nodes 1 $5 \
        --reference "$TMP/oracle.f64" \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" > "$log" 2>&1 || true
    if grep -q '^rep [1-9]' "$log"; then
        awk '/^rep [1-9]/ {s+=$9; g+=$16; n++}
             END {printf "%8.1f M rec/s  %.2f GHz  %7.1f M rec/GHz\n",
                  s/n, g/n, (s/n)/(g/n)}' "$log"
    else
        # Do not swallow the failure the way the first three attempts did.
        echo "FAILED"
        sed 's/^/      | /' "$log" | head -25
    fi
}

echo "########## smoke test: 1 rank under mpirun ##########"
run_mpi "1x8 smoke" 1 8 "$TMP/W24" ""
echo

echo "########## P10-prime: the discriminating pair ##########"
run_mpi "2x8 shared CSR"      2 8 "$TMP/W24" ""
run_mpi "2x8 per-rank CSR"    2 8 "$TMP/W24" "--csr-per-rank _r"
echo "  Equal -> address-space separation is not the mechanism; P10' fails"
echo "  regardless of the absolute numbers."
echo

echo "########## plan.md hybrid ratios, 16 cores fixed (P3) ##########"
for NR in 1 2 4 8 16; do
    run_mpi "$(printf '%dx%d' "$NR" $((16/NR)))" "$NR" $((16/NR)) "$TMP/W24" ""
done
echo "  P3 predicts 2x8 wins -- one rank per socket."
echo

echo "########## SUMMARY ##########"
if command -v python3 >/dev/null && python3 -c 'import pandas' 2>/dev/null; then
python3 - "$TIM" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); d = d[d.warmup == 0]
d["per_rank_csr"] = d.notes.str.extract(r"per_rank_csr=(\d)").fillna("-")
g = (d.groupby(["paradigm","n_ranks","n_threads","timed_region","per_rank_csr"])
       .agg(mrec=("throughput_rec_s", lambda s: s.median()/1e6),
            ghz=("achieved_ghz","median"),
            work_imb=("work_imbalance","median"),
            time_imb=("time_imbalance","median"),
            bit=("bitidentical", lambda s: set(s))).reset_index())
g["per_cycle"] = (g.mrec/g.ghz).round(1)
print(g.round(4).to_string(index=False))
print()
print("  P10': compare 2x8 per_rank_csr=1 against the openmp grouped-16 row.")
print("        Predicted 871 M rec/GHz vs 738. Falsified near 738.")
print("  P3  : compare the five ratios on the `kernel` rows.")
print("  Karp-Flatt needs `end_to_end`; `kernel` amortises the gather away.")
EOF
else
    echo "  (no pandas here -- reduce on the login node)"
fi
echo
echo "timings: $TIM"
