#!/bin/bash
#SBATCH --job-name=e4_align
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH --time=1:30:00
#SBATCH --output=/projects/sb2ea/logs/e4align_%j.out
#SBATCH --error=/projects/sb2ea/logs/e4align_%j.err


set -euo pipefail
export USER=sb2ea

MPI_PREFIX=/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.6
HWLOC_LIB=/opt/ohpc/pub/libs/hwloc/lib
MPIRUN="$MPI_PREFIX/bin/mpirun"
export LD_LIBRARY_PATH="$MPI_PREFIX/lib:$HWLOC_LIB:${LD_LIBRARY_PATH:-}"
[ -x "$MPIRUN" ] || { echo "FATAL: $MPIRUN missing" >&2; exit 1; }

BIN=/projects/sb2ea/bin
PROJ=/projects/sb2ea
LOOKUP=$PROJ/manifest/kernel_lookup.csv
OUT=$PROJ/results/timing
NODE=$(hostname -s)
GIT_SHA="${GIT_SHA:-unknown}"
TMP=/tmp/sb2ea_e4a
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP" "$OUT"
TIM=$OUT/e4a_${NODE}_${SLURM_JOB_ID}.csv
THR=$OUT/e4a_${NODE}_${SLURM_JOB_ID}_threads.csv

for b in "$BIN/kernel_omp" "$BIN/kernel_mpi" "$BIN/stage_aligned"; do
    [ -x "$b" ] || { echo "FATAL: $b missing or not executable" >&2; exit 1; }
done

echo "=== node $NODE  job $SLURM_JOB_ID  git $GIT_SHA  $(date -Is)"
declare -A SEEN
S0=(); S1=()
while IFS=, read -r cpu core sock; do
    [ -z "${cpu:-}" ] && continue
    key="$sock:$core"; [ -n "${SEEN[$key]:-}" ] && continue; SEEN[$key]=1
    if [ "$sock" = "0" ]; then S0+=("$cpu"); else S1+=("$cpu"); fi
done < <(lscpu -p=CPU,CORE,SOCKET | grep -v '^#')
NC0=${#S0[@]}; NC1=${#S1[@]}
ALL=("${S0[@]}" "${S1[@]}")            # grouped order (D44)
echo "cores: socket0=$NC0 socket1=$NC1"
GRP16="$(printf '{%s},' "${ALL[@]}" | sed 's/,$//')"
S0_8="$(printf '{%s},' "${S0[@]}" | sed 's/,$//')"
echo

echo "########## STAGING ##########"
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$GRP16" \
    "$BIN/stage_csr" --src "$PROJ/csr/W24" --dst "$TMP/intl" --mode interleave
echo "--- aligned copy: 16 threads, grouped order, block partition ---"
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$GRP16" \
    "$BIN/stage_aligned" --src "$PROJ/csr/W24" --dst "$TMP/algn" 2>&1 | sed 's/^/  /'
for P in intl algn; do
    printf '  %-6s ' "$P"
    "$BIN/numa_pages" "$TMP/$P/value.f32" | tail -n +2 \
        | awk -F, '{printf "%s=%.4f  ", $4, $7} END {print ""}'
done
echo "  NOTE: both are ~50/50 in AGGREGATE -- that is expected and is why"
echo "  numa_pages cannot distinguish them. The difference is WHICH pages,"
echo "  and the only instrument for that here is throughput."
"$BIN/kernel_serial" --csr "$TMP/intl" --lookup "$LOOKUP" \
    --out "$TMP/oracle.f64" >/dev/null 2>&1
echo "  oracle generated"
echo

omp_run() {   # $1 label  $2 threads  $3 places  $4 csr
    printf '  %-32s ' "$1"
    OMP_PROC_BIND=close OMP_PLACES="$3" \
    "$BIN/kernel_omp" --csr "$4" --lookup "$LOOKUP" --variant W24 \
        --window 24 --threads "$2" --schedule precomputed --partitioner block \
        --tile 0 --repeat $((4 * $2)) --reps 6 --warmup-s 2 \
        --platform broadwell --reference "$TMP/oracle.f64" \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" 2>&1 \
      | awk '/^rep [1-9]/ {s+=$6; g+=$13; n++}
             END {if (n) printf "%8.1f M rec/s  %.2f GHz  %7.1f M rec/GHz\n",
                  s/n, g/n, (s/n)/(g/n); else print "FAILED"}'
}

echo "########## BLOCK A: aligned vs interleaved, pure OpenMP ##########"
omp_run "T=16 grouped, interleaved" 16 "$GRP16" "$TMP/intl"
omp_run "T=16 grouped, ALIGNED"     16 "$GRP16" "$TMP/algn"
echo "  control -- one socket, nothing to align:"
omp_run "T=8 socket0, interleaved"   8 "$S0_8"  "$TMP/intl"
omp_run "T=8 socket0, ALIGNED"       8 "$S0_8"  "$TMP/algn"
echo "  A gain at T=16 with NO gain at T=8 socket0 is the signature of"
echo "  alignment. A gain at both would mean something else changed."
echo


stage_per_rank() {   # $1 = n_ranks
    local nr=$1 r cpu
    for ((r=0;r<nr;r++)); do
        if (( r < (nr + 1) / 2 )); then cpu=${S0[0]}; else cpu=${S1[0]}; fi
        OMP_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES="{$cpu}" \
            "$BIN/stage_csr" --src "$PROJ/csr/W24" \
            --dst "$TMP/pr${nr}_r${r}" --mode interleave >/dev/null 2>&1
    done
    printf '    staged %d per-rank copies\n' "$nr"
}

ranks_for() {   # $1 = n_ranks -> "cpus;cpus;..." at 16/n cpus per rank
    local nr=$1 per=$((16 / $1)) o="" r i idx
    for ((r=0;r<nr;r++)); do
        local g=""
        for ((i=0;i<per;i++)); do idx=$((r * per + i)); g+="${ALL[$idx]},"; done
        o+="${g%,};"
    done
    echo "${o%;}"
}

mpi_run() {   # $1 label  $2 nranks  $3 csr-prefix  $4 extra
    local nr=$2 nt=$((16 / $2)) log="$TMP/run.log"
    printf '  %-32s ' "$1"
    env -u OMP_PLACES OMP_PROC_BIND=false OMP_NUM_THREADS="$nt" \
      "$MPIRUN" -np "$nr" --bind-to none -x LD_LIBRARY_PATH \
        -x OMP_PROC_BIND -x OMP_NUM_THREADS \
      "$BIN/kernel_mpi" --csr "$3" --lookup "$LOOKUP" --variant W24 \
        --window 24 --threads "$nt" --rank-cpus "$(ranks_for "$nr")" \
        --partitioner block --tile 0 --repeat 64 --reps 6 --warmup-s 2 \
        --platform broadwell --nodes 1 $4 \
        --reference "$TMP/oracle.f64" \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" > "$log" 2>&1 || true
    if grep -q '^rep [1-9]' "$log"; then
        awk '/^rep [1-9]/ {s+=$9; g+=$16; n++}
             END {printf "%8.1f M rec/s  %.2f GHz  %7.1f M rec/GHz\n",
                  s/n, g/n, (s/n)/(g/n)}' "$log"
    else
        echo "FAILED"; sed 's/^/      | /' "$log" | grep -v 'mca_base' | head -12
    fi
}

echo "########## BLOCK B: hybrid ratios, shared vs per-rank ##########"
printf '  %-32s %s\n' "" "(16 cores fixed throughout)"
for NR in 1 2 4 8 16; do
    mpi_run "$(printf '%dx%d shared' "$NR" $((16/NR)))" "$NR" "$TMP/intl" ""
done
echo
for NR in 1 2 4 8; do
    stage_per_rank "$NR"
    mpi_run "$(printf '%dx%d PER-RANK' "$NR" $((16/NR)))" "$NR" "$TMP/pr${NR}_r" "--csr-per-rank ''"
    rm -rf "$TMP"/pr${NR}_r*        # 528 MB per copy; reclaim before the next
done
echo "  P3 predicted 2x8. Job 73037 gave 4x4 on the shared CSR. Whether"
echo "  that survives per-rank placement is the question."
echo

echo "########## SUMMARY ##########"
if python3 -c 'import pandas' 2>/dev/null; then
python3 - "$TIM" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); d = d[d.warmup == 0]
d["per_rank"] = d.notes.str.extract(r"per_rank_csr=(\d)").fillna("-")
g = (d.groupby(["paradigm","n_ranks","n_threads","thread_placement",
                "timed_region","per_rank"])
       .agg(mrec=("throughput_rec_s", lambda s: s.median()/1e6),
            ghz=("achieved_ghz","median"),
            work_imb=("work_imbalance","median"),
            time_imb=("time_imbalance","median"),
            bit=("bitidentical", lambda s: ",".join(sorted(set(s))))).reset_index())
g["per_cycle"] = (g.mrec/g.ghz).round(1)
print(g.round(4).to_string(index=False))
EOF
else
    echo "  (no pandas -- reduce on the login node)"
fi
echo
echo "timings: $TIM"
