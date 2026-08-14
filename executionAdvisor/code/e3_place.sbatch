#!/bin/bash
#SBATCH --job-name=e3_place
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH --time=0:40:00
#SBATCH --output=/projects/sb2ea/logs/e3place_%j.out
#SBATCH --error=/projects/sb2ea/logs/e3place_%j.err


set -euo pipefail
export USER=sb2ea

source /opt/ohpc/pub/apps/miniconda/etc/profile.d/conda.sh
conda activate /projects/sb2ea/envs/mimic

BIN=/projects/sb2ea/bin
PROJ=/projects/sb2ea
LOOKUP=$PROJ/manifest/kernel_lookup.csv
OUT=$PROJ/results/timing
NODE=$(hostname -s)
TMP=/tmp/sb2ea_place
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP" "$OUT"

TIM=$OUT/place_${NODE}_${SLURM_JOB_ID}.csv
THR=$OUT/place_${NODE}_${SLURM_JOB_ID}_threads.csv

echo "=== node $NODE  job $SLURM_JOB_ID  $(date -Is)"
echo

# thread-placement sets. lscpu confirms even = socket 0, odd = socket 1.
pl_s0()     { local n=$1 o="" i; for ((i=0;i<n;i++)); do o+="{$((2*i))},";     done; echo "${o%,}"; }
pl_s1()     { local n=$1 o="" i; for ((i=0;i<n;i++)); do o+="{$((2*i+1))},";   done; echo "${o%,}"; }
pl_spread() { local n=$1 o="" i; for ((i=0;i<n;i++)); do o+="{$i},";           done; echo "${o%,}"; }


echo "########## STAGING (3 page placements) ##########"
OMP_NUM_THREADS=1  OMP_PROC_BIND=close OMP_PLACES='{0}' \
    "$BIN/stage_csr" --src "$PROJ/csr/W24" --dst "$TMP/n0"   --mode interleave
OMP_NUM_THREADS=1  OMP_PROC_BIND=close OMP_PLACES='{1}' \
    "$BIN/stage_csr" --src "$PROJ/csr/W24" --dst "$TMP/n1"   --mode interleave
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$(pl_spread 16)" \
    "$BIN/stage_csr" --src "$PROJ/csr/W24" --dst "$TMP/intl" --mode interleave
echo

echo "########## VERIFY placement before timing anything ##########"
for P in n0 n1 intl; do
    printf '  %-5s ' "$P"
    "$BIN/numa_pages" "$TMP/$P/value.f32" | tail -n +2 \
        | awk -F, '{printf "%s=%.4f  ", $4, $7} END {print ""}'
done
echo "  n0 must be ~1.00 on node0, n1 ~1.00 on node1, intl ~0.50/0.50."
echo "  If not, the rest of this job is meaningless."
echo

run_kernel() {   # $1 csr  $2 places  $3 threads
    OMP_PROC_BIND=close OMP_PLACES="$2" \
    "$BIN/kernel_omp" --csr "$1" --lookup "$LOOKUP" \
        --variant W24 --window 24 --threads "$3" \
        --schedule precomputed --partitioner block --tile 128 \
        --repeat $((4 * $3)) --reps 6 --warmup-s 2 \
        --timings "$TIM" --threads-csv "$THR" --job-id "$SLURM_JOB_ID" \
        2>&1 | awk '/^rep [1-9]/ {s+=$6; n++} END {printf "%.1f", s/n}'
}

run_stream() {   # $1 csr  $2 places  $3 threads
    OMP_NUM_THREADS="$3" OMP_PROC_BIND=close OMP_PLACES="$2" \
    "$BIN/csr_stream" --csr "$1" --touch all --reps 6 --tag p 2>/dev/null \
        | awk -F, '$1=="RUN" && $7=="0" {s+=$11; n++} END {printf "%.2f", s/n}'
}

for T in 4 8; do
    echo "########## FULL KERNEL, T=$T  (M rec/s) ##########"
    printf '  %-10s %12s %12s %12s\n' pages 'threads S0' 'threads S1' 'spread'
    for P in n0 n1 intl; do
        printf '  %-10s %12s %12s %12s\n' "$P" \
            "$(run_kernel "$TMP/$P" "$(pl_s0     "$T")" "$T")" \
            "$(run_kernel "$TMP/$P" "$(pl_s1     "$T")" "$T")" \
            "$(run_kernel "$TMP/$P" "$(pl_spread "$T")" "$T")"
    done
    echo

    echo "########## READ-ONLY PROBE, T=$T  (GB/s) ##########"
    printf '  %-10s %12s %12s %12s\n' pages 'threads S0' 'threads S1' 'spread'
    for P in n0 n1 intl; do
        printf '  %-10s %12s %12s %12s\n' "$P" \
            "$(run_stream "$TMP/$P" "$(pl_s0     "$T")" "$T")" \
            "$(run_stream "$TMP/$P" "$(pl_s1     "$T")" "$T")" \
            "$(run_stream "$TMP/$P" "$(pl_spread "$T")" "$T")"
    done
    echo
done

echo "########## HOW TO READ ##########"
echo "  n0/S0 vs n0/S1     : the cost of remote access, isolated."
echo "  n0/S0 vs n1/S1     : socket symmetry. Should match."
echo "  intl row           : all three cells read 50% remote. If S0 still"
echo "                       beats spread here, page locality is NOT the"
echo "                       cause and something else is."
echo "  kernel vs probe    : if the probe's intl row is flat but the"
echo "                       kernel's is not, the effect is in the write"
echo "                       path, since out/tiles are node-local in every"
echo "                       cell and only their SOCKET differs."
echo
echo "timings: $TIM"
