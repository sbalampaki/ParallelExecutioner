#!/bin/bash
#SBATCH --job-name=e3_diag
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH --time=0:45:00
#SBATCH --output=/projects/sb2ea/logs/e3diag_%j.out
#SBATCH --error=/projects/sb2ea/logs/e3diag_%j.err

set -euo pipefail
export USER=sb2ea

source /opt/ohpc/pub/apps/miniconda/etc/profile.d/conda.sh
conda activate /projects/sb2ea/envs/mimic

BIN=/projects/sb2ea/bin
PROJ=/projects/sb2ea
LOOKUP=$PROJ/manifest/kernel_lookup.csv
OUT=$PROJ/results/timing
NODE=$(hostname -s)
TMP=/tmp/sb2ea_diag
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP" "$OUT"

TIM=$OUT/diag_${NODE}_${SLURM_JOB_ID}.csv
THR=$OUT/diag_${NODE}_${SLURM_JOB_ID}_threads.csv

echo "=== node $NODE  job $SLURM_JOB_ID  $(date -Is)"
echo

places_spread() { local n=$1 o="" i; for ((i=0;i<n;i++)); do o+="{$i},"; done; echo "${o%,}"; }
places_even()   { local n=$1 o="" i; for ((i=0;i<n;i++)); do o+="{$((2*i))},"; done; echo "${o%,}"; }

echo "########## Q1a: page placement, NFS copy vs staged copy ##########"
echo "--- /projects (NFS), as the failed one-liner used ---"
"$BIN/numa_pages" "$PROJ/csr/W24/value.f32" | tail -n +2
echo
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$(places_spread 16)" \
    "$BIN/stage_csr" --src "$PROJ/csr/W24" --dst "$TMP/W24" --mode interleave
echo "--- /tmp staged, bound parallel writer (D31) ---"
"$BIN/numa_pages" "$TMP/W24/value.f32" | tail -n +2
echo

echo "########## Q1b: does mbind MIGRATE, or only set policy? ##########"
"$BIN/partition_selftest" --csr "$PROJ/csr/W24" --lookup "$LOOKUP" \
    --variant W24 >/dev/null 2>"$TMP/selftest.err" || true
grep -E 'numa_map' "$TMP/selftest.err" || true
echo "--- placement of the NFS copy AFTER a mbind'd mapping ---"
"$BIN/numa_pages" "$PROJ/csr/W24/value.f32" | tail -n +2
echo "  If this is unchanged from Q1a, mbind set the policy without moving"
echo "  pages, and numa_map.h's mbind_ok is weaker than D33 assumed."
echo

echo "########## Q2a: placement effect, PHYSICAL cores only (T<=8) ##########"
printf '  %-4s %-8s %10s\n' T placement 'M rec/s'
for T in 2 4 6 8; do
    for P in spread socket0; do
        if [ "$P" = spread ]; then PL=$(places_spread "$T"); else PL=$(places_even "$T"); fi
        R=$(OMP_PROC_BIND=close OMP_PLACES="$PL" "$BIN/kernel_omp" \
              --csr "$TMP/W24" --lookup "$LOOKUP" --variant W24 --window 24 \
              --threads "$T" --schedule precomputed --partitioner block \
              --tile 128 --repeat $((4*T)) --reps 6 --warmup-s 2 \
              --timings "$TIM" --threads-csv "$THR" --job-id "$SLURM_JOB_ID" \
              2>&1 | awk '/^rep [1-9]/ {s+=$6; n++} END {printf "%.1f", s/n}')
        printf '  %-4s %-8s %10s\n' "$T" "$P" "$R"
    done
done
echo "  Both sockets should now be equally fed, so spread >= socket0."
echo "  If socket0 still wins, staging did not interleave either."
echo

echo "########## Q2b: the T=14 dip, on the staged copy ##########"
printf '  %-4s %10s %8s %10s %10s\n' T 'M rec/s' GHz work_imb time_imb
for T in 8 10 12 13 14 15 16; do
    OMP_PROC_BIND=close OMP_PLACES="$(places_spread "$T")" \
      "$BIN/kernel_omp" --csr "$TMP/W24" --lookup "$LOOKUP" \
        --variant W24 --window 24 --threads "$T" --schedule precomputed \
        --partitioner block --tile 128 --repeat $((4*T)) --reps 6 --warmup-s 2 \
        --timings "$TIM" --threads-csv "$THR" --job-id "$SLURM_JOB_ID" \
        2>&1 | awk -v t="$T" '/^rep [1-9]/ {s+=$6; g+=$13; w+=$8; i+=$10; n++}
                              END {printf "  %-4s %10.1f %8.2f %10.4f %10.4f\n",
                                   t, s/n, g/n, w/n, i/n}'
done
echo

echo "########## Q2c: SMT on one socket (8 cores + siblings) ##########"
echo "  Not a placement comparison: even ids past 8 threads are SMT"
echo "  siblings of the same 8 cores, so this measures SMT, not sockets."
printf '  %-4s %10s\n' T 'M rec/s'
for T in 8 12 16; do
    R=$(OMP_PROC_BIND=close OMP_PLACES="$(places_even "$T")" \
          "$BIN/kernel_omp" --csr "$TMP/W24" --lookup "$LOOKUP" \
            --variant W24 --window 24 --threads "$T" --schedule precomputed \
            --partitioner block --tile 128 --repeat $((4*T)) --reps 6 \
            --warmup-s 2 --smt \
            --timings "$TIM" --threads-csv "$THR" --job-id "$SLURM_JOB_ID" \
            2>&1 | awk '/^rep [1-9]/ {s+=$6; n++} END {printf "%.1f", s/n}')
    printf '  %-4s %10s\n' "$T" "$R"
done
echo

echo "########## Q3: read-only roof vs mixed read/write roof ##########"
echo "  The one-liner filtered on ',1,' to drop warm-up rows, but"
echo "  --write-out puts write_out=1 in every row, so it dropped all of"
echo "  them. Filtering on the warmup FIELD instead."
printf '  %-4s %12s %12s %10s\n' T 'read GB/s' 'mixed GB/s' 'ratio'
for T in 1 2 4 8 12 16; do
    PL=$(places_spread "$T")
    RD=$(OMP_NUM_THREADS=$T OMP_PROC_BIND=close OMP_PLACES="$PL" \
          "$BIN/csr_stream" --csr "$TMP/W24" --touch all --reps 6 --tag r 2>/dev/null \
          | awk -F, '$1=="RUN" && $7=="0" {s+=$11; n++} END {printf "%.2f", s/n}')
    MX=$(OMP_NUM_THREADS=$T OMP_PROC_BIND=close OMP_PLACES="$PL" \
          "$BIN/csr_stream" --csr "$TMP/W24" --touch all --write-out --reps 6 --tag m 2>/dev/null \
          | awk -F, '$1=="RUN" && $7=="0" {s+=$11; n++} END {printf "%.2f", s/n}')
    printf '  %-4s %12s %12s %10s\n' "$T" "$RD" "$MX" \
        "$(python -c "print(f'{$MX/$RD:.3f}')" 2>/dev/null || echo '-')"
done
echo
echo "  The kernel is a mixed stream. If its ~17.3 GB/s at T=13 sits near"
echo "  the MIXED roof, P6's mechanism is confirmed as well as its"
echo "  location. If it sits well below, the knee is not bandwidth."
echo
echo "########## DONE ##########"
echo "timings: $TIM"
