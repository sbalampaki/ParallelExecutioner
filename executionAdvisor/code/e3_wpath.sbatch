#!/bin/bash
#SBATCH --job-name=e3_wpath
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH --time=0:30:00
#SBATCH --output=/projects/sb2ea/logs/e3wpath_%j.out
#SBATCH --error=/projects/sb2ea/logs/e3wpath_%j.err



set -euo pipefail
export USER=sb2ea

source /opt/ohpc/pub/apps/miniconda/etc/profile.d/conda.sh
conda activate /projects/sb2ea/envs/mimic

BIN=/projects/sb2ea/bin
PROJ=/projects/sb2ea
LOOKUP=$PROJ/manifest/kernel_lookup.csv
OUT=$PROJ/results/timing
NODE=$(hostname -s)
TMP=/tmp/sb2ea_wpath
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP" "$OUT"
TIM=$OUT/wpath_${NODE}_${SLURM_JOB_ID}.csv
THR=$OUT/wpath_${NODE}_${SLURM_JOB_ID}_threads.csv

pl_s0()     { local n=$1 o="" i; for ((i=0;i<n;i++)); do o+="{$((2*i))},"; done; echo "${o%,}"; }
pl_spread() { local n=$1 o="" i; for ((i=0;i<n;i++)); do o+="{$i},";       done; echo "${o%,}"; }

echo "=== node $NODE  job $SLURM_JOB_ID  $(date -Is)"
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$(pl_spread 16)" \
    "$BIN/stage_csr" --src "$PROJ/csr/W24" --dst "$TMP/W24" --mode interleave
"$BIN/numa_pages" "$TMP/W24/value.f32" | tail -n +2 \
    | awk -F, '{printf "  staged %s=%.4f\n", $4, $7}'
echo

echo "########## A. read-only vs read+write, S0 vs spread (GB/s) ##########"
printf '  %-3s %-8s %12s %12s %10s\n' T place 'read only' 'read+write' 'penalty'
for T in 4 8; do
    for P in s0 spread; do
        PL=$([ "$P" = s0 ] && pl_s0 "$T" || pl_spread "$T")
        RD=$(OMP_NUM_THREADS=$T OMP_PROC_BIND=close OMP_PLACES="$PL" \
             "$BIN/csr_stream" --csr "$TMP/W24" --touch all --reps 6 --tag r 2>/dev/null \
             | awk -F, '$1=="RUN" && $7=="0" {s+=$11;n++} END {printf "%.2f", s/n}')
        WR=$(OMP_NUM_THREADS=$T OMP_PROC_BIND=close OMP_PLACES="$PL" \
             "$BIN/csr_stream" --csr "$TMP/W24" --touch all --write-out --reps 6 --tag w 2>/dev/null \
             | awk -F, '$1=="RUN" && $7=="0" {s+=$11;n++} END {printf "%.2f", s/n}')
        printf '  %-3s %-8s %12s %12s %10s\n' "$T" "$P" "$RD" "$WR" \
            "$(python -c "print(f'{100*(1-$WR/$RD):.1f}%')" 2>/dev/null || echo '-')"
    done
done
echo "  Compare the spread-vs-S0 gap in the 'read only' column against the"
echo "  same gap in 'read+write'. If the gap only opens once writes are"
echo "  added, the penalty is the write path."
echo

echo "########## B. kernel write streams: tile 0 vs 128 (M rec/s) ##########"
printf '  %-3s %-6s %10s %10s %10s\n' T tile S0 spread 'penalty'
for T in 4 8; do
    for TL in 0 128; do
        A=$(OMP_PROC_BIND=close OMP_PLACES="$(pl_s0 "$T")" \
            "$BIN/kernel_omp" --csr "$TMP/W24" --lookup "$LOOKUP" \
              --variant W24 --window 24 --threads "$T" --schedule precomputed \
              --partitioner block --tile "$TL" --repeat $((4*T)) --reps 6 \
              --warmup-s 2 --timings "$TIM" --threads-csv "$THR" \
              --job-id "$SLURM_JOB_ID" \
            2>&1 | awk '/^rep [1-9]/ {s+=$6;n++} END {printf "%.1f", s/n}')
        B=$(OMP_PROC_BIND=close OMP_PLACES="$(pl_spread "$T")" \
            "$BIN/kernel_omp" --csr "$TMP/W24" --lookup "$LOOKUP" \
              --variant W24 --window 24 --threads "$T" --schedule precomputed \
              --partitioner block --tile "$TL" --repeat $((4*T)) --reps 6 \
              --warmup-s 2 --timings "$TIM" --threads-csv "$THR" \
              --job-id "$SLURM_JOB_ID" \
            2>&1 | awk '/^rep [1-9]/ {s+=$6;n++} END {printf "%.1f", s/n}')
        printf '  %-3s %-6s %10s %10s %10s\n' "$T" "$TL" "$A" "$B" \
            "$(python -c "print(f'{100*(1-$B/$A):.1f}%')" 2>/dev/null || echo '-')"
    done
done
echo "  tile=0 writes 1280 B/stay once. tile=128 writes it, reads it back,"
echo "  writes it again. If the spread penalty is larger at tile=128, it"
echo "  scales with write traffic."
echo
echo "  NOTE: these are RAW throughputs. Spread runs at a higher clock"
echo "  (2.88 vs 2.69 GHz at T=4), so every penalty here understates the"
echo "  per-cycle penalty by roughly 5 points."
echo
echo "timings: $TIM"
