#!/bin/bash
#SBATCH --job-name=e3_sweep
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2:30:00
#SBATCH --output=/projects/sb2ea/logs/e3_sweep_%j.out
#SBATCH --error=/projects/sb2ea/logs/e3_sweep_%j.err


set -euo pipefail
export USER=sb2ea

source /opt/ohpc/pub/apps/miniconda/etc/profile.d/conda.sh
conda activate /projects/sb2ea/envs/mimic

BIN=/projects/sb2ea/bin
PROJ=/projects/sb2ea
LOOKUP=$PROJ/manifest/kernel_lookup.csv
OUT=$PROJ/results/timing
NODE=$(hostname -s)
GIT_SHA="${GIT_SHA:-unknown}"
SEED="${SEED:-20260728}"

TIMINGS=$OUT/e3_${NODE}_${SLURM_JOB_ID}.csv
THREADS=$OUT/e3_${NODE}_${SLURM_JOB_ID}_threads.csv
mkdir -p "$OUT" "$PROJ/logs"

TMP=/tmp/sb2ea_e3
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"

echo "=== node $NODE  job $SLURM_JOB_ID  git $GIT_SHA  seed $SEED"
echo "=== $(date -Is)"
lscpu | grep -E 'Model name|^CPU\(s\)|NUMA node\(s\)' || true
echo

places() {                      # $1 = n threads -> {0},{1},...,{n-1}
    local n=$1 out="" i
    for ((i = 0; i < n; i++)); do out+="{$i},"; done
    echo "${out%,}"
}

echo "########## STAGING ##########"
for V in W24 Wfull; do
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$(places 16)" \
        "$BIN/stage_csr" --src "$PROJ/csr/$V" --dst "$TMP/$V" --mode interleave
done
echo

echo "########## ORACLES ##########"
for V in W24 Wfull; do
    "$BIN/kernel_serial" --csr "$TMP/$V" --lookup "$LOOKUP" \
        --out "$TMP/oracle_$V.f64" >/dev/null 2>&1
    echo "oracle: $V"
done
echo

gen() {
    local V W T
    # A. thread scaling at W24 . The headline curve.
    for T in 1 2 4 6 8 10 12 14 16; do
        echo "W24 24 $T precomputed block -1 4096 canonical natural"
    done
    # A2. thread scaling at Wfull -- the scaling primary 
    for T in 1 2 4 8 16; do
        echo "Wfull full $T precomputed block -1 4096 canonical natural"
    done
    # B. tile size (P7): C* = 15625/ceil(T/2), so the knee should MOVE with T
    for T in 2 8 16; do
        for C in 512 1024 2048 4096 8192 16384 32768; do
            echo "W24 24 $T precomputed block -1 $C canonical natural"
        done
    done
    # C. partitioners x row order. sorted_desc is A3's adversarial layout and
    #    is what makes scheduling visible above the memory floor.
    for T in 8 16; do
        for P in block cyclic nzbalanced wbalanced contig_opt greedy; do
            for R in canonical sorted_desc; do
                echo "W24 24 $T precomputed $P -1 4096 $R natural"
            done
        done
    done
    # D. runtime schedules x chunk. chunk is the same knob as m at fixed data

    for T in 8 16; do
        echo "W24 24 $T static block -1 4096 canonical natural"
        echo "W24 24 $T guided none 64 4096 canonical natural"
        for K in 1 16 64 256 1024; do
            echo "W24 24 $T static_chunked block_cyclic $K 4096 canonical natural"
            echo "W24 24 $T dynamic none $K 4096 canonical natural"
        done
    done
    # E. arithmetic intensity (D32). Same code, same data, same schedule
    #    only the hit rate moves, so the scaling knee should move with it.
    for T in 1 2 4 8 16; do
        for WL in none natural all; do
            echo "W24 24 $T precomputed block -1 4096 canonical $WL"
        done
    done
    # F. partitioners at Wfull, where the tail is undiluted
    for T in 8 16; do
        for P in block nzbalanced contig_opt greedy; do
            echo "Wfull full $T precomputed $P -1 4096 canonical natural"
        done
    done
}

gen | python -c "
import random, sys
lines = [l for l in sys.stdin.read().splitlines() if l.strip()]
random.Random($SEED).shuffle(lines)
sys.stdout.write('\n'.join(lines) + '\n')
" > "$TMP/configs.txt"

NCFG=$(wc -l < "$TMP/configs.txt")
echo "########## $NCFG CONFIGURATIONS (order randomised, seed $SEED) ##########"
echo

i=0
while read -r V W T SCH PART CHUNK TILE RO WL; do
    i=$((i + 1))

    # K per D17/A4: enough iterations that the timed region exceeds ~1 s.
    # Scaled by thread count because the region shrinks as threads are added.
    if [ "$V" = "W24" ]; then K=$((4 * T)); else K=$T; fi
    [ "$K" -lt 1 ] && K=1

    SMT=""
    [ "$T" -gt 16 ] && SMT="--smt"

    REF=""
    [ "$WL" = "natural" ] && REF="--reference $TMP/oracle_$V.f64"

    CHUNK_ARG=""
    [ "$CHUNK" != "-1" ] && CHUNK_ARG="--chunk $CHUNK"

    printf '[%3d/%3d] %-6s T=%-3s %-15s %-12s chunk=%-5s tile=%-6s %-12s %s\n' \
        "$i" "$NCFG" "$V" "$T" "$SCH" "$PART" "$CHUNK" "$TILE" "$RO" "$WL"

    OMP_PROC_BIND=close OMP_PLACES="$(places "$T")" \
    "$BIN/kernel_omp" \
        --csr "$TMP/$V" --lookup "$LOOKUP" \
        --variant "$V" --window "$W" \
        --threads "$T" --schedule "$SCH" --partitioner "$PART" \
        $CHUNK_ARG --tile "$TILE" --row-order "$RO" --whitelist "$WL" \
        --repeat "$K" --reps 6 --warmup-s 2 \
        --platform broadwell --stage-mode interleave \
        $REF $SMT \
        --timings "$TIMINGS" --threads-csv "$THREADS" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" \
        2>&1 | grep -E '^rep [1-9]|BIT-IDENTITY|migrated' | sed 's/^/          /'
    echo
done < "$TMP/configs.txt"

echo "########## DONE ##########"
echo "timings: $TIMINGS"
echo "threads: $THREADS"
wc -l "$TIMINGS" "$THREADS"
df -h /tmp | tail -n 1
