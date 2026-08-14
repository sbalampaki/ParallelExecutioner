#!/bin/bash
#SBATCH --job-name=e3f
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH --time=2:30:00
#SBATCH --output=/projects/sb2ea/logs/e3f_%j.out
#SBATCH --error=/projects/sb2ea/logs/e3f_%j.err


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
SEED="${SEED:-20260730}"

TIM=$OUT/e3f_${NODE}_${SLURM_JOB_ID}.csv
THR=$OUT/e3f_${NODE}_${SLURM_JOB_ID}_threads.csv
mkdir -p "$OUT"
TMP=/tmp/sb2ea_e3f
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"

echo "=== node $NODE  job $SLURM_JOB_ID  git $GIT_SHA  seed $SEED  $(date -Is)"
echo


pl() {   # $1 = placement, $2 = n threads
    local n=$2 o="" i
    case "$1" in
      socket0|smt) for ((i=0;i<n;i++)); do o+="{$((2*i))},";   done ;;
      socket1)     for ((i=0;i<n;i++)); do o+="{$((2*i+1))},"; done ;;
      spread)      for ((i=0;i<n;i++)); do o+="{$i},";         done ;;
      *) echo "bad placement $1" >&2; return 1 ;;
    esac
    echo "${o%,}"
}

echo "########## STAGING (interleaved; D31) ##########"
for V in W24 Wfull; do
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$(pl spread 16)" \
        "$BIN/stage_csr" --src "$PROJ/csr/$V" --dst "$TMP/$V" --mode interleave
done
"$BIN/numa_pages" "$TMP/W24/value.f32" | tail -n +2 \
    | awk -F, '{printf "  W24 %s=%.4f\n", $4, $7}'
echo
echo "########## ORACLES ##########"
for V in W24 Wfull; do
    "$BIN/kernel_serial" --csr "$TMP/$V" --lookup "$LOOKUP" \
        --out "$TMP/oracle_$V.f64" >/dev/null 2>&1
    echo "oracle: $V"
done
echo

gen() {
    local T P R C S K
    # A. thread scaling, W24, tile 0 -- the headline figure, both branches
    for T in 1 2 4 6 8 10 12 14 16; do
        echo "W24 24 $T spread precomputed block -1 0 canonical natural"
    done
    for T in 1 2 4 6 8; do
        echo "W24 24 $T socket0 precomputed block -1 0 canonical natural"
    done
    echo "W24 24 8 socket1 precomputed block -1 0 canonical natural"   # symmetry
    for T in 12 16; do          # 8 cores + siblings, all on socket 0
        echo "W24 24 $T smt precomputed block -1 0 canonical natural"
    done
    # B. thread scaling, Wfull -- the scaling primary (A4/D17)
    for T in 1 2 4 8 12 16; do
        echo "Wfull full $T spread precomputed block -1 0 canonical natural"
    done
    for T in 1 2 4 8; do
        echo "Wfull full $T socket0 precomputed block -1 0 canonical natural"
    done
    # C. partitioners x row order (A8, D35, A3's adversarial layout)
    for P in block cyclic nzbalanced wbalanced contig_opt greedy; do
        for R in canonical sorted_desc; do
            for T in 8 16; do
                echo "W24 24 $T spread precomputed $P -1 0 $R natural"
            done
            echo "W24 24 8 socket0 precomputed $P -1 0 $R natural"
        done
    done
    # D. partitioners at Wfull, where the tail is undiluted
    for P in block nzbalanced contig_opt greedy; do
        for T in 8 16; do
            echo "Wfull full $T spread precomputed $P -1 0 canonical natural"
        done
        echo "Wfull full 8 socket0 precomputed $P -1 0 canonical natural"
    done
    # E. runtime schedules; chunk is A5's m knob at fixed data
    for T in 8 16; do
        echo "W24 24 $T spread static block -1 0 canonical natural"
        echo "W24 24 $T spread guided none 64 0 canonical natural"
        for K in 16 64 256 1024; do
            echo "W24 24 $T spread static_chunked block_cyclic $K 0 canonical natural"
            echo "W24 24 $T spread dynamic none $K 0 canonical natural"
        done
    done
    echo "W24 24 8 socket0 static block -1 0 canonical natural"
    echo "W24 24 8 socket0 guided none 64 0 canonical natural"
    for K in 16 64 256 1024; do
        echo "W24 24 8 socket0 static_chunked block_cyclic $K 0 canonical natural"
        echo "W24 24 8 socket0 dynamic none $K 0 canonical natural"
    done
    # F. arithmetic intensity (D32) -- same code, same data, only hit rate
    for S in none natural all; do
        for T in 1 2 4 8 16; do
            echo "W24 24 $T spread precomputed block -1 0 canonical $S"
        done
        for T in 1 2 4 8; do
            echo "W24 24 $T socket0 precomputed block -1 0 canonical $S"
        done
    done
    # G. tile sweep, retained as a LABELLED NULL with its mechanism
    for C in 0 1 32 128 512 4096; do
        echo "W24 24 8 spread  precomputed block -1 $C canonical natural"
        echo "W24 24 8 socket0 precomputed block -1 $C canonical natural"
    done
}

gen | python -c "
import random, sys
lines=[l for l in sys.stdin.read().splitlines() if l.strip()]
random.Random($SEED).shuffle(lines)
sys.stdout.write('\n'.join(lines)+'\n')
" > "$TMP/cfg.txt"

N=$(wc -l < "$TMP/cfg.txt")
echo "########## $N CONFIGURATIONS (randomised, seed $SEED) ##########"
echo

i=0
while read -r V W T PLC SCH PART CHUNK TILE RO WL; do
    i=$((i+1))
    if [ "$V" = "W24" ]; then K=$((4*T)); else K=$T; fi
    [ "$K" -lt 1 ] && K=1
    REF=""; [ "$WL" = "natural" ] && REF="--reference $TMP/oracle_$V.f64"
    CH="";  [ "$CHUNK" != "-1" ] && CH="--chunk $CHUNK"

    printf '[%3d/%3d] %-6s T=%-3s %-8s %-14s %-12s chunk=%-5s tile=%-5s %-12s %s\n' \
        "$i" "$N" "$V" "$T" "$PLC" "$SCH" "$PART" "$CHUNK" "$TILE" "$RO" "$WL"

    OMP_PROC_BIND=close OMP_PLACES="$(pl "$PLC" "$T")" \
    "$BIN/kernel_omp" \
        --csr "$TMP/$V" --lookup "$LOOKUP" --variant "$V" --window "$W" \
        --threads "$T" --schedule "$SCH" --partitioner "$PART" \
        $CH --tile "$TILE" --row-order "$RO" --whitelist "$WL" \
        --repeat "$K" --reps 6 --warmup-s 2 \
        --platform broadwell --stage-mode interleave $REF \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" \
        2>&1 | grep -E '^rep [1-9]|BIT-IDENTITY|migrated' | sed 's/^/          /'
    echo
done < "$TMP/cfg.txt"

echo "########## DONE ##########"
wc -l "$TIM" "$THR"
