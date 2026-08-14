#!/bin/bash
#SBATCH --job-name=e3b_tile
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH --time=2:00:00
#SBATCH --output=/projects/sb2ea/logs/e3b_%j.out
#SBATCH --error=/projects/sb2ea/logs/e3b_%j.err


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
SEED="${SEED:-20260729}"

TIMINGS=$OUT/e3b_${NODE}_${SLURM_JOB_ID}.csv
THREADS=$OUT/e3b_${NODE}_${SLURM_JOB_ID}_threads.csv
mkdir -p "$OUT"

TMP=/tmp/sb2ea_e3b
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"

echo "=== node $NODE  job $SLURM_JOB_ID  git $GIT_SHA  seed $SEED"
echo "=== $(date -Is)"
echo "=== L2 boundary C = 205 stays,  L1d boundary C = 26 stays"
echo

places() { local n=$1 out="" i; for ((i=0;i<n;i++)); do out+="{$i},"; done; echo "${out%,}"; }

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
    local T C P R
    # A. tile sweep straddling BOTH per-core boundaries (P7').
    #    0 = direct write; 26 = L1d; 205 = L2; 4096 = the first sweep's value.
    for T in 2 8 16; do
        for C in 0 8 16 26 48 64 128 205 256 512 1024 4096; do
            echo "W24 24 $T precomputed block -1 $C canonical natural"
        done
    done
    # B. same at Wfull, where records-per-stay is 3.6x higher so the output
    #    path is a much smaller share of traffic
    for T in 8 16; do
        for C in 0 26 205 1024 4096; do
            echo "Wfull full $T precomputed block -1 $C canonical natural"
        done
    done
    # C. partitioners on an UNHANDICAPPED baseline (tile 0)
    for T in 8 16; do
        for P in block cyclic nzbalanced wbalanced contig_opt greedy; do
            for R in canonical sorted_desc; do
                echo "W24 24 $T precomputed $P -1 0 $R natural"
            done
        done
    done
    for T in 8 16; do
        for P in block nzbalanced contig_opt greedy; do
            echo "Wfull full $T precomputed $P -1 0 canonical natural"
        done
    done

    for T in 1 2 4 6 8 10 12 13 14 15 16; do
        echo "W24 24 $T precomputed block -1 0 canonical natural"
    done
    for T in 1 2 4 8 12 16; do
        echo "Wfull full $T precomputed block -1 0 canonical natural"
    done

    for T in 1 2 4 8 16; do
        for W in none natural all; do
            echo "W24 24 $T precomputed block -1 0 canonical $W"
        done
    done
}

gen | python -c "
import random, sys
lines=[l for l in sys.stdin.read().splitlines() if l.strip()]
random.Random($SEED).shuffle(lines)
sys.stdout.write('\n'.join(lines)+'\n')
" > "$TMP/configs.txt"

NCFG=$(wc -l < "$TMP/configs.txt")
echo "########## $NCFG CONFIGURATIONS (order randomised, seed $SEED) ##########"
echo

i=0
while read -r V W T SCH PART CHUNK TILE RO WL; do
    i=$((i+1))
    if [ "$V" = "W24" ]; then K=$((4*T)); else K=$T; fi
    [ "$K" -lt 1 ] && K=1
    REF=""; [ "$WL" = "natural" ] && REF="--reference $TMP/oracle_$V.f64"
    CH=""; [ "$CHUNK" != "-1" ] && CH="--chunk $CHUNK"

    printf '[%3d/%3d] %-6s T=%-3s %-13s %-11s tile=%-6s %-12s %s\n' \
        "$i" "$NCFG" "$V" "$T" "$SCH" "$PART" "$TILE" "$RO" "$WL"

    OMP_PROC_BIND=close OMP_PLACES="$(places "$T")" \
    "$BIN/kernel_omp" \
        --csr "$TMP/$V" --lookup "$LOOKUP" --variant "$V" --window "$W" \
        --threads "$T" --schedule "$SCH" --partitioner "$PART" \
        $CH --tile "$TILE" --row-order "$RO" --whitelist "$WL" \
        --repeat "$K" --reps 6 --warmup-s 2 \
        --platform broadwell --stage-mode interleave $REF \
        --timings "$TIMINGS" --threads-csv "$THREADS" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" \
        2>&1 | grep -E '^rep [1-9]|BIT-IDENTITY|migrated' | sed 's/^/          /'
    echo
done < "$TMP/configs.txt"

echo "########## DONE ##########"
wc -l "$TIMINGS" "$THREADS"
