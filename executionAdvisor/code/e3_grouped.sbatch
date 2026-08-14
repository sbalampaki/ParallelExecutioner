#!/bin/bash
#SBATCH --job-name=e3g
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH --time=1:30:00
#SBATCH --output=/projects/sb2ea/logs/e3g_%j.out
#SBATCH --error=/projects/sb2ea/logs/e3g_%j.err



set -euo pipefail
export USER=sb2ea

source /opt/ohpc/pub/apps/miniconda/etc/profile.d/conda.sh
conda activate /projects/sb2ea/envs/mimic

BIN=${BIN_OVERRIDE:-/projects/sb2ea/bin}
PROJ=/projects/sb2ea
LOOKUP=$PROJ/manifest/kernel_lookup.csv
OUT=$PROJ/results/timing
NODE=$(hostname -s)
GIT_SHA="${GIT_SHA:-unknown}"
SEED="${SEED:-20260731}"
TMP=/tmp/sb2ea_e3g
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP" "$OUT"
TIM=$OUT/e3g_${NODE}_${SLURM_JOB_ID}.csv
THR=$OUT/e3g_${NODE}_${SLURM_JOB_ID}_threads.csv

echo "=== node $NODE  job $SLURM_JOB_ID  git $GIT_SHA  seed $SEED  $(date -Is)"

declare -A SEEN
S0=(); S1=(); SMTC=()
while IFS=, read -r cpu core sock; do
    [ -z "${cpu:-}" ] && continue
    key="$sock:$core"
    if [ -n "${SEEN[$key]:-}" ]; then SMTC+=("$cpu"); continue; fi
    SEEN[$key]=1
    if [ "$sock" = "0" ]; then S0+=("$cpu"); else S1+=("$cpu"); fi
done < <(lscpu -p=CPU,CORE,SOCKET | grep -v '^#')
NC0=${#S0[@]}; NC1=${#S1[@]}
echo "cores: socket0=$NC0 socket1=$NC1  smt siblings=${#SMTC[@]}"
echo

# alt and grp use the SAME cpu set for a given T; only the order differs.
places() {   # $1 = alt|grp|s0|smt1  $2 = n
    local k=$1 n=$2 o="" i h
    case "$k" in
      alt)  (( (n+1)/2 > NC0 || n/2 > NC1 )) && return 1
            for ((i=0;i<n;i++)); do h=$((i/2))
                if (( i % 2 == 0 )); then o+="{${S0[$h]}},"; else o+="{${S1[$h]}},"; fi
            done ;;
      grp)  (( (n+1)/2 > NC0 || n/2 > NC1 )) && return 1
            for ((i=0;i<(n+1)/2;i++)); do o+="{${S0[$i]}},"; done
            for ((i=0;i<n/2;i++));     do o+="{${S1[$i]}},"; done ;;
      s0)   (( n > NC0 )) && return 1
            for ((i=0;i<n;i++)); do o+="{${S0[$i]}},"; done ;;
      smt1) (( n > 2*NC0 )) && return 1                # one socket + siblings
            for ((i=0;i<n;i++)); do
                if (( i < NC0 )); then o+="{${S0[$i]}},"
                else o+="{${SMTC[$((i-NC0))]}},"; fi
            done ;;
    esac
    echo "${o%,}"
}
cpus_of() { places "$1" "$2" | tr -d '{}' ; }

echo "########## STAGING (interleaved; D31) ##########"
OMP_NUM_THREADS=$((NC0+NC1)) OMP_PROC_BIND=close OMP_PLACES="$(places alt $((NC0+NC1)))" \
    "$BIN/stage_csr" --src "$PROJ/csr/W24"   --dst "$TMP/W24"   --mode interleave
OMP_NUM_THREADS=$((NC0+NC1)) OMP_PROC_BIND=close OMP_PLACES="$(places alt $((NC0+NC1)))" \
    "$BIN/stage_csr" --src "$PROJ/csr/Wfull" --dst "$TMP/Wfull" --mode interleave
"$BIN/numa_pages" "$TMP/W24/value.f32" | tail -n +2 \
    | awk -F, '{printf "  W24 %s=%.4f\n", $4, $7}'
for V in W24 Wfull; do
    "$BIN/kernel_serial" --csr "$TMP/$V" --lookup "$LOOKUP" \
        --out "$TMP/oracle_$V.f64" >/dev/null 2>&1
done
echo "  oracles generated"
echo

gen() {
    local T O P R K

    for V in "W24 24" "Wfull full"; do
        set -- $V
        for T in 4 6 8 10 12 14 16; do
            for O in alt grp; do
                echo "$1 $2 $T $O precomputed block -1 canonical"
            done
        done
        for T in 2 4 6 8; do
            echo "$1 $2 $T s0 precomputed block -1 canonical"
        done
    done
    # B. partitioners at T=16, both orders, both row orders (D41, A8)
    for P in block cyclic nzbalanced wbalanced contig_opt greedy; do
        for R in canonical sorted_desc; do
            for O in alt grp; do
                echo "W24 24 16 $O precomputed $P -1 $R"
            done
        done
    done
    for P in block nzbalanced contig_opt greedy; do
        for O in alt grp; do
            echo "Wfull full 16 $O precomputed $P -1 canonical"
        done
    done
    # C. schedules at T=16, both orders (D42's 39% spread may shrink)
    for O in alt grp; do
        echo "W24 24 16 $O static block -1 canonical"
        echo "W24 24 16 $O guided none 64 canonical"
        for K in 16 64 256 1024; do
            echo "W24 24 16 $O dynamic none $K canonical"
            echo "W24 24 16 $O static_chunked block_cyclic $K canonical"
        done
    done
    for T in 12 16; do
        echo "W24 24 $T smt1 precomputed block -1 canonical"
    done
}

gen | python -c "
import random, sys
L=[l for l in sys.stdin.read().splitlines() if l.strip()]
random.Random($SEED).shuffle(L)
sys.stdout.write('\n'.join(L)+'\n')
" > "$TMP/cfg.txt"
N=$(wc -l < "$TMP/cfg.txt")
echo "########## $N CONFIGURATIONS (randomised, seed $SEED) ##########"
echo

i=0
while read -r V W T O SCH PART CHUNK RO; do
    i=$((i+1))
    PL=$(places "$O" "$T") || { printf '[%3d/%3d] skipped %s T=%s\n' "$i" "$N" "$O" "$T"; continue; }
    if [ "$V" = "W24" ]; then K=$((4*T)); else K=$T; fi
    CH=""; [ "$CHUNK" != "-1" ] && CH="--chunk $CHUNK"
    SMTF=""; [ "$O" = "smt1" ] && SMTF="--smt"

    printf '[%3d/%3d] %-6s T=%-3s %-5s %-14s %-12s chunk=%-5s %s\n' \
        "$i" "$N" "$V" "$T" "$O" "$SCH" "$PART" "$CHUNK" "$RO"

    OMP_PROC_BIND=close OMP_PLACES="$PL" \
    "$BIN/kernel_omp" --csr "$TMP/$V" --lookup "$LOOKUP" \
        --variant "$V" --window "$W" --threads "$T" \
        --schedule "$SCH" --partitioner "$PART" $CH \
        --tile 0 --row-order "$RO" --whitelist natural \
        --repeat "$K" --reps 6 --warmup-s 2 \
        --platform broadwell --stage-mode interleave $SMTF \
        --reference "$TMP/oracle_$V.f64" \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" 2>&1 \
      | grep -E '^rep [1-9]|BIT-IDENTITY|migrated' | sed 's/^/          /'
    echo
done < "$TMP/cfg.txt"

echo "########## DONE ##########"
wc -l "$TIM" "$THR"
echo
echo "Reduce with:"
echo "  python code/probes/analyze_e3.py '$TIM'"
echo
echo "Ordering is recoverable from threads.csv by counting socket"
echo "transitions across tid order: alternating gives T-1, grouped gives 1."
echo "There is no schema column for it yet (addendum 2 sec 8)."
