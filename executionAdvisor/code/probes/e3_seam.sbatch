#!/bin/bash
#SBATCH --job-name=e3_seam
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=c1,c2,c3,c4,c5
#SBATCH --time=0:45:00
#SBATCH --output=/projects/sb2ea/logs/e3seam_%j.out
#SBATCH --error=/projects/sb2ea/logs/e3seam_%j.err


set -euo pipefail
export USER=sb2ea

BIN=/projects/sb2ea/bin
PROJ=/projects/sb2ea
SRC=$PROJ/csr/W24
LOOKUP=$PROJ/manifest/kernel_lookup.csv
OUT=$PROJ/results/timing
NODE=$(hostname -s)
GIT_SHA="${GIT_SHA:-unknown}"
TMP=/tmp/sb2ea_seam
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP" "$OUT"
TIM=$OUT/seam_${NODE}_${SLURM_JOB_ID}.csv
THR=$OUT/seam_${NODE}_${SLURM_JOB_ID}_threads.csv

for b in "$BIN/kernel_omp" "$BIN/kernel_serial" "$BIN/stage_csr"; do
    [ -x "$b" ] || { echo "FATAL: $b missing" >&2; exit 1; }
done
if "$BIN/kernel_omp" --note x 2>&1 | grep -q 'unknown arg'; then
    echo "FATAL: kernel_omp has no --note; rebuild first" >&2; exit 1
fi

echo "=== node $NODE  job $SLURM_JOB_ID  git $GIT_SHA  $(date -Is)"

declare -A SEEN
S0=(); S1=()
while IFS=, read -r cpu core sock; do
    [ -z "${cpu:-}" ] && continue
    key="$sock:$core"; [ -n "${SEEN[$key]:-}" ] && continue; SEEN[$key]=1
    if [ "$sock" = "0" ]; then S0+=("$cpu"); else S1+=("$cpu"); fi
done < <(lscpu -p=CPU,CORE,SOCKET | grep -v '^#')
NC=${#S0[@]}
echo "cores: socket0=$NC socket1=${#S1[@]}"


places_for_group() {   # $1 = group size g
    local g=$1 i j out=() prev="" seams=0
    for ((i=0; i<NC; i+=g)); do
        for ((j=i; j<i+g && j<NC; j++)); do out+=("${S0[$j]}"); done
        for ((j=i; j<i+g && j<NC; j++)); do out+=("${S1[$j]}"); done
    done
    for c in "${out[@]}"; do
        local s=0; for x in "${S1[@]}"; do [ "$x" = "$c" ] && s=1; done
        [ -n "$prev" ] && [ "$s" != "$prev" ] && seams=$((seams+1))
        prev=$s
    done
    printf '%d ' "$seams"
    printf '{%s},' "${out[@]}" | sed 's/,$//'
}


set_places() {   # $1 = group size g -> sets SEAMS and PLACES
    local out; out=$(places_for_group "$1")
    SEAMS=${out%% *}
    PLACES=${out#* }
}

set_places 1
OMP_NUM_THREADS=16 OMP_PROC_BIND=close \
    OMP_PLACES="$PLACES" \
    "$BIN/stage_csr" --src "$SRC" --dst "$TMP/csr" --mode interleave
"$BIN/kernel_serial" --csr "$TMP/csr" --lookup "$LOOKUP" \
    --out "$TMP/oracle.f64" >/dev/null 2>&1
echo "  staged (2 MiB interleave, alternating writer) and oracle generated"

run() {   # $1 group size
    local g=$1 rc=0
    set_places "$g"
    printf '  g=%-2d seams=%-3d ' "$g" "$SEAMS"
 
    set +e
    OMP_PROC_BIND=close OMP_PLACES="$PLACES" \
    "$BIN/kernel_omp" --csr "$TMP/csr" --lookup "$LOOKUP" --variant W24 \
        --window 24 --threads 16 --schedule precomputed --partitioner block \
        --tile 0 --repeat 64 --reps 6 --warmup-s 2 --platform broadwell \
        --reference "$TMP/oracle.f64" \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" \
        --note "exp=seam;group=$g;seams=$SEAMS" > "$TMP/run.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "FAILED rc=$rc"
        tail -8 "$TMP/run.log" | sed 's/^/      | /'
        set -e; return 0
    fi
 
    awk '/^rep [1-9]/ {s+=$6; g+=$13; w+=$10; t+=$12; n++}
         END {if (n) printf "%8.1f M rec/s  %.2f GHz  %7.1f M rec/GHz  work_imb %.4f  time_imb %.4f\n", s/n, g/n, (s/n)/(g/n), w/n, t/n; else print "NO REPS"}' \
         "$TMP/run.log"
    set -e
    return 0
}

echo; echo "########## SEAM SWEEP, W24 T=16, PALINDROME ##########"
echo "  g=8 is grouped (1 seam), g=1 is alternating (15 seams)."
echo "  D44 measured these two endpoints at +36.5% on Broadwell."
echo
for g in 8 4 2 1 1 2 4 8; do
    run "$g" || echo "    (run g=$g returned non-zero; continuing)"
done
echo "  sweep complete: 8 configurations attempted"

echo
echo "timings: $TIM"
echo "threads: $THR"
cat <<'EOT'

REDUCE WITH:
  python3 code/probes/reduce_seam.py --timings <TIM> --threads <THR>


EOT
