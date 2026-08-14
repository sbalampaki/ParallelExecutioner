#!/bin/bash
#SBATCH --job-name=e3_c11_gran
#SBATCH --partition=research-gpu
#SBATCH --nodelist=c11
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=1:30:00
#SBATCH --output=/projects/sb2ea/logs/e3gran_%j.out
#SBATCH --error=/projects/sb2ea/logs/e3gran_%j.err

set -euo pipefail
export USER=sb2ea

BIN=/projects/sb2ea/bin/snb          # c11 has no AVX2/FMA3 (D27 bound)
PROJ=/projects/sb2ea
SRC=$PROJ/csr
LOOKUP=$PROJ/manifest/kernel_lookup.csv
OUT=$PROJ/results/timing
NODE=$(hostname -s)
GIT_SHA="${GIT_SHA:-unknown}"
TMP=/tmp/sb2ea_gran
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP" "$OUT"
TIM=$OUT/gran_${NODE}_${SLURM_JOB_ID}.csv
THR=$OUT/gran_${NODE}_${SLURM_JOB_ID}_threads.csv

for b in "$BIN/kernel_omp" "$BIN/kernel_serial" "$BIN/stage_csr" \
         "$BIN/numa_verify"; do
    [ -x "$b" ] || { echo "FATAL: $b missing -- c11 needs its own" \
                          "-march=sandybridge build" >&2; exit 1; }
done

if "$BIN/stage_csr" --block-mib 2 2>&1 | grep -q 'unknown arg'; then
    echo "FATAL: stage_csr has no --block-mib. See the NOTE at the foot" >&2
    echo "       of this script; it is a four-line change." >&2; exit 1
fi

echo "=== node $NODE  job $SLURM_JOB_ID  git $GIT_SHA  $(date -Is)"
lscpu | grep -m1 'Model name' || true
df -h /tmp | tail -1

declare -A SEEN
S0=(); S1=()
while IFS=, read -r cpu core sock; do
    [ -z "${cpu:-}" ] && continue
    key="$sock:$core"; [ -n "${SEEN[$key]:-}" ] && continue; SEEN[$key]=1
    if [ "$sock" = "0" ]; then S0+=("$cpu"); else S1+=("$cpu"); fi
done < <(lscpu -p=CPU,CORE,SOCKET | grep -v '^#')
ALL=("${S0[@]}" "${S1[@]}")
ALT=(); for ((i=0;i<${#S0[@]};i++)); do ALT+=("${S0[$i]}" "${S1[$i]}"); done
places(){ printf '{%s},' "$@" | sed 's/,$//'; }
csv(){ local IFS=,; echo "$*"; }
GRP_PL=$(places "${ALL[@]}"); GRP_CSV=$(csv "${ALL[@]}")
ALT_PL=$(places "${ALT[@]}")
echo "cores: socket0=${#S0[@]} socket1=${#S1[@]}"

run() {   # $1 label  $2 variant  $3 tree  $4 sched  $5 chunk
    local part=block; local -a extra=()
    [ "$4" = dynamic ] && { part=none; extra=(--chunk "$5"); }
    printf '  %-42s ' "$1"
    if ! OMP_PROC_BIND=close OMP_PLACES="$GRP_PL" \
      "$BIN/kernel_omp" --csr "$TMP/$3" --lookup "$LOOKUP" --variant "$2" \
        --window "${2#W}" --threads 16 --schedule "$4" \
        --partitioner "$part" "${extra[@]}" \
        --tile 0 --repeat 64 --reps 6 --warmup-s 2 \
        --platform sandybridge --reference "$TMP/oracle_$2.f64" \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" > "$TMP/run.log" 2>&1
    then echo "FAILED"; tail -8 "$TMP/run.log" | sed 's/^/      | /'; return 0; fi
    awk '/^rep [1-9]/ {s+=$6; w+=$9; t+=$11; n++}
         END {if (n) printf "%8.1f M rec/s   work_imb %.4f  time_imb %.4f\n",
              s/n, w/n, t/n; else print "NO REPS"}' "$TMP/run.log"
}


ladder() {   # $1 variant  $2... block sizes, then the palindrome order
    local V=$1; shift
    local -a rungs=("$@")
    echo; echo "########## $V LADDER ##########"
    for B in "${rungs[@]}"; do
        local D="$TMP/${V}_b${B}"
        OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$ALT_PL" \
            "$BIN/stage_csr" --src "$SRC/$V" --dst "$D" \
            --mode interleave --block-mib "$B" 2>/dev/null
        [ -s "$D/offsets.i64" ] || { echo "FATAL: staging failed for $D" >&2; exit 1; }
        printf '  staged %-6s block %2d MiB   page_local_frac: ' "$V" "$B"
        "$BIN/numa_verify" --csr "$D" --threads 16 --cpus "$GRP_CSV" \
            | awk '/value.f32/{v=1} v && /page_local_frac/{print $4; exit}'
    done
    echo "  every rung must read ~0.50 -- aggregate locality is the CONTROL,"
    echo "  held fixed while the period varies, so any effect is period."
    "$BIN/kernel_serial" --csr "$TMP/${V}_b2" --lookup "$LOOKUP" \
        --out "$TMP/oracle_$V.f64" >/dev/null 2>&1
    echo "  oracle generated (c11 has its own object and oracle, D27)"
    echo
    for B in "${rungs[@]}"; do run "$V  block ${B} MiB" "$V" "${V}_b${B}" precomputed -1; done
    echo "  --- reverse pass (palindrome) ---"
    for ((i=${#rungs[@]}-1; i>=0; i--)); do
        B=${rungs[$i]}; run "$V  block ${B} MiB" "$V" "${V}_b${B}" precomputed -1
    done
}

ladder W24 2 3 4 5 6 8 10 12 16 32

echo; echo "########## P15: DYNAMIC AT A SLOW RUNG AND A FAST ONE ##########"
for B in 16 4; do
    run "W24 block ${B} MiB  precomputed" W24 "W24_b${B}" precomputed -1
    run "W24 block ${B} MiB  dynamic(64)" W24 "W24_b${B}" dynamic     64
done
echo "  16 MiB is W24's predicted-worst rung (sd 2.04), 4 MiB a predicted-fast"
echo "  one (sd 0.47). Decorrelation predicts a large dynamic gain at 16 and a"
echo "  small one at 4."
rm -rf "$TMP"/W24_b*

ladder Wfull 2 4 5 8 16 32

echo
echo "timings: $TIM"
echo "threads: $THR"
cat <<'EOT'

READ IT AS:

NOTE -- stage_csr needs --block-mib. Currently `#define BLOCK (2UL << 20)`.
Make it a variable defaulting to 2 MiB, add

    else if (!strcmp(argv[i], "--block-mib") && i + 1 < argc)
        BLOCK = (size_t)atol(argv[++i]) << 20;

and keep the head-alignment logic untouched -- it aligns the grid to the
destination mapping's own boundaries and copies the leading partial block
serially, which matters more as BLOCK grows, not less.
EOT
