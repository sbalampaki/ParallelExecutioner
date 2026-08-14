#!/bin/bash
#SBATCH --job-name=e3_gran2
#SBATCH --partition=research-gpu
#SBATCH --nodelist=c11
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=1:30:00
#SBATCH --output=/projects/sb2ea/logs/e3gran2_%j.out
#SBATCH --error=/projects/sb2ea/logs/e3gran2_%j.err



set -euo pipefail
export USER=sb2ea

BIN=/projects/sb2ea/bin/snb
PROJ=/projects/sb2ea
SRC=$PROJ/csr
LOOKUP=$PROJ/manifest/kernel_lookup.csv
OUT=$PROJ/results/timing
NODE=$(hostname -s)
GIT_SHA="${GIT_SHA:-unknown}"
TMP=/tmp/sb2ea_gran2
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP" "$OUT"
TIM=$OUT/gran2_${NODE}_${SLURM_JOB_ID}.csv
THR=$OUT/gran2_${NODE}_${SLURM_JOB_ID}_threads.csv

for b in "$BIN/kernel_omp" "$BIN/kernel_serial" "$BIN/stage_csr" \
         "$BIN/numa_verify"; do
    [ -x "$b" ] || { echo "FATAL: $b missing" >&2; exit 1; }
done
if "$BIN/stage_csr" --block-kib 4096 2>&1 | grep -q 'unknown arg'; then
    echo "FATAL: stage_csr has no --block-kib. See the NOTE at the foot." >&2
    exit 1
fi

echo "=== node $NODE  job $SLURM_JOB_ID  git $GIT_SHA  $(date -Is)"
lscpu | grep -m1 'Model name' || true

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
ALT_PL=$(places "${ALT[@]}"); GRP_CSV=$(csv "${ALL[@]}")
echo "cores: socket0=${#S0[@]} socket1=${#S1[@]}"

stage() {   # $1 label  $2 variant  $3 block KiB
    local D="$TMP/$1"
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$ALT_PL" \
        "$BIN/stage_csr" --src "$SRC/$2" --dst "$D" \
        --mode interleave --block-kib "$3" 2>/dev/null
    [ -s "$D/offsets.i64" ] || { echo "FATAL: staging failed: $D" >&2; exit 1; }
    printf '  staged %-10s %6d KiB (%.2f MiB)  page_local_frac: ' \
        "$1" "$3" "$(echo "scale=2; $3/1024" | bc)"
    "$BIN/numa_verify" --csr "$D" --threads 16 --cpus "$GRP_CSV" \
        | awk '/value.f32/{v=1} v && /page_local_frac/{print $4; exit}'
}

run() {   # $1 label  $2 variant  $3 tree  $4 threads  $5 sched  $6 chunk
    local part=block; local -a extra=()
    [ "$5" = dynamic ] && { part=none; extra=(--chunk "$6"); }
    printf '  %-40s ' "$1"
    if ! OMP_PROC_BIND=close OMP_PLACES="$(places "${ALL[@]:0:$4}")" \
      "$BIN/kernel_omp" --csr "$TMP/$3" --lookup "$LOOKUP" --variant "$2" \
        --window "${2#W}" --threads "$4" --schedule "$5" \
        --partitioner "$part" "${extra[@]}" \
        --tile 0 --repeat $((4 * $4)) --reps 6 --warmup-s 2 \
        --platform sandybridge --reference "$TMP/oracle_$2.f64" \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" > "$TMP/run.log" 2>&1
    then echo "FAILED"; tail -8 "$TMP/run.log" | sed 's/^/      | /'; return 0; fi
    # Field positions verified against a real kernel_omp rep line:
    #   $6 throughput  $9 "work_imb"  $10 VALUE  $11 "time_imb"  $12 VALUE
    # Job 74020 used $9/$11, which are the LABELS -- awk parses those as 0,
    # which is why every line printed 0.0000.
    awk '/^rep [1-9]/ {s+=$6; w+=$10; t+=$12; n++}
         END {if (n) printf "%8.1f M rec/s   work_imb %.4f  time_imb %.4f\n",
              s/n, w/n, t/n; else print "NO REPS"}' "$TMP/run.log"
}

echo; echo "########## P16: EDGE, W24, T=16 ##########"
EDGE=(4608 4864 5120 5376 5632 5888 6144 6656)     # 4.5 .. 6.5 MiB
for K in "${EDGE[@]}"; do stage "edge_$K" W24 "$K"; done
"$BIN/kernel_serial" --csr "$TMP/edge_4608" --lookup "$LOOKUP" \
    --out "$TMP/oracle_W24.f64" >/dev/null 2>&1
echo "  predicted edge: block = stride/2 = 5.24 MiB = 5366 KiB"
echo
for K in "${EDGE[@]}"; do
    run "W24 edge $(echo "scale=2;$K/1024"|bc) MiB  T=16" W24 "edge_$K" 16 precomputed -1
done
echo "  --- reverse pass ---"
for ((i=${#EDGE[@]}-1; i>=0; i--)); do
    K=${EDGE[$i]}
    run "W24 edge $(echo "scale=2;$K/1024"|bc) MiB  T=16" W24 "edge_$K" 16 precomputed -1
done
rm -rf "$TMP"/edge_*

echo; echo "########## P17: FIXED TREE, VARYING THREAD COUNT ##########"
stage coarse W24 6144      # 6 MiB block, period 12 MiB
stage fine   W24 2048      # 2 MiB block, period  4 MiB
"$BIN/kernel_serial" --csr "$TMP/fine" --lookup "$LOOKUP" \
    --out "$TMP/oracle_W24.f64" >/dev/null 2>&1
echo "  same pages throughout; only T moves, and T alone sets the stride."
echo "  T:      4     8    12    14    16"
echo "  stride 41.9  21.0  14.0  12.0  10.5 MiB"
echo "  ratio   3.49  1.75  1.16  1.00  0.87  (stride/period, coarse tree)"
echo
for T in 4 8 12 14 16 16 14 12 8 4; do
    run "coarse(6 MiB)  T=$T" W24 coarse "$T" precomputed -1
    run "fine  (2 MiB)  T=$T" W24 fine   "$T" precomputed -1
done

echo; echo "########## P18: DYNAMIC ON BOTH TREES, T=16 ##########"
for TREE in coarse fine; do
    run "$TREE  T=16  precomputed" W24 "$TREE" 16 precomputed -1
    run "$TREE  T=16  dynamic(64)" W24 "$TREE" 16 dynamic     64
done
echo "  74020: dynamic recovered the whole step (+16.6% coarse, +7.8% fine)."
echo "  time_imbalance is now printed, so the recovery can be attributed."

echo
echo "timings: $TIM"
echo "threads: $THR"
cat <<'EOT'

READ IT AS:
NOTE -- stage_csr needs --block-kib for sub-MiB resolution. Alongside the
existing --block-mib arm add:

    else if (!strcmp(argv[i], "--block-kib") && i + 1 < argc)
        BLOCK = (size_t)atol(argv[++i]) << 10;

and confirm `head` is bounded by 2 MiB (THP_ALIGN), not by BLOCK -- at a
6.5 MiB block an unbounded head would place 4% of the array on one node
by itself. Rebuild both ISA levels; kernel_core.o is untouched so its
sha256 must not move, and verify_core_objects.sh will say so.
EOT
