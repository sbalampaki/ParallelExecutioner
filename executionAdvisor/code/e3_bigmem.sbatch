#!/bin/bash
#SBATCH --job-name=e3_bigmem
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=0:50:00
#SBATCH --output=/projects/sb2ea/logs/e3big_%j.out
#SBATCH --error=/projects/sb2ea/logs/e3big_%j.err


set -euo pipefail
export USER=sb2ea


HAVE_PY=0
if source /opt/ohpc/pub/apps/miniconda/etc/profile.d/conda.sh 2>/dev/null \
   && conda activate /projects/sb2ea/envs/mimic 2>/dev/null \
   && python -c 'import pandas' 2>/dev/null; then
    HAVE_PY=1
else
    echo "note: python/pandas unavailable here; CSV is still written."
fi


BIN=${BIN_OVERRIDE:-/projects/sb2ea/bin}
PROJ=/projects/sb2ea
LOOKUP=$PROJ/manifest/kernel_lookup.csv
OUT=$PROJ/results/timing
NODE=$(hostname -s)
GIT_SHA="${GIT_SHA:-unknown}"
TMP=/tmp/sb2ea_big
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP" "$OUT"
TIM=$OUT/big_${NODE}_${SLURM_JOB_ID}.csv
THR=$OUT/big_${NODE}_${SLURM_JOB_ID}_threads.csv

echo "=== node $NODE  job $SLURM_JOB_ID  git $GIT_SHA  $(date -Is)"
echo "=== binaries: $BIN"
lscpu | grep -E 'Model name|^CPU\(s\)|Socket|Core\(s\) per socket|NUMA node\(s\)|L2|L3' || true
echo

# --- placement sets derived from real topology 
declare -A SEEN
S0=(); S1=()
while IFS=, read -r cpu core sock; do
    [ -z "${cpu:-}" ] && continue
    key="$sock:$core"
    [ -n "${SEEN[$key]:-}" ] && continue          # skip SMT siblings
    SEEN[$key]=1
    if [ "$sock" = "0" ]; then S0+=("$cpu"); else S1+=("$cpu"); fi
done < <(lscpu -p=CPU,CORE,SOCKET | grep -v '^#')

NC0=${#S0[@]}; NC1=${#S1[@]}
echo "physical cores: socket0=$NC0  socket1=$NC1"
echo "  socket0 cpus: ${S0[*]:0:8} ..."
echo "  socket1 cpus: ${S1[*]:0:8} ..."
echo


places() {   # $1 = socket0|socket1|spread   $2 = n threads
    local kind=$1 n=$2 o="" i h
    case "$kind" in
      socket0) (( n > NC0 )) && return 1
               for ((i=0;i<n;i++)); do o+="{${S0[$i]}},"; done ;;
      socket1) (( n > NC1 )) && return 1
               for ((i=0;i<n;i++)); do o+="{${S1[$i]}},"; done ;;
      spread)  (( (n+1)/2 > NC0 || n/2 > NC1 )) && return 1
               for ((i=0;i<n;i++)); do
                   h=$((i/2))
                   if (( i % 2 == 0 )); then o+="{${S0[$h]}},";
                   else                      o+="{${S1[$h]}},"; fi
               done ;;
      *) return 1 ;;
    esac
    echo "${o%,}"
}

echo "########## STAGING ##########"

if (( NC1 > 0 )); then
    STAGE_N=$(( 2 * (NC0 < NC1 ? NC0 : NC1) ))
    STAGE_PL="$(places spread "$STAGE_N")"
else
    STAGE_N=$NC0
    STAGE_PL="$(places socket0 "$STAGE_N")"
    echo "  single socket: staging bound to socket 0 (interleave is a no-op)"
fi
OMP_NUM_THREADS=$STAGE_N OMP_PROC_BIND=close OMP_PLACES="$STAGE_PL" \
    "$BIN/stage_csr" --src "$PROJ/csr/W24" --dst "$TMP/W24" --mode interleave
"$BIN/numa_pages" "$TMP/W24/value.f32" | tail -n +2 \
    | awk -F, '{printf "  %s=%.4f\n", $4, $7}'

if ! "$BIN/kernel_serial" --csr "$TMP/W24" --lookup "$LOOKUP" \
        --out "$TMP/oracle.f64" >/dev/null 2>&1; then
    echo "FATAL: kernel_serial failed on $NODE." >&2
    echo "  If this is SIGILL, the node lacks an ISA level the binaries were" >&2
    echo "  built for (-march=haswell needs AVX2+FMA3; Sandy Bridge has" >&2
    echo "  neither). Build for this node's -march and re-run with" >&2
    echo "  BIN_OVERRIDE pointing at it." >&2
    exit 1
fi
echo "oracle generated -- binaries execute on $NODE"
echo

run() {   # $1 label  $2 T  $3 placement  $4 partitioner  $5 row_order  $6 tile
    local PL
    if ! PL=$(places "$3" "$2"); then
        printf '  %-24s %s\n' "$1" "skipped (not enough cores for $3 x$2)"
        return 0
    fi
    OMP_PROC_BIND=close OMP_PLACES="$PL" \
    "$BIN/kernel_omp" --csr "$TMP/W24" --lookup "$LOOKUP" \
        --variant W24 --window 24 --threads "$2" \
        --schedule precomputed --partitioner "$4" --tile "$6" \
        --row-order "$5" --repeat $((4 * $2)) --reps 6 --warmup-s 2 \
        --platform cascadelake --stage-mode interleave \
        --reference "$TMP/oracle.f64" \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" \
        >/dev/null 2>"$TMP/e.txt" || true
    printf '  %-24s ' "$1"
    awk '/^rep [1-9]/ {s+=$6; n++} END {printf "%8.1f M rec/s", s/n}' "$TMP/e.txt"
    grep -m1 'placement\|smt\|socket\|spread' /dev/null 2>/dev/null || true
    echo
}

echo "########## PAPI availability ##########"
"$BIN/kernel_omp" --csr "$TMP/W24" --lookup "$LOOKUP" --variant W24 \
    --threads 2 --schedule precomputed --partitioner block --tile 0 \
    --repeat 1 --reps 2 --warmup-s 0.5 --timings /dev/null \
    --threads-csv /dev/null --job-id 0 2>&1 >/dev/null | grep -E '^PAPI' || true
echo

echo "########## A. single-socket scaling (does 96.6% hold at 18 cores?) ####"
for T in 1 2 4 6 8 10 12 16 18; do
    run "socket0 T=$T" "$T" socket0 block canonical 0
done
echo

echo "########## B. spread scaling ##########"
for T in 2 4 8 16 24 32 36; do
    run "spread T=$T" "$T" spread block canonical 0
done
echo

echo "########## C. P8: cross-socket at matched thread count ##########"
if (( NC1 == 0 )); then
    echo "  single-socket node -- P8 is not testable here. It needs a"
    echo "  two-socket part; c11 (Sandy Bridge, 2x8) is the candidate."
fi
for T in 4 8 16; do
    run "T=$T socket0" "$T" socket0 block canonical 0
    run "T=$T socket1" "$T" socket1 block canonical 0
    run "T=$T spread"  "$T" spread  block canonical 0
done
echo

echo "########## D. locality: sorted_desc with balance held at 1.000 ######"
run "block canonical"    8 socket0 block      canonical   0
run "block sorted_desc"  8 socket0 block      sorted_desc 0
run "nzbal canonical"    8 socket0 nzbalanced canonical   0
run "nzbal sorted_desc"  8 socket0 nzbalanced sorted_desc 0
echo

echo "########## E. P9: tile harm onset should move with per-core L2 ######"
echo "  P7  (shared L3) predicts the onset does not move: L3 is 19.25 MB"
echo "      here against 20 MB on Broadwell, a 0.96x change."
echo "  P7' (per-core L2) predicts it moves 4x: L2 is 1 MB against 256 KB,"
echo "      so C = 1048576/1280 = 819 rather than 205, and the onset should"
echo "      shift from between 512-4096 to between 2048-16384."
echo "  The two caches scale by very different factors between these parts,"
echo "  which is what makes this a sharper test than one platform allows."
for C in 0 128 512 819 2048 4096 16384; do
    run "tile=$C" 8 socket0 block canonical "$C"
done
echo

echo "########## SUMMARY: counters per record ##########"
if [ "$HAVE_PY" = "1" ]; then
python - "$TIM" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); d = d[d.warmup == 0]
if d.empty:
    print("  no rows"); raise SystemExit
if (d.tot_cyc <= 0).all():
    print("  PAPI not linked on this platform -- rebuild there with papi loaded.")
d["work"] = d.n_records * d.k_iterations
d["tile"] = d.notes.str.extract(r"tile=(\d+)").astype(float)
g = (d.groupby(["thread_placement","n_threads","partitioner","row_order","tile"])
       .agg(mrec=("throughput_rec_s", lambda s: s.median()/1e6),
            ghz=("achieved_ghz","median"), cyc=("tot_cyc","median"),
            ins=("tot_ins","median"), tcm=("l3_tcm","median"),
            tca=("l3_tca","median"), work=("work","median"),
            timb=("time_imbalance","median")).reset_index())
g["IPC"]            = g.ins / g.cyc
g["cyc_per_rec"]    = g.cyc / g.work
g["ins_per_rec"]    = g.ins / g.work
g["L3miss_per_rec"] = g.tcm / g.work
g["rec_per_cycle"]  = g.mrec*1e6 / (g.ghz*1e9)
cols = ["thread_placement","n_threads","partitioner","row_order","tile",
        "mrec","ghz","rec_per_cycle","IPC","cyc_per_rec","ins_per_rec",
        "L3miss_per_rec","timb"]
print(g[cols].round(4).to_string(index=False))
print()
print("  P8: L3miss_per_rec for spread vs socket0 at matched T.")
print("      Broadwell showed +21.9% at T=8. Reproducing here makes it a")
print("      property of multi-socket systems; not reproducing makes the")
print("      Broadwell result platform-specific and the headline needs it.")
print("  P9: the tile size at which throughput starts to fall should move")
print("      from ~512 (Broadwell, 256 KB L2) to ~2048 (Cascade Lake, 1 MB).")
print("  ins_per_rec is the control: same binary, same work, so it must be")
print("      flat. On Broadwell it held at 24.3-25.0 across everything.")
EOF
else
    echo "  skipped (no python here). Reduce on the login node:"
    echo "    python code/probes/analyze_e3.py \'$TIM\'"
fi
echo
echo "timings: $TIM"
