#!/bin/bash
#SBATCH --job-name=e4_papi
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH --nodelist=c6
#SBATCH --time=1:00:00
#SBATCH --output=/projects/sb2ea/logs/e4papi_%j.out
#SBATCH --error=/projects/sb2ea/logs/e4papi_%j.err



set -euo pipefail
export USER=sb2ea

BIN=/projects/sb2ea/bin
PROJ=/projects/sb2ea
SRC=$PROJ/csr/W24
LOOKUP=$PROJ/manifest/kernel_lookup.csv
OUT=$PROJ/results/timing
NODE=$(hostname -s)
GIT_SHA="${GIT_SHA:-unknown}"
TMP=/tmp/sb2ea_e4p
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP" "$OUT"
TIM=$OUT/e4p_${NODE}_${SLURM_JOB_ID}.csv
THR=$OUT/e4p_${NODE}_${SLURM_JOB_ID}_threads.csv


PAPI_FLAG="${PAPI_FLAG:-}"


export LD_LIBRARY_PATH="/opt/ohpc/pub/libs/papi/6.0.0/lib:${LD_LIBRARY_PATH:-}"

for b in "$BIN/kernel_omp" "$BIN/kernel_serial" "$BIN/stage_csr" \
         "$BIN/stage_aligned" "$BIN/numa_verify"; do
    [ -x "$b" ] || { echo "FATAL: $b missing" >&2; exit 1; }
done

echo "=== node $NODE  job $SLURM_JOB_ID  git $GIT_SHA  $(date -Is)"
echo "=== PAPI_FLAG='$PAPI_FLAG'"

# --- topology, derived 
declare -A SEEN
S0=(); S1=()
while IFS=, read -r cpu core sock; do
    [ -z "${cpu:-}" ] && continue
    key="$sock:$core"; [ -n "${SEEN[$key]:-}" ] && continue; SEEN[$key]=1
    if [ "$sock" = "0" ]; then S0+=("$cpu"); else S1+=("$cpu"); fi
done < <(lscpu -p=CPU,CORE,SOCKET | grep -v '^#')

GRP=("${S0[@]}" "${S1[@]}")
REV=("${S1[@]}" "${S0[@]}")              # reversed: gives the anti tree
ALT=(); for ((i=0;i<${#S0[@]};i++)); do ALT+=("${S0[$i]}" "${S1[$i]}"); done

places(){ printf '{%s},' "$@" | sed 's/,$//'; }
csv(){ local IFS=,; echo "$*"; }
GRP_PL=$(places "${GRP[@]}"); GRP_CSV=$(csv "${GRP[@]}")
REV_PL=$(places "${REV[@]}")
ALT_PL=$(places "${ALT[@]}")
S0_PL=$(places  "${S0[@]}");  S0_CSV=$(csv "${S0[@]}")
echo "cores: socket0=${#S0[@]} socket1=${#S1[@]}"

# --- stage the four trees 
echo; echo "########## STAGING ##########"
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$GRP_PL" \
    "$BIN/stage_csr" --src "$SRC" --dst "$TMP/intl_grp" --mode interleave
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$ALT_PL" \
    "$BIN/stage_csr" --src "$SRC" --dst "$TMP/intl_alt" --mode interleave
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$GRP_PL" \
    "$BIN/stage_aligned" --src "$SRC" --dst "$TMP/algn" >/dev/null
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$REV_PL" \
    "$BIN/stage_aligned" --src "$SRC" --dst "$TMP/anti" >/dev/null
echo "  four trees staged"

"$BIN/kernel_serial" --csr "$TMP/algn" --lookup "$LOOKUP" \
    --out "$TMP/oracle.f64" >/dev/null 2>&1
echo "  oracle generated"


echo; echo "########## MEASURED page_local_frac AT T=16 GROUPED ##########"
for P in anti intl_alt intl_grp algn; do
    printf '  %-9s ' "$P"
    "$BIN/numa_verify" --csr "$TMP/$P" --threads 16 --cpus "$GRP_CSV" \
        | awk '/value.f32/{v=1} v && /page_local_frac/{print $4; exit}'
done
echo "  predicted: anti 0.0000  intl_alt ~0.50  intl_grp 0.5494  algn 1.0000"


run() {   # $1 label  $2 threads  $3 places  $4 tree
    printf '  %-30s ' "$1"
    # `if !` puts this in condition context, so `set -e` cannot kill the job
    # before the failure is reported.
    if ! OMP_PROC_BIND=close OMP_PLACES="$3" \
    "$BIN/kernel_omp" --csr "$TMP/$4" --lookup "$LOOKUP" --variant W24 \
        --window 24 --threads "$2" --schedule precomputed --partitioner block \
        --tile 0 --repeat $((4 * $2)) --reps 6 --warmup-s 2 \
        --platform broadwell --reference "$TMP/oracle.f64" \
        $PAPI_FLAG \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" > "$TMP/run.log" 2>&1
    then
        echo "FAILED"; tail -20 "$TMP/run.log" | sed 's/^/      /'; return 1
    fi
    echo "ok"
    grep -Ei 'papi|error|warn' "$TMP/run.log" | sed 's/^/      /' || true
}


echo; echo "########## PREFLIGHT ##########"
run "papi preflight (algn)" 16 "$GRP_PL" algn

if awk -F, 'NR==1 {for (i=1;i<=NF;i++) if ($i=="l3_tcm") c=i;
                   if (!c) {print "no l3_tcm column in header" > "/dev/stderr"; exit 1}; next}
            $c != -1 {found=1}
            END {exit !found}' "$TIM"; then
    echo "  l3_tcm is populated -- counters are live"
    awk -F, 'NR==1{for(i=1;i<=NF;i++){if($i=="l3_tcm")m=i; if($i=="l3_tca")a=i;
                                      if($i=="tot_cyc")y=i; if($i=="tot_ins")n=i}; next}
             NR==2{printf "  first row: l3_tcm=%s l3_tca=%s tot_cyc=%s tot_ins=%s\n",
                          $m,$a,$y,$n}' "$TIM"
else
    echo "  FATAL: every l3_tcm is -1. kernel_omp is linked against PAPI"
    echo "  (libpapi.so.6.0, libpfm.so.4) and enables it by default, so this"
    echo "  means pw_init() failed at runtime, not that a flag is missing."
    echo "  Check perf access on this node:"
    echo "    cat /proc/sys/kernel/perf_event_paranoid"
    echo "    dmesg 2>/dev/null | grep -i perf | tail"
    exit 1
fi
rm -f "$TIM" "$THR"          # discard the preflight rows


echo; echo "########## T=16 GROUPED, PALINDROME ORDER ##########"
for P in algn intl_grp intl_alt anti anti intl_alt intl_grp algn; do
    run "T=16 grouped  $P" 16 "$GRP_PL" "$P"
done


echo; echo "########## T=8 SOCKET0 ANCHOR ##########"
run "T=8 socket0  algn  (0.50 local)" 8 "$S0_PL" algn
run "T=8 socket0  anti  (0.50 local)" 8 "$S0_PL" anti

echo
echo "timings: $TIM"
echo "threads: $THR"
cat <<'EOT'

REDUCE WITH:
  python3 - <<'PY'
  import pandas as pd
  t = pd.read_csv("<TIM>")
  t = t[t.warmup == 0]
  t["l3_per_rec"]  = t.l3_tcm / (t.n_records * t.k_iterations)
  t["cyc_per_rec"] = t.tot_cyc / (t.n_records * t.k_iterations)
  t["ins_per_rec"] = t.tot_ins / (t.n_records * t.k_iterations)
  t["mrec_ghz"]    = t.throughput_rec_s / 1e6 / t.achieved_ghz
  print(t.groupby(["n_threads","notes"])
          [["l3_per_rec","cyc_per_rec","ins_per_rec","mrec_ghz"]].median())
  PY


EOT
