#!/bin/bash
#SBATCH --job-name=e3_papi
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH --time=0:40:00
#SBATCH --output=/projects/sb2ea/logs/e3papi_%j.out
#SBATCH --error=/projects/sb2ea/logs/e3papi_%j.err


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
TMP=/tmp/sb2ea_papi
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP" "$OUT"
TIM=$OUT/papi_${NODE}_${SLURM_JOB_ID}.csv
THR=$OUT/papi_${NODE}_${SLURM_JOB_ID}_threads.csv

pl() {
    local n=$2 o="" i
    case "$1" in
      socket0|smt) for ((i=0;i<n;i++)); do o+="{$((2*i))},";   done ;;
      socket1)     for ((i=0;i<n;i++)); do o+="{$((2*i+1))},"; done ;;
      spread)      for ((i=0;i<n;i++)); do o+="{$i},";         done ;;
    esac
    echo "${o%,}"
}

echo "=== node $NODE  job $SLURM_JOB_ID  $(date -Is)"
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$(pl spread 16)" \
    "$BIN/stage_csr" --src "$PROJ/csr/W24" --dst "$TMP/W24" --mode interleave
"$BIN/kernel_serial" --csr "$TMP/W24" --lookup "$LOOKUP" \
    --out "$TMP/oracle.f64" >/dev/null 2>&1
echo

run() {   # $1 label  $2 threads  $3 placement  $4 partitioner  $5 row_order
    OMP_PROC_BIND=close OMP_PLACES="$(pl "$3" "$2")" \
    "$BIN/kernel_omp" --csr "$TMP/W24" --lookup "$LOOKUP" \
        --variant W24 --window 24 --threads "$2" \
        --schedule precomputed --partitioner "$4" --tile 0 \
        --row-order "$5" --repeat $((4 * $2)) --reps 6 --warmup-s 2 \
        --platform broadwell --stage-mode interleave \
        --reference "$TMP/oracle.f64" \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" \
        > /dev/null 2>"$TMP/err.txt" || true
    grep -m1 'PAPI:' "$TMP/err.txt" >/dev/null 2>&1 || true
    printf '  %-26s ' "$1"
    awk '/^rep [1-9]/ {s+=$6; n++} END {printf "%8.1f", s/n}' "$TMP/err.txt"
}

echo "########## PAPI availability ##########"
"$BIN/kernel_omp" --csr "$TMP/W24" --lookup "$LOOKUP" --variant W24 \
    --threads 2 --schedule precomputed --partitioner block --tile 0 \
    --repeat 1 --reps 2 --warmup-s 0.5 --timings /dev/null \
    --threads-csv /dev/null --job-id 0 2>&1 >/dev/null | grep -E '^PAPI' || true
echo

echo "########## (1) cross-socket, T=8 and T=16, block/canonical ##########"
for CFG in "T8 socket0:8:socket0" "T8 socket1:8:socket1" "T8 spread:8:spread" \
           "T4 socket0:4:socket0" "T4 spread:4:spread" \
           "T16 spread:16:spread" "T16 smt(1sock):16:smt"; do
    L=${CFG%%:*}; R=${CFG#*:}; T=${R%%:*}; P=${R##*:}
    run "$L" "$T" "$P" block canonical; echo " M rec/s"
done
echo

echo "########## (2) sorted_desc, T=8 socket0 ##########"
echo "  nzbalanced holds work_imb at 1.000 in both orders, so any"
echo "  difference here is locality rather than balance."
for CFG in "block canonical:block:canonical" "block sorted_desc:block:sorted_desc" \
           "nzbal canonical:nzbalanced:canonical" "nzbal sorted_desc:nzbalanced:sorted_desc"; do
    L=${CFG%%:*}; R=${CFG#*:}; PA=${R%%:*}; RO=${R##*:}
    run "$L" 8 socket0 "$PA" "$RO"; echo " M rec/s"
done
echo

echo "########## SUMMARY: counters per record ##########"
python - "$TIM" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
d = d[d.warmup == 0]
if d.empty or (d.tot_cyc <= 0).all():
    print("  no counters recorded -- PAPI was not linked in. Rebuild with")
    print("  `module load papi` visible to build_kernel.sh.")
    raise SystemExit
# counters are totals over the K-looped region, so normalise by the
# records actually processed: n_records * k_iterations.
d["work"] = d.n_records * d.k_iterations
g = (d.groupby(["thread_placement", "n_threads", "partitioner", "row_order"])
       .agg(mrec=("throughput_rec_s", lambda s: s.median()/1e6),
            ghz=("achieved_ghz", "median"),
            cyc=("tot_cyc", "median"), ins=("tot_ins", "median"),
            tcm=("l3_tcm", "median"), tca=("l3_tca", "median"),
            work=("work", "median")).reset_index())
g["IPC"]          = g.ins / g.cyc
g["cyc_per_rec"]  = g.cyc / g.work
g["ins_per_rec"]  = g.ins / g.work
g["L3miss_per_rec"] = g.tcm / g.work
g["L3_miss_pct"]  = 100 * g.tcm / g.tca
cols = ["thread_placement","n_threads","partitioner","row_order","mrec",
        "IPC","cyc_per_rec","ins_per_rec","L3miss_per_rec","L3_miss_pct"]
print(g[cols].round(4).to_string(index=False))
print()
print("  READ AS:")
print("    L3miss_per_rec HIGHER for spread than socket0  -> cache is the")
print("      cross-socket mechanism, and the effect is finally named.")
print("    L3miss_per_rec EQUAL but IPC lower for spread  -> the stall is")
print("      not cache. Uncore frequency remains the standing hypothesis,")
print("      and it needs MSR access we do not have -- so it stays a named")
print("      hypothesis and the effect is reported by elimination.")
print("    ins_per_rec should be IDENTICAL everywhere: same binary, same")
print("      work. If it is not, something other than placement differs.")
EOF
echo
echo "timings: $TIM"
