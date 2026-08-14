#!/usr/bin/env bash
#SBATCH -p research-cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 0:30:00
#SBATCH -J stream
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH -o /projects/sb2ea/logs/stream-%j.out
#SBATCH -e /projects/sb2ea/logs/stream-%j.err

set -euo pipefail
export USER=sb2ea

PROJ=/projects/sb2ea
BIN=$PROJ/bin/stream_cpu
OUT=$PROJ/results/stream
mkdir -p "$OUT"
CSV=$OUT/stream_$(hostname)_${SLURM_JOB_ID}.csv

test -x "$BIN" || { echo "FATAL: $BIN not built. Run build_stream.sh on the login node."; exit 1; }

echo "host    : $(hostname)"
echo "date    : $(date -Is)"
echo "nproc   : $(nproc)"
numactl --hardware || true
echo

( while :; do
    awk -F: '/MHz/{s+=$2; n++} END{if(n)printf "%s clock_avg_mhz=%.0f\n", strftime("%H:%M:%S"), s/n}' \
      /proc/cpuinfo
    sleep 5
  done ) > "$OUT/clock_${SLURM_JOB_ID}.log" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

echo "config,threads,bind,kernel,best_rate_mb_s,avg_time_s,min_time_s,max_time_s" > "$CSV"

run_stream () {          # $1 label   $2 nthreads   $3... launcher
    local label=$1 nt=$2; shift 2
    echo "=================================================================="
    echo ">>> $label   OMP_NUM_THREADS=$nt   launcher: $*"
    echo "=================================================================="
    OMP_NUM_THREADS=$nt OMP_PROC_BIND=close OMP_PLACES=cores OMP_DISPLAY_ENV=verbose \
        "$@" "$BIN" 2>&1 | tee /tmp/stream_$$.txt
    awk -v L="$label" -v NT="$nt" -v B="$*" '
        /^(Copy|Scale|Add|Triad):/ {
            gsub(":","",$1);
            printf "%s,%s,\"%s\",%s,%s,%s,%s,%s\n", L, NT, B, $1, $2, $3, $4, $5
        }' /tmp/stream_$$.txt >> "$CSV"
    rm -f /tmp/stream_$$.txt
    echo
}

for NT in 1 2 4 8 16; do
    run_stream "node" "$NT" env
done


if command -v numactl >/dev/null; then
    run_stream "socket0" 8 numactl --cpunodebind=0 --membind=0
    run_stream "socket1" 8 numactl --cpunodebind=1 --membind=1
    # deliberate NUMA-remote case: useful as a labelled contrast on Day 12
    run_stream "remote"  8 numactl --cpunodebind=0 --membind=1
else
    echo "WARN: numactl absent; per-socket numbers skipped."
    echo "      Fall back to an explicit place list for socket 0:"
    echo "      OMP_PLACES='{0},{2},{4},{6},{8},{10},{12},{14}'"
fi

# ---- SMT, labelled separately, never as '32 cores' -------------------
run_stream "smt" 32 env

kill $SAMPLER 2>/dev/null || true
echo "clock samples: $OUT/clock_${SLURM_JOB_ID}.log"
sort -t= -k2 -n "$OUT/clock_${SLURM_JOB_ID}.log" | sed -n '1p;$p' || true

echo
echo "=== results -> $CSV ==="
column -s, -t < "$CSV"
