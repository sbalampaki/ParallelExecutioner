#!/bin/bash
#SBATCH --job-name=numa_roof
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH --time=0:45:00
#SBATCH --output=/projects/sb2ea/logs/numa_roof_%j.out
#SBATCH --error=/projects/sb2ea/logs/numa_roof_%j.err


set -euo pipefail

export USER=sb2ea                      # LDAP does not resolve in jobs
BIN=/projects/sb2ea/bin
SRC_CSR=${SRC_CSR:-/projects/sb2ea/csr/Wfull}
RESULTS=/projects/sb2ea/results/day6
STAMP="${SLURM_JOB_ID}"
mkdir -p "$RESULTS" /projects/sb2ea/logs

# git is not installed on compute nodes; the sha is passed in at submit time.
GIT_SHA="${GIT_SHA:-unknown}"

TMP=/tmp/sb2ea_day6
SER="$TMP/csr_serial"
INT="$TMP/csr_interleave"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

echo "=== node: $(hostname)  job: $SLURM_JOB_ID  git: $GIT_SHA"
echo "=== csr : $SRC_CSR"
echo "=== date: $(date -Is)"
nproc
lscpu | grep -E 'Model name|Socket|NUMA node' || true
echo

mkdir -p "$TMP"


places_spread() {                       # $1 = n threads
  local n=$1 out="" i
  for ((i = 0; i < n; i++)); do out+="{$i},"; done
  echo "${out%,}"
}
places_socket0() {                      # $1 = n threads (max 8)
  local n=$1 out="" i
  for ((i = 0; i < n; i++)); do out+="{$((i * 2))},"; done
  echo "${out%,}"
}

run_probe() {                           # $1 tag  $2 csr  $3 nthreads  $4 places  $5 touch
  local tag=$1 csr=$2 nt=$3 places=$4 touch=$5
  echo "--- $tag  threads=$nt  touch=$touch  places=$places"
  OMP_NUM_THREADS="$nt" \
  OMP_PROC_BIND=close \
  OMP_PLACES="$places" \
  "$BIN/csr_stream" --csr "$csr" --touch "$touch" --reps 6 --tag "$tag" \
    | tee -a "$RESULTS/probe_${STAMP}.csv"
  echo
}

echo "########## STAGING ##########"
# Serial stage: single-threaded, i.e. what a plain `cp` does today.
"$BIN/stage_csr" --src "$SRC_CSR" --dst "$SER" --mode serial

# Interleaved stage: OpenMP writer bound alternately across both sockets.
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$(places_spread 16)" \
  "$BIN/stage_csr" --src "$SRC_CSR" --dst "$INT" --mode interleave
echo


echo "########## TASK 1a: PAGE PLACEMENT ##########"
{
  "$BIN/numa_pages" "$SER/itemid.i32" "$SER/t.f32" "$SER/value.f32"
  "$BIN/numa_pages" "$INT/itemid.i32" "$INT/t.f32" "$INT/value.f32"
} | tee "$RESULTS/numa_pages_${STAMP}.csv"
echo


echo "########## TASK 1b: 8-THREAD PLACEMENT CONTRAST ##########"
run_probe "t1_serialstage_socket0" "$SER" 8 "$(places_socket0 8)" all
run_probe "t1_serialstage_spread"  "$SER" 8 "$(places_spread  8)" all
run_probe "t1_interstage_socket0"  "$INT" 8 "$(places_socket0 8)" all
run_probe "t1_interstage_spread"   "$INT" 8 "$(places_spread  8)" all


echo "########## TASK 2: ROOF SWEEP (12 B/record) ##########"
for NT in 1 2 3 4 5 6 8 10 12 14 16; do
  run_probe "t2_roof_all_${NT}" "$INT" "$NT" "$(places_spread "$NT")" all
done

# Same sweep at 4 B/record.  This reproduces the traffic profile of the
# 0%-whitelist fixture IF the kernel branches on slot[id] before loading
# t[] and value[].  The pair separates traversal traffic from aggregation.
echo "########## TASK 2b: 4 B/record CONTRAST ##########"
for NT in 1 2 4 8 16; do
  run_probe "t2_roof_item_${NT}" "$INT" "$NT" "$(places_spread "$NT")" itemid
done


echo "########## TASK 2c: WITH OUTPUT WRITE ##########"
for NT in 1 4 8 16; do
  echo "--- t2_write_${NT}"
  OMP_NUM_THREADS="$NT" OMP_PROC_BIND=close OMP_PLACES="$(places_spread "$NT")" \
    "$BIN/csr_stream" --csr "$INT" --touch all --write-out --reps 6 \
                      --tag "t2_write_${NT}" \
    | tee -a "$RESULTS/probe_${STAMP}.csv"
  echo
done

echo "########## DONE ##########"
echo "results: $RESULTS/probe_${STAMP}.csv"
echo "         $RESULTS/numa_pages_${STAMP}.csv"
df -h /tmp | tail -n 1
