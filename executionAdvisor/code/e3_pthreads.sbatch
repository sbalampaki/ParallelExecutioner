#!/bin/bash
#SBATCH --job-name=e3_pth
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH --time=1:15:00
#SBATCH --output=/projects/sb2ea/logs/e3pth_%j.out
#SBATCH --error=/projects/sb2ea/logs/e3pth_%j.err


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
TMP=/tmp/sb2ea_pth
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP" "$OUT"
TIM=$OUT/pth_${NODE}_${SLURM_JOB_ID}.csv
THR=$OUT/pth_${NODE}_${SLURM_JOB_ID}_threads.csv

echo "=== node $NODE  job $SLURM_JOB_ID  git $GIT_SHA  $(date -Is)"

declare -A SEEN
S0=(); S1=(); SMT=()
while IFS=, read -r cpu core sock; do
    [ -z "${cpu:-}" ] && continue
    key="$sock:$core"
    if [ -n "${SEEN[$key]:-}" ]; then SMT+=("$cpu"); continue; fi
    SEEN[$key]=1
    if [ "$sock" = "0" ]; then S0+=("$cpu"); else S1+=("$cpu"); fi
done < <(lscpu -p=CPU,CORE,SOCKET | grep -v '^#')
NC0=${#S0[@]}; NC1=${#S1[@]}; NSMT=${#SMT[@]}
echo "cores: socket0=$NC0 socket1=$NC1  smt siblings=$NSMT"
echo

cpus_of() {   # $1 = socket0|spread|all32   $2 = n
    local kind=$1 n=$2 o="" i h
    case "$kind" in
      socket0) (( n > NC0 )) && return 1
               for ((i=0;i<n;i++)); do o+="${S0[$i]},"; done ;;
      spread)  (( (n+1)/2 > NC0 || n/2 > NC1 )) && return 1
               for ((i=0;i<n;i++)); do h=$((i/2))
                   if (( i % 2 == 0 )); then o+="${S0[$h]},"; else o+="${S1[$h]},"; fi
               done ;;
      all32)   (( n > NC0 + NC1 + NSMT )) && return 1
               local all=("${S0[@]}" "${S1[@]}" "${SMT[@]}")
               for ((i=0;i<n;i++)); do o+="${all[$i]},"; done ;;
    esac
    echo "${o%,}"
}
places_of() {   # OMP_PLACES form of the same list
    local l; l=$(cpus_of "$1" "$2") || return 1
    echo "{${l//,/\},\{}}"
}

echo "########## STAGING ##########"
OMP_NUM_THREADS=$((NC0+NC1)) OMP_PROC_BIND=close \
OMP_PLACES="$(places_of spread $((NC0+NC1)))" \
    "$BIN/stage_csr" --src "$PROJ/csr/W24" --dst "$TMP/W24" --mode interleave
"$BIN/kernel_serial" --csr "$TMP/W24" --lookup "$LOOKUP" \
    --out "$TMP/oracle.f64" >/dev/null 2>&1
echo "oracle generated"
echo

pth() {   # $1 label $2 T $3 placement $4 row_order $5 chunk|nosteal
    local CP; CP=$(cpus_of "$3" "$2") || { printf '  %-30s skipped\n' "$1"; return 0; }
    local FLAG="--chunk $5"
    [ "$5" = "nosteal" ] && FLAG="--no-steal"
    printf '  %-30s ' "$1"
    "$BIN/kernel_pthreads" --csr "$TMP/W24" --lookup "$LOOKUP" \
        --variant W24 --window 24 --threads "$2" --cpus "$CP" \
        --partitioner block --row-order "$4" --tile 0 $FLAG \
        --repeat $((4 * $2)) --reps 6 --warmup-s 2 \
        --platform broadwell --reference "$TMP/oracle.f64" \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" 2>&1 \
      | awk '/^rep [1-9]/ {s+=$6; w+=$8; t+=$10; n++}
             END {printf "%8.1f M rec/s   work_imb %.4f   time_imb %.4f\n",
                  s/n, w/n, t/n}'
}

echo "########## A1. stealing vs none, CANONICAL (expect a null) ##########"
for P in socket0 spread; do
    T=8; [ "$P" = spread ] && T=16
    pth "$P T=$T no-steal"      "$T" "$P" canonical nosteal
    for CH in 1 16 64 256 1024; do
        pth "$P T=$T steal chunk=$CH" "$T" "$P" canonical "$CH"
    done
done
echo

echo "########## A2. stealing vs none, SORTED_DESC (the real test) #########"
echo "  D40 predicts work_imbalance falls sharply and throughput does not,"
echo "  because the sorted_desc penalty is locality rather than imbalance."
for P in socket0 spread; do
    T=8; [ "$P" = spread ] && T=16
    pth "$P T=$T no-steal"      "$T" "$P" sorted_desc nosteal
    for CH in 16 64 256; do
        pth "$P T=$T steal chunk=$CH" "$T" "$P" sorted_desc "$CH"
    done
done
echo

echo "########## B. threads = 32 (SMT across both sockets) ##########"
for T in 16 24 32; do
    PL=$(places_of all32 "$T") || { echo "  T=$T skipped"; continue; }
    printf '  omp  T=%-3s ' "$T"
    OMP_PROC_BIND=close OMP_PLACES="$PL" \
    "$BIN/kernel_omp" --csr "$TMP/W24" --lookup "$LOOKUP" \
        --variant W24 --window 24 --threads "$T" --schedule precomputed \
        --partitioner block --tile 0 --repeat $((4 * T)) --reps 6 \
        --warmup-s 2 --smt --platform broadwell \
        --reference "$TMP/oracle.f64" \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" 2>&1 \
      | awk '/^rep [1-9]/ {s+=$6; g+=$13; n++}
             END {printf "%8.1f M rec/s   %.2f GHz\n", s/n, g/n}'
done
echo

echo "########## SUMMARY ##########"
python - "$TIM" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); d = d[d.warmup == 0]
d["steals"] = d.notes.str.extract(r"steals=(\d+)").astype(float)
d["work"] = d.n_records * d.k_iterations
g = (d.groupby(["paradigm","thread_placement","n_threads","row_order",
                "schedule","chunk_size"])
       .agg(mrec=("throughput_rec_s", lambda s: s.median()/1e6),
            ghz=("achieved_ghz","median"),
            work_imb=("work_imbalance","median"),
            time_imb=("time_imbalance","median"),
            steals=("steals","median"),
            tcm=("l3_tcm","median"), work=("work","median"),
            bit=("bitidentical", lambda s: (s=="pass").all())).reset_index())
g["L3miss_per_rec"] = g.tcm / g.work
print(g.round(4).to_string(index=False))
print()
print("  A: on canonical, stealing should be a null or a small loss.")
print("  A2: on sorted_desc, work_imbalance should FALL sharply while")
print("      throughput barely moves -- D40 confirmed by an instrument that")
print("      manipulates imbalance directly. If throughput jumps instead,")
print("      D40 is wrong and imbalance mattered after all.")
print("  bit: must be True everywhere. The stealing schedule is")
print("      nondeterministic run to run, so byte-identical output is the")
print("      strongest form of D27's claim available.")
EOF
echo
echo "timings: $TIM"
