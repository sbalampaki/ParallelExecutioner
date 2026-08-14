#!/bin/bash
#SBATCH --job-name=e4_partsweep
#SBATCH --partition=research-cpu
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=2
#SBATCH --exclusive
#SBATCH --exclude=c1,c2,c3,c4,c5
#SBATCH --time=2:00:00
#SBATCH --output=/projects/sb2ea/logs/e4ps_%j.out
#SBATCH --error=/projects/sb2ea/logs/e4ps_%j.err


set -euo pipefail
export USER=sb2ea

BIN=/projects/sb2ea/bin
PROJ=/projects/sb2ea
SRC=$PROJ/csr
LOOKUP=$PROJ/manifest/kernel_lookup.csv
OUT=$PROJ/results/timing
GIT_SHA="${GIT_SHA:-unknown}"
TMP=/tmp/sb2ea_e4ps
TIM=$OUT/e4ps_${SLURM_JOB_ID}.csv
THR=$OUT/e4ps_${SLURM_JOB_ID}_threads.csv
mkdir -p "$OUT"

MPI_PREFIX=/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.6
HWLOC_LIB=/opt/ohpc/pub/libs/hwloc/lib
MPIRUN="$MPI_PREFIX/bin/mpirun"
export LD_LIBRARY_PATH="$MPI_PREFIX/lib:$HWLOC_LIB:${LD_LIBRARY_PATH:-}"
[ -x "$MPIRUN" ] || { echo "FATAL: $MPIRUN missing" >&2; exit 1; }

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NN=${#NODES[@]}
echo "=== job $SLURM_JOB_ID  git $GIT_SHA  $(date -Is)"
echo "=== $NN nodes: ${NODES[*]}   p = 16 x N"
[ "$NN" -ge 2 ] || { echo "FATAL: need >=2 nodes" >&2; exit 1; }

hostlist()  { local IFS=,; echo "${NODES[*]:0:$1}"; }
hostslots() { local i o=""; for ((i=0;i<$1;i++)); do o+="${NODES[$i]}:2,"; done; echo "${o%,}"; }

declare -A SEEN
S0=(); S1=()
while IFS=, read -r cpu core sock; do
    [ -z "${cpu:-}" ] && continue
    key="$sock:$core"; [ -n "${SEEN[$key]:-}" ] && continue; SEEN[$key]=1
    if [ "$sock" = "0" ]; then S0+=("$cpu"); else S1+=("$cpu"); fi
done < <(lscpu -p=CPU,CORE,SOCKET | grep -v '^#')
ALT=(); for ((i=0;i<${#S0[@]};i++)); do ALT+=("${S0[$i]}" "${S1[$i]}"); done
places(){ printf '{%s},' "$@" | sed 's/,$//'; }
csv(){ local IFS=,; echo "$*"; }
ALT_PL=$(places "${ALT[@]}")
rankcpus_for() { local i o=""; for ((i=0;i<$1;i++)); do
    o+="$(csv "${S0[@]}");$(csv "${S1[@]}");"; done; echo "${o%;}"; }
echo "cores: socket0=${#S0[@]} socket1=${#S1[@]}"

stage_all() {   # $1 variant  $2 n_nodes
    local V=$1 N=$2
    srun -N "$N" --ntasks-per-node=1 --nodelist="$(hostlist "$N")" bash -c "
        set -e
        mkdir -p $TMP
        for r in 0 1; do
            g=\$(( 2 * \${SLURM_PROCID:-0} + r ))
            d=$TMP/${V}_rank\$g
            cpu=\$([ \$r = 0 ] && echo ${S0[0]} || echo ${S1[0]})
            OMP_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=\"{\$cpu}\" \
                $BIN/stage_csr --src $SRC/$V --dst \$d --mode interleave 2>/dev/null
            [ -s \$d/offsets.i64 ] || { echo \"FATAL: \$(hostname) \$d\" >&2; exit 1; }
        done" || return 1
    echo "    staged $V on $N nodes"
}

run() {   # $1 partitioner  $2 n_nodes
    local P=$1 N=$2 NR=$((2 * $2)) p=$((16 * $2))
    printf '  p=%-3d %-11s ' "$p" "$P"
    env -u OMP_PLACES OMP_PROC_BIND=false OMP_NUM_THREADS=8 \
      "$MPIRUN" -np "$NR" --map-by ppr:2:node --bind-to none \
        --host "$(hostslots "$N")" -x LD_LIBRARY_PATH \
        -x OMP_PROC_BIND -x OMP_NUM_THREADS \
      "$BIN/kernel_mpi" --csr "$TMP/Wfull_" --csr-per-rank rank \
        --lookup "$LOOKUP" --variant Wfull --window full \
        --threads 8 --rank-cpus "$(rankcpus_for "$N")" \
        --partitioner "$P" --tile 0 --repeat 64 --reps 6 --warmup-s 2 \
        --platform broadwell --nodes "$N" \
        --reference "$TMP/oracle.f64" \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" \
        --note "exp=partsweep;partitioner=$P;p=$p;nodes=$N" \
        > "$TMP/run.log" 2>&1 || true
    if grep -q '^rep [1-9]' "$TMP/run.log"; then
        awk '/^rep [1-9]/ {s+=$9; g+=$16; w+=$13; t+=$15; n++}
             END {printf "%9.1f M rec/s  %7.1f M rec/GHz  work_imb %.4f  time_imb %.4f\n",
                  s/n, (s/n)/(g/n), w/n, t/n}' "$TMP/run.log"
    else
        echo "FAILED"
        grep -v mca_base "$TMP/run.log" | head -10 | sed 's/^/      | /'
    fi
}

PARTS=(block nzbalanced contig_opt wbalanced greedy cyclic)

echo; echo "########## p = $((16*NN)) (N=$NN) ##########"
stage_all Wfull "$NN"
srun -N1 --ntasks=1 --nodelist="${NODES[0]}" \
    "$BIN/kernel_serial" --csr "$TMP/Wfull_rank0" --lookup "$LOOKUP" \
    --out "$TMP/oracle.f64" >/dev/null 2>&1
srun -N "$NN" --ntasks-per-node=1 bash -c "cp $TMP/oracle.f64 $TMP/ 2>/dev/null || true" || true
echo "  oracle generated"
echo "  E0 predicts at p=80: block 1.1294  nzbalanced 1.0124"
echo "                       contig_opt 1.0016  greedy 1.0001"
echo
for P in "${PARTS[@]}"; do run "$P" "$NN"; done
echo "  --- reverse pass ---"
for ((i=${#PARTS[@]}-1; i>=0; i--)); do run "${PARTS[$i]}" "$NN"; done

echo; echo "########## p = 16 (N=1), THE CONTROL ##########"
echo "  E0 predicts at p=16: block 1.0494  nzbalanced 1.0006"
echo "                       contig_opt 1.0003  greedy 1.0000"
echo "  A partitioner effect the same size here as at p=80 is not the tail."
echo
for P in "${PARTS[@]}"; do run "$P" 1; done
echo "  --- reverse pass ---"
for ((i=${#PARTS[@]}-1; i>=0; i--)); do run "${PARTS[$i]}" 1; done

srun -N "$NN" --ntasks-per-node=1 bash -c "rm -rf $TMP" || true

echo
echo "timings: $TIM"
echo "threads: $THR"
cat <<'EOT'

REDUCE WITH:
  python3 code/probes/reduce_partsweep.py --timings <TIM> \
      --predictions /projects/sb2ea/results/day6/cv_work_partitions_extended.csv

EOT
