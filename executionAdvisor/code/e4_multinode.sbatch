#!/bin/bash
#SBATCH --job-name=e4_multinode
#SBATCH --partition=research-cpu
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=2
#SBATCH --exclusive
#SBATCH --exclude=c1,c2,c3,c4,c5
#SBATCH --time=3:00:00
#SBATCH --output=/projects/sb2ea/logs/e4mn_%j.out
#SBATCH --error=/projects/sb2ea/logs/e4mn_%j.err

set -euo pipefail
export USER=sb2ea

BIN=/projects/sb2ea/bin
PROJ=/projects/sb2ea
SRC=$PROJ/csr
LOOKUP=$PROJ/manifest/kernel_lookup.csv
OUT=$PROJ/results/timing
GIT_SHA="${GIT_SHA:-unknown}"
TMP=/tmp/sb2ea_e4mn
TIM=$OUT/e4mn_${SLURM_JOB_ID}.csv
THR=$OUT/e4mn_${SLURM_JOB_ID}_threads.csv
mkdir -p "$OUT"

MPI_PREFIX=/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.6
HWLOC_LIB=/opt/ohpc/pub/libs/hwloc/lib
MPIRUN="$MPI_PREFIX/bin/mpirun"
export LD_LIBRARY_PATH="$MPI_PREFIX/lib:$HWLOC_LIB:${LD_LIBRARY_PATH:-}"
[ -x "$MPIRUN" ] || { echo "FATAL: $MPIRUN missing" >&2; exit 1; }

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NN=${#NODES[@]}
echo "=== job $SLURM_JOB_ID  git $GIT_SHA  $(date -Is)"
echo "=== allocated $NN nodes: ${NODES[*]}"
[ "$NN" -ge 4 ] || { echo "FATAL: need at least 4 nodes, got $NN" >&2; exit 1; }

hostlist() { local IFS=,; echo "${NODES[*]:0:$1}"; }
# OpenMPI treats a bare hostname in --host as ONE slot, so `--host c6` with
# -np 2 fails with "not enough slots". The allocation gives 2 tasks/node
# (--ntasks-per-node=2), so the slot count has to be stated explicitly.
hostslots() { local i o=""; for ((i=0;i<$1;i++)); do o+="${NODES[$i]}:2,"; done; echo "${o%,}"; }

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
ALT_PL=$(places "${ALT[@]}")

rankcpus_for() {   # $1 = n_nodes
    local i o=""
    for ((i=0; i<$1; i++)); do o+="$(csv "${S0[@]}");$(csv "${S1[@]}");"; done
    echo "${o%;}"
}
echo "cores: socket0=${#S0[@]} socket1=${#S1[@]}"
echo "rank-cpus at N=1: $(rankcpus_for 1)"


echo; echo "########## P21: NFS AGGREGATE READ BANDWIDTH ##########"
echo "  cold reads of Wfull/itemid.i32 from /projects, N nodes at once"
printf '  %-6s %-12s %-12s %s\n' "N" "wall_s" "agg_MB/s" "per_node_MB/s"
SZ=0
for f in itemid.i32 t.f32 value.f32; do
    SZ=$((SZ + $(stat -c %s "$SRC/Wfull/$f")))
done
printf '  reading %.2f GB per node with O_DIRECT (bypasses page cache)\n' \
    "$(awk -v s=$SZ 'BEGIN{print s/1e9}')"
for N in 1 2 4 "$NN"; do
    [ "$N" -gt "$NN" ] && continue
    T0=$(date +%s.%N)
    srun -N "$N" --ntasks-per-node=1 --nodelist="$(hostlist "$N")" bash -c "
        for f in itemid.i32 t.f32 value.f32; do
            dd if=$SRC/Wfull/\$f of=/dev/null bs=8M iflag=direct
        done" >/dev/null 2>&1
    T1=$(date +%s.%N)
    awk -v t0="$T0" -v t1="$T1" -v n="$N" -v sz="$SZ" 'BEGIN{
        d=t1-t0; agg=n*sz/d/1e6;
        printf "  %-6d %-12.3f %-12.1f %.1f\n", n, d, agg, agg/n}'
done
echo "  10 GbE point-to-point is 1153 MB/s (platform.md). Where the"
echo "  aggregate flattens is the shared-storage ceiling, and it is a"
echo "  result in its own right, not a nuisance."


stage_all() {   # $1 = variant  $2 = n_nodes
    local V=$1 N=$2
    srun -N "$N" --ntasks-per-node=1 --nodelist="$(hostlist "$N")" bash -c "
        set -e
        mkdir -p $TMP
        # kernel_mpi builds the path as prefix + suffix + GLOBAL rank, so on
        # node i the two copies must be named rank(2i) and rank(2i+1), not
        # rank0/rank1. SLURM_PROCID is the node index here (1 task/node).
        for r in 0 1; do
            g=\$(( 2 * \${SLURM_PROCID:-0} + r ))
            d=$TMP/${V}_rank\$g
            cpu=\$([ \$r = 0 ] && echo ${S0[0]} || echo ${S1[0]})
            OMP_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=\"{\$cpu}\" \
                $BIN/stage_csr --src $SRC/$V --dst \$d --mode interleave 2>/dev/null
            [ -s \$d/offsets.i64 ] || { echo \"FATAL: \$(hostname) \$d\" >&2; exit 1; }
            echo \"      \$(hostname) staged \$(basename \$d)\"
        done" || return 1
    echo "    staged $V on $N nodes, 2 node-local copies each, verified"
}

run_mpi() {   # $1 label  $2 variant  $3 n_nodes  $4 note
    local V=$2 N=$3 NR=$((2 * $3))
    printf '  %-40s ' "$1"
    env -u OMP_PLACES OMP_PROC_BIND=false OMP_NUM_THREADS=8 \
      "$MPIRUN" -np "$NR" --map-by ppr:2:node --bind-to none \
        --host "$(hostslots "$N")" -x LD_LIBRARY_PATH \
        -x OMP_PROC_BIND -x OMP_NUM_THREADS \
      "$BIN/kernel_mpi" --csr "$TMP/${V}_" --csr-per-rank rank \
        --lookup "$LOOKUP" --variant "$V" --window "${V#W}" \
        --threads 8 --rank-cpus "$(rankcpus_for "$N")" \
        --partitioner block --tile 0 --repeat 64 --reps 6 --warmup-s 2 \
        --platform broadwell --nodes "$N" \
        --reference "$TMP/oracle_$V.f64" \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" \
        --note "$4" > "$TMP/run.log" 2>&1 || true
    if grep -q '^rep [1-9]' "$TMP/run.log"; then
        awk '/^rep [1-9]/ {s+=$9; g+=$16; n++}
             END {printf "%9.1f M rec/s  %.2f GHz  %8.1f M rec/GHz\n",
                  s/n, g/n, (s/n)/(g/n)}' "$TMP/run.log"
    else
        echo "FAILED"; grep -v mca_base "$TMP/run.log" | head -12 | sed 's/^/      | /'
    fi
}


echo; echo "########## P19: STRONG SCALING, Wfull, 2x8 per node ##########"
stage_all Wfull "$NN"
srun -N1 --ntasks=1 --nodelist="${NODES[0]}" \
    "$BIN/kernel_serial" --csr "$TMP/Wfull_rank0" --lookup "$LOOKUP" \
    --out "$TMP/oracle_Wfull.f64" >/dev/null 2>&1
srun -N "$NN" --ntasks-per-node=1 bash -c \
    "cp $TMP/oracle_Wfull.f64 $TMP/ 2>/dev/null || true" || true
echo "  oracle generated and distributed"
echo
for N in 1 2 4 "$NN"; do
    [ "$N" -gt "$NN" ] && continue
    run_mpi "strong  Wfull  N=$N  (${N}x2x8)" Wfull "$N" \
            "exp=strong;variant=Wfull;nodes=$N;ranks_per_node=2"
done
echo "  Both timed regions are written. Karp-Flatt uses end_to_end."


echo; echo "########## P20: WEAK SCALING ##########"
for pair in "1 W24" "2 W48" "4 Wfull"; do
    set -- $pair; N=$1; V=$2
    [ -d "$SRC/$V" ] || { echo "  $V absent, skipping N=$N"; continue; }
    [ "$N" -gt "$NN" ] && continue
    stage_all "$V" "$N" || continue
    srun -N1 --ntasks=1 --nodelist="${NODES[0]}" \
        "$BIN/kernel_serial" --csr "$TMP/${V}_rank0" --lookup "$LOOKUP" \
        --out "$TMP/oracle_$V.f64" >/dev/null 2>&1
    srun -N "$N" --ntasks-per-node=1 --nodelist="$(hostlist "$N")" bash -c \
        "cp $TMP/oracle_$V.f64 $TMP/ 2>/dev/null || true" || true
    run_mpi "weak    $V  N=$N" "$V" "$N" \
            "exp=weak;variant=$V;nodes=$N;ranks_per_node=2"
done
echo "  The variant ladder is approximate; report as indicative."

srun -N "$NN" --ntasks-per-node=1 bash -c "rm -rf $TMP" || true

echo
echo "timings: $TIM"
echo "threads: $THR"
cat <<'EOT'

REDUCE WITH:
  python3 code/probes/reduce_multinode.py --timings <TIM>

  Cell identity comes from the `notes` column now, not launch order:
  exp=strong|weak, variant, nodes, ranks_per_node.


EOT
