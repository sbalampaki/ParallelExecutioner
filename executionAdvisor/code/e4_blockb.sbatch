#!/bin/bash
#SBATCH --job-name=e4_blockb
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH --nodelist=c6
#SBATCH --time=1:30:00
#SBATCH --output=/projects/sb2ea/logs/e4b_%j.out
#SBATCH --error=/projects/sb2ea/logs/e4b_%j.err


set -euo pipefail
export USER=sb2ea

BIN=/projects/sb2ea/bin
PROJ=/projects/sb2ea
SRC=$PROJ/csr/W24
LOOKUP=$PROJ/manifest/kernel_lookup.csv
OUT=$PROJ/results/timing
NODE=$(hostname -s)
GIT_SHA="${GIT_SHA:-unknown}"
TMP=/tmp/sb2ea_e4b
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP" "$OUT"
TIM=$OUT/e4b_${NODE}_${SLURM_JOB_ID}.csv
THR=$OUT/e4b_${NODE}_${SLURM_JOB_ID}_threads.csv

MPI_PREFIX=/opt/ohpc/pub/mpi/openmpi4-gnu12/4.1.6
HWLOC_LIB=/opt/ohpc/pub/libs/hwloc/lib
MPIRUN="$MPI_PREFIX/bin/mpirun"
# DT_RUNPATH is not transitive (trap #3): kernel_mpi finds libmpi, libmpi
# does not find libhwloc. Export it AND forward it with -x.
export LD_LIBRARY_PATH="$MPI_PREFIX/lib:$HWLOC_LIB:${LD_LIBRARY_PATH:-}"
[ -x "$MPIRUN" ] || { echo "FATAL: $MPIRUN missing" >&2; exit 1; }

for b in "$BIN/kernel_omp" "$BIN/kernel_mpi" "$BIN/kernel_serial" \
         "$BIN/stage_csr" "$BIN/numa_verify"; do
    [ -x "$b" ] || { echo "FATAL: $b missing" >&2; exit 1; }
done

echo "=== node $NODE  job $SLURM_JOB_ID  git $GIT_SHA  $(date -Is)"

declare -A SEEN
S0=(); S1=()
while IFS=, read -r cpu core sock; do
    [ -z "${cpu:-}" ] && continue
    key="$sock:$core"; [ -n "${SEEN[$key]:-}" ] && continue; SEEN[$key]=1
    if [ "$sock" = "0" ]; then S0+=("$cpu"); else S1+=("$cpu"); fi
done < <(lscpu -p=CPU,CORE,SOCKET | grep -v '^#')

ALL=("${S0[@]}" "${S1[@]}")                      # grouped order (D44)
ALT=(); for ((i=0;i<${#S0[@]};i++)); do ALT+=("${S0[$i]}" "${S1[$i]}"); done

places(){ printf '{%s},' "$@" | sed 's/,$//'; }
csv(){ local IFS=,; echo "$*"; }
GRP_PL=$(places "${ALL[@]}"); GRP_CSV=$(csv "${ALL[@]}")
ALT_PL=$(places "${ALT[@]}")
S0_CSV=$(csv "${S0[@]}"); S1_CSV=$(csv "${S1[@]}")
echo "cores: socket0=${#S0[@]} socket1=${#S1[@]}"

echo; echo "########## STAGING SHARED TREES ##########"
# grouped places -> 8 blocks node0, 8 blocks node1, 32 MiB period.
# This is what job 73063's whole Block B ran on.
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$GRP_PL" \
    "$BIN/stage_csr" --src "$SRC" --dst "$TMP/shared_grp" --mode interleave
# alternating places -> true 2 MiB interleave, which is what stage_csr.c's
# own header specifies and what nothing has ever actually been run on.
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$ALT_PL" \
    "$BIN/stage_csr" --src "$SRC" --dst "$TMP/shared_alt" --mode interleave

for P in shared_grp shared_alt; do
    printf '  %-11s T=16 grouped page_local_frac: ' "$P"
    "$BIN/numa_verify" --csr "$TMP/$P" --threads 16 --cpus "$GRP_CSV" \
        | awk '/value.f32/{v=1} v && /page_local_frac/{print $4; exit}'
done
echo "  expected: shared_grp 0.5494   shared_alt ~0.5005"

"$BIN/kernel_serial" --csr "$TMP/shared_alt" --lookup "$LOOKUP" \
    --out "$TMP/oracle.f64" >/dev/null 2>&1
echo "  oracle generated"


stage_per_rank() {   # $1 = n_ranks
    local nr=$1 r cpu d
    for ((r=0; r<nr; r++)); do
        if (( r < (nr + 1) / 2 )); then cpu=${S0[0]}; else cpu=${S1[0]}; fi
        d="$TMP/pr${nr}_rank${r}"
        OMP_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES="{$cpu}" \
            "$BIN/stage_csr" --src "$SRC" --dst "$d" --mode interleave 2>/dev/null
       
        [ -s "$d/offsets.i64" ] || {
            echo "FATAL: no $d/offsets.i64 -- per-rank staging failed" >&2
            exit 1; }
    done
    printf '    staged %2d per-rank copies, all verified present\n' "$nr"
}

ranks_for() {   # $1 = n_ranks -> "cpus;cpus;..." at 16/n cpus per rank
    local nr=$1 per=$((16 / $1)) o="" r i idx
    for ((r=0; r<nr; r++)); do
        local g=""
        for ((i=0; i<per; i++)); do idx=$((r * per + i)); g+="${ALL[$idx]},"; done
        o+="${g%,};"
    done
    echo "${o%;}"
}

omp_run() {   # $1 label  $2 threads  $3 places  $4 tree
    printf '  %-34s ' "$1"
    OMP_PROC_BIND=close OMP_PLACES="$3" \
    "$BIN/kernel_omp" --csr "$TMP/$4" --lookup "$LOOKUP" --variant W24 \
        --window 24 --threads "$2" --schedule precomputed --partitioner block \
        --tile 0 --repeat $((4 * $2)) --reps 6 --warmup-s 2 \
        --platform broadwell --reference "$TMP/oracle.f64" \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" > "$TMP/run.log" 2>&1 \
        || { echo "FAILED"; tail -8 "$TMP/run.log" | sed 's/^/      | /'; return 0; }
    awk '/^rep [1-9]/ {s+=$6; g+=$13; n++}
         END {if (n) printf "%8.1f M rec/s  %.2f GHz  %7.1f M rec/GHz\n",
              s/n, g/n, (s/n)/(g/n); else print "NO REPS"}' "$TMP/run.log"
}

mpi_run() {   
    local lab=$1 nr=$2 pre=$3; shift 3
    local nt=$((16 / nr)) log="$TMP/run.log"
    printf '  %-34s ' "$lab"
    env -u OMP_PLACES OMP_PROC_BIND=false OMP_NUM_THREADS="$nt" \
      "$MPIRUN" -np "$nr" --bind-to none -x LD_LIBRARY_PATH \
        -x OMP_PROC_BIND -x OMP_NUM_THREADS \
      "$BIN/kernel_mpi" --csr "$pre" --lookup "$LOOKUP" --variant W24 \
        --window 24 --threads "$nt" --rank-cpus "$(ranks_for "$nr")" \
        --partitioner block --tile 0 --repeat 64 --reps 6 --warmup-s 2 \
        --platform broadwell --nodes 1 "$@" \
        --reference "$TMP/oracle.f64" \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" > "$log" 2>&1 || true
    if grep -q '^rep [1-9]' "$log"; then
        awk '/^rep [1-9]/ {s+=$9; g+=$16; n++}
             END {printf "%8.1f M rec/s  %.2f GHz  %7.1f M rec/GHz\n",
                  s/n, g/n, (s/n)/(g/n)}' "$log"
    else
        echo "FAILED"
        sed 's/^/      | /' "$log" | grep -v 'mca_base' | head -12
    fi
}

# --- anchors 
echo; echo "########## OPENMP ANCHORS (T=16 grouped) ##########"
omp_run "openmp T=16  shared_grp" 16 "$GRP_PL" shared_grp
omp_run "openmp T=16  shared_alt" 16 "$GRP_PL" shared_alt
echo "  job 73918 medians: shared_grp 688.6, shared_alt 741.2 M rec/GHz."
echo "  These two runs say whether this node is in the same state today."


echo; echo "########## P12a: 2x8, THREE ARMS, PALINDROME ##########"
stage_per_rank 2
printf '  per-rank copy 0 (expect 1.0000): '
"$BIN/numa_verify" --csr "$TMP/pr2_rank0" --threads 8 --cpus "$S0_CSV" \
    | awk '/value.f32/{v=1} v && /page_local_frac/{print $4; exit}'
printf '  per-rank copy 1 (expect 1.0000): '
"$BIN/numa_verify" --csr "$TMP/pr2_rank1" --threads 8 --cpus "$S1_CSV" \
    | awk '/value.f32/{v=1} v && /page_local_frac/{print $4; exit}'

for arm in grp alt prk prk alt grp; do
    case $arm in
      grp) mpi_run "2x8 shared_grp" 2 "$TMP/shared_grp" ;;
      alt) mpi_run "2x8 shared_alt" 2 "$TMP/shared_alt" ;;
      prk) mpi_run "2x8 PER-RANK"   2 "$TMP/pr2_" --csr-per-rank rank ;;
    esac
done
rm -rf "$TMP"/pr2_rank*

echo; echo "########## P12b: RATIO SWEEP ON BOTH TREES ##########"
printf '  %-34s %s\n' "" "(16 cores fixed throughout)"
for NR in 1 2 4 8 16; do
    L=$(printf '%dx%d' "$NR" $((16 / NR)))
    mpi_run "$L shared_grp" "$NR" "$TMP/shared_grp"
    mpi_run "$L shared_alt" "$NR" "$TMP/shared_alt"
    stage_per_rank "$NR"
    mpi_run "$L PER-RANK"   "$NR" "$TMP/pr${NR}_" --csr-per-rank rank
    rm -rf "$TMP"/pr${NR}_rank*      # 528 MB a copy; reclaim before the next
    echo
done

echo "timings: $TIM"
echo "threads: $THR"
cat <<'EOT'

EOT
