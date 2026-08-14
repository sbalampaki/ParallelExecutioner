#!/bin/bash
#SBATCH --job-name=e3_order
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH --time=0:50:00
#SBATCH --output=/projects/sb2ea/logs/e3ord_%j.out
#SBATCH --error=/projects/sb2ea/logs/e3ord_%j.err


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
TMP=/tmp/sb2ea_ord
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP" "$OUT"
TIM=$OUT/ord_${NODE}_${SLURM_JOB_ID}.csv
THR=$OUT/ord_${NODE}_${SLURM_JOB_ID}_threads.csv

echo "=== node $NODE  job $SLURM_JOB_ID  git $GIT_SHA  $(date -Is)"

declare -A SEEN
S0=(); S1=()
while IFS=, read -r cpu core sock; do
    [ -z "${cpu:-}" ] && continue
    key="$sock:$core"; [ -n "${SEEN[$key]:-}" ] && continue; SEEN[$key]=1
    if [ "$sock" = "0" ]; then S0+=("$cpu"); else S1+=("$cpu"); fi
done < <(lscpu -p=CPU,CORE,SOCKET | grep -v '^#')
NC0=${#S0[@]}; NC1=${#S1[@]}
echo "physical cores: socket0=$NC0 socket1=$NC1"
echo

places() {   # $1 = alt|grp|s0   $2 = n threads
    local kind=$1 n=$2 o="" i h
    case "$kind" in
      alt) (( (n+1)/2 > NC0 || n/2 > NC1 )) && return 1
           for ((i=0;i<n;i++)); do h=$((i/2))
               if (( i % 2 == 0 )); then o+="{${S0[$h]}},"; else o+="{${S1[$h]}},"; fi
           done ;;
      grp) (( (n+1)/2 > NC0 || n/2 > NC1 )) && return 1
           for ((i=0;i<(n+1)/2;i++)); do o+="{${S0[$i]}},"; done
           for ((i=0;i<n/2;i++));     do o+="{${S1[$i]}},"; done ;;
      s0)  (( n > NC0 )) && return 1
           for ((i=0;i<n;i++)); do o+="{${S0[$i]}},"; done ;;
    esac
    echo "${o%,}"
}

echo "########## STAGING: two page placements ##########"
OMP_NUM_THREADS=$((NC0+NC1)) OMP_PROC_BIND=close OMP_PLACES="$(places alt $((NC0+NC1)))" \
    "$BIN/stage_csr" --src "$PROJ/csr/W24" --dst "$TMP/intl" --mode interleave
OMP_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES="{${S0[0]}}" \
    "$BIN/stage_csr" --src "$PROJ/csr/W24" --dst "$TMP/n0" --mode interleave
for P in intl n0; do
    printf '  %-5s ' "$P"
    "$BIN/numa_pages" "$TMP/$P/value.f32" | tail -n +2 \
        | awk -F, '{printf "%s=%.4f  ", $4, $7} END {print ""}'
done
"$BIN/kernel_serial" --csr "$TMP/intl" --lookup "$LOOKUP" \
    --out "$TMP/oracle.f64" >/dev/null 2>&1
echo "  oracle generated"
echo

run() {   # $1 label  $2 T  $3 ordering  $4 csr
    local PL; PL=$(places "$3" "$2") || { printf '  %-28s skipped\n' "$1"; return 0; }
    printf '  %-28s ' "$1"
    OMP_PROC_BIND=close OMP_PLACES="$PL" \
    "$BIN/kernel_omp" --csr "$TMP/$4" --lookup "$LOOKUP" \
        --variant W24 --window 24 --threads "$2" --schedule precomputed \
        --partitioner block --tile 0 --repeat $((4 * $2)) --reps 6 \
        --warmup-s 2 --platform broadwell --stage-mode interleave \
        --reference "$TMP/oracle.f64" \
        --timings "$TIM" --threads-csv "$THR" \
        --job-id "$SLURM_JOB_ID" --git-sha "$GIT_SHA" 2>&1 \
      | awk '/^rep [1-9]/ {s+=$6; g+=$13; n++}
             END {printf "%8.1f M rec/s  %.2f GHz  %7.1f M rec/GHz\n",
                  s/n, g/n, (s/n)/(g/n)}'
}

for CSR in intl n0; do
    echo "########## CSR pages: $CSR ##########"
    for T in 2 4 8 12 16; do
        run "T=$T alternating" "$T" alt "$CSR"
        run "T=$T grouped"     "$T" grp "$CSR"
    done
    run "T=8 socket0 (control)" 8 s0 "$CSR"
    echo
done

echo "########## SUMMARY ##########"
python - "$TIM" "$THR" <<'EOF'
import sys, pandas as pd, numpy as np
r = pd.read_csv(sys.argv[1]); r = r[r.warmup == 0]
t = pd.read_csv(sys.argv[2])

# The schema has no column for thread ordering, but it is recoverable: count
# socket transitions across tid order. Alternating gives T-1 transitions,
# socket-grouped gives 1, single-socket gives 0.
def order_of(g):
    s = g.sort_values("tid").socket.values
    if len(np.unique(s)) < 2:
        return "socket0"
    return "grouped" if (s[1:] != s[:-1]).sum() <= 1 else "alternating"
ordr = t.groupby("run_id").apply(order_of, include_groups=False).rename("ordering")
r = r.merge(ordr, on="run_id", how="left")

r["csr"] = np.where(r.notes.str.contains("n0"), "n0", "intl")   # fallback
r["work"] = r.n_records * r.k_iterations
r["rec_per_ghz"] = r.throughput_rec_s / (r.achieved_ghz * 1e9) * 1e9 / 1e6

g = (r.groupby(["ordering", "n_threads"])
       .agg(mrec=("throughput_rec_s", lambda s: s.median()/1e6),
            ghz=("achieved_ghz", "median"),
            rec_per_ghz=("rec_per_ghz", "median"),
            l3=("l3_tcm", "median"), work=("work", "median"),
            ipc=("tot_ins", "median"),
            cyc=("tot_cyc", "median"),
            timb=("time_imbalance", "median"),
            bit=("bitidentical", lambda s: (s == "pass").all())).reset_index())
g["L3miss_per_rec"] = g.l3 / g.work
g["IPC"] = g.ipc / g.cyc
cols = ["ordering","n_threads","mrec","ghz","rec_per_ghz","IPC",
        "L3miss_per_rec","timb","bit"]
print(g[cols].round(4).to_string(index=False))

print()
print("  ORDERING EFFECT, per cycle and in L3 misses:")
piv = g.pivot_table(index="n_threads", columns="ordering",
                    values=["rec_per_ghz", "L3miss_per_rec"])
if ("rec_per_ghz","grouped") in piv and ("rec_per_ghz","alternating") in piv:
    a = piv[("rec_per_ghz","alternating")]; b = piv[("rec_per_ghz","grouped")]
    ma = piv[("L3miss_per_rec","alternating")]; mb = piv[("L3miss_per_rec","grouped")]
    out = pd.DataFrame({"alt": a, "grp": b, "gain_%": 100*(b/a - 1),
                        "miss_alt": ma, "miss_grp": mb,
                        "miss_chg_%": 100*(mb/ma - 1)})
    print(out.round(4).to_string())
print()
print("  READ AS:")
print("    gain LARGE and miss_chg NEGATIVE, on BOTH page placements")
print("      -> ordering is a real L3-footprint effect, independent of page")
print("         placement. Every earlier sweep used `alternating` and")
print("         understated 16-thread scaling; the headline must be redone.")
print("    gain present on intl but ABSENT on n0")
print("      -> ordering interacts with page placement, and the earlier")
print("         alternating numbers were confounded by staging.")
print("    gain SMALL on both -> the T=16 observation was noise or a")
print("      confound elsewhere; the existing headline stands.")
print()
print("  bit must be True everywhere.")
EOF
echo
echo "  NOTE: the two page placements are in separate CSR directories, so"
echo "  split them by the run order above -- the first block is intl, the"
echo "  second n0. The schema has no page-placement column; that is a gap"
echo "  worth closing if ordering turns out to be a reported axis."
echo
echo "timings: $TIM"
