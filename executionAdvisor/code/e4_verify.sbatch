#!/bin/bash
#SBATCH --job-name=e4_verify
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH --nodelist=c6
#SBATCH --time=0:30:00
#SBATCH --output=/projects/sb2ea/logs/e4verify_%j.out
#SBATCH --error=/projects/sb2ea/logs/e4verify_%j.err



set -euo pipefail
export USER=sb2ea                       # LDAP does not resolve in jobs

BIN=/projects/sb2ea/bin
PROJ=/projects/sb2ea
SRC=$PROJ/csr/W24
TMP=/tmp/sb2ea_e4v
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP"

for b in "$BIN/stage_csr" "$BIN/stage_aligned" "$BIN/numa_verify" "$BIN/numa_pages"; do
    [ -x "$b" ] || { echo "FATAL: $b missing or not executable" >&2; exit 1; }
done

echo "=== node $(hostname -s)  job $SLURM_JOB_ID  $(date -Is)"

declare -A SEEN
S0=(); S1=()
while IFS=, read -r cpu core sock; do
    [ -z "${cpu:-}" ] && continue
    key="$sock:$core"; [ -n "${SEEN[$key]:-}" ] && continue; SEEN[$key]=1
    if [ "$sock" = "0" ]; then S0+=("$cpu"); else S1+=("$cpu"); fi
done < <(lscpu -p=CPU,CORE,SOCKET | grep -v '^#')

GRP=("${S0[@]}" "${S1[@]}")                       # grouped order (D44)
ALT=()                                            # alternating order
for ((i=0; i<${#S0[@]}; i++)); do ALT+=("${S0[$i]}" "${S1[$i]}"); done

places() { printf '{%s},' "$@" | sed 's/,$//'; }  # -> OMP_PLACES
csv()    { local IFS=,; echo "$*"; }              # -> numa_verify --cpus

GRP_PL=$(places "${GRP[@]}");  GRP_CSV=$(csv "${GRP[@]}")
ALT_PL=$(places "${ALT[@]}")
S0_PL=$(places  "${S0[@]}");   S0_CSV=$(csv  "${S0[@]}")

echo "cores: socket0=${#S0[@]} socket1=${#S1[@]}"
echo "grouped     order: $GRP_CSV"
echo "alternating order: $(csv "${ALT[@]}")"
echo

# --- stage 
echo "########## STAGING ##########"
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$GRP_PL" \
    "$BIN/stage_csr" --src "$SRC" --dst "$TMP/intl_grp" --mode interleave
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$ALT_PL" \
    "$BIN/stage_csr" --src "$SRC" --dst "$TMP/intl_alt" --mode interleave
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES="$GRP_PL" \
    "$BIN/stage_aligned" --src "$SRC" --dst "$TMP/algn" | sed 's/^/  /'
echo

# --- aggregate
echo "########## AGGREGATE (what 73063 reported) ##########"
for P in intl_grp intl_alt algn; do
    printf '  %-9s ' "$P"
    "$BIN/numa_pages" "$TMP/$P/value.f32" | tail -n +2 \
        | awk -F, '{printf "%s=%.4f  ", $4, $7} END {print ""}'
done
echo "  predicted: intl_grp 0.5238   intl_alt 0.5000   algn 0.4995"
echo

# --- per-partition, the actual measurement 
for P in intl_grp intl_alt algn; do
    echo "########## $P -- PER-PARTITION LOCALITY ##########"
    echo "--- as the T=16 grouped kernel cell reads it ---"
    "$BIN/numa_verify" --csr "$TMP/$P" --threads 16 --cpus "$GRP_CSV" \
        | sed 's/^/  /'
    echo "--- as the T=8 socket0 kernel cell reads it ---"
    "$BIN/numa_verify" --csr "$TMP/$P" --threads 8  --cpus "$S0_CSV" \
        | sed 's/^/  /'
    echo
done

echo "########## WHAT TO LOOK FOR ##########"
cat <<'EOT'
  algn at T=16 grouped : local_frac 1.00 for every thread.
      Anything less means stage_aligned did not place what it claimed and
      Block A's null is uninterpretable rather than informative.

  algn at T=8 socket0  : 1.00 for tid 0-3, 0.00 for tid 4-7.
      Confirms the control received no treatment
EOT
