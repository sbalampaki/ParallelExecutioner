#!/usr/bin/env bash
#SBATCH -p research-cpu
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH -J kserial
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH -o /projects/sb2ea/logs/kernel_serial-%j.out
#SBATCH -e /projects/sb2ea/logs/kernel_serial-%j.err
#

set -euo pipefail
export USER=sb2ea

PROJ=/projects/sb2ea
BIN=$PROJ/bin/kernel_serial
LOOK=$PROJ/manifest/kernel_lookup.csv
OUT=$PROJ/results/kernel
mkdir -p "$OUT" "$PROJ/features"

test -x "$BIN"  || { echo "FATAL: $BIN not built (run build_kernel.sh on the login node)"; exit 1; }
test -f "$LOOK" || { echo "FATAL: $LOOK missing (run feature_spec.sbatch first)"; exit 1; }

echo "host   : $(hostname)"
echo "date   : $(date -Is)"
sha256sum "$LOOK"          # the kernel must read the frozen spec
echo


echo "=================== W24 tile invariance ==================="
"$BIN" --csr "$PROJ/csr/W24" --lookup "$LOOK" --verify-tiles \
       --out "$PROJ/features/W24_serial.f64"

echo
echo "=================== Wfull tile invariance ================="
"$BIN" --csr "$PROJ/csr/Wfull" --lookup "$LOOK" --verify-tiles \
       --out "$PROJ/features/Wfull_serial.f64"


CSV=$OUT/tile_sweep_$(hostname)_${SLURM_JOB_ID}.csv
echo "window,tile,rep,repeat_k,per_iter_s,records_per_s" > "$CSV"


NREP=6

for W in W24 Wfull; do
  for C in 64 256 1024 4096 8192 12288 16384 20480 24576 32768 65536; do
    if [ "$W" = "W24" ]; then K=50; else K=15; fi
    for REP in $(seq 0 $((NREP - 1))); do
      "$BIN" --csr "$PROJ/csr/$W" --lookup "$LOOK" \
             --tile "$C" --repeat "$K" >/dev/null 2>/tmp/k_$$.err || true
      PER=$(awk '/per-iter/{print $3}'   /tmp/k_$$.err)
      THR=$(awk '/throughput/{print $3}' /tmp/k_$$.err)
      echo "$W,$C,$REP,$K,$PER,$THR" >> "$CSV"
      rm -f /tmp/k_$$.err
    done
    # median of reps 1..N-1, run 0 discarded
    MED=$(awk -F, -v w="$W" -v c="$C" '$1==w && $2==c && $3>0 {print $6}' "$CSV" \
          | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')
    printf '  %-6s tile=%-6s K=%-3s  median %s M rec/s\n' "$W" "$C" "$K" "$MED"
  done
done

echo
echo "=== $CSV ==="
column -s, -t < "$CSV"

rm -f /tmp/k_$$.err
