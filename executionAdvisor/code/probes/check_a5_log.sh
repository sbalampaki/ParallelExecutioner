#!/usr/bin/env bash

set -uo pipefail

LOG="${1:-$(ls -t /projects/sb2ea/logs/a5_boot_*.out 2>/dev/null | head -1)}"
EXPECT="${2:-90.11}"

TOL="${TOL:-0.5}"

if [ -z "${LOG:-}" ] || [ ! -f "$LOG" ]; then
    echo "no log found; pass one explicitly:" >&2
    echo "  bash $0 /projects/sb2ea/logs/a5_boot_<JOBID>.out" >&2
    exit 2
fi

echo "log: $LOG"
echo

read -r -d '' PRELUDE <<'AWK'
  /COST = RECORDS/                    { sec = "records" }
  /COST = WORK/                       { sec = "work"    }
  /^[A-Za-z][A-Za-z0-9_]* +\(cost =/  { var = $1 }
  /^ *p = [0-9]+ /                    { pp  = $3 }
AWK

echo "=============================================================="
echo "1. CROSS-CHECK — Wfull p=96 canonical draw, cost=records"
echo "=============================================================="
awk -v expect="$EXPECT" "
  $PRELUDE
  sec == \"records\" && var == \"Wfull\" && pp == \"96\" && /canonical draw/ {
      v = \$3; gsub(/%/, \"\", v)
      d = v - expect; if (d < 0) d = -d
      printf \"  bootstrap canonical draw : %s%%\n\", v
      printf \"  cv_work.py block         : %s%%\n\", expect
      if (d > 0.005) {
          printf \"\n  ** MISMATCH (%.4f pp) — STOP.\n\", d
          print  \"  The two scripts disagree about what a block partition is.\"
          print  \"  Nothing else in this log is meaningful until that is resolved.\"
          exit 1
      }
      print \"\n  OK — two independent code paths agree.\"
      found = 1
      exit 0
  }
  END { if (!found && !NR) exit 3 }
" "$LOG"
RC1=$?
if [ $RC1 -eq 0 ]; then
    :
elif [ $RC1 -eq 1 ]; then
    echo
    echo "STOPPING. Fix the disagreement before reading anything else."
    exit 1
else
    echo "  could not locate the Wfull p=96 records row — check the log ran fully"
fi
echo

echo "=============================================================="
echo "2. BLOCK vs CYCLIC — canonical gap is noise, boot gap must be ~0"
echo "=============================================================="
echo "  block and cyclic are the SAME distribution under a random"
echo "  permutation, so any canonical gap is single-draw noise and the"
echo "  bootstrap gap must converge to zero."
echo
printf "  %-8s %-22s %-5s %11s %11s\n" cost variant p "canon gap" "boot gap"
printf "  %-8s %-22s %-5s %11s %11s\n" ----- ---------------------- ----- ----------- -----------
awk -v tol="$TOL" "
  $PRELUDE
  /block vs cyclic/ {
      printf \"  %-8s %-22s %-5s %8s pp %8s pp%s\n\",
             sec, var, \"p=\" pp, \$9, \$17,
             ((\$17 < tol && \$17 > -tol) ? \"\" : \"   <-- BOOT GAP LARGE (raise --nboot?)\")
  }
" "$LOG"
echo
echo "  Largest canonical gaps (each one is pure noise, by construction):"
awk "
  $PRELUDE
  /block vs cyclic/ { g = \$9; if (g < 0) g = -g; printf \"%8.2f  %-8s %-22s p=%s\n\", g, sec, var, pp }
" "$LOG" | sort -rn | head -6 | sed 's/^/    /'
echo
echo "  Compare the top figure against A8's canonical 90.11 / 83.99 = 6.12 pp."
echo "  That gap was never a strategy effect."
echo

echo "=============================================================="
echo "3. CALIBRATION VERDICT — is the D8 seed driving anything?"
echo "=============================================================="
grep -A 12 'CALIBRATION VERDICT' "$LOG" | sed 's/^/  /'
echo
echo "  inline flags raised during the run:"
if grep -q 'FLAG:' "$LOG"; then
    grep -n 'FLAG:' "$LOG" | sed 's/^/    /'
    echo
    echo "  A flagged seed is fixable NOW and not after the sweep."
    echo "  If you reseed: ONCE, blind, and record the original percentile."
    echo "  Do not search over seeds — that is the failure D30 prevents."
else
    echo "    none"
fi
echo

echo "=============================================================="
echo "context: the finite-p term A5 approximates"
echo "=============================================================="
awk '/E\[max of p standard normals\]/,/ratio > 1/' "$LOG" | head -8 | sed 's/^/  /'
echo
echo "next: python code/probes/verify_a5_bootstrap.py \\"
echo "        --boot-records /projects/sb2ea/results/day6/a5_bootstrap_records.csv \\"
echo "        --boot-work    /projects/sb2ea/results/day6/a5_bootstrap_work.csv \\"
echo "        --partitions   /projects/sb2ea/results/day6/cv_work_partitions.csv"
