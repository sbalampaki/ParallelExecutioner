#!/usr/bin/env bash


set -euo pipefail

REPO=/home/sb2ea/loadimbalance
DATA=/projects/sb2ea



ACK='dropcounts\.json|cohort_summary\.json'

mkdir -p "$REPO"/results/{manifest,e0,stream,day4,day5,day6,kernel,timing}

copy () {   # copy <src-glob> <dest-dir>
    local n=0
    for f in $1; do
        [ -e "$f" ] || continue
        cp -u "$f" "$2/" && n=$((n+1))
    done
    printf '  %-46s %d file(s)\n' "$(basename "$1")" "$n"
}

echo "sync $DATA -> $REPO"

# --- aggregate manifests  --------------
copy "$DATA/manifest/*.json"                "$REPO/results/manifest"

# ---  distribution characterization ---------------------------------
copy "$DATA/results/e0/*.csv"               "$REPO/results/e0"
copy "$DATA/results/e0/*.png"               "$REPO/results/e0"

copy "$DATA/manifest/feature_spec.csv"      "$REPO/results/day4"
copy "$DATA/manifest/range_table.csv"       "$REPO/results/day4"
copy "$DATA/manifest/feature_coverage.csv"  "$REPO/results/day4"
copy "$DATA/manifest/kernel_lookup.csv"     "$REPO/results/day4"
copy "$DATA/manifest/kernel_lookup.sha256"  "$REPO/results/day4"
copy "$DATA/manifest/gate_rejects.csv"  "$REPO/results/day4"

copy "$DATA/results/stream/*.csv"           "$REPO/results/stream"

copy "$DATA/results/day6/probe_*.csv"            "$REPO/results/day6"
copy "$DATA/results/day6/numa_pages_*.csv"       "$REPO/results/day6"
copy "$DATA/results/day6/cv_work.csv"            "$REPO/results/day6"
copy "$DATA/results/day6/cv_work_partitions.csv" "$REPO/results/day6"
copy "$DATA/results/day6/a5_bootstrap_*.csv"     "$REPO/results/day6"

copy "$DATA/results/timing/*.csv"           "$REPO/results/timing"


copy "$DATA/results/kernel/*.csv"           "$REPO/results/kernel"


copy "$DATA/results/day5/*.json"            "$REPO/results/day5"
copy "$DATA/results/day5/*.csv"             "$REPO/results/day5"


echo
echo "post-sync check:"
HITS=$(grep -rlIE 'subject_id|stay_id|hadm_id' "$REPO/results" 2>/dev/null | grep -vE "$ACK" || true)
if [ -n "$HITS" ]; then
    echo "  !! IDENTIFIERS PRESENT -- do not commit:"
    echo "$HITS"
    exit 1
fi
echo "  clean."
echo
cd "$REPO" && git status --short
