#!/usr/bin/env bash
#SBATCH -p research-cpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 1:00:00
#SBATCH -J feat_spec
#SBATCH --exclude=c3,c4,c5
#SBATCH -o /projects/sb2ea/logs/feature_spec-%j.out
#SBATCH -e /projects/sb2ea/logs/feature_spec-%j.err

set -euo pipefail
export USER=sb2ea                   


REPO=/home/sb2ea/loadimbalance       
PROJ=/projects/sb2ea                 

CONDA_BASE=/opt/ohpc/pub/apps/miniconda
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$PROJ/envs/mimic"

mkdir -p "$PROJ/manifest" "$PROJ/work/duckdb_tmp" "$PROJ/logs"

echo "host      : $(hostname)"
echo "started   : $(date -Is)"
echo "duckdb    : $(python -c 'import duckdb; print(duckdb.__version__)')"
echo "git sha   : $(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
echo

python "$REPO/code/run_sql.py" "$REPO/code/feature_spec.sql"

echo
echo "finished  : $(date -Is)"
echo "artifacts :"
ls -l "$PROJ/manifest/feature_spec.csv" \
      "$PROJ/manifest/range_table.csv" \
      "$PROJ/manifest/feature_coverage.csv" \
      "$PROJ/manifest/kernel_lookup.csv"

sha256sum "$PROJ/manifest/kernel_lookup.csv" | tee "$PROJ/manifest/kernel_lookup.sha256"
