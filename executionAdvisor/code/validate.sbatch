#!/usr/bin/env bash
#SBATCH -p research-cpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 0:30:00
#SBATCH -J mimic-validate
#SBATCH --exclude=c3,c4,c5
#SBATCH -o /projects/sb2ea/logs/validate-%j.out
#SBATCH -e /projects/sb2ea/logs/validate-%j.err


set -euo pipefail

export USER=sb2ea


module load miniconda/3 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/sb2ea/envs/mimic

export DUCKDB_THREADS="${DUCKDB_THREADS:-8}"
export DUCKDB_MEMORY_LIMIT="${DUCKDB_MEMORY_LIMIT:-24GB}"

echo "host=$(hostname)  start=$(date -Is)"

python /projects/sb2ea/code/validate.py "$@"
rc=$?

echo "end=$(date -Is)  rc=$rc"
exit $rc
