#!/usr/bin/env bash
#SBATCH -p research-cpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 3:00:00
#SBATCH -J mimic-csr
#SBATCH --exclude=c3,c4,c5
#SBATCH -o /projects/sb2ea/logs/csr-%j.out
#SBATCH -e /projects/sb2ea/logs/csr-%j.err



set -euo pipefail

export USER=sb2ea

# Lmod is non-functional on compute nodes; sbatch inherits PATH from the
# login node, so this is best-effort and must not abort the job.

CONDA_BASE=/opt/ohpc/pub/apps/miniconda

[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] || { echo "no conda at $CONDA_BASE" >&2; exit 1; }

source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate /projects/sb2ea/envs/mimic

export DUCKDB_THREADS="${DUCKDB_THREADS:-8}"
export DUCKDB_MEMORY_LIMIT="${DUCKDB_MEMORY_LIMIT:-32GB}"

echo "host=$(hostname)  start=$(date -Is)"

python /projects/sb2ea/code/csr.py "$@"
rc=$?

echo "end=$(date -Is)  rc=$rc"
exit $rc
