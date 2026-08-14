#!/usr/bin/env bash
#SBATCH -p research-cpu
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH -J mimic-convert
#SBATCH --exclude=c3,c4,c5
#SBATCH -o /projects/sb2ea/logs/convert-%j.out
#SBATCH -e /projects/sb2ea/logs/convert-%j.err



set -euo pipefail

export USER=sb2ea                     # LDAP does not resolve in jobs

# Lmod is non-functional on compute nodes (see platform.md); sbatch inherits
# PATH from the login node, so this is best-effort and must not abort the job.
module load miniconda/3 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /projects/sb2ea/envs/mimic

# ':-' so that --export=ALL,DUCKDB_THREADS=N can override these.
export DUCKDB_THREADS="${DUCKDB_THREADS:-8}"
export DUCKDB_MEMORY_LIMIT="${DUCKDB_MEMORY_LIMIT:-24GB}"

echo "host=$(hostname)  start=$(date -Is)"
echo "cpus=${SLURM_CPUS_PER_TASK:-?}  threads=${DUCKDB_THREADS}  mem=${DUCKDB_MEMORY_LIMIT}"

python /projects/sb2ea/code/convert.py "$@"

echo "end=$(date -Is)"
