#!/usr/bin/env bash
#SBATCH -p research-cpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0:30:00
#SBATCH -J mimic-e0
#SBATCH --exclude=c3,c4,c5
#SBATCH -o /projects/sb2ea/logs/e0-%j.out
#SBATCH -e /projects/sb2ea/logs/e0-%j.err

# Untimed analysis, not a benchmark: no --exclusive.

set -euo pipefail

export USER=sb2ea

# Source conda by absolute path: Lmod is non-functional on compute nodes and
# inherited PATH is not reliable across login sessions (job 72298 died 127).
CONDA_BASE=/opt/ohpc/pub/apps/miniconda
[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ] || {
    echo "no conda at $CONDA_BASE" >&2; exit 1; }
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /projects/sb2ea/envs/mimic

echo "host=$(hostname)  start=$(date -Is)"

python /projects/sb2ea/code/e0_distribution.py "$@"
rc=$?

echo "end=$(date -Is)  rc=$rc"
exit $rc
