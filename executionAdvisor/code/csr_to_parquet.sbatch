#!/usr/bin/env bash
#SBATCH -p research-cpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 1:00:00
#SBATCH -J csr2pq
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH -o /projects/sb2ea/logs/csr_to_parquet-%j.out
#SBATCH -e /projects/sb2ea/logs/csr_to_parquet-%j.err
#
# Rebuild work/events_full.parquet from csr/Wfull/.
# Untimed preparation -> no --exclusive.
#
#   sbatch code/csr_to_parquet.sbatch
#
set -euo pipefail
export USER=sb2ea

REPO=/home/sb2ea/loadimbalance
PROJ=/projects/sb2ea

CONDA_BASE=/opt/ohpc/pub/apps/miniconda
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$PROJ/envs/mimic"

mkdir -p "$PROJ/work" "$PROJ/logs"

echo "host    : $(hostname)"
echo "started : $(date -Is)"
echo "git sha : ${GIT_SHA:-n/a}"
echo

python "$REPO/code/csr_to_parquet.py" --window Wfull \
       --out "$PROJ/work/events_full.parquet"

echo
echo "finished: $(date -Is)"
ls -lh "$PROJ/work/events_full.parquet"
