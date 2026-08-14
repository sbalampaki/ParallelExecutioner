#!/bin/bash
#SBATCH --job-name=cv_work
#SBATCH --partition=research-cpu
#SBATCH --nodes=1
#SBATCH --exclude=c2,c3,c4,c5
#SBATCH --time=0:30:00
#SBATCH --output=/projects/sb2ea/logs/cv_work_%j.out
#SBATCH --error=/projects/sb2ea/logs/cv_work_%j.err


set -euo pipefail
export USER=sb2ea

source /opt/ohpc/pub/apps/miniconda/etc/profile.d/conda.sh
conda activate /projects/sb2ea/envs/mimic

CSR=/projects/sb2ea/csr
OUT=/projects/sb2ea/results/day6
mkdir -p "$OUT"

ARGS=""
for V in Wfull W48 W24 W12 W6 \
         sizematched_Wfull sizematched_W48 sizematched_W24 \
         sizematched_W12 sizematched_W6 \
         shape_W24_trim0 shape_W24_trim1 shape_W24_trim5 shape_W24_trim10; do
  if [ -d "$CSR/$V" ]; then
    ARGS="$ARGS --csr $CSR/$V"
  else
    echo "skip (absent): $CSR/$V"
  fi
done

python "$HOME/loadimbalance/code/probes/cv_work.py" \
  $ARGS \
  --lookup /projects/sb2ea/manifest/kernel_lookup.csv \
  --lookup-sha /projects/sb2ea/manifest/kernel_lookup.sha256 \
  --p 96 \
  --out "$OUT/cv_work.csv"
