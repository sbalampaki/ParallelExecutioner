#!/bin/bash
#SBATCH -p research-cpu -N2 --ntasks-per-node=1 --exclusive --exclude=c2 -t 0:15:00 -o pp_conda.out
export USER=sb2ea
module load miniconda/3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /projects/sb2ea/envs/mimic
echo "nodes: $SLURM_JOB_NODELIST"
srun -l /projects/sb2ea/nodeinfo.sh
mpirun --mca plm ssh -np 2 --map-by ppr:1:node --bind-to core --report-bindings $HOME/bin/pp_conda
exit 0
