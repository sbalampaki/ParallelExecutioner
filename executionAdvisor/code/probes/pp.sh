#!/bin/bash
#SBATCH -p research-cpu -N2 --ntasks-per-node=1 --exclusive -t 0:15:00 -o pp.out
srun -n2 --ntasks-per-node=1 $HOME/bin/pp
exit 0
