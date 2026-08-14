#!/bin/bash
#SBATCH -p research-cpu -N1 --exclusive -t 0:10:00 -o probe.out
hostname; lscpu | grep -E 'Model name|^CPU\(s\)|Socket|Core\(s\) per|L3|NUMA node'
free -g; df -h /tmp /dev/shm; touch /tmp/$USER.t && echo TMP_WRITABLE
ip -br link; ibstat 2>/dev/null || echo NO_IB
ls /usr/bin/lstopo* /usr/bin/hwloc* 2>/dev/null
