#!/usr/bin/env bash

#SBATCH --nodes=1                   # 1 node
#SBATCH --ntasks-per-node=1         # 1 tasks per node
#SBATCH --time=24:00:00             # time limits: 24 hours
#SBATCH --error=cardDownloader.err    # standard error file
#SBATCH --output=cardDownloader.out   # standard output file
#SBATCH --partition=cpuextralong    # partition name
#SBATCH --mem=10G
#SBATCH --mail-user=adamej14@fel.cvut.cz # where send info about job
#SBATCH --mail-type=ALL             # what to send, valid type values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

ml Python/3.13.5-GCCcore-14.3.0
srun python3 cardDownloader.py -f all-cards-20251018215107.json -s /mnt/personal/adamej14/images