#!/usr/bin/env bash

#SBATCH --nodes=1                   # 1 node
#SBATCH --ntasks-per-node=1         # 1 tasks per node
#SBATCH --time=24:00:00             # time limits: 24 hours
#SBATCH --error=scraper.err    # standard error file
#SBATCH --output=scraper.out   # standard output file
#SBATCH --partition=cpuextralong    # partition name
#SBATCH --mem=10G
#SBATCH --mail-user=adamej14@fel.cvut.cz # where send info about job
#SBATCH --mail-type=END             # what to send, valid type values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

ml Python/3.13.5-GCCcore-14.3.0
ml Python-bundle-PyPI/2025.07-GCCcore-14.3.0
srun python3 img_scraper.py -f default-cards-20260105101020.json -s /mnt/personal/adamej14/dataset
