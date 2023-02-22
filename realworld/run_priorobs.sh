#!/bin/bash -l
#SBATCH --job-name=prior_obs
#SBATCH --output=prior_obs.out
#SBATCH --mail-user=damian.ndiwago@uni.lu
#SBATCH --mail-type=END,FAIL
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time=2-00:00:00
#
#SBATCH -p batch
#SBATCH --qos=normal


module load lang/Python/3.8.6-GCCcore-10.2.0
source $HOME/damian/damian-venv/bin/activate

python3 cppthreebuckets.py

deactivate
