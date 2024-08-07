#!/bin/bash -l
#SBATCH -J ppc2conver6_days2_60_iris
#SBATCH --output=ppc2conver6_2days2_60_iris.out
#SBATCH --mail-user=damian.ndiwago@uni.lu
#SBATCH --mail-type=END,FAIL
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 30
#BATCH --mem=1TB
#SBATCH -p batch
#SBATCH --qos long
#SBATCH --time=7-00:00:00
echo "== Starting run at $(date)"
echo "== Job name: ${SLURM_JOB_NAME}"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir: ${SLURM_SUBMIT_DIR}"
echo "== Number of tasks: ${SLURM_NTASKS}"


module load lang/Python/3.8.6-GCCcore-10.2.0
source $HOME/damian/damian-venv/bin/activate

parallel -j96 --plus python3 ../ppc21trialbucsrea.py --num {1} --output_dir ./output6_7days2iris_{1} ::: 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 

