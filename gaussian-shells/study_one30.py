from pathlib import Path
import os
import subprocess

scratch = Path(os.environ['SCRATCH'])


dimensions = [30]

for dimension in dimensions:
    for run in range(0, 20):
        job_directory = scratch / "thesis-ideas" / "gaussian-shells" / \
            "study_one" / \
            f"dimension-{str(dimension).zfill(1)}-run-{str(run).zfill(3)}"
        if not job_directory.is_dir():
            os.makedirs(job_directory)

        job_file = job_directory / "launcher.sh"

        with open(job_file, "w") as f:
            f.write("#!/bin/bash -l\n")
            f.write("#SBATCH -N 1\n")
            f.write("#SBATCH --ntasks-per-node=1\n")
            f.write("#SBATCH -c 10\n")
            f.write("#SBATCH -p batch\n")
            f.write("#SBATCH --qos normal\n")
            f.write("#SBATCH -t 00:05:00\n")
            f.write(f"#SBATCH -o {job_directory}/slurm-%A_%a.out\n")
            f.write("module load lang/Python/3.8.6-GCCcore-10.2.0\n")
            f.write("source $HOME/damian/damian-venv/bin/activate\n")
            f.write('echo "== Starting run at $(date)"\n')
            f.write('echo "== Job name: ${SLURM_JOB_NAME}"\n')
            f.write('echo "== Job ID: ${SLURM_JOBID}"\n')
            f.write('echo "== Node list: ${SLURM_NODELIST}"\n')
            f.write('echo "== Submit dir: ${SLURM_SUBMIT_DIR}"\n')
            f.write('echo "== Number of tasks: ${SLURM_NTASKS}"\n')
            f.write(
                f"python3 $HOME/thesis-ideas/gaussian-shells/gaussian_shell.py --dimension {dimension} --output_dir {job_directory}")

        subprocess.run(["sbatch", job_file])
