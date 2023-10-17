import numpy as np
from pathlib import Path
import os

if os.getenv("SLURM_JOB_ID") is None:
    raise RuntimeError("Not running inside a SLURM job!")

scratch = Path(os.environ['SCRATCH'])
study_directory = scratch / "thesis-ideas" / "gaussian-shells" / "study_one"

dimensions = [2, 5, 10, 20, 30]


for dimension in dimensions:
    marginal_likelihoods = np.zeros(20)
    for run in range(0, 20):
        job_directory = study_directory / \
            f"dimension-{str(dimension).zfill(1)}-run-{str(run).zfill(3)}"
        with np.load(job_directory / "results.npz") as results:
            marginal_likelihoods[run] = results["marginal_likelihood"]

    print(
        f"dimension {dimension}: {np.mean(marginal_likelihoods)} \\pm {np.sqrt(np.var(marginal_likelihoods))}")
