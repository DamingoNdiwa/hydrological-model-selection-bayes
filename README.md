# Hydrological model selection using Bayes factors

## Introduction

This repository contains supporting code for the paper "Selecting a conceptual
hydrological model using Bayes' factors computed with Replica Exchange
Hamiltonian Monte Carlo" by Mingo et al.

*TODO: Place full citation to pre-print, accepted article and Zenodo version.*

This code is licensed under the GNU Lesser General Public License version 3 or
later, see `COPYING` and `COPYING.LESSER`.

Some studies were executed on the University of Luxembourg HPC systems. The
launch scripts are specific to that cluster and are included in the
`hpc_scripts/` subdirectory for reference. It would be necessary to modify
these job launch scripts to run these studies on another cluster.

## Dependencies

The primary dependencies are JAX and Tensorflow Probability (using the JAX
backend). Secondary dependencies for pre and post-processing include various
plotting and statistical libraries. 

The code is compatible with Python 3.10 and the dependencies specified in the
`requirements.txt` file:

    pip install -r requirements.txt

The file `requirements-not-fixed.txt` contains the same dependencies without
fixed version requirements.

## Megala Creek data `data/`

The folder `megala_creek_australia/` contains the raw data for the Magala Creek
catchment in text format. A script `prepare_megala_creek_data.py` places the
data into a Pandas dataframe saved at `megala_creek_australia.pkl.gz` for
straightforward ingestion into the main scripts. Note that the dataset contains
missing values on some days.

## Gaussian shells `gaussian_shells/`

This folder contains scripts for the Gaussian shell examples.

For a basic plot using the proposed algorithm, run:

    python3 gaussian_shell.py

## Experiment one - synthetic discharge `experiment_one/`

The folder `experiment_one/` contains codes for the first experiment where 
the calibration data is generated from a forward run of model two.

1. Generate data by executing `generate_data.py`.
2. For the marginal likelihood calculation execute `twobuckets.py`,
   `threebuckets.py` and `fourbuckets.py`.
3. The subfolder `ppc/` contains codes for the posterior predictive checks.
4. The results can be transferred for post-processing in which cases the
   scripts are in the root `post_processing/` folder.

## Experiment two - synthetic discharge `experiment_two/` 

The folder `experiment_two/` contains codes for the second experiment where 
the calibration data is generated from a forward run of model three.

1. Generate data by executing `generate_data.py`.
2. For the marginal likelihood calculation execute `twobuckets.py`,
   `threebuckets.py` and `fourbuckets.py`.
3. The subfolder `ppc/` contains codes for the posterior predictive checks.
4. The results can be transferred for post-processing in which cases the
   scripts are in the root `post_processing/` folder.

## Experiment three - real discharge `real_world/` 

The folder `real_world/` contains codes for the third experiment where the
models are calibrated and assessed against real data from Magala Creek.

## Postprocessing - `postprocess/` 

The folder `postprocess/` contains scripts for generating posterior plots and
chain convergence diagnostics from the results produced in experiments one, two
and three. 
