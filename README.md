# HdromodSelectRephmc
This repository contains code for the paper "xxxx". The requirement.txt file includes the Python packages used for the article.

## data
Contains the data and scripts for preprocessing.
## gaussian-shells
This folder has scripts for the Gaussian shell examples.
1. First, run the scripts gaussian_shell.py, study_one.py, study_one20.py, study_one30.py in any order and finally study_one_postprocess.py.
2.  The script study_one_postprocess.py needs to be run on HPC because many scripts run in parallel.
3.  The plotGaussianshell.py and Gaussian_nuts_mala_rephmc.py can be run in any order.

## studyone
The folder studyone contains codes for the first experiment. 

1. First, generate data by running the run_studyone_data.sh on the HPC cluster.
2. Then run the three other files that end in sh. The files give each model's parameter estimates and log marginal likelihood.
3. The results can be transferred for further processing in which cases the codes are in the post processing folder.
4. The subfolder ppc contains codes for the posterior predictive check. Here first, generate the data by running run_studyone_data.sh. Then, run the other scripts ending in sh to get the results. The results can be transferred for post-processing.
5. The models can be run many times in parallel by specifying the number of times in the files ending in sh.
## studytwo
The folder studytwo contains codes for the second experiment.

* First run run_studytwo_data.sh to generate the data. 
* Then run the other scripts ending in sh and save the results for post-processing.
* The subfolder ppc is for the posterior predictive.

## realworld
The folder realworld contains scripts when real observed discharge is used. The scripts can be run in any order, and shell files can be run to get the results.

* The file run_ppc3bucsrea.sh is to get the results for the posterior predictive check for the chosen model $M_3$.
* The script ppc_cppp.py should be run after run_twobucs.sh, run_threebucs.sh, and un_fourbucs.sh. 
* The script ppc_cppp.py uses results from the models to do posterior predictive checks.
## post_process

The folder post_process  has codes
* To check for convergence with the real-world discharge data as examples. 
* For the pair plots in with real-world discharge data as examples.
