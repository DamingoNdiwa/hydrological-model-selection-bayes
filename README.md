# HdromoSelectRehmc
This repository contains code for the paper "xxxx". The requirement.txt file includes the Python packages used for the article.

## Study one
The folder studyone contains codes for the first experiment. 

1. First, generate data by running the run_studyone_data.sh on the HPC cluster.
2. Then run the three other files that end in sh. The files give each model's parameter estimates and log marginal likelihood.
3.  The results can be transferred to the for further processing in which cases the codes are in the post_processing folder.
4. The subfolder ppc contains codes for the posterior predictive check. Here first, generate the data by running run_studyone_data.sh. Then, run the other scripts ending in sh to get the results. The results can be transferred for post-processing.

## Study two
The folder studytwo contains codes for the second experiment.

1. First run run_studytwo_data.sh to generate the data. 
2.  Then run the other scripts ending in sh and save the results for post-processing.
3.  The subfolder ppc is for the posterior predictive.

## Real world
The folder realworld contains scripts when real observed discharge is used. The scripts can be run in any order, and shell files can be run to get the results.

The file run_ppc3bucsrea.sh is to get the results for the posterior predictive check for the chosen model $M_3$. The script cppthreebuckets.py is to get samples from the prior predictive distribution, which are used to calculate the prior calibrated posterior predictive p-value $pcppp$.

## Post process
The folder post_process has code for plots in the article and posterior predictive checks for the chosen model $M_3$ and codes for the selected model's NSE and KGE.

