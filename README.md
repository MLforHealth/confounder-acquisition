# Confounding Feature Acquisition for Causal Effect Estimation
# About
This repository contains codes to generate simulated data using IHDP dataset, and implement acquisition strategies, and evaluate their effectiveness in combination with common treatment effect estimation models. Python environment is specified in [environment.yml](environment.yml) and [requirements.txt](requirements.txt). 
# IHDP
This contains raw IHDP data used for simulation.
# Code
## simulate.py
This script generates simulated data using IHDP dataset. Various parameters that influence treatment assignment, outcome generations, and missingness mechanism can be set. An example that corresponds to experiments in the paper is given in [generate_ihdp_simulated_a.sh](code/bash/generate_ihdp_simulated_a.sh).

## simulate_al_ihdp.py
This script implements confounding feature acquisition strategies using different models. Examples of running the script can be found at [CMGP_fact.sh](code/bash/CMGP_fact.sh) and [DR_a_exp.sh](code/bash/DR_a_exp.sh).

## causal_models.py
This script is called by simulate_al_ihdp.py to implement CMGP model. We used the implementation of CMGP at [Repo](https://bitbucket.org/mvdschaar/mlforhealthlabpub/src/c9a65d2c44fa73c9266cbe8ddcc69b16ff424a73/alg/causal_multitask_gaussian_processes_ite/).

# Paper
If you use this code in your research, please cite the following [paper](https://arxiv.org/abs/2011.08753):
```
Wang, S., Yi, S.E., Joshi, S., & Ghassemi, M. (2020). Confounding Feature Acquisition for Causal Effect Estimation. arXiv:2011.08753.
```
# References
Alaa, Ahmed M., and Mihaela van der Schaar. [Bayesian inference of individualized treatment effects using multi-task gaussian processes](http://papers.nips.cc/paper/6934-bayesian-inference-of-individualized-treatment-effects-using-multi-task-gaussian-processes) Advances in Neural Information Processing Systems. 2017.

Tibshirani, Julie, et al. [Package ‘grf’](http://cran.ms.unimelb.edu.au/web/packages/grf/grf.pdf) (2018).
