#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=2:0:0
#PBS -l select=1:ncpus=4:mem=20gb:scratch_local=10gb
#PBS -N anlp_cv01

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
PYTHON_SCRIPT=/storage/plzen1/home/zapped99/anlp/anlp-2024_zappe_dominik/run_cv01.py

singularity run $CONTAINER pip install wandb --user
singularity run $CONTAINER python -m wandb login --relogin ff0893cd7836ab91e9386aa470ed0837f2479f9b

singularity run $CONTAINER python $PYTHON_SCRIPT -model $m -optimizer $o -lr $l -dp $d
