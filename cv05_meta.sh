#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l walltime=8:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=32gb:scratch_local=32gb
#PBS -N anlp_cv05

DATADIR=/storage/plzen1/home/zapped99/anlp/anlp-2024_zappe_dominik
cd $SCRATCHDIR

cp -r $DATADIR/* .

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
PYTHON_SCRIPT=cv05/main05.py

singularity run $CONTAINER pip3 install -r /storage/plzen1/home/zapped99/anlp/anlp-2024_zappe_dominik/requirements.txt --user
singularity run $CONTAINER python3 -m wandb login --relogin ff0893cd7836ab91e9386aa470ed0837f2479f9b

singularity run --nv $CONTAINER python3 $PYTHON_SCRIPT -task $task -model_type $model_type -lr $lr -batch_size $batch_size -seq_len $seq_len

clean_scratch
