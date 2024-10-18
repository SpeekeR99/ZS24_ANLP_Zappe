#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=2:0:0
#PBS -l select=1:ncpus=4:mem=32gb:scratch_local=32gb:plzen=True
#PBS -N anlp_cv02

DATADIR=/storage/plzen1/home/zapped99/anlp/anlp-2024_zappe_dominik
cd $SCRATCHDIR

cp -r $DATADIR/* .

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
PYTHON_SCRIPT=run_cv02.py

singularity run $CONTAINER pip install -r /storage/plzen1/home/zapped99/anlp/anlp-2024_zappe_dominik/requirements.txt --user
singularity run $CONTAINER python -m wandb login --relogin ff0893cd7836ab91e9386aa470ed0837f2479f9b

singularity run $CONTAINER python $PYTHON_SCRIPT -lr $lr -optimizer $opt -random_emb $re -emb_training $et -emb_projection $ep -final_metric $fm -vocab_size $vs -lr_scheduler $lr_s
