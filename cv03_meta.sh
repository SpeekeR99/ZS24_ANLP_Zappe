#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=2:0:0
#PBS -l select=1:ncpus=4:mem=32gb:scratch_local=32gb:plzen=True
#PBS -N anlp_cv03

DATADIR=/storage/plzen1/home/zapped99/anlp/anlp-2024_zappe_dominik
cd $SCRATCHDIR

cp -r $DATADIR/* .

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
PYTHON_SCRIPT=run_cv03.py

singularity run $CONTAINER pip install -r /storage/plzen1/home/zapped99/anlp/anlp-2024_zappe_dominik/requirements.txt --user
singularity run $CONTAINER python -m wandb login --relogin ff0893cd7836ab91e9386aa470ed0837f2479f9b

singularity run $CONTAINER python $PYTHON_SCRIPT -model $model -vocab_size $vocab_size -seq_len $seq_len -batches $batch -batch_size $bs -lr $lr -activation $act -random_emb $re -emb_training $et -emb_projection $ep -proj_size $proj_size -gradient_clip $gradient_clip -n_kernel $n_kernels -cnn_architecture $cnn_architecture
