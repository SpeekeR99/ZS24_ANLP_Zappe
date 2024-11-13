#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=2:0:0
#PBS -l select=1:ncpus=4:mem=32gb:scratch_local=32gb:plzen=True
#PBS -N anlp_cv04

DATADIR=/storage/plzen1/home/zapped99/anlp/anlp-2024_zappe_dominik
cd $SCRATCHDIR

cp -r $DATADIR/* .

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
PYTHON_SCRIPT=cv04/main04.py

singularity run $CONTAINER pip3 install -r /storage/plzen1/home/zapped99/anlp/anlp-2024_zappe_dominik/requirements.txt --user
singularity run $CONTAINER python3 -m wandb login --relogin ff0893cd7836ab91e9386aa470ed0837f2479f9b

singularity run $CONTAINER python3 $PYTHON_SCRIPT --model_type $model_type --learning_rate $lr --l2_alpha $l2_alpha --data_dir $datadir --labels $label --eval_steps $eval_steps --num_train_epochs $num_train_epochs --task $task --per_device_train_batch_size $batch_size --per_device_eval_batch_size $batch_size --output_dir output --do_predict --do_train --do_eval --eval_dataset_batches 200 --logging_steps 50 --warmup_steps 4000 --dropout_probs 0.05 --lstm_hidden_dimension 128 --num_lstm_layers 2 --embedding_dimension 128
if $model_type != "CZERT" && $model_type != "SLAVIC"; then
    singularity run $CONTAINER python3 $PYTHON_SCRIPT --no_bias --model_type $model_type --learning_rate $lr --l2_alpha $l2_alpha --data_dir $datadir --labels $label --eval_steps $eval_steps --num_train_epochs $num_train_epochs --task $task --per_device_train_batch_size $batch_size --per_device_eval_batch_size $batch_size --output_dir output --do_predict --do_train --do_eval --eval_dataset_batches 200 --logging_steps 50 --warmup_steps 4000 --dropout_probs 0.05 --lstm_hidden_dimension 128 --num_lstm_layers 2 --embedding_dimension 128
fi

clean_scratch
