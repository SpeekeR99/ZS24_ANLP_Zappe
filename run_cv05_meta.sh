#!/bin/bash

# Grid search
# model_types=("UWB-AIR/Czert-B-base-cased" "ufal/robeczech-base" "fav-kky/FERNET-C5")
# tasks=("sts" "sentiment")
# lrs=(0.001 0.0001 0.00001)
# batch_sizes=(32 64)
# seq_lens=(64 128)

# for model_type in "${model_types[@]}"; do
#     for task in "${tasks[@]}"; do
#         for lr in "${lrs[@]}"; do
#             for batch_size in "${batch_sizes[@]}"; do
#                 for seq_len in "${seq_lens[@]}"; do
#                     echo "Model type: $model_type, Task: $task, LR: $lr, Batch size: $batch_size, Seq len: $seq_len"
#                     qsub -v task=$task,model_type=$model_type,lr=$lr,batch_size=$batch_size,seq_len=$seq_len cv05_meta.sh
#                 done
#             done
#         done
#     done
# done

# Best runs
for i in {1..10}; do
    qsub -v task="sts",model_type="UWB-AIR/Czert-B-base-cased",lr=0.00001,batch_size=64,seq_len=64 cv05_meta.sh
    qsub -v task="sts",model_type="ufal/robeczech-base",lr=0.00001,batch_size=32,seq_len=64 cv05_meta.sh
    qsub -v task="sts",model_type="fav-kky/FERNET-C5",lr=0.00001,batch_size=32,seq_len=64 cv05_meta.sh

    qsub -v task="sentiment",model_type="UWB-AIR/Czert-B-base-cased",lr=0.0001,batch_size=64,seq_len=128 cv05_meta.sh
    qsub -v task="sentiment",model_type="ufal/robeczech-base",lr=0.00001,batch_size=32,seq_len=128 cv05_meta.sh
    qsub -v task="sentiment",model_type="fav-kky/FERNET-C5",lr=0.0001,batch_size=64,seq_len=128 cv05_meta.sh
done
