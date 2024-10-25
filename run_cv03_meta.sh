#!/bin/bash

model_mean="mean"
model_cnn="cnn"
# General HPs
vocab_size=20000
seq_len=100
batches=(100000 500000)
batch_sizes=(32 64)
lrs=(0.001 0.0001 0.00001 0.000001)
activations=("relu" "gelu")
random_embs=("true" "false")
emb_trainings=("true" "false")
emb_projections=("true" "false")
proj_size=128
gradient_clip=0.5
# CNN specific HPs
n_kernels=64
cnn_architectures=("A" "B" "C")

for batch in "${batches[@]}"; do
    for bs in "${batch_sizes[@]}"; do
        for lr in "${lrs[@]}"; do
            for act in "${activations[@]}"; do
                for re in "${random_embs[@]}"; do
                    for et in "${emb_trainings[@]}"; do
                        for ep in "${emb_projections[@]}"; do
                            echo $model_mean $batch $bs $lr $act $re $et $ep
                            qsub -v model="$model_mean",vocab_size="$vocab_size",seq_len="$seq_len",batch="$batch",bs="$bs",lr="$lr",act="$act",re="$re",et="$et",ep="$ep",proj_size="$proj_size",gradient_clip="$gradient_clip",n_kernels="$n_kernels",cnn_architecture="A" cv03_meta.sh
                        done
                    done
                done
            done
        done
    done
done

for batch in "${batches[@]}"; do
    for bs in "${batch_sizes[@]}"; do
        for lr in "${lrs[@]}"; do
            for act in "${activations[@]}"; do
                for re in "${random_embs[@]}"; do
                    for et in "${emb_trainings[@]}"; do
                        for ep in "${emb_projections[@]}"; do
                            for cnn_architecture in "${cnn_architectures[@]}"; do
                                echo $model_mean $batch $bs $lr $act $re $et $ep $cnn_architecture
                                qsub -v model="$model_mean",vocab_size="$vocab_size",seq_len="$seq_len",batch="$batch",bs="$bs",lr="$lr",act="$act",re="$re",et="$et",ep="$ep",proj_size="$proj_size",gradient_clip="$gradient_clip",n_kernels="$n_kernels",cnn_architecture="$cnn_architecture" cv03_meta.sh
                            done
                        done
                    done
                done
            done
        done
    done
done
