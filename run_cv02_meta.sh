#!/bin/bash

batch_sizes=(1000)
lrs=(0.01 0.001 0.0001 0.00001)
optimizers=("sgd" "adam")
lr_schedulers=("multiStepLR" "expLR")
random_embs=("true" "false")
emb_trainings=("true" "false")
emb_projections=("true" "false")
final_metrics=("cos" "neural")
vocab_sizes=(20000 50000)

for opt in "${optimizers[@]}"; do
    for lr_s in "${lr_schedulers[@]}"; do
        for lr in "${lrs[@]}"; do
            for fm in "${final_metrics[@]}"; do
                for vs in "${vocab_sizes[@]}"; do
                    for re in "${random_embs[@]}"; do
                        for et in "${emb_trainings[@]}"; do
                            for ep in "${emb_projections[@]}"; do
                                echo $opt $lr_s $lr $fm $vs $re $et $ep
                                qsub -v opt="$opt",lr_s="$lr_s",lr="$lr",fm="$fm",vs="$vs",re="$re",et="$et",ep="$ep" cv02_meta.sh
                            done
                        done
                    done
                done
            done
        done
    done
done
