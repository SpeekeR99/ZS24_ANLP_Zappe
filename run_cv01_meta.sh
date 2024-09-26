#!/bin/bash

models=("dense" "cnn")
optimizers=("sgd" "adam")
lrs=(0.1 0.01 0.001 0.0001 0.00001)
dps=(0 0.1 0.3 0.5)

for m in "${models[@]}"; do
    for o in "${optimizers[@]}"; do
        for l in "${lrs[@]}"; do
            for d in "${dps[@]}"; do
                echo $m $o $l $d
                qsub cv01_meta.sh
            done
        done
    done
done