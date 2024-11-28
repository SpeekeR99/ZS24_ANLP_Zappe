#!/bin/bash

# Grid search
model_types=("UWB-AIR/Czert-B-base-cased" "ufal/robeczech-base" "fav-kky/FERNET-C5")
tasks=("sts" "sentiment")

for model_type in "${model_types[@]}"; do
    for task in "${tasks[@]}"; do
        echo "Model type: $model_type, Task: $task"
        qsub -v model_type="$model_type",task="$task" cv05_meta.sh
    done
done
