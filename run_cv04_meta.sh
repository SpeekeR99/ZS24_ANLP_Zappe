#!/bin/bash

# # Grid search
# model_types=("RNN" "LSTM")
# lrs=(0.001 0.0001)
# l2_alphas=(0.01 0)

# # Differences between tasks
# datadirs=("cv04/data" "cv04/data-mt")
# labels=("cv04/data/labels.txt" "cv04/data-mt/labels.txt")
# eval_steps=(200 300)
# num_train_epochs=(1000 10)
# tasks=("NER" "TAGGING")

# batch_size=8

# for model_type in "${model_types[@]}"; do
#     for lr in "${lrs[@]}"; do
#         for l2_alpha in "${l2_alphas[@]}"; do
#             echo NER $model_type $lr $l2_alpha
#             qsub -v model_type="$model_type",lr="$lr",l2_alpha="$l2_alpha",datadir="${datadirs[0]}",label="${labels[0]}",eval_steps="${eval_steps[0]}",num_train_epochs="${num_train_epochs[0]}",task="${tasks[0]}",batch_size="$batch_size" cv04_meta.sh
#         done
#     done
# done

# for model_type in "${model_types[@]}"; do
#     for lr in "${lrs[@]}"; do
#         for l2_alpha in "${l2_alphas[@]}"; do
#             echo TAGGING $model_type $lr $l2_alpha
#             qsub -v model_type="$model_type",lr="$lr",l2_alpha="$l2_alpha",datadir="${datadirs[1]}",label="${labels[1]}",eval_steps="${eval_steps[1]}",num_train_epochs="${num_train_epochs[1]}",task="${tasks[1]}",batch_size="$batch_size" cv04_meta.sh
#         done
#     done
# done

# # CZERT, SLAVIC

# czert_model="CZERT"
# slavic_model="SLAVIC"
# lr=0.0001
# eval_steps=100
# num_train_epochs=50
# batch_size=32

# qsub -v model_type="$czert_model",lr="$lr",datadir="${datadirs[0]}",label="${labels[0]}",eval_steps="$eval_steps",num_train_epochs="$num_train_epochs",task="${tasks[0]}",batch_size="$batch_size" cv04_meta.sh
# qsub -v model_type="$slavic_model",lr="$lr",datadir="${datadirs[0]}",label="${labels[0]}",eval_steps="$eval_steps",num_train_epochs="$num_train_epochs",task="${tasks[0]}",batch_size="$batch_size" cv04_meta.sh

# eval_steps=300
# num_train_epochs=10

# qsub -v model_type="$czert_model",lr="$lr",datadir="${datadirs[1]}",label="${labels[1]}",eval_steps="$eval_steps",num_train_epochs="$num_train_epochs",task="${tasks[1]}",batch_size="$batch_size" cv04_meta.sh
# qsub -v model_type="$slavic_model",lr="$lr",datadir="${datadirs[1]}",label="${labels[1]}",eval_steps="$eval_steps",num_train_epochs="$num_train_epochs",task="${tasks[1]}",batch_size="$batch_size" cv04_meta.sh

# Extended experiments

czert_model="CZERT"
freeze_layers=(2 4 6)
for freeze_layer in "${freeze_layers[@]}"; do
    qsub -v model_type="$czert_model",freeze_layer="$freeze_layer" cv04_meta.sh
done

bert_model="BERT"
lrs=(0.001 0.0001)
l2_alphas=(0.01 0)
for lr in "${lrs[@]}"; do
    for l2_alpha in "${l2_alphas[@]}"; do
        qsub -v model_type="$bert_model",lr="$lr",l2_alpha="$l2_alpha" cv04_meta.sh
    done
done
