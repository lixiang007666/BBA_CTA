#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Override these defaults with DATASET_ROOT, MODEL_ROOT, and LOG_ROOT as needed.
dataset_root=${DATASET_ROOT:-$script_dir/data/Fundus}
model_root=${MODEL_ROOT:-$script_dir/models}
path_save_log=${LOG_ROOT:-$script_dir/logs}

#Dataset [RIM_ONE_r3, REFUGE, ORIGA, REFUGE_Valid, Drishti_GS]
Source=${SOURCE_DATASET:-ORIGA}
Target=${TARGET_DATASETS:-RIM_ONE_r3}

optimizer=Adam
lr=${LEARNING_RATE:-0.05}

alpha=0.3
gamma=0.75
eta=0.05
bn_tau=0.01

#Command
cd "$script_dir/OPTIC"
echo "PEOA source=$Source target=$Target lr=$lr"
CUDA_VISIBLE_DEVICES=0 python BBA.py \
--dataset_root "$dataset_root" --model_root "$model_root" --path_save_log "$path_save_log" \
--Source_Dataset "$Source" --target_datasets $Target \
--optimizer "$optimizer" --lr "$lr" \
--alpha "$alpha" --gamma "$gamma" --eta "$eta" --bn_tau "$bn_tau" \
--iters 2 --batch_size 1 "${@}"
