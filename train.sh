#!/bin/bash

# Example values — you can modify these as needed
MODEL_TYPE="CNNLSTM"
DATASET_PATH="./data/dataset_2005028_3k.json"
NORMALIZED="False"

# Run the training script
python train.py \
  --model=${MODEL_TYPE} \
  --dataset_path=${DATASET_PATH} \
  --normalized=${NORMALIZED}
