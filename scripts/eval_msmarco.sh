#!/bin/bash

MODEL_PATH=$1
OUT_DIR=$2
TEMP_DIR=./temp_dir/
BATCH_SIZE=1
P_MAX_LEN=512
Q_MAX_LEN=32

if [[ $# -eq 0 || -z "$1" || -z "$2" ]]
  then
    echo "eval_msmarco.sh <MODEL_PATH> <OUT_DIR>"
    exit -1
fi


python -m tevatron.driver.encode \
  --output_dir=$TEMP_DIR \
  --model_name_or_path ${MODEL_PATH} \
  --fp16 \
  --per_device_eval_batch_size $BATCH_SIZE \
  --p_max_len $P_MAX_LEN \
  --dataset_name Tevatron/msmarco-passage \
  --encoded_save_path $OUT_DIR/corpus_msmarco-passage.pkl


