#!/bin/bash

MODEL_PATH_OR_NAME=$1
MODEL_OUT=$2
EXTRA_ARGS=$3
LOG_FILE=$4


if [ -z "$MODEL_PATH_OR_NAME" ] || [ -z "$MODEL_OUT" ] || [ -z "$LOG_FILE" ]
then
  echo "usage: MODEL_PATH_OR_NAME MODEL_OUT EXTRA_ARGS LOG_FILE"
  exit -1
fi

if [ ! -z "$MODEL_CACHE_DIR" ]
then
      echo "MODEL_CACHE_DIR=${MODEL_CACHE_DIR}"
      EXTRA_ARGS="${EXTRA_ARGS} --cache_dir ${MODEL_CACHE_DIR}"
fi

if [ ! -z "$DATA_CACHE_DIR" ]
then
      echo "DATA_CACHE_DIR=${DATA_CACHE_DIR}"
      EXTRA_ARGS="${EXTRA_ARGS} --data_cache_dir ${DATA_CACHE_DIR}"
fi

echo "EXTRA_ARGS=${EXTRA_ARGS}"

mkdir -p ${MODEL_OUT}


TEST_SETS=('dl19' 'dl20')
for split in "${TEST_SETS[@]}"
do
  echo "eval ${split}"
  # get embeds
  python -m tevatron.driver.qpp \
  --output_dir=${MODEL_OUT} \
  --model_name_or_path ${MODEL_PATH_OR_NAME} \
  --qpp_save_path ${MODEL_OUT}/${split}_msmarco-passage.txt \
  --fp16 \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --p_max_len 256 \
  --exclude_title \
  --q_max_len 32 \
  --encode_is_qry \
  --dataset_name Tevatron/msmarco-passage/${split} \
  --encoded_save_path ${MODEL_OUT}/${split}_msmarco-passage.pkl \
  ${EXTRA_ARGS} >>${LOG_FILE} 2>&1
done