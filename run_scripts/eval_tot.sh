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

export MODEL_CACHE_DIR=/scratch-shared/sbhargav/.hf_model_cache
export DATA_CACHE_DIR=/scratch-shared/sbhargav/.hf_data_cache
export IR_DATASETS_HOME=/scratch-shared/sbhargav/.ird_cache/data
export IR_DATASETS_TMP=/scratch-shared/sbhargav/.ird_cache/temp
export HF_HOME=/scratch-shared/sbhargav/.hf_data_cache

export MODEL_PATH_OR_NAME=${MODEL_PATH_OR_NAME}
export MODEL_OUT=${MODEL_OUT}
export EXTRA_ARGS=${EXTRA_ARGS}
export LOG_FILE=${LOG_FILE}

DATA_PATH=/projects/0/prjs0907/data
TOP_K=1000
BATCH_SIZE=128

echo "encoding TOT Corpus"
# dataset name still needs to be set to ensure the right
# preprocessor is set
python -m tevatron.driver.encode \
  --output_dir=${MODEL_OUT} \
  --model_name_or_path ${MODEL_PATH_OR_NAME} \
  --fp16 \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --p_max_len 512 \
  --exclude_title \
  --q_max_len 512 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --hf_disk_dataset ${DATA_PATH}/ToT/trec-corpus \
  --encoded_save_path ${MODEL_OUT}/tot_corpus.pkl \
  ${EXTRA_ARGS} >>${LOG_FILE} 2>&1


TEST_SETS=('reddit-test' 'reddit-validation' 'trec-dev' 'trec-train' 'reddit-train' 'trec-corpus' 'trec-test')
for split in "${TEST_SETS[@]}"
do
  echo "eval ${split}"
  # get embeds
  python -m tevatron.driver.encode \
  --output_dir=${MODEL_OUT} \
  --model_name_or_path ${MODEL_PATH_OR_NAME} \
  --fp16 \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --p_max_len 512 \
  --exclude_title \
  --q_max_len 512 \
  --encode_is_qry \
  --dataset_name Tevatron/msmarco-passage \
  --hf_disk_dataset ${DATA_PATH}/ToT/trec-corpus \
  --encoded_save_path ${MODEL_OUT}/tot-${split}.pkl \
  ${EXTRA_ARGS} >>${LOG_FILE} 2>&1


  # obtain run
  python -m tevatron.faiss_retriever \
  --query_reps ${MODEL_OUT}/tot-${split}.pkl \
  --passage_reps ${MODEL_OUT}/tot_corpus.pkl \
  --depth ${TOP_K} \
  --batch_size  ${BATCH_SIZE} \
  --save_text \
  --save_ranking_to ${RESULTS_DIR}/tot-${split}.run >>${LOG_FILE} 2>&1
done