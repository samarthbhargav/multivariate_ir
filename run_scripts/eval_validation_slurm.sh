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

export MODEL_CACHE_DIR=/ivi/ilps/projects/multivariate_ir/.hf_model_cache
export DATA_CACHE_DIR=/ivi/ilps/projects/multivariate_ir/.hf_data_cache
export IR_DATASETS_HOME=/ivi/ilps/projects/multivariate_ir/.ird_cache/data
export IR_DATASETS_TMP=/ivi/ilps/projects/multivariate_ir/.ird_cache/temp
export HF_HOME=/ivi/ilps/projects/multivariate_ir/.hf_data_cache

export MODEL_PATH_OR_NAME=${MODEL_PATH_OR_NAME}
export MODEL_OUT=${MODEL_OUT}
export EXTRA_ARGS=${EXTRA_ARGS}
export LOG_FILE=${LOG_FILE}


if [ -z "$LOG_FILE" ]
then
  echo "provide MODEL_OUT!"
  exit -1
fi
echo "logging to ${LOG_FILE}"

if [ -z "$MODEL_OUT" ]
then
  echo "provide MODEL_OUT!"
  exit -1
fi
echo "MODEL_OUT: ${MODEL_OUT}"

if [ -z "$MODEL_PATH_OR_NAME" ]
then
  echo "provide MODEL_PATH_OR_NAME!"
  exit -1
fi
echo "MODEL_PATH_OR_NAME: ${MODEL_PATH_OR_NAME}"

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

TOP_K=1000
BATCH_SIZE=512
METRICS="ndcg_cut_10,map,recip_rank,recip_rank_cut_10,recall_1000"
RESULTS_DIR=${MODEL_OUT}/runs

mkdir -p ${RESULTS_DIR}
mkdir -p ${MODEL_OUT}


########################################################################################################################
#### MS-MARCO Validation ####
########################################################################################################################

## Encode MS-MARCO Passage
# encode corpus
echo "encoding MS-MARCO-Passage"
for s in $(seq -f "%02g" 0 19)
do
  echo "shard ${s}"
  python -m tevatron.driver.encode \
    --output_dir=${MODEL_OUT} \
    --model_name_or_path ${MODEL_PATH_OR_NAME} \
    --fp16 \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --p_max_len 256 \
    --exclude_title \
    --q_max_len 32 \
    --dataset_name Tevatron/msmarco-passage-corpus \
    --encoded_save_path ${MODEL_OUT}/corpus_emb.${s}.pkl \
    --encode_num_shard 20 \
    --encode_shard_index ${s} \
    ${EXTRA_ARGS} >>${LOG_FILE} 2>&1
done

echo "encoding validation"
# get embeds
python -m tevatron.driver.encode \
  --output_dir=${MODEL_OUT} \
  --model_name_or_path ${MODEL_PATH_OR_NAME} \
  --fp16 \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --p_max_len 256 \
  --exclude_title \
  --q_max_len 32 \
  --encode_is_qry \
  --dataset_name Tevatron/msmarco-passage/dev \
  --hf_disk_dataset /ivi/ilps/projects/multivariate_ir/experiments_gs/data/validation \
  --encoded_save_path ${MODEL_OUT}/validation_msmarco-passage.pkl \
  ${EXTRA_ARGS} >>${LOG_FILE} 2>&1

python -m tevatron.faiss_retriever \
  --query_reps ${MODEL_OUT}/validation_msmarco-passage.pkl \
  --passage_reps ${MODEL_OUT}/'corpus_emb.*.pkl' \
  --depth ${TOP_K} \
  --batch_size  ${BATCH_SIZE} \
  --save_text \
  --save_ranking_to ${RESULTS_DIR}/validation_msmarco-passage.run >>${LOG_FILE} 2>&1




