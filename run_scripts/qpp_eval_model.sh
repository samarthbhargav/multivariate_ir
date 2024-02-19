#!/bin/bash

MODEL_PATH_OR_NAME=$1
MODEL_OUT=$2
EXTRA_ARGS=$3
LOG_FILE=$4
BATCH_SIZE=128
DATA_PATH=/projects/0/prjs0907/data

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


echo "running on perturbations"
echo "MSMARCO-Med Corpus"
python -m tevatron.driver.qpp \
  --output_dir=${MODEL_OUT} \
  --model_name_or_path ${MODEL_PATH_OR_NAME} \
  --qpp_save_path ${MODEL_OUT}/msmarco-med-corpus.txt \
  --fp16 \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --p_max_len 256 \
  --exclude_title \
  --q_max_len 32 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --hf_disk_dataset ${DATA_PATH}/msmarco-med/corpus \
  ${EXTRA_ARGS} >>${LOG_FILE} 2>&1

PERCS=(0.1 0.5 1.0)
for perc in "${PERCS[@]}"
do
  python -m tevatron.driver.qpp \
  --output_dir=${MODEL_OUT} \
  --model_name_or_path ${MODEL_PATH_OR_NAME} \
  --qpp_save_path ${MODEL_OUT}/msmarco-med-corpus-${perc}.txt \
  --fp16 \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --p_max_len 256 \
  --exclude_title \
  --q_max_len 32 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --hf_disk_dataset ${DATA_PATH}/msmarco-med-perturbed-${perc}/corpus \
  ${EXTRA_ARGS} >>${LOG_FILE} 2>&1
done

SPLIT=('train' 'validation')
echo "MSMARCO splits"
for split in "${SPLIT[@]}"
do
  echo "${split}"
  python -m tevatron.driver.qpp \
  --output_dir=${MODEL_OUT} \
  --model_name_or_path ${MODEL_PATH_OR_NAME} \
  --qpp_save_path ${MODEL_OUT}/msmarco-${split}.txt \
  --fp16 \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --p_max_len 256 \
  --encode_is_qry \
  --exclude_title \
  --q_max_len 32 \
  --dataset_name Tevatron/msmarco-passage \
  --hf_disk_dataset ${DATA_PATH}/msmarco/${split} \
  ${EXTRA_ARGS} >>${LOG_FILE} 2>&1


  for perc in "${PERCS[@]}"
  do
      echo "${split} -- ${perc}"
      python -m tevatron.driver.qpp \
      --output_dir=${MODEL_OUT} \
      --model_name_or_path ${MODEL_PATH_OR_NAME} \
      --qpp_save_path ${MODEL_OUT}/msmarco-perturbed-${perc}-${split}.txt \
      --fp16 \
      --per_device_eval_batch_size ${BATCH_SIZE} \
      --p_max_len 256 \
      --encode_is_qry \
      --exclude_title \
      --q_max_len 32 \
      --dataset_name Tevatron/msmarco-passage \
      --hf_disk_dataset ${DATA_PATH}/msmarco-perturbed-${perc}/${split} \
      ${EXTRA_ARGS} >>${LOG_FILE} 2>&1
  done
done

TEST_SETS=('dl19' 'dl20' 'dev')
for split in "${TEST_SETS[@]}"
do
  echo "eval ${split}"
  python -m tevatron.driver.qpp \
  --output_dir=${MODEL_OUT} \
  --model_name_or_path ${MODEL_PATH_OR_NAME} \
  --qpp_save_path ${MODEL_OUT}/msmarco-${split}.txt \
  --fp16 \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --p_max_len 256 \
  --exclude_title \
  --q_max_len 32 \
  --encode_is_qry \
  --dataset_name Tevatron/msmarco-passage/${split} \
  ${EXTRA_ARGS} >>${LOG_FILE} 2>&1
done

echo "scifact corpus"
python -m tevatron.driver.qpp \
--output_dir=${MODEL_OUT} \
--model_name_or_path ${MODEL_PATH_OR_NAME} \
--fp16 \
--per_device_eval_batch_size ${BATCH_SIZE} \
--p_max_len 512 \
--dataset_name Tevatron/scifact-corpus \
--qpp_save_path ${MODEL_OUT}/corpus_scifact.txt \
${EXTRA_ARGS} >>${LOG_FILE} 2>&1

echo "scifact dev"
python -m tevatron.driver.qpp \
--output_dir=${MODEL_OUT} \
--model_name_or_path ${MODEL_PATH_OR_NAME} \
--fp16 \
--per_device_eval_batch_size ${BATCH_SIZE} \
--q_max_len 64 \
--encode_is_qry \
--dataset_name Tevatron/scifact/dev \
--qpp_save_path ${MODEL_OUT}/dev_scifact.txt \
${EXTRA_ARGS} >>${LOG_FILE} 2>&1


BIER_DATASETS=("fiqa" "trec-covid" "cqadupstack-android" "cqadupstack-english" "cqadupstack-gaming" "cqadupstack-gis" "cqadupstack-wordpress" "cqadupstack-physics" "cqadupstack-programmers" "cqadupstack-stats" "cqadupstack-tex" "cqadupstack-unix" "cqadupstack-webmasters" "cqadupstack-wordpress" )
BATCH_SIZE=128
for bds in "${BIER_DATASETS[@]}"
do
    echo "eval $bds"

    echo "encoding corpus"
    python -m tevatron.driver.qpp \
    --output_dir=${MODEL_OUT} \
    --model_name_or_path ${MODEL_PATH_OR_NAME} \
    --fp16 \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --p_max_len 512 \
    --dataset_name Tevatron/beir-corpus:${bds} \
    --qpp_save_path ${MODEL_OUT}/bier_${bds}-corpus.txt \
    ${EXTRA_ARGS} >>${LOG_FILE} 2>&1

    echo "encoding test queries"
    python -m tevatron.driver.qpp \
    --output_dir=${MODEL_OUT} \
    --model_name_or_path ${MODEL_PATH_OR_NAME} \
    --fp16 \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --dataset_name Tevatron/beir:${bds}/test \
    --qpp_save_path ${MODEL_OUT}/bier_${bds}_test.txt \
    --q_max_len 512 \
    --encode_is_qry \
    ${EXTRA_ARGS} >>${LOG_FILE} 2>&1
done
