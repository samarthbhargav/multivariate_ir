#!/bin/bash

if [[ $# -eq 0 || -z "$1" || -z "$2" ]]
  then
    echo "eval_msmarco.sh <MODEL_PATH> <OUT_DIR>"
    exit -1
fi

MODEL_PATH=$1
OUT_DIR=$2
TEMP_DIR=./temp_dir/
BATCH_SIZE=1
P_MAX_LEN=128
Q_MAX_LEN=32
DATA_CACHE_DIR=/ivi/ilps/personal/sbharga/hf_data_cache
MODEL_CACHE_DIR=/ivi/ilps/personal/sbharga/hf_model_cache
RESULTS_DIR=./final_results/tas_b
METRICS="ndcg_cut_10,map,recip_rank"
mkdir -p $RESULTS_DIR


python -m tevatron.driver.encode \
  --output_dir=$TEMP_DIR \
  --model_name_or_path ${MODEL_PATH} \
  --fp16 \
  --per_device_eval_batch_size $BATCH_SIZE \
  --p_max_len $P_MAX_LEN \
  --dataset_name Tevatron/msmarco-passage \
  --encoded_save_path $OUT_DIR/corpus_msmarco-passage.pkl \
  --cache_dir $MODEL_CACHE_DIR \
  --data_cache_dir $DATA_CACHE_DIR

# this requires much more mem because we're not sharding
srun -p gpu --gres=gpu:1 --mem=96G --time=24:00:00 --exclude=ilps-cn111,ilps-cn108 python -m tevatron.driver.encode \
  --output_dir=/ivi/ilps/personal/sbharga/temp_dir \
  --model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
  --fp16 \
  --per_device_eval_batch_size 512 \
  --p_max_len 128 \
  --q_max_len 32 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --encoded_save_path /ivi/ilps/personal/sbharga/mvrl/embeds/corpus_msmarco-passage.pkl \
  --cache_dir /ivi/ilps/personal/sbharga/hf_model_cache \
  --data_cache_dir /ivi/ilps/personal/sbharga/hf_data_cache



TEST_SETS=('dev' 'dl19' 'dl20')

for split in "${TEST_SETS[@]}"
do
  echo "eval $split"
  # get embeds
  srun -p gpu --gres=gpu:1 --mem=32G --time=24:00:00 --exclude=ilps-cn111,ilps-cn108 python -m tevatron.driver.encode \
  --output_dir=/ivi/ilps/personal/sbharga/temp_dir \
  --model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
  --fp16 \
  --per_device_eval_batch_size 512 \
  --p_max_len 128 \
  --q_max_len 32 \
  --encode_is_qry \
  --dataset_name Tevatron/msmarco-passage/$split \
  --encoded_save_path /ivi/ilps/personal/sbharga/mvrl/embeds/$split_msmarco-passage.pkl \
  --cache_dir /ivi/ilps/personal/sbharga/hf_model_cache \
  --data_cache_dir /ivi/ilps/personal/sbharga/hf_data_cache

  # run, also require more mem
  srun -p gpu --gres=gpu:1 --mem=64G --time=24:00:00 --exclude=ilps-cn111,ilps-cn108 python -m tevatron.faiss_retriever \
  --query_reps /ivi/ilps/personal/sbharga/mvrl/embeds/$split_msmarco-passage.pkl \
  --passage_reps /ivi/ilps/personal/sbharga/mvrl/embeds/corpus_msmarco-passage.pkl \
  --depth 100 \
  --batch_size 512 \
  --save_text \
  --save_ranking_to /ivi/ilps/personal/sbharga/mvrl/embeds/$split_msmarco-passage.run
done


# evaluate
python eval_run.py --input /ivi/ilps/personal/sbharga/mvrl/embeds/dev_msmarco-passage.run \
      --dataset msmarco-passage/dev/judged --metrics $METRICS  \
      --output $RESULTS_DIR/dev_msmarco-passage.json

python eval_run.py --input /ivi/ilps/personal/sbharga/mvrl/embeds/dl19_msmarco-passage.run \
      --dataset msmarco-passage/trec-dl-2019/judged --metrics $METRICS  \
      --output $RESULTS_DIR/dl19.json

python eval_run.py --input /ivi/ilps/personal/sbharga/mvrl/embeds/dl20_msmarco-passage.run \
      --dataset msmarco-passage/trec-dl-2020/judged --metrics $METRICS  \
      --output $RESULTS_DIR/dl20.json

