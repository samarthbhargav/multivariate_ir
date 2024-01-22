
if [ "$#" -ne 3 ]
  then
  echo "USAGE: ./eval_msmarco.sh MODEL_PATH_OR_NAME MODEL_OUT EXTRA_ARGS"
  exit 1
fi


MODEL_NAME=$1
MODEL_OUT=$2
EXTRA_ARGS=$3

TEMP_DIR=/ivi/ilps/projects/multivariate_ir/.temp/
MODEL_CACHE_DIR=/ivi/ilps/projects/multivariate_ir/.hf_model_cache
DATA_CACHE_DIR=/ivi/ilps/projects/multivariate_ir/.hf_data_cache
TOP_K=100
BATCH_SIZE=512
METRICS="ndcg_cut_10,map,recip_rank,recip_rank_cut_10"
RESULTS_DIR=${MODEL_OUT}/runs
export IR_DATASETS_HOME=/ivi/ilps/projects/multivariate_ir/.ird_cache/data
export IR_DATASETS_TMP=/ivi/ilps/projects/multivariate_ir/.ird_cache/temp
export HF_HOME=/ivi/ilps/projects/multivariate_ir/.hf_data_cache

mkdir -p ${RESULTS_DIR}
mkdir -p ${MODEL_OUT}



python -m tevatron.driver.encode \
  --output_dir=${TEMP_DIR} \
  --model_name_or_path ${MODEL_NAME} \
  --fp16 \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --p_max_len 128 \
  --exclude_title \
  --q_max_len 32 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --encoded_save_path ${MODEL_OUT}/corpus_msmarco-passage.pkl \
  --cache_dir ${MODEL_CACHE_DIR} \
  --data_cache_dir ${DATA_CACHE_DIR} \
  ${EXTRA_ARGS}



TEST_SETS=('dev' 'dl19' 'dl20')
for split in "${TEST_SETS[@]}"
do
  echo "eval ${split}"
  # get embeds
  python -m tevatron.driver.encode \
  --output_dir=${TEMP_DIR} \
  --model_name_or_path ${MODEL_NAME} \
  --fp16 \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --p_max_len 128 \
  --exclude_title \
  --q_max_len 32 \
  --encode_is_qry \
  --dataset_name Tevatron/msmarco-passage/${split} \
  --encoded_save_path ${MODEL_OUT}/${split}_msmarco-passage.pkl \
  --cache_dir ${MODEL_CACHE_DIR} \
  --data_cache_dir ${DATA_CACHE_DIR} \
  ${EXTRA_ARGS}


  # obtain run
  python -m tevatron.faiss_retriever \
  --query_reps ${MODEL_OUT}/${split}_msmarco-passage.pkl \
  --passage_reps ${MODEL_OUT}/corpus_msmarco-passage.pkl \
  --depth ${TOP_K} \
  --batch_size  ${BATCH_SIZE} \
  --save_text \
  --save_ranking_to ${RESULTS_DIR}/${split}_msmarco-passage.run
done


python eval_run.py --input ${RESULTS_DIR}/dev_msmarco-passage.run \
      --dataset msmarco-passage/dev/judged --metrics ${METRICS}  \
      --output ${RESULTS_DIR}/dev_msmarco-passage.json

python eval_run.py --input ${RESULTS_DIR}/dl19_msmarco-passage.run \
      --dataset msmarco-passage/trec-dl-2019/judged --metrics ${METRICS}  \
      --output ${RESULTS_DIR}/dl19.json

python eval_run.py --input ${RESULTS_DIR}/dl20_msmarco-passage.run \
      --dataset msmarco-passage/trec-dl-2020/judged --metrics ${METRICS}  \
      --output ${RESULTS_DIR}/dl20.json