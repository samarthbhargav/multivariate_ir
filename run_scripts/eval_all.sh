

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
#### MS-MARCO ####
########################################################################################################################

## Encode MS-MARCO Passage
echo "encoding MS-MARCO-Passage"
python -m tevatron.driver.encode \
  --output_dir=${MODEL_OUT} \
  --model_name_or_path ${MODEL_PATH_OR_NAME} \
  --fp16 \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --p_max_len 256 \
  --exclude_title \
  --q_max_len 32 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --encoded_save_path ${MODEL_OUT}/corpus_msmarco-passage.pkl \
  ${EXTRA_ARGS} >>${LOG_FILE} 2>&1



TEST_SETS=('dev' 'dl19' 'dl20')
for split in "${TEST_SETS[@]}"
do
  echo "eval ${split}"
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
  --dataset_name Tevatron/msmarco-passage/${split} \
  --encoded_save_path ${MODEL_OUT}/${split}_msmarco-passage.pkl \
  ${EXTRA_ARGS} >>${LOG_FILE} 2>&1


  # obtain run
  python -m tevatron.faiss_retriever \
  --query_reps ${MODEL_OUT}/${split}_msmarco-passage.pkl \
  --passage_reps ${MODEL_OUT}/corpus_msmarco-passage.pkl \
  --depth ${TOP_K} \
  --batch_size  ${BATCH_SIZE} \
  --save_text \
  --save_ranking_to ${RESULTS_DIR}/${split}_msmarco-passage.run >>${LOG_FILE} 2>&1
done


python ../eval_run.py --input ${RESULTS_DIR}/dev_msmarco-passage.run \
      --dataset msmarco-passage/dev/judged --metrics ${METRICS}  \
      --output ${RESULTS_DIR}/dev_msmarco-passage.json >>${LOG_FILE} 2>&1

python ../eval_run.py --input ${RESULTS_DIR}/dl19_msmarco-passage.run \
      --dataset msmarco-passage/trec-dl-2019/judged --metrics ${METRICS}  \
      --output ${RESULTS_DIR}/dl19.json >>${LOG_FILE} 2>&1

python ../eval_run.py --input ${RESULTS_DIR}/dl20_msmarco-passage.run \
      --dataset msmarco-passage/trec-dl-2020/judged --metrics ${METRICS}  \
      --output ${RESULTS_DIR}/dl20.json >>${LOG_FILE} 2>&1


########################################################################################################################
#### SciFact ####
########################################################################################################################
# smaller batch size for SciFact
BATCH_SIZE=256

echo "encoding SciFact corpus"
python -m tevatron.driver.encode \
--output_dir=${MODEL_OUT} \
--model_name_or_path ${MODEL_PATH_OR_NAME} \
--fp16 \
--per_device_eval_batch_size ${BATCH_SIZE} \
--p_max_len 512 \
--dataset_name Tevatron/scifact-corpus \
--encoded_save_path ${MODEL_OUT}/corpus_scifact.pkl \
${EXTRA_ARGS} >>${LOG_FILE} 2>&1

echo "eval scifact"
python -m tevatron.driver.encode \
--output_dir=${MODEL_OUT} \
--model_name_or_path ${MODEL_PATH_OR_NAME} \
--fp16 \
--per_device_eval_batch_size ${BATCH_SIZE} \
--q_max_len 64 \
--encode_is_qry \
--dataset_name Tevatron/scifact/dev \
--encoded_save_path ${MODEL_OUT}/dev_scifact.pkl \
${EXTRA_ARGS} >>${LOG_FILE} 2>&1


# obtain run
python -m tevatron.faiss_retriever \
--query_reps ${MODEL_OUT}/dev_scifact.pkl \
--passage_reps ${MODEL_OUT}/corpus_scifact.pkl \
--depth ${TOP_K} \
--batch_size ${BATCH_SIZE} \
--save_text \
--save_ranking_to ${RESULTS_DIR}/dev_scifact.run >>${LOG_FILE} 2>&1


python ../eval_run.py --input ${RESULTS_DIR}/dev_scifact.run \
--hf_dataset Tevatron/scifact/dev --metrics ${METRICS}  \
--output ${RESULTS_DIR}/dev_scifact.json >>${LOG_FILE} 2>&1


########################################################################################################################
#### BIER datasets ####
########################################################################################################################

BIER_DATASETS=( "fiqa" "trec-covid" "cqadupstack-android" "cqadupstack-english" "cqadupstack-gaming" "cqadupstack-gis" "cqadupstack-wordpress" "cqadupstack-physics" "cqadupstack-programmers" "cqadupstack-stats" "cqadupstack-tex" "cqadupstack-unix" "cqadupstack-webmasters" "cqadupstack-wordpress" )
BATCH_SIZE=128
for bds in "${BIER_DATASETS[@]}"
do
    echo "eval $bds"

    echo "encoding corpus"
    python -m tevatron.driver.encode \
    --output_dir=${MODEL_OUT} \
    --model_name_or_path ${MODEL_PATH_OR_NAME} \
    --fp16 \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --p_max_len 512 \
    --dataset_name Tevatron/beir-corpus:${bds} \
    --encoded_save_path ${MODEL_OUT}/corpus_bier_${bds}.pkl \
    ${EXTRA_ARGS} >>${LOG_FILE} 2>&1

    echo "encoding test queries"
    python -m tevatron.driver.encode \
    --output_dir=${MODEL_OUT} \
    --model_name_or_path ${MODEL_PATH_OR_NAME} \
    --fp16 \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --dataset_name Tevatron/beir:${bds}/test \
    --encoded_save_path ${MODEL_OUT}/test_bier_${bds}.pkl \
    --q_max_len 512 \
    --encode_is_qry \
    ${EXTRA_ARGS} >>${LOG_FILE} 2>&1

    echo "obtaining run"
    python -m tevatron.faiss_retriever \
    --query_reps ${MODEL_OUT}/test_bier_${bds}.pkl \
    --passage_reps ${MODEL_OUT}/corpus_bier_${bds}.pkl \
    --depth 1000 \
    --batch_size ${BATCH_SIZE} \
    --save_text \
    --save_ranking_to ${RESULTS_DIR}/bier_test_${bds}.run >>${LOG_FILE} 2>&1


    python ../eval_run.py --input ${RESULTS_DIR}/bier_test_${bds}.run \
      --hf_dataset Tevatron/beir:${bds}/test --metrics ${METRICS}  \
      --output ${RESULTS_DIR}/bier_test_${bds}.json >>${LOG_FILE} 2>&1
done