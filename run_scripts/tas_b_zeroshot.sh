MODEL_NAME=sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco
TEMP_DIR=/ivi/ilps/projects/multivariate_ir/.temp/
MODEL_OUT=/ivi/ilps/projects/multivariate_ir/experiments/tas_b_zeroshot
MODEL_CACHE_DIR=/ivi/ilps/projects/multivariate_ir/.hf_model_cache
DATA_CACHE_DIR=/ivi/ilps/projects/multivariate_ir/.hf_data_cache
TOP_K=100
BATCH_SIZE=512
METRICS="ndcg_cut_10,map,recip_rank,recip_rank_cut_10,recall_1000"
RESULTS_DIR=${MODEL_OUT}/runs
export IR_DATASETS_HOME=/ivi/ilps/projects/multivariate_ir/.ird_cache/data
export IR_DATASETS_TMP=/ivi/ilps/projects/multivariate_ir/.ird_cache/temp
export HF_HOME=/ivi/ilps/projects/multivariate_ir/.hf_data_cache

mkdir -p ${RESULTS_DIR}
mkdir -p ${MODEL_OUT}


########################################################################################################################
#### MS-MARCO ####
########################################################################################################################

## Encode MS-MARCO Passage
echo "encoding MS-MARCO-Passage"
#python -m tevatron.driver.encode \
#  --output_dir=${TEMP_DIR} \
#  --model_name_or_path ${MODEL_NAME} \
#  --fp16 \
#  --per_device_eval_batch_size ${BATCH_SIZE} \
#  --p_max_len 200 \
#  --exclude_title \
#  --q_max_len 30 \
#  --dataset_name Tevatron/msmarco-passage-corpus \
#  --encoded_save_path ${MODEL_OUT}/corpus_msmarco-passage.pkl \
#  --cache_dir ${MODEL_CACHE_DIR} \
#  --data_cache_dir ${DATA_CACHE_DIR}



TEST_SETS=('dev' 'dl19' 'dl20')
for split in "${TEST_SETS[@]}"
do
  echo "eval ${split}"
  # get embeds
  #  python -m tevatron.driver.encode \
  #  --output_dir=${TEMP_DIR} \
  #  --model_name_or_path ${MODEL_NAME} \
  #  --fp16 \
  #  --per_device_eval_batch_size ${BATCH_SIZE} \
  #  --p_max_len 200 \
  #  --exclude_title \
  #  --q_max_len 30 \
  #  --encode_is_qry \
  #  --dataset_name Tevatron/msmarco-passage/${split} \
  #  --encoded_save_path ${MODEL_OUT}/${split}_msmarco-passage.pkl \
  #  --cache_dir ${MODEL_CACHE_DIR} \
  #  --data_cache_dir ${DATA_CACHE_DIR}
  #
  #
  #  # obtain run
  #  python -m tevatron.faiss_retriever \
  #  --query_reps ${MODEL_OUT}/${split}_msmarco-passage.pkl \
  #  --passage_reps ${MODEL_OUT}/corpus_msmarco-passage.pkl \
  #  --depth ${TOP_K} \
  #  --batch_size  ${BATCH_SIZE} \
  #  --save_text \
  #  --save_ranking_to ${RESULTS_DIR}/${split}_msmarco-passage.run
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


########################################################################################################################
#### SciFact ####
########################################################################################################################
# smaller batch size for SciFact
BATCH_SIZE=256

echo "encoding SciFact corpus"
#python -m tevatron.driver.encode \
#  --output_dir=${TEMP_DIR} \
#  --model_name_or_path ${MODEL_NAME} \
#  --fp16 \
#  --per_device_eval_batch_size ${BATCH_SIZE} \
#  --p_max_len 512 \
#  --dataset_name Tevatron/scifact-corpus \
#  --encoded_save_path ${MODEL_OUT}/corpus_scifact.pkl \
#  --cache_dir ${MODEL_CACHE_DIR} \
#  --data_cache_dir ${DATA_CACHE_DIR} \
#
#echo "eval scifact"
#python -m tevatron.driver.encode \
#--output_dir=${TEMP_DIR} \
#--model_name_or_path ${MODEL_NAME} \
#--fp16 \
#--per_device_eval_batch_size ${BATCH_SIZE} \
#--q_max_len 64 \
#--encode_is_qry \
#--dataset_name Tevatron/scifact/dev \
#--encoded_save_path ${MODEL_OUT}/dev_scifact.pkl \
#--cache_dir ${MODEL_CACHE_DIR} \
#--data_cache_dir ${DATA_CACHE_DIR}
#
#
## obtain run
#python -m tevatron.faiss_retriever \
#--query_reps ${MODEL_OUT}/dev_scifact.pkl \
#--passage_reps ${MODEL_OUT}/corpus_scifact.pkl \
#--depth ${TOP_K} \
#--batch_size ${BATCH_SIZE} \
#--save_text \
#--save_ranking_to ${RESULTS_DIR}/dev_scifact.run


python eval_run.py --input ${RESULTS_DIR}/dev_scifact.run \
      --hf_dataset Tevatron/scifact/dev --metrics ${METRICS}  \
      --output ${RESULTS_DIR}/dev_scifact.json


########################################################################################################################
#### BIER datasets ####
########################################################################################################################

BIER_DATASETS=( "fiqa" "trec-covid" "cqadupstack-android" "cqadupstack-english" "cqadupstack-gaming" "cqadupstack-gis" "cqadupstack-wordpress" "cqadupstack-physics" "cqadupstack-programmers" "cqadupstack-stats" "cqadupstack-tex" "cqadupstack-unix" "cqadupstack-webmasters" "cqadupstack-wordpress" )
BATCH_SIZE=128
for bds in "${BIER_DATASETS[@]}"
do
    echo "eval $bds"

    echo "encoding corpus"
    #    python -m tevatron.driver.encode \
    #    --output_dir=${TEMP_DIR} \
    #    --model_name_or_path ${MODEL_NAME} \
    #    --fp16 \
    #    --per_device_eval_batch_size ${BATCH_SIZE} \
    #    --p_max_len 512 \
    #    --dataset_name Tevatron/beir-corpus:${bds} \
    #    --encoded_save_path ${MODEL_OUT}/corpus_bier_${bds}.pkl \
    #    --cache_dir ${MODEL_CACHE_DIR} \
    #    --data_cache_dir ${DATA_CACHE_DIR}
    #
    #    echo "encoding test queries"
    #    python -m tevatron.driver.encode \
    #    --output_dir=${TEMP_DIR} \
    #    --model_name_or_path ${MODEL_NAME} \
    #    --fp16 \
    #    --per_device_eval_batch_size ${BATCH_SIZE} \
    #    --dataset_name Tevatron/beir:${bds}/test \
    #    --encoded_save_path ${MODEL_OUT}/test_bier_${bds}.pkl \
    #    --q_max_len 512 \
    #    --encode_is_qry \
    #    --cache_dir ${MODEL_CACHE_DIR} \
    #    --data_cache_dir ${DATA_CACHE_DIR}
    #
    #    echo "obtaining run"
    #    python -m tevatron.faiss_retriever \
    #    --query_reps ${MODEL_OUT}/test_bier_${bds}.pkl \
    #    --passage_reps ${MODEL_OUT}/corpus_bier_${bds}.pkl \
    #    --depth 1000 \
    #    --batch_size ${BATCH_SIZE} \
    #    --save_text \
    #    --save_ranking_to ${RESULTS_DIR}/bier_test_${bds}.run


    python eval_run.py --input ${RESULTS_DIR}/bier_test_${bds}.run \
      --hf_dataset Tevatron/beir:${bds}/test --metrics ${METRICS}  \
      --output ${RESULTS_DIR}/bier_test_${bds}.json
done