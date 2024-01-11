MODEL_NAME=sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco
TEMP_DIR=/ivi/ilps/personal/sbharga/temp_dir
MODEL_OUT=/ivi/ilps/personal/sbharga/mvrl/tas_b_zeroshot/
MODEL_CACHE_DIR=/ivi/ilps/personal/sbharga/hf_model_cache
DATA_CACHE_DIR=/ivi/ilps/personal/sbharga/hf_data_cache
TOP_K=100
BATCH_SIZE=512
METRICS="ndcg_cut_10,map,recip_rank"
RESULTS_DIR=final_results/tas_b_zeroshot/


mkdir -p $RESULTS_DIR
mkdir -p $MODEL_OUT


########################################################################################################################
#### MS-MARCO ####
########################################################################################################################

## Encode MS-MARCO Passage
echo "encoding MS-MARCO-Passage"
python -m tevatron.driver.encode \
  --output_dir=$TEMP_DIR \
  --model_name_or_path $MODEL_NAME \
  --fp16 \
  --per_device_eval_batch_size $BATCH_SIZE \
  --p_max_len 128 \
  --exclude_title \
  --q_max_len 32 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --encoded_save_path $MODEL_OUT/corpus_msmarco-passage.pkl \
  --cache_dir $MODEL_CACHE_DIR \
  --data_cache_dir $DATA_CACHE_DIR



TEST_SETS=('dev' 'dl19' 'dl20')
for split in "${TEST_SETS[@]}"
do
  echo "eval $split"
  # get embeds
  python -m tevatron.driver.encode \
  --output_dir=$TEMP_DIR \
  --model_name_or_path $MODEL_NAME \
  --fp16 \
  --per_device_eval_batch_size $BATCH_SIZE \
  --p_max_len 128 \
  --exclude_title \
  --q_max_len 32 \
  --encode_is_qry \
  --dataset_name Tevatron/msmarco-passage/$split \
  --encoded_save_path $MODEL_OUT/$split_msmarco-passage.pkl \
  --cache_dir $MODEL_CACHE_DIR \
  --data_cache_dir $DATA_CACHE_DIR


  # obtain run
  python -m tevatron.faiss_retriever \
  --query_reps $MODEL_OUT/$split_msmarco-passage.pkl \
  --passage_reps $MODEL_OUT/corpus_msmarco-passage.pkl \
  --depth $TOP_K \
  --batch_size  $BATCH_SIZE \
  --save_text \
  --save_ranking_to $MODEL_OUT/$split_msmarco-passage.run
done


python eval_run.py --input $MODEL_OUT/dev_msmarco-passage.run \
      --dataset msmarco-passage/dev/judged --metrics $METRICS  \
      --output $RESULTS_DIR/dev_msmarco-passage.json

python eval_run.py --input $MODEL_OUT/dl19_msmarco-passage.run \
      --dataset msmarco-passage/trec-dl-2019/judged --metrics $METRICS  \
      --output $RESULTS_DIR/dl19.json

python eval_run.py --input $MODEL_OUT/dl20_msmarco-passage.run \
      --dataset msmarco-passage/trec-dl-2020/judged --metrics $METRICS  \
      --output $RESULTS_DIR/dl20.json


########################################################################################################################
#### SciFact ####
########################################################################################################################
# smaller batch size for SciFact
BATCH_SIZE=256

echo "encoding SciFact corpus"
python -m tevatron.driver.encode \
  --output_dir=$TEMP_DIR \
  --model_name_or_path $MODEL_NAME \
  --fp16 \
  --per_device_eval_batch_size $BATCH_SIZE \
  --p_max_len 512 \
  --dataset_name Tevatron/scifact-corpus \
  --encoded_save_path $MODEL_OUT/corpus_scifact.pkl \
  --cache_dir $MODEL_CACHE_DIR \
  --data_cache_dir $DATA_CACHE_DIR \

echo "eval scifact"

BATCH_SIZE=
TEST_SETS=('dev' 'test')
for split in "${TEST_SETS[@]}"
do
  echo "eval $split"
  # get embeds

  python -m tevatron.driver.encode \
  --output_dir=$TEMP_DIR \
  --model_name_or_path $MODEL_NAME \
  --fp16 \
  --per_device_eval_batch_size $BATCH_SIZE \
  --q_max_len 64 \
  --encode_is_qry \
  --dataset_name Tevatron/scifact/$split \
  --encoded_save_path $MODEL_OUT/$split_scifact.pkl \
  --cache_dir $MODEL_CACHE_DIR \
  --data_cache_dir $DATA_CACHE_DIR


  # obtain run
  python -m tevatron.faiss_retriever \
  --query_reps $MODEL_OUT/$split_scifact.pkl \
  --passage_reps $MODEL_OUT/corpus_scifact.pkl \
  --depth $TOP_K \
  --batch_size $BATCH_SIZE \
  --save_text \
  --save_ranking_to $MODEL_OUT/$split_scifact.run


done


## TODO: eval_run.py


BIER_DATASETS=( "fiqa" "trec-covid" "cqadupstack-android" "cqadupstack-english" "cqadupstack-gaming" "cqadupstack-gis" "cqadupstack-wordpress" "cqadupstack-physics" "cqadupstack-programmers" "cqadupstack-stats" "cqadupstack-tex" "cqadupstack-unix" "cqadupstack-webmasters" "cqadupstack-wordpress" )
BATCH_SIZE=128
for bds in "${BIER_DATASETS[@]}"
do
    echo "eval $bds"

    echo "encoding corpus"
    python -m tevatron.driver.encode \
    --output_dir=$TEMP_DIR \
    --model_name_or_path $MODEL_NAME \
    --fp16 \
    --per_device_eval_batch_size $BATCH_SIZE \
    --p_max_len 512 \
    --dataset_name Tevatron/beir-corpus:${bds} \
    --encoded_save_path $MODEL_OUT/corpus_bier_${bds}.pkl \
    --cache_dir $MODEL_CACHE_DIR \
    --data_cache_dir $DATA_CACHE_DIR

    echo "encoding test queries"
    python -m tevatron.driver.encode \
    --output_dir=$TEMP_DIR \
    --model_name_or_path $MODEL_NAME \
    --fp16 \
    --per_device_eval_batch_size $BATCH_SIZE \
    --dataset_name Tevatron/beir:${bds}/test \
    --encoded_save_path $MODEL_OUT/test_bier_${bds}.pkl \
    --q_max_len 512 \
    --encode_is_qry \
    --cache_dir $MODEL_CACHE_DIR \
    --data_cache_dir $DATA_CACHE_DIR

    echo "obtaining run"
    python -m tevatron.faiss_retriever \
    --query_reps $MODEL_OUT/test_bier_${bds}.pkl \
    --passage_reps $MODEL_OUT/corpus_bier_${bds}.pkl \
    --depth 1000 \
    --batch_size $BATCH_SIZE \
    --save_text \
    --save_ranking_to $MODEL_OUT/bier_test_${bds}.run


    ## TODO: eval_run.py
done