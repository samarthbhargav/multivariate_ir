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
#### SciFact ####
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


