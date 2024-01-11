
# Scifact

srun -p gpu --gres=gpu:1 --mem=96G --time=24:00:00 --exclude=ilps-cn111,ilps-cn108 python -m tevatron.driver.encode \
  --output_dir=/ivi/ilps/personal/sbharga/temp_dir \
  --model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
  --fp16 \
  --per_device_eval_batch_size 256 \
  --p_max_len 512 \
  --dataset_name Tevatron/scifact-corpus \
  --encoded_save_path /ivi/ilps/personal/sbharga/mvrl/tas_b_zeroshot/corpus_scifact.pkl \
  --cache_dir /ivi/ilps/personal/sbharga/hf_model_cache \
  --data_cache_dir /ivi/ilps/personal/sbharga/hf_data_cache




srun -p gpu --gres=gpu:1 --mem=96G --time=24:00:00 --exclude=ilps-cn111,ilps-cn108 python -m tevatron.driver.encode \
  --output_dir=/ivi/ilps/personal/sbharga/temp_dir \
  --model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
  --fp16 \
  --per_device_eval_batch_size 256 \
  --q_max_len 64 \
  --encode_is_qry \
  --dataset_name Tevatron/scifact/dev \
  --encoded_save_path /ivi/ilps/personal/sbharga/mvrl/tas_b_zeroshot/dev_scifact.pkl \
  --cache_dir /ivi/ilps/personal/sbharga/hf_model_cache \
  --data_cache_dir /ivi/ilps/personal/sbharga/hf_data_cache

srun -p gpu --gres=gpu:1 --mem=96G --time=24:00:00 --exclude=ilps-cn111,ilps-cn108 python -m tevatron.driver.encode \
  --output_dir=/ivi/ilps/personal/sbharga/temp_dir \
  --model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
  --fp16 \
  --per_device_eval_batch_size 256 \
  --q_max_len 64 \
  --encode_is_qry \
  --dataset_name Tevatron/scifact/test \
  --encoded_save_path /ivi/ilps/personal/sbharga/mvrl/tas_b_zeroshot/test_scifact.pkl \
  --cache_dir /ivi/ilps/personal/sbharga/hf_model_cache \
  --data_cache_dir /ivi/ilps/personal/sbharga/hf_data_cache


srun -p gpu --gres=gpu:1 --mem=96G --time=24:00:00 --exclude=ilps-cn111,ilps-cn108 python -m tevatron.faiss_retriever \
  --query_reps /ivi/ilps/personal/sbharga/mvrl/tas_b_zeroshot/dev_scifact.pkl \
  --passage_reps /ivi/ilps/personal/sbharga/mvrl/tas_b_zeroshot/corpus_scifact.pkl  \
  --depth 100 \
  --batch_size 256 \
  --save_text \
  --save_ranking_to /ivi/ilps/personal/sbharga/mvrl/tas_b_zeroshot/dev_scifact.run

srun -p gpu --gres=gpu:1 --mem=96G --time=24:00:00 --exclude=ilps-cn111,ilps-cn108 python -m tevatron.faiss_retriever \
  --query_reps /ivi/ilps/personal/sbharga/mvrl/tas_b_zeroshot/test_scifact.pkl \
  --passage_reps /ivi/ilps/personal/sbharga/mvrl/tas_b_zeroshot/corpus_scifact.pkl  \
  --depth 100 \
  --batch_size 256 \
  --save_text \
  --save_ranking_to /ivi/ilps/personal/sbharga/mvrl/tas_b_zeroshot/test_scifact.run

python eval_run.py --inpt