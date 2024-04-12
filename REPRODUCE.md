# Steps to Reproduce the Experiments in the Paper

## Setup 
```
# After Cloning Repo and cd'ing into repo
conda create --name multivariate_ir python=3.8
conda activate multivariate_ir
conda install faiss-gpu pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install -e .
pip install accelerate -U && pip install pytrec_eval ir_datasets notebook

## for installing GradCache
git clone https://github.com/luyug/GradCache
cd GradCache 
pip install .
cd ..

```

Execute the following lines to prepare the datasets used in evaluation
```
DATA_DIR=/path/to/data
HF_CACHE=/path/to/hf/cache
python prepare_msmarco.py $DATA_DIR $HF_CACHE 

## setup experiment root
EXP_ROOT=/path/to/exp/root
```

## Baseline models

### TAS-B (pre-trained)

We used the [DistilBERT TAS-B pretrained on MS-MARCO](https://huggingface.co/sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco) for 
our zero-shot run. 

#### Hyperparam search
None

#### Obtaining run
Execute the following lines to obtain the trained model:

```
MODEL_PATH_OR_NAME=sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco
MODEL_OUT=$EXP_ROOT/tasb_b_zeroshot/
EXTRA_ARGS=""
LOG_FILE=$EXP_ROOT/tasb_b_zeroshot/eval.log
export MODEL_PATH_OR_NAME=${MODEL_PATH_OR_NAME}
export MODEL_OUT=${MODEL_OUT}
export EXTRA_ARGS=${EXTRA_ARGS}
export LOG_FILE=${LOG_FILE}

/bin/bash eval.sh
``` 


### DPR Model

#### Hyperparam search

```
cd params
python make_job.py --model_config hyperparams/dpr.json --job_config job_config/job_config_6jobs.json \
        --dest dpr_hs_db --exp_root_folder $EXP_ROOT --exp_name dpr_hs_db \
        --job_template ./templates/job_template.sh --cmd "python -m tevatron.driver.train"
sbatch dpr_hs_db.job

cd ../

# after job finishes, find the path to the best model
python best_model.py --experiment_path $EXP_ROOT/dpr_hs_db

BEST_MODEL=/path/to/model
sh run_scripts/eval.sh $BEST_MODEL \ 
           $BEST_MODEL \
           "" \
           $BEST_MODEL/eval.log 
```



### MRL Model (no distillation)

#### Hyperparam search

```
cd params
python make_job.py --model_config hyperparams/mvrl_no_distill_tasb.json \
        --job_config job_config/job_config.json \
        --dest mvrl_nd \
        --exp_root_folder $EXP_ROOT \
        --exp_name mvrl_nd \
        --job_template ./templates/job_template.sh \
        --cmd "python -m tevatron.driver.train"

sbatch mvrl_nd.job

cd ../

# after job finishes, find the path to the best model, and save the extra params to EXTRA_PARAMS
python best_model.py --experiment_path $EXP_ROOT/mvrl_nd
# additional params to pass to the model e.g., the \beta used to train the original model 
EXTRA_PARAMS="--var_activation_param_b BEST_BETA --OTHER_PARAM VAL "
BEST_MODEL=/path/to/model
sh run_scripts/eval.sh $BEST_MODEL \ 
           $BEST_MODEL \
           $EXTRA_PARAMS \
           $BEST_MODEL/eval.log
```

If you want to skip the hyper-param search:

MVRL with TAS-B checkpoint
```
python -m tevatron.driver.train \
--do_train \
--do_eval  \
--model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir $HF_CACHE \
--cache_dir $HF_CACHE \
--train_dir $DATA_DIR/train \
--val_dir $DATA_DIR/validation \
--per_device_train_batch_size 15 \
--per_device_eval_batch_size 105 \
--train_n_passages 31 \
--q_max_len 32 \
--p_max_len 256 \
--max_steps 150000 \
--evaluation_strategy steps \
--eval_steps 25000 \
--save_steps 25000 \
--metric_for_best_model mrr \
--disable_distributed  \
--warmup_ratio 0.1 \
--fp16  --fp16_full_eval  \
--exclude_title  \
--model_type mvrl_no_distill \
--add_var_token  \
--embed_formulation updated \
--var_activation softplus \
--learning_rate 7e-06 \
--var_activation_param_b 2.5 \
--keep_data_in_memory \
--output_dir $EXP_ROOT/mrl_nd_model/

```

MVRL with TAS-B checkpoint, using logvar activation function
```
python -m tevatron.driver.train \
--do_train \
--do_eval  \
--model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir $HF_CACHE \
--cache_dir $HF_CACHE \
--train_dir $DATA_DIR/train \
--val_dir $DATA_DIR/validation \
--per_device_train_batch_size 15 \
--per_device_eval_batch_size 105 \
--train_n_passages 31 \
--q_max_len 32 \
--p_max_len 256 \
--max_steps 150000 \
--evaluation_strategy steps \
--eval_steps 25000 \
--save_steps 25000 \
--metric_for_best_model mrr \
--disable_distributed  \
--warmup_ratio 0.1 \
--fp16  --fp16_full_eval  \
--exclude_title  \
--model_type mvrl_no_distill \
--add_var_token  \
--embed_formulation updated \
--var_activation logvar \
--learning_rate 7e-06 \
--keep_data_in_memory \
--output_dir $EXP_ROOT/mrl_nd_lv_model/

```

Alternative training strategy: MVRL, use embedding formulation during training, rather than the full KL loss. To use the (incorrect) formulation in the original paper,
use `--embed_formulation original` instead

```
python -m tevatron.driver.train \
--do_train \
--do_eval  \
--model_name_or_path distilbert-base-uncased \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir $HF_CACHE \
--cache_dir $HF_CACHE \
--train_dir $DATA_DIR/train \
--val_dir $DATA_DIR/validation \
--per_device_train_batch_size 15 \
--per_device_eval_batch_size 105 \
--train_n_passages 31 \
--q_max_len 32 \
--p_max_len 256 \
--max_steps 150000 \
--evaluation_strategy steps \
--eval_steps 25000 \
--save_steps 25000 \
--metric_for_best_model mrr \
--disable_distributed  \
--warmup_ratio 0.1 \
--fp16  --fp16_full_eval  \
--exclude_title  \
--model_type mvrl_no_distill \
--add_var_token  \
--embed_during_train \
--embed_formulation updated \
--var_activation softplus \
--learning_rate 7e-06 \
--var_activation_param_b 2.5 \
--keep_data_in_memory \
--output_dir $EXP_ROOT/mvrl_nd_edt/
``` 



MVRL ND, TASB With corrected dimensionality to make eventual dim ~= 768 & fixed to 768
```
python -m tevatron.driver.train \
--do_train \
--do_eval  \
--model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir $HF_CACHE \
--cache_dir $HF_CACHE \
--train_dir $DATA_DIR/train \
--val_dir $DATA_DIR/validation \
--per_device_train_batch_size 15 \
--per_device_eval_batch_size 105 \
--train_n_passages 31 \
--q_max_len 32 \
--p_max_len 256 \
--max_steps 150000 \
--evaluation_strategy steps \
--eval_steps 25000 \
--save_steps 25000 \
--metric_for_best_model mrr \
--disable_distributed  \
--warmup_ratio 0.1 \
--fp16  --fp16_full_eval  \
--exclude_title  \
--model_type mvrl_no_distill \
--add_var_token  \
--embed_formulation updated \
--var_activation softplus \
--learning_rate 7e-06 \
--var_activation_param_b 2.5 \
--keep_data_in_memory \
--projection_dim 255 \
--output_dir $EXP_ROOT/mvrl_255/

```

### MRL (distillation)



### CLDRD


## QPP 

1. Preprocess: 
    
    ```
        python -m qpp.preprocess --path datasets/trec-dl/
    ```
   
2. Run QPP on the MVRL models:
    ```
    TODO
    ```
3. Build actual performance files:

    BM25:
    ```
       
    python -m pyserini.index.lucene \
            --collection JsonCollection \
            --generator DefaultLuceneDocumentGenerator \
            --threads 16 \
            --storePositions \
            --storeDocvectors \
            --storeRaw \
            -input datasets/trec-dl/corpus/ \
            -index datasets/trec-dl/corpus_index/
    
    
    python -m pyserini.search.lucene \
            --bm25 --hits 1000 --threads 16 --batch-size 64 \
            --index datasets/trec-dl/corpus_index/ \
            --topics datasets/trec-dl/dl19/queries.tsv \
            --output datasets/trec-dl/runs/dl19-bm25-1000.txt
    
    python -m pyserini.search.lucene \
            --bm25 --hits 1000 --threads 16 --batch-size 64 \
            --index datasets/trec-dl/corpus_index/ \
            --topics datasets/trec-dl/dl20/queries.tsv \
            --output datasets/trec-dl/runs/dl20-bm25-1000.txt
   
   python -m pyserini.search.lucene \
            --bm25 --hits 1000 --threads 16 --batch-size 64 \
            --index datasets/trec-dl/corpus_index/ \
            --topics datasets/trec-dl/dev/queries.tsv \
            --output datasets/trec-dl/runs/dev-bm25-1000.txt
    
    
    mkdir -p datasets/actual_performances/
    
    python -u qpp.evaluation_retrieval  \
            --run datasets/trec-dl/runs/dl19-bm25-1000.txt \
            --qrel datasets/trec-dl/dl19/qrel.txt \
            --output_path datasets/actual_performances/dl19_bm25.json
    
    python -u qpp.evaluation_retrieval  \
            --run datasets/trec-dl/runs/dl20-bm25-1000.txt \
            --qrel datasets/trec-dl/dl20/qrel.txt \
            --output_path datasets/actual_performances/dl20_bm25.json
   
   python -m qpp.evaluation_retrieval  \
            --run datasets/trec-dl/runs/dev-bm25-1000.txt \
            --qrel datasets/trec-dl/dev/qrel.txt \
            --output_path datasets/actual_performances/dev_bm25.json  
 

    ```
   
   Convert run files from DPR & TASB to the required format as well:
   ```
    export QPP_METRIC="ndcg_cut_10"
    export QPP_METRIC_NAME="ndcg@10" 
    python -m qpp.convert_run_for_qpp --path runs/dpr/dl19_msmarco-passage.run \
        --output datasets/actual_performances/dl19_dpr.json \
        --metric ${QPP_METRIC} --metric_name ${QPP_METRIC_NAME}\
        --ir_dataset_name msmarco-passage/trec-dl-2019/judged
    
    python -m qpp.convert_run_for_qpp --path runs/dpr/dl20_msmarco-passage.run \
        --output datasets/actual_performances/dl20_dpr.json \
        --metric ${QPP_METRIC} --metric_name ${QPP_METRIC_NAME}\
        --ir_dataset_name msmarco-passage/trec-dl-2020/judged
    
    python -m qpp.convert_run_for_qpp --path runs/dpr/dev_msmarco-passage.run \
        --output datasets/actual_performances/dev_dpr.json \
        --metric ${QPP_METRIC} --metric_name ${QPP_METRIC_NAME}\
        --ir_dataset_name msmarco-passage/dev/small
    
    python -m qpp.convert_run_for_qpp --path runs/tasb/dl19_msmarco-passage.run \
            --output datasets/actual_performances/dl19_tasb.json \
            --metric ${QPP_METRIC} --metric_name ${QPP_METRIC_NAME}\
            --ir_dataset_name msmarco-passage/trec-dl-2019/judged
    
    python -m qpp.convert_run_for_qpp --path runs/tasb/dl20_msmarco-passage.run \
            --output datasets/actual_performances/dl20_tasb.json \
            --metric ${QPP_METRIC} --metric_name ${QPP_METRIC_NAME}\
            --ir_dataset_name msmarco-passage/trec-dl-2020/judged
   
    python -m qpp.convert_run_for_qpp --path runs/tasb/dev_msmarco-passage.run \
            --output datasets/actual_performances/dev_tasb.json \
            --metric ${QPP_METRIC} --metric_name ${QPP_METRIC_NAME}\
            --ir_dataset_name msmarco-passage/dev/small
   
    
 
 
   ```
  

   
3. Run Baselines & Methods
   
   Run QPP pre-retrieval baselines :
   ```
        # run this from the repo root
        sh run_scripts/qpp_baselines.sh
   ```
     
4. Evaluate
    ```
   python -m qpp.evaluate --actual bm25,datasets/actual_performances/dl19_bm25.json \
                           dpr,datasets/actual_performances/dl19_dpr.json \
                           tasb,datasets/actual_performances/dl19_tasb.json \
                           mvrl_nd,datasets/actual_performances/dl19_mvrl_nd.json \
                           --predicted_dir qpp_output/pre-retrieval/dl19/ \
                           --metric ndcg@10 \
                           --output qpp_output/dl19.csv
   
   python -m qpp.evaluate --actual bm25,datasets/actual_performances/dl20_bm25.json \
                           dpr,datasets/actual_performances/dl20_dpr.json \
                           tasb,datasets/actual_performances/dl20_tasb.json \
                           mvrl_nd,datasets/actual_performances/dl20_mvrl_nd.json \
                           --predicted_dir qpp_output/pre-retrieval/dl20/ \
                           --metric ndcg@10 \
                           --output qpp_output/dl20.csv
    ```
    
 


### MRL-QPP Evaluation

To evaluate the QPP for the MRL model, first generate the QPP results:
```
sh qpp_eval_model.sh \
  /path/to/model \
  /path/to/output/ \
  "extra parameters " \
  /path/to/logfile
```

## Gathering Results

1. Run `python gather_results.py --input_dir $EXP_ROOT --output_dir notebooks/gathered_results`

Run `notebooks/gather_results_final.ipynb`