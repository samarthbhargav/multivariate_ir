# Steps to Reproduce the Experiments in the Paper

## Setup 
```
# After Cloning Repo and cd'ing into repo
conda create --name multivariate_ir python=3.8
conda activate multivariate_ir
conda install faiss-gpu pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install -e .
pip install accelerate -U && pip install pytrec_eval ir_datasets notebook

```

Execute the following lines to prepare the datasets used in evaluation
```
python prepare_msmarco.py /ivi/ilps/projects/multivariate_ir/data /ivi/ilps/projects/multivariate_ir/.hf_data_cache
```

## Baseline models

### TAS-B (0Shot, pre-trained)

We used the [DistilBERT TAS-B pretrained on MS-MARCO](https://huggingface.co/sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco) for 
our zero-shot run. 


#### Hyperparam search
None

#### Obtaining run
Execute the following command to obtain the trained model:

```
sh run_scripts/tas_b_zeroshot.sh
``` 


### DR Model

#### Hyperparam search


```
python make_job.py --model_config hyperparams/dpr.json --job_config job_config/job_config_6jobs.json \
        --dest dpr_hs_db --exp_root_folder /home/sbhargav/multivariate_ir_experiments/experiments --exp_name dpr_hs_db \
        --job_template ./templates/job_template_snellius.sh --cmd "python -m tevatron.driver.train"

python best_model.py --experiment_path /home/sbhargav/multivariate_ir_experiments/experiments/dpr_hs_db

sh eval_snellius.sh /home/sbhargav/multivariate_ir_experiments/experiments/dpr_hs_db/3 \ 
                    /home/sbhargav/multivariate_ir_experiments/experiments/dpr_hs_db/3 \
                    "" \
                    /home/sbhargav/multivariate_ir_experiments/experiments/dpr_hs_db/3/eval_log.log 
```



#### Training

#### Obtaining run

### MVRL Model (no distillation)

#### Hyperparam search

```
python make_job.py --model_config hyperparams/mvrl_no_distill_db.json \
        --job_config job_config/job_config_12jobs.json \
        --dest mvrl_nd_db \
        --exp_root_folder /scratch-shared/sbhargav/multivariate_ir_experiments/experiments \
        --exp_name mvrl_nd_db \
        --job_template ./templates/job_template_snellius.sh \
        --cmd "python -m tevatron.driver.train"

```

TODO
#### Obtaining run



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
    
    
    mkdir -p datasets/actual_performances/
    
    python -u evaluation_retrieval.py  \
            --run datasets/trec-dl/runs/dl19-bm25-1000.txt \
            --qrel datasets/trec-dl/dl19/qrel.txt \
            --output_path datasets/actual_performances/dl19_bm25.json
    
    python -u evaluation_retrieval.py  \
            --run datasets/trec-dl/runs/dl20-bm25-1000.txt \
            --qrel datasets/trec-dl/dl20/qrel.txt \
            --output_path datasets/actual_performances/dl20_bm25.json  
 

    ```
   
   Convert run files from DPR & TASB to the required format as well:
   ```
        export QPP_METRIC="ndcg_cut_10"
        export QPP_METRIC_NAME="ndcg@10" 
        python -m qpp.convert_run_for_qpp --path runs/dpr/dl19.run \
            --output datasets/actual_performances/dl19_dpr.json \
            --metric ${QPP_METRIC} --metric_name ${QPP_METRIC_NAME}\
            --ir_dataset_name msmarco-passage/trec-dl-2019/judged
   
        python -m qpp.convert_run_for_qpp --path runs/dpr/dl20.run \
            --output datasets/actual_performances/dl20_dpr.json \
            --metric ${QPP_METRIC} --metric_name ${QPP_METRIC_NAME}\
            --ir_dataset_name msmarco-passage/trec-dl-2020/judged
   
       python -m qpp.convert_run_for_qpp --path runs/tasb/dl19.run \
                --output datasets/actual_performances/dl19_tasb.json \
                --metric ${QPP_METRIC} --metric_name ${QPP_METRIC_NAME}\
                --ir_dataset_name msmarco-passage/trec-dl-2019/judged
   
       python -m qpp.convert_run_for_qpp --path runs/tasb/dl20.run \
                --output datasets/actual_performances/dl20_tasb.json \
                --metric ${QPP_METRIC} --metric_name ${QPP_METRIC_NAME}\
                --ir_dataset_name msmarco-passage/trec-dl-2020/judged
 
   ```
   
3. Run Baselines & Methods
   
   Run QPP pre-retrieval baselines :
   ```
        # run this from the repo root
        sh run_scripts/qpp_baselines.sh
   ```
   
   Run QPP on MVRL models
   ```
    qpp_output/pre-retrieval/dl19/MVRL_
   ```
   
4. Evaluate
    ```
   python -m qpp.evaluate --actual bm25,datasets/actual_performances/dl19_bm25.json \
                           dpr,datasets/actual_performances/dl19_dpr.json \
                           tasb,datasets/actual_performances/dl19_tasb.json \
                           --predicted_dir qpp_output/pre-retrieval/dl19/ \
                           --metric ndcg@10 \
                           --output qpp_output/dl19.csv
   
   python -m qpp.evaluate --actual bm25,datasets/actual_performances/dl20_bm25.json \
                           dpr,datasets/actual_performances/dl20_dpr.json \
                           tasb,datasets/actual_performances/dl20_tasb.json \
                           --predicted_dir qpp_output/pre-retrieval/dl20/ \
                           --metric ndcg@10 \
                           --output qpp_output/dl20.csv
    ```
    
                           
   
