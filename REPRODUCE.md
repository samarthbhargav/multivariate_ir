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
```



#### Training

#### Obtaining run

### MVRL Model (no distillation)

#### Hyperparam search

```
python make_job.py --model_config hyperparams/mvrl_no_distill_db.json --job_config job_config/job_config_2jobs.json \
        --dest mvrl_nd_db --exp_root_folder /home/sbhargav/multivariate_ir_experiments/experiments --exp_name mvrl_nd_db \
        --job_template ./templates/job_template_snellius.sh --cmd "python -m tevatron.driver.train"

python make_job.py --model_config hyperparams/mvrl_no_distill_tasb.json --job_config job_config/job_config_2jobs.json \
        --dest mvrl_nd_tasb --exp_root_folder /home/sbhargav/multivariate_ir_experiments/experiments --exp_name mvrl_nd_tasb \
        --job_template ./templates/job_template_snellius.sh --cmd "python -m tevatron.driver.train"
```

TODO
#### Obtaining run




