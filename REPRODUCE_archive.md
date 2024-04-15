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
srun -p gpu --time=01:00:00 --mem=24G --gres=gpu:a6000:1 sh run_scripts/tas_b_zeroshot.sh
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
python best_model.py /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/mvrl_nd_db/


python make_job.py --model_config hyperparams/mvrl_no_distill_logvar_db.json \
        --job_config job_config/job_config_6jobs.json \
        --dest mvrl_nd_db_logvar \
        --exp_root_folder /scratch-shared/sbhargav/multivariate_ir_experiments/experiments \
        --exp_name mvrl_nd_db_logvar \
        --job_template ./templates/job_template_snellius.sh \
        --cmd "python -m tevatron.driver.train"

```

MVRL with TAS-B checkpoint
```
srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:1 python -m tevatron.driver.train \
--do_train \
--do_eval  \
--model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir /scratch-shared/sbhargav/.hf_data_cache \
--cache_dir /scratch-shared/sbhargav/.hf_model_cache \
--train_dir /scratch-shared/sbhargav/data/msmarco/train \
--val_dir /scratch-shared/sbhargav/data/msmarco/validation \
--per_device_train_batch_size 14 \
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
--output_dir /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/mvrl_nd_tasb_2/ >> tasb_train_2.log 2>&1 &

```

MVRL with TAS-B checkpoint, using logvar activation function
```
srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:1 python -m tevatron.driver.train \
--do_train \
--do_eval  \
--model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir /projects/0/prjs0907/.hf_data_cache \
--cache_dir /projects/0/prjs0907/.hf_model_cache \
--train_dir /projects/0/prjs0907/data/msmarco/train \
--val_dir /projects/0/prjs0907/data/msmarco/validation \
--per_device_train_batch_size 14 \
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
--output_dir /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar/ >> mvrl_nd_tasb_logvar.log 2>&1 &

```


MVRL with TAS-B checkpoint, using logvar activation function. Trained with 512 batch size using gradient accumulation

max_steps with bz 14 = 150000
max_steps with bz 512 = x
--> x = (14 * 150000) / 512 ~= 4000

eval_steps = 500 (needs to divide 4k)
```
### SLURM version for testing
srun -p gpu --time=96:00:00 --mem=24G -c12 --gres=gpu:a6000:1 python -m tevatron.driver.train \
--do_train \
--do_eval  \
--model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir /ivi/ilps/projects/multivariate_ir/.hf_data_cache \
--cache_dir /ivi/ilps/projects/multivariate_ir/.hf_model_cache \
--train_dir /ivi/ilps/projects/multivariate_ir/data/msmarco-med/train \
--val_dir /ivi/ilps/projects/multivariate_ir/data/msmarco-med/validation \
--overwrite_output_dir \
--per_device_train_batch_size 512 \
--grad_cache \
--gc_q_chunk_size 400 \
--gc_p_chunk_size 400 \
--per_device_eval_batch_size 105 \
--train_n_passages 31 \
--q_max_len 32 \
--p_max_len 256 \
--max_steps 40 \
--evaluation_strategy steps \
--eval_steps 20 \
--save_steps 20 \
--metric_for_best_model mrr \
--warmup_ratio 0.1 \
--fp16  --fp16_full_eval  \
--exclude_title  \
--model_type mvrl_no_distill \
--add_var_token  \
--var_activation logvar \
--embed_formulation full_kl \
--negatives_x_device \
--learning_rate 7e-06 \
--output_dir /ivi/ilps/projects/multivariate_ir/experiments/mvrl_nd_tasb_logvar_512/ >> mvrl_nd_tasb_logvvar_512.log 2>&1 &

## Snellius version
srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:1 python -m tevatron.driver.train \
--do_train \
--do_eval  \
--model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir /projects/0/prjs0907/.hf_data_cache \
--cache_dir /projects/0/prjs0907/.hf_model_cache \
--train_dir /projects/0/prjs0907/data/msmarco/train \
--val_dir /projects/0/prjs0907/data/msmarco/validation \
--per_device_train_batch_size 512 \
--grad_cache \
--gc_q_chunk_size 300 \
--gc_p_chunk_size 300 \
--per_device_eval_batch_size 64 \
--train_n_passages 31 \
--q_max_len 32 \
--p_max_len 256 \
--max_steps 4000 \
--evaluation_strategy steps \
--eval_steps 500 \
--save_steps 500 \
--metric_for_best_model mrr \
--warmup_ratio 0.1 \
--fp16  --fp16_full_eval  \
--exclude_title  \
--model_type mvrl_no_distill \
--add_var_token  \
--embed_formulation full_kl \
--negatives_x_device \
--var_activation logvar \
--learning_rate 7e-06 \
--keep_data_in_memory \
--output_dir /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512/ >> mvrl_nd_tasb_logvvar_512.log 2>&1 &

srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:1 python -m tevatron.driver.train \
--do_train \
--do_eval  \
--model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir /projects/0/prjs0907/.hf_data_cache \
--cache_dir /projects/0/prjs0907/.hf_model_cache \
--train_dir /projects/0/prjs0907/data/msmarco/train \
--val_dir /projects/0/prjs0907/data/msmarco/validation \
--per_device_train_batch_size 256 \
--grad_cache \
--gc_q_chunk_size 300 \
--gc_p_chunk_size 300 \
--per_device_eval_batch_size 105 \
--train_n_passages 31 \
--q_max_len 32 \
--p_max_len 256 \
--max_steps 8000 \
--evaluation_strategy steps \
--eval_steps 1000 \
--save_steps 1000 \
--metric_for_best_model mrr \
--warmup_ratio 0.1 \
--fp16  --fp16_full_eval  \
--exclude_title  \
--model_type mvrl_no_distill \
--add_var_token  \
--embed_formulation full_kl \
--negatives_x_device \
--var_activation logvar \
--learning_rate 7e-06 \
--keep_data_in_memory \
--output_dir /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256/ >> mvrl_nd_tasb_logvvar_256.log 2>&1 &


srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:1 python -m tevatron.driver.train \
--do_train \
--do_eval  \
--model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir /projects/0/prjs0907/.hf_data_cache \
--cache_dir /projects/0/prjs0907/.hf_model_cache \
--train_dir /projects/0/prjs0907/data/msmarco/train \
--val_dir /projects/0/prjs0907/data/msmarco/validation \
--per_device_train_batch_size 128 \
--grad_cache \
--gc_q_chunk_size 300 \
--gc_p_chunk_size 300 \
--per_device_eval_batch_size 105 \
--train_n_passages 31 \
--q_max_len 32 \
--p_max_len 256 \
--max_steps 16000 \
--evaluation_strategy steps \
--eval_steps 2000 \
--save_steps 2000 \
--metric_for_best_model mrr \
--warmup_ratio 0.1 \
--fp16  --fp16_full_eval  \
--exclude_title  \
--model_type mvrl_no_distill \
--add_var_token  \
--embed_formulation full_kl \
--negatives_x_device \
--var_activation logvar \
--learning_rate 7e-06 \
--keep_data_in_memory \
--output_dir /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128/ >> mvrl_nd_tasb_logvvar_128.log 2>&1 &


# MultiGPU
LOCAL_RANK=0,1 CUDA_VISIBLE_DEVICES=0,1 srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:2 python -m torch.distributed.launch \
--nproc_per_node=6 --use-env -m tevatron.driver.train  \
--do_train \
--do_eval  \
--model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir /projects/0/prjs0907/.hf_data_cache \
--cache_dir /projects/0/prjs0907/.hf_model_cache \
--train_dir /projects/0/prjs0907/data/msmarco/train \
--val_dir /projects/0/prjs0907/data/msmarco/validation \
--per_device_train_batch_size 14 \
--per_device_eval_batch_size 105 \
--train_n_passages 31 \
--q_max_len 32 \
--p_max_len 256 \
--max_steps 75000 \
--evaluation_strategy steps \
--eval_steps 1000 \
--save_steps 1000 \
--negatives_x_device \
--metric_for_best_model mrr \
--warmup_ratio 0.1 \
--fp16  --fp16_full_eval  \
--exclude_title  \
--model_type mvrl_no_distill \
--add_var_token  \
--embed_formulation updated \
--var_activation logvar \
--learning_rate 7e-06 \
--keep_data_in_memory \
--output_dir /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_2gpu/ >> mvrl_nd_tasb_logvar_2gpu.log 2>&1 &



# SLURM
LOCAL_RANK=0,1 CUDA_VISIBLE_DEVICES=0,1 srun -p gpu --time=96:00:00 --mem=55G -c12 --gres=gpu:a6000:2 python -m torch.distributed.launch \
--nproc_per_node=2 --use-env -m tevatron.driver.train  \
--do_train \
--model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir /ivi/ilps/projects/multivariate_ir/.hf_data_cache \
--cache_dir /ivi/ilps/projects/multivariate_ir/.hf_model_cache \
--train_dir /ivi/ilps/projects/multivariate_ir/data/msmarco/train \
--val_dir /ivi/ilps/projects/multivariate_ir/data/msmarco/validation \
--per_device_train_batch_size 14 \
--per_device_eval_batch_size 105 \
--train_n_passages 31 \
--q_max_len 32 \
--p_max_len 256 \
--max_steps 75000 \
--logging_steps 150 \
--save_steps 1000 \
--negatives_x_device \
--metric_for_best_model mrr \
--warmup_ratio 0.1 \
--fp16  --fp16_full_eval  \
--exclude_title  \
--model_type mvrl_no_distill \
--add_var_token  \
--embed_formulation updated \
--var_activation logvar \
--learning_rate 7e-06 \
--keep_data_in_memory \
--output_dir /ivi/ilps/projects/multivariate_ir/experiments/mvrl_nd_tasb_logvar_2gpu/ >> mvrl_nd_tasb_logvar_2gpu.log 2>&1 &









```

MVRL, use embeds during training, rather than the full KL loss
```
srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:1 python -m tevatron.driver.train \
--do_train \
--do_eval  \
--model_name_or_path distilbert-base-uncased \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir /scratch-shared/sbhargav/.hf_data_cache \
--cache_dir /scratch-shared/sbhargav/.hf_model_cache \
--train_dir /scratch-shared/sbhargav/data/msmarco/train \
--val_dir /scratch-shared/sbhargav/data/msmarco/validation \
--per_device_train_batch_size 14 \
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
--output_dir /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/mvrl_nd_db_edt_updated/ >> mvrl_nd_db_edt_updated.log 2>&1 &

srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:1 python -m tevatron.driver.train \
--do_train \
--do_eval  \
--model_name_or_path distilbert-base-uncased \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir /scratch-shared/sbhargav/.hf_data_cache \
--cache_dir /scratch-shared/sbhargav/.hf_model_cache \
--train_dir /scratch-shared/sbhargav/data/msmarco/train \
--val_dir /scratch-shared/sbhargav/data/msmarco/validation \
--per_device_train_batch_size 14 \
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
--output_dir /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/mvrl_nd_db_edt_original/ >> mvrl_nd_db_edt_original.log 2>&1 &

``` 

MC-Dropout model (1) base layer not frozen (2) base layer frozen (3) tas-b not frozen (4) tas-b frozen
```
srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:1 python -m tevatron.driver.train \
--do_train  \
--do_eval  \
--model_name_or_path distilbert-base-uncased \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir /projects/0/prjs0907/.hf_data_cache \
--cache_dir /projects/0/prjs0907/.hf_model_cache \
--train_dir /projects/0/prjs0907/data/msmarco/train \
--val_dir /projects/0/prjs0907/data/msmarco/validation \
--per_device_train_batch_size 14 \
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
--fp16  \
--fp16_full_eval  \
--exclude_title  \
--model_type stochastic \
--learning_rate 7e-06 \
--output_dir /projects/0/prjs0907/multivariate_ir_experiments/experiments/stoch_db/ >> stoch_db.log 2>&1 &


srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:1 python -m tevatron.driver.train \
--do_train  \
--do_eval  \
--model_name_or_path distilbert-base-uncased \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir /projects/0/prjs0907/.hf_data_cache \
--cache_dir /projects/0/prjs0907/.hf_model_cache \
--train_dir /projects/0/prjs0907/data/msmarco/train \
--val_dir /projects/0/prjs0907/data/msmarco/validation \
--per_device_train_batch_size 14 \
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
--fp16  \
--fp16_full_eval  \
--exclude_title  \
--model_type stochastic \
--freeze_base_model \
--learning_rate 7e-06 \
--output_dir /projects/0/prjs0907/multivariate_ir_experiments/experiments/stoch_db_frozen/ >> stoch_db_frozen.log 2>&1 &


srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:1 python -m tevatron.driver.train \
--do_train  \
--do_eval  \
--model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir /projects/0/prjs0907/.hf_data_cache \
--cache_dir /projects/0/prjs0907/.hf_model_cache \
--train_dir /projects/0/prjs0907/data/msmarco/train \
--val_dir /projects/0/prjs0907/data/msmarco/validation \
--per_device_train_batch_size 14 \
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
--fp16  \
--fp16_full_eval  \
--exclude_title  \
--model_type stochastic \
--learning_rate 7e-06 \
--output_dir /projects/0/prjs0907/multivariate_ir_experiments/experiments/stoch_tasb/ >> stoch_tasb.log 2>&1 &


srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:1 python -m tevatron.driver.train \
--do_train  \
--do_eval  \
--model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir /projects/0/prjs0907/.hf_data_cache \
--cache_dir /projects/0/prjs0907/.hf_model_cache \
--train_dir /projects/0/prjs0907/data/msmarco/train \
--val_dir /projects/0/prjs0907/data/msmarco/validation \
--per_device_train_batch_size 14 \
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
--fp16  \
--fp16_full_eval  \
--exclude_title  \
--model_type stochastic \
--freeze_base_model \
--learning_rate 7e-06 \
--output_dir /projects/0/prjs0907/multivariate_ir_experiments/experiments/stoch_tasb_frozen/ >> stoch_tasb_frozen.log 2>&1 &

```


MVRL ND, TASB With corrected dimensionality to make eventual dim ~= 768 & fixed to 768
```
srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:1 python -m tevatron.driver.train \
--do_train \
--do_eval  \
--model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
--dataset_name Tevatron/msmarco-passage \
--data_cache_dir /projects/0/prjs0907/.hf_data_cache \
--cache_dir /projects/0/prjs0907/.hf_model_cache \
--train_dir /projects/0/prjs0907/data/msmarco/train \
--val_dir /projects/0/prjs0907/data/msmarco/validation \
--per_device_train_batch_size 14 \
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
--output_dir /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_dim255/ >> mvrl_nd_tasb_dim255.log 2>&1 &

```


#### Evaluation

```

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db/14 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db/14/original \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation original --var_activation softplus --var_activation_param_b 2.5" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db/14/original/eval.log 

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db/14 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db/14/updated \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation softplus --var_activation_param_b 2.5" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db/14/updated/eval.log

srun -p gpu --gres=gpu:1 --mem=24G -c12 --time=48:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db/14 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db/14/mean \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation mean --var_activation softplus --var_activation_param_b 2.5" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db/14/mean/eval.log 

 

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_2 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_2/original \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation original --var_activation softplus --var_activation_param_b 2.5" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_2/original/eval.log   

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh eval_snellius.sh \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/mvrl_nd_tasb_2 \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/mvrl_nd_tasb_2/updated \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation softplus --var_activation_param_b 2.5" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_2/updated/eval.log

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_2 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_2/mean \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation mean --var_activation softplus --var_activation_param_b 2.5" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_2/mean/eval.log &

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db_logvar/3 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db_logvar/3/original \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation original --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db_logvar/3/original/eval.log  

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db_logvar/3 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db_logvar/3/updated \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db_logvar/3/updated/eval.log

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db_logvar/3 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db_logvar/3/mean/ \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation mean --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db_logvar/3/mean/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db_edt_updated \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db_edt_updated/updated \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation softplus --var_activation_param_b 2.5 " \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db_edt_updated/updated/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db_edt_original \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db_edt_original/original \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation original --var_activation softplus --var_activation_param_b 2.5 " \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db_edt_original/original/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar/updated \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar/updated/eval.log &

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar/original \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation original --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar/original/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar/mean \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation mean --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar/mean/eval.log &

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_2gpu \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_2gpu/updated \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_2gpu/updated/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512/updated \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512/updated/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512/mean \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation mean --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512/mean/eval.log &

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512/original \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation original --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512/original/eval.log &



srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256/updated \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256/updated/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256/mean \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation mean --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256/mean/eval.log &

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256/original \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation original --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256/original/eval.log &




srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128/updated \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128/updated/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128/mean \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation mean --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128/mean/eval.log &

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128/original \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation original --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128/original/eval.log &



srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/stoch_db \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/stoch_db \
            "--model_type stochastic " \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/stoch_db/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/stoch_db_frozen \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/stoch_db_frozen \
            "--model_type stochastic " \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/stoch_db_frozen/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=24:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/stoch_tasb_frozen \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/stoch_tasb_frozen \
            "--model_type stochastic " \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/stoch_tasb_frozen/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/stoch_tasb \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/stoch_tasb \
            "--model_type stochastic " \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/stoch_tasb/eval.log &


srun -p gpu --gres=gpu:1 --mem=240G -c12 --time=96:00:00 sh eval_snellius.sh \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/stoch_db \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/stoch_db/mrl \
            "--model_type stochastic_mrl " \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/stoch_db/mrl/eval.log &


srun -p gpu --gres=gpu:1 --mem=240G -c12 --time=96:00:00 sh eval_snellius.sh \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/stoch_db_frozen \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/stoch_db_frozen/mrl \
            "--model_type stochastic_mrl " \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/stoch_db_frozen/mrl/eval.log &


srun -p gpu --gres=gpu:1 --mem=240G -c12 --time=24:00:00 sh eval_snellius.sh \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/stoch_tasb_frozen \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/stoch_tasb_frozen/mrl \
            "--model_type stochastic_mrl " \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/stoch_tasb_frozen/mrl/eval.log &


srun -p gpu --gres=gpu:1 --mem=240G -c12 --time=96:00:00 sh eval_snellius.sh \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/stoch_tasb \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/stoch_tasb/mrl \
            "--model_type stochastic_mrl " \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/stoch_tasb/mrl/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=24:00:00 sh eval_snellius.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl/mean \
            "--model_type mvrl --add_var_token  --embed_formulation mean --var_activation softplus --var_activation_param_b 2.5" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl/mean/eval.log &





srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=24:00:00 sh eval_snellius.sh \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/mvrl_nd_tasb_dim255 \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/mvrl_nd_tasb_dim255/updated \
            "--model_type mvrl_no_distill --projection_dim 255 --add_var_token  --embed_formulation updated --var_activation softplus --var_activation_param_b 2.5 " \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/mvrl_nd_tasb_dim255/updated/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=24:00:00 sh eval_snellius.sh \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/mvrl_nd_tasb_dim255 \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/mvrl_nd_tasb_dim255/original \
            "--model_type mvrl_no_distill --projection_dim 255 --add_var_token  --embed_formulation original --var_activation softplus --var_activation_param_b 2.5 " \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/mvrl_nd_tasb_dim255/original/eval.log &

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=24:00:00 sh eval_snellius.sh \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/mvrl_nd_tasb_dim255 \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/mvrl_nd_tasb_dim255/mean \
            "--model_type mvrl_no_distill --projection_dim 255 --add_var_token  --embed_formulation mean --var_activation softplus --var_activation_param_b 2.5 " \
            /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/mvrl_nd_tasb_dim255/mean/eval.log &



```



```
srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_slurm.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512/updated \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512/updated/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_slurm.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512/mean \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation mean --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512/mean/eval.log &

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_slurm.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512/original \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation original --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_512/original/eval.log &



srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_slurm.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256/updated \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256/updated/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_slurm.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256/mean \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation mean --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256/mean/eval.log &

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_slurm.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256/original \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation original --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_256/original/eval.log &




srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_slurm.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128/updated \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128/updated/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_slurm.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128/mean \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation mean --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128/mean/eval.log &

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=96:00:00 sh eval_slurm.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128/original \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation original --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar_128/original/eval.log &




```


#### QPP Evaluation


```

srun -p gpu --gres=gpu:1 --mem=24G --time=12:00:00 qpp_eval_all.sh 

```

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
   
    
    python -m qpp.convert_run_for_qpp --path runs/mvrl/dl19_msmarco-passage.run \
            --output datasets/actual_performances/dl19_mvrl.json \
            --metric ${QPP_METRIC} --metric_name ${QPP_METRIC_NAME}\
            --ir_dataset_name msmarco-passage/trec-dl-2019/judged
    
    python -m qpp.convert_run_for_qpp --path runs/mvrl/dl20_msmarco-passage.run \
            --output datasets/actual_performances/dl20_mvrl.json \
            --metric ${QPP_METRIC} --metric_name ${QPP_METRIC_NAME}\
            --ir_dataset_name msmarco-passage/trec-dl-2020/judged
   
    python -m qpp.convert_run_for_qpp --path runs/mvrl/dev_msmarco-passage.run \
            --output datasets/actual_performances/dev_mvrl.json \
            --metric ${QPP_METRIC} --metric_name ${QPP_METRIC_NAME}\
            --ir_dataset_name msmarco-passage/dev/small
   
    
 
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
    
    
   
           
   python -m qpp.evaluate_one --actual bm25,datasets/actual_performances/dl19_bm25.json \
           dpr,datasets/actual_performances/dl19_dpr.json \
           tasb,datasets/actual_performances/dl19_tasb.json \
           mvrl_nd,datasets/actual_performances/dl19_mvrl_nd.json \
           --predicted /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar/qpp-25000/msmarco-dl19.txt \
           --metric ndcg@10 
           
   python -m qpp.evaluate_one --actual bm25,datasets/actual_performances/dl19_bm25.json \
           dpr,datasets/actual_performances/dl19_dpr.json \
           tasb,datasets/actual_performances/dl19_tasb.json \
           mvrl_nd,datasets/actual_performances/dl19_mvrl_nd.json \
           --predicted /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar/qpp-50000/msmarco-dl19.txt \
           --metric ndcg@10
           
           
   python -m qpp.evaluate_one --actual bm25,datasets/actual_performances/dl19_bm25.json \
           dpr,datasets/actual_performances/dl19_dpr.json \
           tasb,datasets/actual_performances/dl19_tasb.json \
           mvrl_nd,datasets/actual_performances/dl19_mvrl_nd.json \
           --predicted /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar/qpp-100000/msmarco-dl19.txt \
           --metric ndcg@10
   
   python -m qpp.evaluate_one --actual bm25,datasets/actual_performances/dl19_bm25.json \
           dpr,datasets/actual_performances/dl19_dpr.json \
           tasb,datasets/actual_performances/dl19_tasb.json \
           mvrl_nd,datasets/actual_performances/dl19_mvrl_nd.json \
           --predicted /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar/qpp-125000/msmarco-dl19.txt \
           --metric ndcg@10
           
   python -m qpp.evaluate_one --actual bm25,datasets/actual_performances/dl19_bm25.json \
           dpr,datasets/actual_performances/dl19_dpr.json \
           tasb,datasets/actual_performances/dl19_tasb.json \
           mvrl_nd,datasets/actual_performances/dl19_mvrl_nd.json \
           --predicted /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar/qpp-150000/msmarco-dl19.txt \
           --metric ndcg@10
                           
   
   python -m qpp.evaluate_one --actual bm25,datasets/actual_performances/dl20_bm25.json \
           dpr,datasets/actual_performances/dl20_dpr.json \
           tasb,datasets/actual_performances/dl20_tasb.json \
           mvrl_nd,datasets/actual_performances/dl20_mvrl_nd.json \
           --predicted /ivi/ilps/projects/multivariate_ir/experiments_gs/Snellius/MVRL_TASB_MiniLM_pseudolabels_CL_1_b_25_lr_5106_embed_during_train_scale/qpp/msmarco-dl20.txt \
           --metric ndcg@10
           
   python -m qpp.evaluate_one --actual bm25,datasets/actual_performances/dl20_bm25.json \
           dpr,datasets/actual_performances/dl20_dpr.json \
           tasb,datasets/actual_performances/dl20_tasb.json \
           mvrl_nd,datasets/actual_performances/dl20_mvrl_nd.json \
           --predicted /ivi/ilps/projects/multivariate_ir/experiments_gs/Snellius/MVRL_TASB_MiniLM_pseudolabels_CL_1_b_25_lr_5106_embed_during_train_clamp/qpp/msmarco-dl20.txt \
           --metric ndcg@10

### Get representations for visualizations

```
srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh get_representations.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db/14 \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db/14/rep/ \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation original --var_activation softplus --var_activation_param_b 2.5" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_db/14/rep/rep.log


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh get_representations.sh \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar/rep/ \
            "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation logvar" \
            /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl_nd_tasb_logvar/rep/rep.log


```                           
   
### Data Prep for perturbation
```


```

