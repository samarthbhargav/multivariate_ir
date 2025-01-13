

```
cp vocab.txt tokenizer_config.json tokenizer.json special_tokens_map.json config.json 
```


```


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh eval_snellius.sh \
    /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_stg3_gc_32 \
    /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_stg3_gc_32 \
    "" \
    /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_stg3_gc_32/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh eval_snellius.sh \
    /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_stg3_gc_64 \
    /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_stg3_gc_64 \
    "" \
    /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_stg3_gc_64/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh eval_snellius.sh \
    /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_stg3_gc_128 \
    /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_stg3_gc_128 \
    "" \
    /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_stg3_gc_128/eval.log &

srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh eval_snellius.sh \
    /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_stg3_gc_256 \
    /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_stg3_gc_256 \
    "" \
    /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_stg3_gc_256/eval.log &


srun -p gpu --gres=gpu:1 --mem=120G -c12 --time=48:00:00 sh eval_snellius.sh \
    /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_stg3_gc_512 \
    /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_stg3_gc_512 \
    "" \
    /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_stg3_gc_512/eval.log &



```


## 512 

### Stage 1

```
CUDA_VISIBLE_DEVICES=0 srun -p gpu --gres=gpu:1 --mem=120G --time=99:00:00 -c12 python -m tevatron.driver.train_DRD \
  --model_name_or_path sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
  --teacher_model_name_or_path cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --evaluation_strategy steps \
  --do_train \
  --do_eval \
  --exclude_title \
  --kd_type cldrd \
  --pseudolabels \
  --ann_neg_num 30 \
  --group_1 5 \
  --group_2 12 \
  --group_3 13 \
  --group_1_size 5 \
  --group_2_size 45 \
  --group_3_size 150 \
  --train_n_passages 1 \
  --grad_cache \
  --gc_q_chunk_size 15 \
  --gc_p_chunk_size 15 \
  --dataset_name Tevatron/msmarco-passage \
  --train_dir /scratch-shared/sbhargav/data/train_reranked_MiniLM_200_cmp_no_title \
  --val_dir /scratch-shared/sbhargav/data/validation \
  --fp16 \
  --fp16_full_eval \
  --learning_rate 5e-6 \
  --q_max_len 32 \
  --p_max_len 256 \
  --warmup_ratio 0.1 \
  --logging_steps 15 \
  --evaluation_strategy steps \
  --cache_dir /scratch-shared/sbhargav/data/cache_models \
  --data_cache_dir /scratch-shared/sbhargav/data/train_reranked_MiniLM_200_cmp_no_title \
  --disable_distributed \
  --output_dir /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_gc_512_2 \
  --per_device_train_batch_size 512 \
  --max_steps 2941  \
  --eval_steps 735  \
  --save_steps 735 
```

### Stage 2

```
CUDA_VISIBLE_DEVICES=0 srun -p gpu --gres=gpu:1 --mem=120G --time=99:00:00 -c12 python -m tevatron.driver.train_DRD \
  --teacher_model_name_or_path cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --evaluation_strategy steps \
  --do_train \
  --do_eval \
  --exclude_title \
  --kd_type cldrd \
  --pseudolabels \
  --ann_neg_num 30 \
  --group_1 10 \
  --group_2 10 \
  --group_3 10 \
  --group_1_size 10 \
  --group_2_size 40 \
  --group_3_size 150 \
  --train_n_passages 1 \
  --grad_cache \
  --gc_q_chunk_size 15 \
  --gc_p_chunk_size 15 \
  --dataset_name Tevatron/msmarco-passage \
  --train_dir /scratch-shared/sbhargav/data/train_reranked_MiniLM_200_cmp_no_title \
  --val_dir /scratch-shared/sbhargav/data/validation \
  --fp16 \
  --fp16_full_eval \
  --learning_rate 5e-6 \
  --q_max_len 32 \
  --p_max_len 256 \
  --warmup_ratio 0.1 \
  --logging_steps 15 \
  --evaluation_strategy steps \
  --cache_dir /scratch-shared/sbhargav/data/cache_models \
  --data_cache_dir /scratch-shared/sbhargav/data/train_reranked_MiniLM_200_cmp_no_title \
  --disable_distributed \
  --model_name_or_path /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_gc_512/TODO/ \
  --load_model_from_disk \
  --output_dir /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_stg2_gc_512 \
  --per_device_train_batch_size 512 \
  --max_steps 1472  \
  --eval_steps 735  \
  --save_steps 735


```


### Stage 3

```

```


## 256

### Stage 1 

```

```

### Stage 2

```




CUDA_VISIBLE_DEVICES=0 srun -p gpu --gres=gpu:1 --mem=120G --time=99:00:00 -c12 python -m tevatron.driver.train_DRD \
  --teacher_model_name_or_path cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --evaluation_strategy steps \
  --do_train \
  --do_eval \
  --exclude_title \
  --kd_type cldrd \
  --pseudolabels \
  --ann_neg_num 30 \
  --group_1 10 \
  --group_2 10 \
  --group_3 10 \
  --group_1_size 10 \
  --group_2_size 40 \
  --group_3_size 150 \
  --train_n_passages 1 \
  --grad_cache \
  --gc_q_chunk_size 15 \
  --gc_p_chunk_size 15 \
  --dataset_name Tevatron/msmarco-passage \
  --train_dir /scratch-shared/sbhargav/data/train_reranked_MiniLM_200_cmp_no_title \
  --val_dir /scratch-shared/sbhargav/data/validation \
  --fp16 \
  --fp16_full_eval \
  --learning_rate 5e-6 \
  --q_max_len 32 \
  --p_max_len 256 \
  --warmup_ratio 0.1 \
  --logging_steps 15 \
  --evaluation_strategy steps \
  --cache_dir /scratch-shared/sbhargav/data/cache_models \
  --data_cache_dir /scratch-shared/sbhargav/data/train_reranked_MiniLM_200_cmp_no_title \
  --disable_distributed \  
  --load_model_from_disk \
  --model_name_or_path /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_gc_256/checkpoint-5880 \
  --output_dir /scratch-shared/sbhargav/multivariate_ir_experiments/experiments/gradcache/cldrd_stg2_gc_256 \
  --max_steps 2941 \
  --eval_steps 1470 \
  --save_steps 1470



```

### Stage 3


## 128

### Stage 1 
### Stage 2
### Stage 3



## 64

### Stage 1 
### Stage 2
### Stage 3

## 32

 
### Stage 1 
### Stage 2
### Stage 3