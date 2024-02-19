DATA_ROOT=.
N_PROC=4


dsets=('msmarco-small' 'msmarco-med')
percs=(0.1 0.5 1.0)
for dset in "${dsets[@]}"
do
   for perc in "${percs[@]}"
   do
     python perturb_dataset.py --is_query \
        --hf_disk_dataset ${DATA_ROOT}/${dset}/train \
        --n_proc ${N_PROC} \
        --word_swap_pct ${perc} \
        --out  ${DATA_ROOT}/${dset}-perturbed-${perc}/train

     python perturb_dataset.py --is_query \
          --hf_disk_dataset ${DATA_ROOT}/${dset}/validation \
          --n_proc ${N_PROC} \
          --word_swap_pct ${perc} \
          --out  ${DATA_ROOT}/${dset}-perturbed-${perc}/validation

     python perturb_dataset.py \
          --hf_disk_dataset ${DATA_ROOT}/${dset}/corpus \
          --n_proc ${N_PROC} \
          --word_swap_pct ${perc} \
          --out  ${DATA_ROOT}/${dset}-perturbed-${perc}/corpus
   done
done


for perc in "${percs[@]}"
do
  python perturb_dataset.py --is_query \
        --hf_disk_dataset ${DATA_ROOT}/msmarco/train \
        --n_proc ${N_PROC} \
        --word_swap_pct ${perc} \
        --out  ${DATA_ROOT}/msmarco-perturbed-${perc}/train

  python perturb_dataset.py --is_query \
       --hf_disk_dataset ${DATA_ROOT}/msmarco/validation \
       --n_proc ${N_PROC} \
       --word_swap_pct ${perc} \
       --out  ${DATA_ROOT}/msmarco-perturbed-${perc}/validation

  python perturb_dataset.py --is_query \
    --hf_dataset Tevatron/msmarco-passage \
    --hf_dataset_split dev \
    --n_proc ${N_PROC} \
    --word_swap_pct ${perc} \
    --out  ${DATA_ROOT}/msmarco-perturbed-${perc}/dev/

  python perturb_dataset.py --is_query \
    --hf_dataset Tevatron/msmarco-passage \
    --hf_dataset_split dl19 \
    --n_proc ${N_PROC} \
    --word_swap_pct ${perc} \
    --out  ${DATA_ROOT}/msmarco-perturbed-${perc}/dl19/

  python perturb_dataset.py --is_query \
    --hf_dataset Tevatron/msmarco-passage \
    --hf_dataset_split dl20 \
    --n_proc ${N_PROC} \
    --word_swap_pct ${perc} \
    --out  ${DATA_ROOT}/msmarco-perturbed-${perc}/dl20/


  python perturb_dataset.py \
    --hf_dataset Tevatron/msmarco-passage-corpus \
    --n_proc ${N_PROC} \
    --word_swap_pct ${perc} \
    --out  ${DATA_ROOT}/msmarco-perturbed-${perc}/corpus
done



