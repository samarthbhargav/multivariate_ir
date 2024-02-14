DATA_ROOT=.
N_PROC=4


dsets=('msmarco-small' 'msmarco-med' 'msmarco')
#dsets=('msmarco-small')
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

