PRE_RETR_CMD="python -m qpp.pre_retrieval"

#datasets=('dl19' 'dl20')
datasets=('dev')
for dset in "${datasets[@]}"
do
    ${PRE_RETR_CMD} \
        --mode precomputation \
        --query_path datasets/trec-dl/${dset}/queries.tsv \
        --index_path datasets/trec-dl/corpus_index/ \
        --output_path qpp_output/pre-retrieval/${dset}

    ${PRE_RETR_CMD} \
    --mode baselines \
    --query_path datasets/trec-dl/${dset}/queries.tsv \
    --index_path datasets/trec-dl/corpus_index/ \
    --qrels_path datasets/trec-dl/${dset}/qrel.txt \
    --output_path qpp_output/pre-retrieval/${dset}

done

#
#python -u ./unsupervisedQPP/pre_retrieval.py \
#    --mode PPL-QPP \
#    --query_path datasets/trec-dl/dl19/queries.tsv \
#    --index_path datasets/trec-dl/corpus_index/ \
#    --qrels_path datasets/trec-dl/dl19/qrel.txt \
#    --output_path ./output/pre-retrieval/dl19 \
#    --LM gpt2-xl \
#    --qpp_names VAR-std-sum \
#    --alpha 1
#
#python -u ./unsupervisedQPP/pre_retrieval.py \
#    --mode PPL-QPP \
#    --query_path datasets/trec-dl/dl20/queries.tsv \
#    --index_path datasets/trec-dl/corpus_index/ \
#    --qrels_path datasets/trec-dl/dl20/qrel.txt \
#    --output_path ./output/pre-retrieval/dl20 \
#    --LM gpt2-xl \
#    --qpp_names VAR-std-sum \
#    --alpha TODO
