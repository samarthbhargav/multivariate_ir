PRE_RETR_CMD="python -m qpp.pre_retrieval"

datasets=('dev' 'dl19' 'dl20')
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