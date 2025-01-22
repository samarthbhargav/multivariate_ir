import argparse
import gc
import glob
import random
from tqdm import tqdm

from datasets import load_dataset, load_from_disk, concatenate_datasets


def get_dataset_info(dataset_name):
    info = dataset_name.split("/")
    dataset_split = info[-1] if len(info) == 3 else "train"
    dataset_name = "/".join(info[:-1]) if len(info) == 3 else "/".join(info)
    dataset_language = "default"
    if ":" in dataset_name:
        dataset_name, dataset_language = dataset_name.split(":")

    return dataset_name, dataset_split, dataset_language


def main(args):
    if args.dataset:
        dataset = load_from_disk(args.dataset)
    else:

        dataset_name, dataset_split, dataset_language = get_dataset_info(args.hf_dataset)
        dataset = load_dataset(dataset_name,
                               dataset_language,
                               cache_dir=args.data_cache_dir,
                               split=dataset_split)

    if args.corpus:
        corpus = load_from_disk(args.corpus)
    else:
        corpus_name, corpus_split, corpus_language = get_dataset_info(args.hf_corpus)

        corpus = load_dataset(corpus_name,
                              corpus_language,
                              cache_dir=args.data_cache_dir,
                              split=corpus_split)

    # load rankings
    qpreds_dict = {}
    ranking_files = glob.glob(args.rankings)
    for ranking_file in tqdm(ranking_files, desc="reading ranking files"):
        qpreds_data_tmp = open(ranking_file, 'rt').readlines()

        for line in qpreds_data_tmp:
            qpreds_data_tmp = line.replace('\n', '').split('\t')
            qid = qpreds_data_tmp[0]

            if qid in qpreds_dict.keys():
                if len(qpreds_dict[qid]) <= 250:
                    qpreds_dict[qid].update({qpreds_data_tmp[1]: qpreds_data_tmp[2]})

            else:
                qpreds_dict[qid] = {qpreds_data_tmp[1]: qpreds_data_tmp[2]}

    # docid is same as the position on dataset (or not)
    if not args.docid_is_pos:
        docid2pos = dict(zip(corpus["docid"], range(len(corpus["docid"]))))

    # sample from rankings
    num_shards = args.shards
    for shard_idx in tqdm(range(num_shards), desc="shards"):
        shard = dataset.shard(num_shards=num_shards, index=shard_idx, contiguous=True)

        ann_negatives = []
        for q_entry in shard:

            qid = q_entry["query_id"]
            rankings_dict = qpreds_dict[qid]

            if args.remove_qrel_positives:
                positive_docids = [positive["docid"] for positive in q_entry["positive_passages"]]
                for pos_docid in positive_docids:
                    rankings_dict.pop(pos_docid, None)

            keys = random.sample(list(rankings_dict.keys()), args.k) if args.random else list(rankings_dict.keys())[
                                                                                         :args.k]
            values = [rankings_dict[k] for k in keys]

            hard_negs_ann = []
            for docid, score in dict(zip(keys, values)).items():
                doc = corpus[int(docid)] if args.docid_is_pos else corpus[docid2pos[docid]]
                title, text = doc["title"], doc["text"]
                hard_negs_ann.append({"docid": docid, "title": doc["title"], "text": doc["text"], "score": score})

            ann_negatives.append(hard_negs_ann)

        shard = shard.add_column("ann_negatives", ann_negatives)
        shard.save_to_disk(args.output + f"/shard_{shard_idx}")

    # remove from memory
    del corpus
    del dataset
    del ann_negatives
    gc.collect()

    # combine the shards and save to new location
    dataset = concatenate_datasets([
        load_from_disk(args.output + f"/shard_{shard_idx}")
        for shard_idx in range(num_shards)
    ])

    dataset.save_to_disk(args.output + "/combined")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=str, help="glob, location of rankings file", required=True)
    parser.add_argument("--hf_dataset", default=None, type=str, help="huggingface dataset name")
    parser.add_argument("--hf_corpus", default=None, type=str, help="huggingface dataset name")
    parser.add_argument("--dataset", default=None, type=str, help="dataset name")
    parser.add_argument("--corpus", default=None, type=str, help="dataset name")
    parser.add_argument("--shards", default=None, type=int, help="number of shards (when saving the new dataset)")
    parser.add_argument("--data_cache_dir", help="here do you want to store the data downloaded from huggingface",
                        default=None)
    parser.add_argument('--remove_qrel_positives',
                        help="remove the known positives from the list of retrieved passages", action="store_true")
    parser.add_argument("--k", type=int, help="number of hard negatives from ANN", default=10, choices=range(1, 1000))
    parser.add_argument('--docid_is_pos', help="docid is the position in the dataset", action="store_true")
    parser.add_argument('--random', help="random selection of k ANN negatives", action="store_true")
    parser.add_argument("--output", default=None, help="if provided, saves the results in a JSON file")
    args = parser.parse_args()

    main(args)

