import datasets
import random
import numpy as np
import torch
import sys
import os

if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    path = sys.argv[1]
    hf_cache_dir = sys.argv[2]
    print("hf cache dir", hf_cache_dir)
    print("result root ", path)

    os.makedirs(path, exist_ok=True)
    dataset = datasets.load_dataset(
        "Tevatron/msmarco-passage",
        "default",
        cache_dir=hf_cache_dir
    )

    # split the training set into training/validation
    dataset = dataset["train"]

    dev_size = 6980
    train_size = len(dataset)
    # sample val set equal to the dev set size
    splits = dataset.train_test_split(test_size=6980 / train_size)

    # ['614670', '549277', '611068', '515554', '1158539']
    print(splits["train"]["query_id"][:5])
    # ['280161', '220852', '994332', '451240', '605646']
    print(splits["test"]["query_id"][:5])

    splits["train"].save_to_disk(os.path.join(path, "msmarco/train"))
    splits["test"].save_to_disk(os.path.join(path, "msmarco/validation"))

    # create a tiny dataset for quick testing
    n = 20
    sm_train = splits["train"].train_test_split(test_size=n / len(splits["train"]))["test"]
    sm_train.save_to_disk(
        os.path.join(path, "msmarco-small/train"))
    n = 5
    sm_val = splits["test"].train_test_split(test_size=n / len(splits["test"]))["test"]
    sm_val.save_to_disk(
        os.path.join(path, "msmarco-small/validation"))

    n_docs = 100
    doc_ids = set()
    for s in [sm_train, sm_val]:
        for q in s:
            for doc in q["positive_passages"]:
                doc_ids.add(doc["docid"])

    print(len(doc_ids), "positives")
    corpus = datasets.load_dataset("Tevatron/msmarco-passage-corpus",
                                   cache_dir=hf_cache_dir)

    oth_corpus = corpus.filter(lambda _: _["docid"] not in doc_ids).shuffle()["train"].select(
        list(range(n_docs - len(doc_ids))))
    pos_corpus = corpus.filter(lambda _: _["docid"] in doc_ids)
    corpus_subset = datasets.concatenate_datasets([pos_corpus["train"], oth_corpus])
    print(corpus_subset)
    corpus_subset.save_to_disk(os.path.join(path, "msmarco-small/corpus"))

    n = 10000
    med_train = splits["train"].train_test_split(test_size=n / len(splits["train"]))["test"]
    med_train.save_to_disk(
        os.path.join(path, "msmarco-med/train"))

    # save the entire validation as is
    med_val = splits["test"]
    med_val.save_to_disk(
        os.path.join(path, "msmarco-med/validation"))

    n_docs = 50000
    doc_ids = set()
    for s in [med_train, med_val]:
        for q in s:
            for doc in q["positive_passages"]:
                doc_ids.add(doc["docid"])

    print(len(doc_ids), "positives")

    # sample remaining docs from the corpus

    oth_corpus = corpus.filter(lambda _: _["docid"] not in doc_ids).shuffle()["train"].select(
        list(range(n_docs - len(doc_ids))))
    pos_corpus = corpus.filter(lambda _: _["docid"] in doc_ids)
    corpus_subset = datasets.concatenate_datasets([pos_corpus["train"], oth_corpus])
    print(corpus_subset)
    corpus_subset.save_to_disk(os.path.join(path, "msmarco-med/corpus"))
