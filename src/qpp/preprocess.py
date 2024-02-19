import pathlib
import ir_datasets
import json
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("preprocess")
    parser.add_argument("--path", required=True, help="root location of datasets folder")
    args = parser.parse_args()
    path = pathlib.Path(args.path)

    dataset = ir_datasets.load("msmarco-passage")
    os.makedirs(path / "corpus", exist_ok=True)

    corpus_path = path / "corpus" / "data.jsonl"
    if os.path.exists(corpus_path):
        print(f"{corpus_path} already exists!")
    else:
        with open(corpus_path, "w") as writer:
            for doc in dataset.docs_iter():
                writer.write(json.dumps({"id": doc.doc_id, "contents": doc.text}) + "\n")

    for name, dl_set in [("dl19", "msmarco-passage/trec-dl-2019/judged"),
                         ("dl20", "msmarco-passage/trec-dl-2020/judged")]:
        dataset = ir_datasets.load(dl_set)

        res_dir = path / name
        os.makedirs(res_dir, exist_ok=True)

        if os.path.exists(res_dir / "queries.tsv"):
            print(res_dir / "queries.tsv", "already exists!")
        else:
            with open(res_dir / "queries.tsv", "w") as writer:
                for query in dataset.queries_iter():
                    writer.write(f"{query.query_id}\t{query.text}\n")

        if os.path.exists(res_dir / f"qrel.txt"):
            print(res_dir / f"qrel.txt", "already exists!")
        else:
            with open(res_dir / f"qrel.txt", "w") as writer:
                for qrel in dataset.qrels_iter():
                    # query-number 0 document-id relevance
                    writer.write(f"{qrel.query_id}\t0\t{qrel.doc_id}\t{qrel.relevance}\n")
