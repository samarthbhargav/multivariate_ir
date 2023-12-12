import os
import json
import ir_datasets
import argparse

a = """
{
   "query_id": "<query id>",
   "query": "<query text>",
   "positive_passages": [
     {"docid": "<passage id>", "title": "<passage title>", "text": "<passage body>"}
   ],
   "negative_passages": [
     {"docid": "<passage id>", "title": "<passage title>", "text": "<passage body>"}
   ]
}
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()
    os.makedirs(args.dest, exist_ok=True)

    datasets = ["msmarco-passage/trec-dl-2019/judged", "msmarco-passage/trec-dl-2020/judged"]

    for dataset_name in datasets:
        print("dataset:", dataset_name)
        dataset = ir_datasets.load(dataset_name)

        queries = {}
        for q in dataset.queries_iter():
            queries[q.query_id] = {
                "query_id": q.query_id,
                "query": q.text,
                "positive_passages": [],
                "negative_passages": []
            }

        qrels = []
        for qrel in dataset.qrels_iter():
            # TOPIC      ITERATION      DOCUMENT#      RELEVANCY
            qrels.append(f"{qrel.query_id} Q0 {qrel.doc_id} {qrel.relevance}")

        with open(os.path.join(args.dest, dataset_name.split("/")[-2] + "_topics.jsonl"), "w") as writer:
            writer.write("\n".join([json.dumps(_) for _ in queries.values()]))

        with open(os.path.join(args.dest, dataset_name.split("/")[-2] + ".qrel"), "w") as writer:
            writer.write("\n".join(qrels))
