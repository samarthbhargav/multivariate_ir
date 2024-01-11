import os
import argparse
import ir_datasets


def get_generic_query(query):
    return {
        "query_id": q.query_id,
        "query": q.text,
        "positive_passages": [],
        "negative_passages": []
    }

def get_generic_document(document):
    return

query_funcs = {"beir/fiqa"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser("ird_setup")
    parser.add_argument("--dataset", help="name of ir_dataset dataset", required=True)
    parser.add_argument("--splits", help="splits to process", required=True)
    parser.add_argument("--out", help="(folder) to store resulting files")

    args = parser.parse_args()

    dataset = ir_datasets.load(args.dataset)

    # write corpus

    # write queries
    queries = {}
    for q in dataset.queries_iter():
        # TODO TITLE
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
    # write corpus
