import argparse
import json

import pytrec_eval
import ir_datasets
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser("convert_run_for_qpp")
    parser.add_argument("--path", required=True, help="path to run file")
    parser.add_argument("--ir_dataset_name", required=True, help="name of ir_dataset to obtain QRELS")
    parser.add_argument("--output", required=True, help="path to output file")
    parser.add_argument("--metric", required=True, help="metric to use")
    parser.add_argument("--metric_name", required=True, help="metric name")
    args = parser.parse_args()

    dataset = ir_datasets.load(args.ir_dataset_name)

    qrels = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

    print(f"loaded {len(qrels)} qrels from {args.ir_dataset_name}")

    run = defaultdict(dict)
    with open(args.path) as reader:
        for line in reader:
            qid, doc_id, rel = line.split()
            run[qid][doc_id] = float(rel)
            assert qid in qrels
        print(f"loaded {len(run)} quries from {args.path}")

    assert len(run) == len(qrels)
    res = pytrec_eval.RelevanceEvaluator(qrels, measures=[args.metric]).evaluate(run)
    conv_res = defaultdict(dict)
    for qid, perfs in res.items():
        conv_res[qid][args.metric_name] = perfs[args.metric]
    print(f"writing result to {args.output}")
    with open(args.output, "w") as writer:
        json.dump(conv_res, writer)
