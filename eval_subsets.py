import datasets
import pytrec_eval
from collections import defaultdict
import json
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("eval_subsets")
    parser.add_argument("--run_path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=None)

    args = parser.parse_args()
    run_path = args.run_path
    data = args.data

    run = defaultdict(dict)
    with open(run_path) as reader:
        for i, line in enumerate(reader):
            qid, doc_id, score = line.split()
            run[qid][doc_id] = float(score)

    val = datasets.load_from_disk(data)

    hf_qrel = defaultdict(dict)
    for i in range(len(val)):
        q = val[i]
        qid = q["query_id"]
        hf_qrel[qid] = {}
        for pos in q["positive_passages"]:
            hf_qrel[qid][pos["docid"]] = 1

    assert run.keys() == hf_qrel.keys()

    measures = ["recall_5", "ndcg_cut_100", "recip_rank", "recall_1000"]
    res = pytrec_eval.RelevanceEvaluator(hf_qrel, measures).evaluate(run)
    mean_res = {}
    for measure in measures:
        mean_ = np.mean([_[measure] for _ in res.values()])
        mean_res[measure] = mean_
        if not args.out:
            print(measure, round(mean_, 4))

    if args.out:
        print(args.out)
        with open(args.out, "w") as writer:
            json.dump(mean_res, writer, indent=2)
