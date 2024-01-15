import argparse
import copy
from collections import defaultdict
import json
import pytrec_eval
import ir_datasets

import numpy as np

SUPP_HF_DATASETS = {
    "Tevatron/scifact/dev": "beir/scifact/test",
    "Tevatron/beir:fiqa/test": "beir/fiqa/test",
    "Tevatron/beir:trec-covid/test": "beir/trec-covid",
}

for domain in ["android", "english", "gaming", "gis", "mathematica", "physics",
               "programmers", "stats", "tex", "unix", "webmasters", "wordpress"]:
    SUPP_HF_DATASETS[f"Tevatron/beir:cqadupstack-{domain}/test"] = f"beir/cqadupstack/{domain}"


def evaluate(run, qrels, metrics):
    """
    Return qid -> {metric_1 : res_1, ...}
    """
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    eval_res_queries = evaluator.evaluate(run)
    return eval_res_queries


def compute_and_merge_mrr_cut(orig_run, qrels, metrics, prev_eval):
    assert all([_.startswith("recip_rank_cut_") for _ in metrics])

    eval_res_queries = copy.deepcopy(prev_eval)
    for metric in metrics:
        k = int(metric.lstrip("recip_rank_cut_"))
        run = {}
        for qid, r in orig_run.items():
            run[qid] = {}
            for d, s in sorted(r.items(), key=lambda _: -_[1])[:k]:
                run[qid][d] = s

        for qid, qvals in evaluate(run, qrels, ["recip_rank"]).items():
            eval_res_queries[qid][metric] = qvals["recip_rank"]

    agg = defaultdict(list)
    for qid, qvals in eval_res_queries.items():
        for metric, met_val in qvals.items():
            agg[metric].append(met_val)
    return agg, eval_res_queries


if __name__ == '__main__':
    parser = argparse.ArgumentParser("eval_run")
    parser.add_argument("--input", help="location of run", required=True)
    parser.add_argument("--metrics", help="list of metrics, csv", required=True)
    parser.add_argument("--dataset", help="name of dataset in ir_datasets", default=None)
    parser.add_argument("--hf_dataset", help="name of dataset in ir_datasets", default=None)
    parser.add_argument("--output", default=None, help="if provided, saves the results in a JSON file")
    args = parser.parse_args()

    print(f"args: {args}")
    assert args.dataset is not None or args.hf_dataset is not None
    if args.hf_dataset:
        assert args.hf_dataset in SUPP_HF_DATASETS

    if args.dataset:
        dataset = ir_datasets.load(args.dataset)
    else:
        dataset = ir_datasets.load(SUPP_HF_DATASETS[args.hf_dataset])

    qrels = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

    print(f"loaded {len(qrels)} queries from {dataset}")

    run = defaultdict(dict)
    # read run file
    with open(args.input) as reader:
        cur_qid = None
        rank = 0
        for line in reader:
            qid, docid, score = line.split()
            run[qid][docid] = float(score)

    print(f"loaded run with {len(run)} from {args.input}")

    metrics = args.metrics.split(",")
    print(f"evaluating: {metrics}")
    assert len(metrics) > 0

    # temporarily remove unsupported metrics
    mrr_cut = [_ for _ in metrics if _.startswith("recip_rank_cut_")]
    metrics = [_ for _ in metrics if _ not in mrr_cut]

    eval_res_queries = evaluate(run, qrels, metrics)
    # compute the MRR@K by cutting off the run at K, because trec_eval doesn't support @K
    agg, eval_res_queries = compute_and_merge_mrr_cut(run, qrels, mrr_cut, eval_res_queries)

    eval_res = {}
    for metric, values in agg.items():
        m, s = (np.mean(values), np.std(values))
        eval_res[metric] = (m, s)
        print(f"\t{metric:<20}: {m: 0.4f} ({s:0.4f})")

    if args.output:
        print(f"writing output to {args.output}")
        with open(args.output, "w") as writer:
            json.dump({"aggregated_result": eval_res, "query_level": eval_res_queries}, writer)
