import argparse
from collections import defaultdict
import json
import pytrec_eval
import ir_datasets
import logging
import numpy as np

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("eval_run")
    parser.add_argument("--input", help="location of run", required=True)
    parser.add_argument("--metrics", help="list of metrics, csv", required=True)
    parser.add_argument("--dataset", help="name of dataset in ir_datasets", required=True)
    parser.add_argument("--output", default=None, help="if provided, saves the results in a JSON file")

    args = parser.parse_args()

    run = defaultdict(dict)
    # read run file
    with open(args.input) as reader:
        cur_qid = None
        rank = 0
        for line in reader:
            qid, docid, score = line.split()
            run[qid][docid] = float(score)

    logger.info(f"loaded run with {len(run)} from {args.input}")

    dataset = ir_datasets.load(args.dataset)
    qrels = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

    logger.info(f"loaded {len(qrels)} queries from {dataset}")

    metrics = args.metrics.split(",")
    logger.info(f"evaluating: {metrics}")
    assert len(metrics) > 0

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    eval_res_queries = evaluator.evaluate(run)
    agg = defaultdict(list)
    for qid, qvals in eval_res_queries.items():
        for metric, met_val in qvals.items():
            agg[metric].append(met_val)

    eval_res = {}
    for metric, values in agg.items():
        m, s = (np.mean(values), np.std(values))
        eval_res[metric] = (m, s)
        logger.info(f"\t{metric:<20}: {m: 0.4f} ({s:0.4f})")

    if args.output:
        with open(args.output, "w") as writer:
            json.dump({"aggregated_result": eval_res, "query_level": eval_res_queries}, writer)
