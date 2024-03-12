import os
import json
import argparse
from scipy.stats import kendalltau, pearsonr, spearmanr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", type=str, default="")
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--actual", type=str, required=True, nargs="+")
    parser.add_argument("--predicted", type=str, required=True)

    args = parser.parse_args()

    # read actual performance
    actual_performances = []
    for ap_file in args.actual:
        ap_name, ap_file = ap_file.split(",")
        if ap_name.startswith("."):
            continue
        print(f"Reading g.t performance: {ap_name} ({ap_file})")
        with open(ap_file) as reader:
            actual_performances.append((ap_name, json.load(reader)))

    # read predicted performance
    pred_performance = {}
    pp_name = args.predicted
    pp_file = os.path.join(args.predicted_dir, args.predicted)
    print(f"Reading Predictions: {pp_name} ({pp_file})")
    with open(pp_file) as reader:
        for line in reader:
            qid, perf = line.split()
            pred_performance[qid] = float(perf)

    for ap_name, ap in actual_performances:
        qids = sorted(list(ap.keys()))
        actuals = [ap[qid][args.metric] for qid in qids]

        assert len(ap) == len(pred_performance)
        assert all([qid in pred_performance for qid in qids])

        preds = [pred_performance[qid] for qid in qids]

        spearman_res = spearmanr(actuals, preds)
        pearson_res = pearsonr(actuals, preds)
        ktau_res = kendalltau(actuals, preds)

        print(f"{ap_name}")
        print(f"\tSpearman Correlation:: {spearman_res.statistic:0.4f}")
        print(f"\tPearson Correlation :: {pearson_res.statistic:0.4f}")
        print(f"\tKendall's Tau       :: {ktau_res.statistic:0.4f}")
