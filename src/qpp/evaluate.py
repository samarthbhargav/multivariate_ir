import os
import json
import argparse
from scipy.stats import kendalltau, pearsonr, spearmanr
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", type=str, default="")
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--actual", type=str, required=True, nargs="+")
    parser.add_argument("--predicted_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

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
    pred_performances = {}
    for pp_file in os.listdir(args.predicted_dir):
        if ".json" in pp_file or pp_file.startswith("."):
            continue

        pp_name = pp_file
        pp_file = os.path.join(args.predicted_dir, pp_file)
        print(f"Reading Predictions: {pp_name} ({pp_file})")
        pred_performances[pp_name] = {}
        with open(pp_file) as reader:
            for line in reader:
                qid, perf = line.split()
                pred_performances[pp_name][qid] = float(perf)

    result_rows = []
    for ap_name, ap in actual_performances:
        qids = sorted(list(ap.keys()))
        actuals = [ap[qid][args.metric] for qid in qids]
        for pp_name, preds in pred_performances.items():
            assert len(ap) == len(preds)
            assert all([qid in preds for qid in qids])

            preds = [preds[qid] for qid in qids]

            spearman_res = spearmanr(actuals, preds)
            pearson_res = pearsonr(actuals, preds)
            ktau_res = kendalltau(actuals, preds)

            result_rows.append({
                "ref": ap_name,
                "method": pp_name,
                "spearman": spearman_res.statistic,
                "spearman_pval": spearman_res.pvalue,
                "pearson": pearson_res.statistic,
                "pearson_pval": pearson_res.pvalue,
                "ktau": ktau_res.statistic,
                "ktau_pval": ktau_res.pvalue
            })

    print(f"saving {len(result_rows)} rows to {args.output}")
    pd.DataFrame(result_rows).to_csv(args.output, index=False)
