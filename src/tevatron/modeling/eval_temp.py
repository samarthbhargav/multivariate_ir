import datasets
import pytrec_eval
from collections import defaultdict

import sys
import numpy as np

run_path = sys.argv[1]
data = sys.argv[2]
negate = sys.argv[3]

if negate == "yes":
    negate = -1
else:
    negate = 1

run = defaultdict(dict)
with open(run_path) as reader:
    for i, line in enumerate(reader):
        qid, doc_id, score = line.split()
        run[qid][doc_id] = negate * float(score)

        if i % 1000 == 0:
            print("read", i, "lines")

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
for measure in measures:
    mean_ = np.mean([_[measure] for _ in res.values()])
    print(measure, round(mean_, 4))
