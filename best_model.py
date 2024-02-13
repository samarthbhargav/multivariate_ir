import os
import json
import pathlib
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("best_model")
    parser.add_argument("--experiment_path", required=True,
                        help="location of the top level directory (sub-directories are models)")

    args = parser.parse_args()
    path = pathlib.Path(args.experiment_path)
    print(f"exploring path: {path}")

    best_model = None
    best_mrr = float("-inf")
    for model_num in os.listdir(path):
        model_path = path / model_num
        print("\t", model_path)
        with open(model_path / "eval_result.json") as reader:
            eval_result = json.load(reader)

        print("\t", eval_result)

        if eval_result["eval_mrr"] > best_mrr:
            best_mrr = eval_result["eval_mrr"]
            best_model = model_path

    print(f"best_mrr: {best_mrr}")
    print(f"best_model: {best_model}")
