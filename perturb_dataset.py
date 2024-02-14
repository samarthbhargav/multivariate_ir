import os
import datasets
import argparse
import torch
import numpy as np
import random
from textattack.transformations import CompositeTransformation, WordInnerSwapRandom
from textattack.constraints.pre_transformation import RepeatModification

from textattack.augmentation import Augmenter


def augment(text, pct_swap=0.5):
    transformation = CompositeTransformation(
        [WordInnerSwapRandom(), ]
    )
    # Set up constraints
    constraints = [RepeatModification()]
    # Create augmenter with specified parameters
    augmenter = Augmenter(
        transformation=transformation,
        constraints=constraints,
        pct_words_to_swap=pct_swap,
        transformations_per_example=1,
    )
    # Augment!
    return augmenter.augment(text)[0]


def perturb_query(row, pct_swap):
    row["query"] = augment(row["query"], pct_swap)
    return row


def perturb_document(row, pct_swap):
    row["text"] = augment(row["text"], pct_swap)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser("perturb_dataset")
    parser.add_argument("--hf_disk_dataset", default=None, help="path to dataset stored on disk")
    parser.add_argument("--hf_dataset", default=None, help="hf dataset name")
    parser.add_argument("--hf_dataset_split", default=None, help="hf dataset name")
    parser.add_argument("--is_query", action="store_true", default=False)
    parser.add_argument("--word_swap_pct", required=True, type=float)
    parser.add_argument("--n_proc", required=True, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert args.hf_disk_dataset is not None or args.hf_dataset is not None

    if args.hf_disk_dataset is not None:
        dataset = datasets.load_from_disk(args.hf_disk_dataset)
    else:
        dataset = datasets.load_dataset(args.hf_dataset, split=args.hf_dataset_split)

    perturb_fn = perturb_query if args.is_query else perturb_document
    print(f"[{args.word_swap_pct}]input: {dataset}; output: {args.out}")
    field_name = "query" if args.is_query else "text"
    out_dataset = dataset.map(perturb_fn,
                              batched=False,
                              load_from_cache_file=False,
                              fn_kwargs={"pct_swap": args.word_swap_pct},
                              num_proc=args.n_proc)
    print(f"\n####\nEX inp: {dataset[0][field_name]}\nEX out: {out_dataset[0][field_name]}\n####")
    out_dataset.save_to_disk(args.out)
