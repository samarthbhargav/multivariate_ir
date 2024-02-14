import os
import datasets

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
    root_path = "./"
    for pct in [0.1, 0.5, 1.0]:
        for dset in ["msmarco-small", "msmarco-med", "msmarco-train"]:
            input_path = os.path.join(root_path, dset, "corpus")
            output = os.path.join(root_path, dset + f"-perturbed-{pct}", "corpus")

            inp_dataset = datasets.load_from_disk(input_path)

            out_dataset = inp_dataset.map(perturb_document,
                                          batched=False,
                                          load_from_cache_file=False,
                                          fn_kwargs={"pct_swap": pct})

            out_dataset.save_to_disk(output)
            print(f"[{pct}]input: {input_path}; output: {output}")
            print(f"EX inp: {inp_dataset[0]['text']}\nEX out: {out_dataset[0]['text']}\n\n")

            for split in ["train", "validation"]:
                input_path = os.path.join(root_path, dset, split)
                output = os.path.join(root_path, dset + f"-perturbed-{pct}", split)

                inp_dataset = datasets.load_from_disk(input_path)

                out_dataset = inp_dataset.map(perturb_query,
                                              batched=False,
                                              load_from_cache_file=False,
                                              fn_kwargs={"pct_swap": pct})

                out_dataset.save_to_disk(output)
                print(f"[{pct}]input: {input_path}; output: {output}")
                print(f"EX inp: {inp_dataset[0]['query']}\nEX out: {out_dataset[0]['query']}\n\n")

        break
