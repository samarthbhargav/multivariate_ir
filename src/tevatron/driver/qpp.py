import logging
import os
import sys
from contextlib import nullcontext
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

from tevatron.arguments import DataArguments, ModelArguments, MVRLTrainingArguments
from tevatron.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.data import EncodeCollator, EncodeDataset
from tevatron.datasets import HFQueryDataset

from tevatron.modeling.dense_mvrl import MVRLDenseModel

from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QPPArguments:
    method: str = field(default="norm", metadata={"help": "how to compute QPP of a query"})
    qpp_save_path: str = field(default=None, metadata={"help": "location of predicted scores"})


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, MVRLTrainingArguments, QPPArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, mvrl_args, qpp_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, mvrl_args, qpp_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments
        mvrl_args: MVRLTrainingArguments
        qpp_args: QPPArguments

    assert qpp_args.qpp_save_path is not None, "provide qpp_save_path"

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError("Multi-GPU encoding is not supported.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )

    assert mvrl_args.model_type.startswith("mvrl")
    assert data_args.add_var_token, "This flag has to be enabled for MVRL models"
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    assert "[VAR]" in tokenizer.all_special_tokens
    model = MVRLDenseModel.load(
        model_args=model_args,
        mvrl_args=mvrl_args,
        model_name_or_path=model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir
    )

    text_max_length = data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len
    if data_args.encode_is_qry:
        encode_dataset = HFQueryDataset(
            tokenizer=tokenizer, data_args=data_args, cache_dir=data_args.data_cache_dir or model_args.cache_dir
        )
    else:
        raise ValueError("Only supported for queries")
        # encode_dataset = HFCorpusDataset(
        #     tokenizer=tokenizer, data_args=data_args, cache_dir=data_args.data_cache_dir or model_args.cache_dir
        # )

    encode_dataset = EncodeDataset(
        encode_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
        tokenizer,
        max_len=text_max_length,
    )

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=EncodeCollator(tokenizer, max_length=text_max_length, padding="max_length"),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    predicted = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()
    for batch_ids, batch in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                if data_args.encode_is_qry:
                    q_reps = model.encode_query(batch)
                    q_reps_mean, q_reps_var = q_reps
                    if qpp_args.method == "norm":
                        predicted.append(torch.linalg.vector_norm(q_reps_var, dim=1))
                    else:
                        raise ValueError(qpp_args.method)
                else:
                    raise ValueError()

    predicted = np.concatenate(predicted).tolist()

    logger.info(f"saving to {qpp_args.qpp_save_path}")

    with open(qpp_args.qpp_save_path, "w") as writer:
        for qid, pp in zip(lookup_indices, predicted):
            writer.write(f"{qid}\t{pp}\n")


if __name__ == "__main__":
    main()
