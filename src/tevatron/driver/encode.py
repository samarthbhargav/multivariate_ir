import logging
import os
import pickle
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
from tevatron.datasets import HFCorpusDataset, HFQueryDataset
from tevatron.modeling import DenseModel, EncoderOutput
from tevatron.modeling.dense_mvrl import MVRLDenseModel

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, MVRLTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, mvrl_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, mvrl_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments
        mvrl_args: MVRLTrainingArguments

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

    if mvrl_args.model_type.startswith("mvrl"):
        assert data_args.add_var_token, "This flag has to be enabled for MVRL models"

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if mvrl_args.model_type == "default":
        model = DenseModel.load(
            model_name_or_path=model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif mvrl_args.model_type == "mvrl_no_distill":
        assert "[VAR]" in tokenizer.all_special_tokens
        model = MVRLDenseModel.load(
            model_args=model_args,
            mvrl_args=mvrl_args,
            model_name_or_path=model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir
        )
    else:
        raise NotImplementedError(mvrl_args.model_type)

    text_max_length = data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len
    if data_args.encode_is_qry:
        encode_dataset = HFQueryDataset(
            tokenizer=tokenizer, data_args=data_args, cache_dir=data_args.data_cache_dir or model_args.cache_dir
        )
    else:
        encode_dataset = HFCorpusDataset(
            tokenizer=tokenizer, data_args=data_args, cache_dir=data_args.data_cache_dir or model_args.cache_dir
        )
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
    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()
    doc_priors = {}
    for batch_ids, batch in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                if data_args.encode_is_qry:
                    model_output: EncoderOutput = model(query=batch)
                    encoded.append(model_output.q_reps.cpu().detach().numpy())
                else:
                    model_output: EncoderOutput = model(passage=batch)
                    encoded.append(model_output.p_reps.cpu().detach().numpy())

                    if mvrl_args.model_type.startswith("mvrl"):
                        doc_prior = model.doc_prior(model_output.extra["p_mean"], model_output.extra["p_var"])
                        for pid, prior in zip(batch_ids, doc_prior.cpu().detach().numpy().tolist()):
                            doc_priors[pid] = float(prior)

    encoded = np.concatenate(encoded)
    logger.info(f"saving to {data_args.encoded_save_path}")
    with open(data_args.encoded_save_path, "wb") as f:
        pickle.dump((encoded, lookup_indices), f)

    if len(doc_priors) > 0:
        doc_prior_path = data_args.encoded_save_path + "_doc_prior"
        logger.info(f"saving doc priors to {doc_prior_path}")
        with open(doc_prior_path, "wb") as f:
            pickle.dump(doc_priors, f)


if __name__ == "__main__":
    main()
