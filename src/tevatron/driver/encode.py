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

from tevatron.arguments import DataArguments, ModelArguments, MVRLTrainingArguments, StochasticArguments
from tevatron.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.data import EncodeCollator, EncodeDataset
from tevatron.datasets import HFCorpusDataset, HFQueryDataset
from tevatron.modeling import DenseModel, EncoderOutput
from tevatron.modeling.dense_mvrl import MVRLDenseModel
from tevatron.modeling.dense_stochastic import StochasticDenseModel

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, MVRLTrainingArguments, StochasticArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, mvrl_args, stoch_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, mvrl_args, stoch_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments
        mvrl_args: MVRLTrainingArguments
        stoch_args: StochasticArguments

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

    encode_kwargs = {}
    if mvrl_args.model_type == "default":
        model = DenseModel.load(
            model_name_or_path=model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif mvrl_args.model_type.startswith("mvrl"):
        assert "[VAR]" in tokenizer.all_special_tokens
        model = MVRLDenseModel.load(
            model_args=model_args,
            mvrl_args=mvrl_args,
            model_name_or_path=model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir
        )
    elif mvrl_args.model_type == "stochastic":
        model = StochasticDenseModel.load(
            model_name_or_path=model_args.model_name_or_path,
            model_args=model_args,
            stoch_args=stoch_args
        )
        encode_kwargs["n_iters"] = stoch_args.n_iters
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
    for batch_ids, batch in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                if data_args.encode_is_qry:
                    model_output: EncoderOutput = model(query=batch, **encode_kwargs)
                    if mvrl_args.model_type == "stochastic":
                        # get the mean vector only
                        model_output.q_reps = StochasticDenseModel.get_mean_var(model_output.q_reps)[0]
                    encoded.append(model_output.q_reps.cpu().detach().numpy())
                else:
                    model_output: EncoderOutput = model(passage=batch, **encode_kwargs)
                    if mvrl_args.model_type == "stochastic":
                        # get the mean vector only
                        model_output.p_reps = StochasticDenseModel.get_mean_var(model_output.p_reps)[0]
                    encoded.append(model_output.p_reps.cpu().detach().numpy())
    encoded = np.concatenate(encoded)
    logger.info(f"saving to {data_args.encoded_save_path}")
    with open(data_args.encoded_save_path, "wb") as f:
        pickle.dump((encoded, lookup_indices), f)


if __name__ == "__main__":
    main()
