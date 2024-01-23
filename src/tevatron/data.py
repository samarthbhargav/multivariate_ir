import logging
import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import BatchEncoding, DataCollatorWithPadding, PreTrainedTokenizer

from .arguments import DataArguments
from .trainer import TevatronTrainer

logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: TevatronTrainer = None,
            is_validation=False,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)
        self.is_validation = is_validation

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.prepare_for_model(
            text_encoding,
            truncation="only_first",
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group["query"]
        encoded_query = self.create_one_example(qry, is_query=True)

        encoded_passages = []
        group_positives = group["positives"]
        group_negatives = group["negatives"]

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
            if self.is_validation:
                pos_psg_id = group["eval_meta"]["positive_passages"][0]
        else:
            _ = (_hashed_seed + epoch) % len(group_positives)
            pos_psg = group_positives[_]
            if self.is_validation:
                pos_psg_id = group["eval_meta"]["positive_passages"][_]

        encoded_passages.append(self.create_one_example(pos_psg))
        if self.is_validation:
            # qid, doc_id, label (assuming binary relevance? TODO make more general?)
            labels = [
                (group["eval_meta"]["query_id"], pos_psg_id, 1)
            ]

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            neg_idx = random.choices(list(range(len(group_negatives))), k=negative_size)
            negs = [group_negatives[_] for _ in neg_idx]
            if self.is_validation:
                neg_psg_ids = [group["eval_meta"]["negative_passages"][_] for _ in neg_idx]
        elif self.data_args.train_n_passages == 1:
            negs = []
            if self.is_validation:
                neg_psg_ids = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
            if self.is_validation:
                neg_psg_ids = group["eval_meta"]["negative_passages"][:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            if self.is_validation:
                negs = [x for x in zip(group_negatives, group["eval_meta"]["negative_passages"])]
                random.Random(_hashed_seed).shuffle(negs)
                negs = negs * 2
                negs = negs[_offset: _offset + negative_size]
                neg_psg_ids = [_[1] for _ in negs]
                negs = [_[0] for _ in negs]
            else:
                negs = [x for x in group_negatives]
                random.Random(_hashed_seed).shuffle(negs)
                negs = negs * 2
                negs = negs[_offset: _offset + negative_size]

        for i, neg_psg in enumerate(negs):
            encoded_passages.append(self.create_one_example(neg_psg))
            if self.is_validation:
                # qid, doc_id, label (assuming binary relevance? TODO make more general?)
                labels.append((
                    (group["eval_meta"]["query_id"], neg_psg_ids[i], 0)
                ))

        if self.is_validation:
            # returns 1, N, N
            return encoded_query, encoded_passages, labels
        return encoded_query, encoded_passages


class EncodeDataset(Dataset):
    input_keys = ["text_id", "text"]

    def __init__(self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_len=128):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        text_id, text = (self.encode_data[item][f] for f in self.input_keys)
        encoded_text = self.tok.prepare_for_model(
            text,
            max_length=self.max_len,
            truncation="only_first",
            padding=False,
            return_token_type_ids=False
        )
        return text_id, encoded_text


@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        # if eval_meta is included, then it's from the validation set
        if len(features[0]) == 2:
            is_train = True
        else:
            is_train = False

        qq = [f[0] for f in features]
        dd = [f[1] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding="max_length",
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding="max_length",
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        if is_train:
            return q_collated, d_collated
        else:
            labels = [f[2] for f in features]
            return q_collated, d_collated, labels


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        collated_features = super().__call__(text_features)
        return text_ids, collated_features
