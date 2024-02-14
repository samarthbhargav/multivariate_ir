import datasets
import logging
import random
from dataclasses import dataclass
from datasets import load_dataset, load_from_disk
from tevatron.arguments import DataArguments
from tevatron.datasets.preprocessor import VAR_PREFIX
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import List, Tuple

logger = logging.getLogger(__name__)


class DistilPreProcessor:
    def __init__(self, student_tokenizer, teacher_tokenizer, query_max_length=32, text_max_length=256, separator=" ",
                 ann_negatives=False, exclude_title=False, add_var_token=False):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator
        self.ann_negatives = ann_negatives
        self.exclude_title = exclude_title
        self.add_var_token = add_var_token

    def _preproc(self, text, title):
        if not self.exclude_title:
            text = title + self.separator + text if title else text

        if self.add_var_token:
            text = VAR_PREFIX + self.separator + text

        return text

    def _preproc_teacher(self, text, title):
        # do not add VAR token to the teacher!
        if not self.exclude_title:
            text = title + self.separator + text if title else text
        return text

    def __call__(self, example):
        eval_meta = {
            "query_id": example["query_id"],
            "positive_passages": [],
            "negative_passages": []
        }

        student_query = self.student_tokenizer.encode(
            self._preproc(example["query"], None), add_special_tokens=False, max_length=self.query_max_length,
            truncation=True
        )
        teacher_query = self.teacher_tokenizer.encode(
            example["query"], add_special_tokens=False, max_length=self.query_max_length,
            truncation=True
        )
        student_positives = []
        teacher_positives = []
        for pos in example["positive_passages"]:
            student_positives.append(
                self.student_tokenizer.encode(
                    self._preproc(pos["text"], pos["title"]),
                    add_special_tokens=False,
                    max_length=self.text_max_length,
                    truncation=True
                )
            )
            teacher_positives.append(
                self.teacher_tokenizer.encode(
                    self._preproc_teacher(pos["text"], pos["title"]),
                    add_special_tokens=False,
                    max_length=self.text_max_length,
                    truncation=True
                )
            )
            eval_meta["positive_passages"].append(pos["docid"])

        student_negatives = []
        teacher_negatives = []
        for neg in example["negative_passages"]:
            student_negatives.append(
                self.student_tokenizer.encode(
                    self._preproc(neg["text"], neg["title"]),
                    add_special_tokens=False,
                    max_length=self.text_max_length,
                    truncation=True
                )
            )
            teacher_negatives.append(
                self.teacher_tokenizer.encode(
                    self._preproc_teacher(neg["text"], neg["title"]),
                    add_special_tokens=False,
                    max_length=self.text_max_length,
                    truncation=True
                )
            )
            eval_meta["negative_passages"].append(neg["docid"])

        if self.ann_negatives:
            student_ann_negatives = []
            teacher_ann_negatives = []
            for neg in example["ann_negatives"]:
                student_ann_negatives.append(
                    self.student_tokenizer.encode(
                        self._preproc(neg["text"], neg["title"]),
                        add_special_tokens=False,
                        max_length=self.text_max_length,
                        truncation=True
                    )
                )
                teacher_ann_negatives.append(
                    self.teacher_tokenizer.encode(
                        self._preproc_teacher(neg["text"], neg["title"]),
                        add_special_tokens=False,
                        max_length=self.text_max_length,
                        truncation=True
                    )
                )

            return {
                "student_query": student_query,
                "student_positives": student_positives,
                "student_negatives": student_negatives,
                "student_ann_negatives": student_ann_negatives,
                "teacher_query": teacher_query,
                "teacher_positives": teacher_positives,
                "teacher_negatives": teacher_negatives,
                "teacher_ann_negatives": teacher_ann_negatives,
                "eval_meta": eval_meta
            }
        else:
            return {
                "student_query": student_query,
                "student_positives": student_positives,
                "student_negatives": student_negatives,
                "teacher_query": teacher_query,
                "teacher_positives": teacher_positives,
                "teacher_negatives": teacher_negatives,
                "eval_meta": eval_meta
            }


class HFDistilTrainDataset:
    def __init__(
            self,
            student_tokenizer: PreTrainedTokenizer,
            teacher_tokenizer: PreTrainedTokenizer,
            data_args: DataArguments,
            cache_dir: str,
            is_validation: bool = False
    ):

        if is_validation:
            data_files = data_args.val_dir
            self.neg_num = data_args.eval_n_passages - 1
            self.ann_neg_num = 0
        else:
            data_files = data_args.train_dir
            self.neg_num = data_args.train_n_passages - 1
            self.ann_neg_num = data_args.ann_neg_num

        if data_files:
            logger.info(f"loading from disk: {data_files}")
            self.dataset = load_from_disk(data_files)
        else:
            self.dataset = load_dataset(
                data_args.dataset_name,
                data_args.dataset_language,
                data_files=data_files,
                cache_dir=cache_dir
            )[data_args.dataset_split]

        self.is_validation = is_validation
        self.preprocessor = None if data_args.dataset_name == "json" else DistilPreProcessor
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.separator = " "
        self.add_var_token = data_args.add_var_token
        self.exclude_title = data_args.exclude_title
        logger.info(f"exclude_title:{self.exclude_title}; add_var_token: {self.add_var_token}")
        logger.info(f"ann_negatives:{self.ann_neg_num}; is_validation: {self.is_validation}")

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        ann_negs = True if self.ann_neg_num != 0 else False
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(
                    student_tokenizer=self.student_tokenizer,
                    teacher_tokenizer=self.teacher_tokenizer,
                    query_max_length=self.q_max_len,
                    text_max_length=self.p_max_len,
                    separator=self.separator,
                    ann_negatives=ann_negs,
                    exclude_title=self.exclude_title,
                    add_var_token=self.add_var_token
                ),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on train dataset",
                load_from_cache_file=False,
            )
        return self.dataset


class DistilTrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            student_tokenizer: PreTrainedTokenizer,
            teacher_tokenizer: PreTrainedTokenizer,
            is_validation=False
    ):
        self.train_data = dataset
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.data_args = data_args
        self.total_len = len(self.train_data)
        self.is_validation = is_validation

    def create_student_example(self, text_encoding: List[int], is_query=False):
        item = self.student_tokenizer.prepare_for_model(
            text_encoding,
            truncation="only_first",
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def create_teacher_example(self, query_encoding: List[int], text_encoding: List[int]):
        item = self.teacher_tokenizer.prepare_for_model(
            query_encoding,
            text_encoding,
            truncation="only_first",
            max_length=self.data_args.q_max_len + self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=True,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding], List[BatchEncoding]]:
        group = self.train_data[item]
        student_qry = group["student_query"]
        student_positives = group["student_positives"]
        student_negatives = group["student_negatives"]
        teacher_qry = group["teacher_query"]
        teacher_positives = group["teacher_positives"]
        teacher_negatives = group["teacher_negatives"]

        encoded_student_query = self.create_student_example(student_qry, is_query=True)
        encoded_student_passages = []
        encoded_teacher_pairs = []

        if (not self.is_validation) and (self.data_args.ann_neg_num > 0) and self.data_args.pseudolabels:
            # to train with pseudolabels only, discard the true positive from qrel
            # and only work on the samples found at ann_negatives

            student_ann_negatives = group["student_ann_negatives"]
            teacher_ann_negatives = group["teacher_ann_negatives"]

            idxs = random.choices(list(range(self.data_args.group_1_size)), k=self.data_args.group_1)
            idxs.extend(random.choices(list(range(self.data_args.group_1_size, self.data_args.group_2_size)),
                                       k=self.data_args.group_2))
            idxs.extend(random.choices(list(range(self.data_args.group_1_size + self.data_args.group_2_size,
                                                  self.data_args.group_1_size + self.data_args.group_2_size + self.data_args.group_3_size)),
                                       k=self.data_args.group_3))

            for psg_idx in idxs:
                encoded_teacher_pairs.append(
                    self.create_teacher_example(teacher_qry, teacher_ann_negatives[psg_idx]))
                encoded_student_passages.append(self.create_student_example(student_ann_negatives[psg_idx]))

        else:
            if self.data_args.positive_passage_no_shuffle:
                pos_psg_idx = 0
            else:
                pos_psg_idx = random.sample(list(range(len(student_positives))), 1)[0]
            encoded_teacher_pairs.append(self.create_teacher_example(teacher_qry, teacher_positives[pos_psg_idx]))
            encoded_student_passages.append(self.create_student_example(student_positives[pos_psg_idx]))

            if self.is_validation:
                labels = [
                    (group["eval_meta"]["query_id"], group["eval_meta"]["positive_passages"][pos_psg_idx], 1)
                ]

            negative_size = self.data_args.train_n_passages - 1
            if len(student_negatives) < negative_size:
                negs_idxs = random.choices(list(range(len(student_negatives))), k=negative_size)
            elif self.data_args.negative_passage_no_shuffle:
                negs_idxs = list(range(len(student_negatives)))[:negative_size]
            else:
                negs_idxs = random.sample(list(range(len(student_negatives))), negative_size)

            student_negs_token_ids = []
            teacher_negs_token_ids = []
            for neg_psg_idx in negs_idxs:

                student_negs_token_ids.append(student_negatives[neg_psg_idx])
                teacher_negs_token_ids.append(teacher_negatives[neg_psg_idx])

                encoded_teacher_pairs.append(self.create_teacher_example(teacher_qry, teacher_negatives[neg_psg_idx]))
                encoded_student_passages.append(self.create_student_example(student_negatives[neg_psg_idx]))
                if self.is_validation:
                    labels.append(
                        (group["eval_meta"]["query_id"], group["eval_meta"]["negative_passages"][neg_psg_idx], 0)
                    )

            # don't load ANN negatives for validation
            if not self.is_validation and self.data_args.ann_neg_num > 0:
                student_ann_negatives = group["student_ann_negatives"]
                teacher_ann_negatives = group["teacher_ann_negatives"]

                # remove hard negatives that are already in the soft negatives
                student_ann_negatives = [token_ids for token_ids in student_ann_negatives if
                                         token_ids not in student_negs_token_ids]
                teacher_ann_negatives = [token_ids for token_ids in teacher_ann_negatives if
                                         token_ids not in teacher_negs_token_ids]

                ann_negative_size = self.data_args.ann_neg_num
                if len(student_ann_negatives) < ann_negative_size:
                    negs_idxs = random.choices(list(range(len(student_ann_negatives))), k=ann_negative_size)
                elif self.data_args.negative_passage_no_shuffle:
                    negs_idxs = list(range(len(student_ann_negatives)))[:ann_negative_size]
                else:
                    negs_idxs = random.sample(list(range(len(student_ann_negatives))), ann_negative_size)

                for neg_psg_idx in negs_idxs:
                    encoded_teacher_pairs.append(
                        self.create_teacher_example(teacher_qry, teacher_ann_negatives[neg_psg_idx]))
                    encoded_student_passages.append(self.create_student_example(student_ann_negatives[neg_psg_idx]))

            if self.is_validation:
                return encoded_student_query, encoded_student_passages, encoded_teacher_pairs, labels

        return encoded_student_query, encoded_student_passages, encoded_teacher_pairs


@dataclass
class DistilTrainCollator:
    tokenizer: PreTrainedTokenizerBase
    teacher_tokenizer: PreTrainedTokenizerBase
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        if len(features[0]) == 3:
            is_train = True
        else:
            is_train = False

        qq = [f[0] for f in features]
        dd = [f[1] for f in features]
        pp = [f[2] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])
        if isinstance(pp[0], list):
            pp = sum(pp, [])

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
        p_collated = self.teacher_tokenizer.pad(
            pp,
            padding="max_length",
            max_length=self.max_q_len + self.max_p_len,
            return_tensors="pt",
        )

        if is_train:
            return q_collated, d_collated, p_collated
        else:
            labels = [f[3] for f in features]
            return q_collated, d_collated, p_collated, labels

