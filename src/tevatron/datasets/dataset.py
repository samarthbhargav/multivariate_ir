import logging

from datasets import load_dataset, load_from_disk
from transformers import PreTrainedTokenizer

from ..arguments import DataArguments
from .preprocessor import CorpusPreProcessor, QueryPreProcessor, TrainPreProcessor

DEFAULT_PROCESSORS = [TrainPreProcessor, QueryPreProcessor, CorpusPreProcessor]
PROCESSOR_INFO = {
    "Tevatron/wikipedia-nq": DEFAULT_PROCESSORS,
    "Tevatron/wikipedia-trivia": DEFAULT_PROCESSORS,
    "Tevatron/wikipedia-curated": DEFAULT_PROCESSORS,
    "Tevatron/wikipedia-wq": DEFAULT_PROCESSORS,
    "Tevatron/wikipedia-squad": DEFAULT_PROCESSORS,
    "Tevatron/scifact": DEFAULT_PROCESSORS,
    "Tevatron/msmarco-passage": DEFAULT_PROCESSORS,
    "json": [None, None, None],
}

logger = logging.getLogger(__name__)


class HFTrainDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str, is_validation=False):
        if is_validation:
            data_files = data_args.val_dir
            self.neg_num = data_args.eval_n_passages - 1
            self.ann_neg_num = 0
        else:
            data_files = data_args.train_dir
            self.neg_num = data_args.train_n_passages - 1
            self.ann_neg_num = data_args.ann_neg_num

        if data_files:
            self.dataset = load_from_disk(data_files)
        else:
            assert not is_validation
            self.dataset = load_dataset(
                data_args.dataset_name,
                data_args.dataset_language,
                cache_dir=cache_dir,
                use_auth_token=True,
            )[data_args.dataset_split]

        self.preprocessor = (
            PROCESSOR_INFO[data_args.dataset_name][0]
            if data_args.dataset_name in PROCESSOR_INFO
            else DEFAULT_PROCESSORS[0]
        )
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.keep_data_in_memory = data_args.keep_data_in_memory
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)
        self.add_var_token = data_args.add_var_token
        self.exclude_title = data_args.exclude_title
        self.is_validation = is_validation
        logger.info(f"exclude_title:{self.exclude_title}; add_var_token: {self.add_var_token}")
        logger.info(f"ann_negatives:{self.ann_neg_num}; is_validation: {self.is_validation}")

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        ann_negs = True if self.ann_neg_num != 0 else False
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len,
                                  self.p_max_len,
                                  separator=self.separator,
                                  exclude_title=self.exclude_title,
                                  add_var_token=self.add_var_token,
                                  ann_negatives=ann_negs),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on train dataset",
                load_from_cache_file=False,
                keep_in_memory=self.keep_data_in_memory
            )
        return self.dataset


class HFQueryDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        if data_args.hf_disk_dataset is not None:
            logger.info(f"loading dataset from {data_args.hf_disk_dataset}")
            self.dataset = load_from_disk(data_args.hf_disk_dataset)
        else:
            data_files = data_args.encode_in_path
            if data_files:
                data_files = {data_args.dataset_split: data_files}
            self.dataset = load_dataset(
                data_args.dataset_name,
                data_args.dataset_language,
                data_files=data_files,
                cache_dir=cache_dir,
                use_auth_token=True,
            )[data_args.dataset_split]
        self.preprocessor = (
            PROCESSOR_INFO[data_args.dataset_name][1]
            if data_args.dataset_name in PROCESSOR_INFO
            else DEFAULT_PROCESSORS[1]
        )
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.proc_num = data_args.dataset_proc_num
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)
        self.add_var_token = data_args.add_var_token

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.q_max_len,
                                  separator=self.separator,
                                  add_var_token=self.add_var_token),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization",
                load_from_cache_file=False
            )
        return self.dataset


class HFCorpusDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments, cache_dir: str):
        if data_args.hf_disk_dataset is not None:
            logger.info(f"loading dataset from {data_args.hf_disk_dataset}")
            self.dataset = load_from_disk(data_args.hf_disk_dataset)
        else:
            data_files = data_args.encode_in_path
            if data_files:
                data_files = {data_args.dataset_split: data_files}
            self.dataset = load_dataset(
                data_args.dataset_name,
                data_args.dataset_language,
                data_files=data_files,
                cache_dir=cache_dir,
                use_auth_token=True,
            )[data_args.dataset_split]
        script_prefix = data_args.dataset_name
        if script_prefix.endswith("-corpus"):
            script_prefix = script_prefix[:-7]
        self.preprocessor = (
            PROCESSOR_INFO[script_prefix][2] if script_prefix in PROCESSOR_INFO else DEFAULT_PROCESSORS[2]
        )
        self.tokenizer = tokenizer
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.separator = getattr(self.tokenizer, data_args.passage_field_separator, data_args.passage_field_separator)

        self.exclude_title = data_args.exclude_title
        logger.info(f"exclude_title:{self.exclude_title}")
        self.add_var_token = data_args.add_var_token

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.p_max_len, separator=self.separator,
                                  exclude_title=self.exclude_title,
                                  add_var_token=self.add_var_token),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenization",
                load_from_cache_file=False
            )
        return self.dataset
