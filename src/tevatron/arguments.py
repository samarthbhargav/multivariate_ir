import os
from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # modeling
    untie_encoder: bool = field(default=False, metadata={"help": "no weight sharing between qry passage encoders"})

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)
    normalize: bool = field(default=False)

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )


@dataclass
class DataArguments:
    train_dir: str = field(default=None, metadata={"help": "Path to train directory"})
    val_dir: str = field(default=None, metadata={"help": "Path to validation directory"})
    dataset_name: str = field(default=None, metadata={"help": "huggingface dataset name"})
    passage_field_separator: str = field(default=" ")
    dataset_proc_num: int = field(default=12, metadata={"help": "number of proc used in dataset preprocess"})
    train_n_passages: int = field(default=8)
    eval_n_passages: int = field(default=31)
    positive_passage_no_shuffle: bool = field(default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"}
    )

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    hf_disk_dataset: str = field(default=None,
                                 metadata={"help": "Path to dataset (loaded using datasets.load_from_disk)"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )

    exclude_title: bool = field(default=False)
    add_var_token: bool = field(default=False)

    ann_neg_num: int = field(
        default=0,
        metadata={
            "help": "The number of ANN hard negatives. If dataset does not contain column ann_negatives, ann_neg_num should be 0"
        },
    )

    pseudolabels: bool = field(default=False)
    group_1_size: int = field(default=5)
    group_2_size: int = field(default=45)
    group_3_size: int = field(default=150)
    group_1: int = field(default=5)
    group_2: int = field(default=12)
    group_3: int = field(default=13)

    def __post_init__(self):
        if self.dataset_name is not None:
            info = self.dataset_name.split("/")
            self.dataset_split = info[-1] if len(info) == 3 else "train"
            self.dataset_name = "/".join(info[:-1]) if len(info) == 3 else "/".join(info)
            self.dataset_language = "default"
            if ":" in self.dataset_name:
                self.dataset_name, self.dataset_language = self.dataset_name.split(":")
        else:
            self.dataset_name = "json"
            self.dataset_split = "train"
            self.dataset_language = "default"
        if self.train_dir is not None:
            if os.path.isdir(self.train_dir):
                files = os.listdir(self.train_dir)
                # change all train directory paths to absolute
                self.train_dir = os.path.join(os.path.abspath(os.getcwd()), self.train_dir)
                self.train_path = [
                    os.path.join(self.train_dir, f) for f in files if f.endswith("jsonl") or f.endswith("json")
                ]
            else:
                self.train_path = [self.train_dir]
        else:
            self.train_path = None


@dataclass
class TevatronTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)

    disable_distributed: bool = field(default=False, metadata={"help": "If set, disables distributed training"})
    early_stopping_patience: int = field(default=0,
                                         metadata={"help": "If > 0, enables early stopping, set to patience "})
    early_stopping_threshold: float = field(default=0, metadata={"help": "early stopping threshold"})

    load_model_from_disk: bool = field(default=False, metadata={"help": "load model that was trained with tevatron"})


@dataclass
class MVRLTrainingArguments:
    model_type: str = field(default="default", metadata={"help": "either default or mvrl"})
    var_activation: str = field(default="softplus", metadata={"help": "either softplus or logvar"})
    var_activation_param_b: float = field(default=None,
                                          metadata={"help": "the param 'beta' for var_activation=softplus"})
    embed_during_train: bool = field(default=False, metadata={
        "help": "if set, uses the embedding similarity instead of KL in the loss"})
    embed_formulation: str = field(default="original",
                                   metadata={"help": "whether to use the 'original' or 'updated' formulation"})

