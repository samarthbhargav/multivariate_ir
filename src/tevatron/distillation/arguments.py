from dataclasses import dataclass, field
from typing import Optional

from tevatron.arguments import ModelArguments, TevatronTrainingArguments


@dataclass
class DistilModelArguments(ModelArguments):
    teacher_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    teacher_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as teacher_model_name"}
    )
    teacher_tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as teacher_model_name"}
    )


@dataclass
class DistilTrainingArguments(TevatronTrainingArguments):
    teacher_temp: float = field(default=1)
    student_temp: float = field(default=1)
    kd_type: str = field(default="cldrd") #kl, cldrd, drd_labels, drd
    softmax_norm: bool = field(default=False)
    kd_in_batch_negs: bool = field(default=False)
    group_1: int = field(default=5)
    group_2: int = field(default=12)
    group_3: int = field(default=13)
