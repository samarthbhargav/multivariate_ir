import logging
import os
import sys
import json

import torch
import pandas as pd

from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed, EarlyStoppingCallback

from tevatron.arguments import DataArguments, MVRLTrainingArguments
from tevatron.distillation.arguments import DistilModelArguments, DistilTrainingArguments
from tevatron.distillation.data import DistilTrainCollator, DistilTrainDataset, HFDistilTrainDataset
from tevatron.distillation.trainer import DistilTrainer, ListwiseDistilTrainer, ListwiseDistilLabelsTrainer, ListwiseDistilPseudolabelsTrainer
from tevatron.driver.train import compute_metrics
from tevatron.modeling import DenseModel
from tevatron.modeling.dense_mvrl import MVRLDenseModel
from tevatron.reranker.modeling import RerankerModel

logger = logging.getLogger(__name__)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group(rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def main():
    parser = HfArgumentParser((DistilModelArguments, DataArguments, DistilTrainingArguments, MVRLTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, mvrl_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, mvrl_args = parser.parse_args_into_dataclasses()
        model_args: DistilModelArguments
        data_args: DataArguments
        training_args: DistilTrainingArguments
        mvrl_args: MVRLTrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    if not training_args.disable_distributed:
        setup(rank=training_args.local_rank, world_size=torch.cuda.device_count())

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

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

    if training_args.load_model_from_disk:
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
        else:
            raise NotImplementedError(mvrl_args.model_type)

    else:

        if mvrl_args.model_type == "default":
            model = DenseModel.build(
                model_args,
                training_args,
                config=config,
                cache_dir=model_args.cache_dir,
            )
        elif mvrl_args.model_type == "mvrl_no_distill":
            raise ValueError(f"wrong model_type: {mvrl_args.model_type}")
        elif mvrl_args.model_type == "mvrl":
            model = MVRLDenseModel.build(
                model_args,
                training_args,
                mvrl_args,
                config=config,
                cache_dir=model_args.cache_dir,
            )
        else:
            raise NotImplementedError(mvrl_args.model_type)

    if mvrl_args.model_type.startswith("mvrl"):
        logger.info(f"adding VAR token! special tokens before: {tokenizer.all_special_tokens}")
        tokenizer.add_special_tokens({'additional_special_tokens': ['[VAR]']}, replace_additional_special_tokens=False)
        model.resize_token_space(len(tokenizer))
        logger.info(f"special tokens after: {tokenizer.all_special_tokens}")

    teacher_config = AutoConfig.from_pretrained(
        model_args.teacher_config_name if model_args.teacher_config_name else model_args.teacher_model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        model_args.teacher_tokenizer_name
        if model_args.teacher_tokenizer_name
        else model_args.teacher_model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    teacher_model = RerankerModel.load(
        model_name_or_path=model_args.teacher_model_name_or_path,
        config=teacher_config,
        cache_dir=model_args.cache_dir,
    )
    teacher_model.to(training_args.device)
    teacher_model.eval()

    train_dataset = HFDistilTrainDataset(
        student_tokenizer=tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        data_args=data_args,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir,
    )

    if training_args.local_rank > 0:
        print("Waiting for main process to perform the mapping")
        if not training_args.disable_distributed:
            torch.distributed.barrier()

    train_dataset = DistilTrainDataset(data_args, train_dataset.process(), tokenizer, teacher_tokenizer)

    if training_args.do_eval and data_args.val_dir:
        val_dataset = HFDistilTrainDataset(student_tokenizer=tokenizer,
                                           teacher_tokenizer=teacher_tokenizer,
                                           data_args=data_args,
                                           cache_dir=data_args.data_cache_dir or model_args.cache_dir,
                                           is_validation=True)
        logger.info(f"validation dataset: {val_dataset.dataset}")
        val_dataset = DistilTrainDataset(data_args,
                                         val_dataset.process(),
                                         tokenizer,
                                         teacher_tokenizer,
                                         is_validation=True)
    else:
        val_dataset = None
        logger.info(f"no validation dataset selected!")

    if training_args.local_rank == 0:
        print("Loading results from main process")
        if not training_args.disable_distributed:
            torch.distributed.barrier()

    if training_args.early_stopping_patience > 0:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience,
                                           early_stopping_threshold=training_args.early_stopping_threshold)]
    else:
        callbacks = None

    if training_args.kd_type== "drd":
        trainer = ListwiseDistilTrainer(
            teacher_model=teacher_model,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
            data_collator=DistilTrainCollator(
                tokenizer, teacher_tokenizer=teacher_tokenizer, max_p_len=data_args.p_max_len,
                max_q_len=data_args.q_max_len
            ),
        )
    elif training_args.kd_type == "drd_labels":
        trainer = ListwiseDistilLabelsTrainer(
            teacher_model=teacher_model,
            data_args=data_args,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
            data_collator=DistilTrainCollator(
                tokenizer, teacher_tokenizer=teacher_tokenizer, max_p_len=data_args.p_max_len,
                max_q_len=data_args.q_max_len
            ),
        )
    elif training_args.kd_type == "cldrd":
        trainer = ListwiseDistilPseudolabelsTrainer(
            teacher_model=teacher_model,
            data_args=data_args,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
            data_collator=DistilTrainCollator(
                tokenizer, teacher_tokenizer=teacher_tokenizer, max_p_len=data_args.p_max_len,
                max_q_len=data_args.q_max_len
            ),
        )
    elif training_args.kd_type == "kl":
        trainer = DistilTrainer(
            teacher_model=teacher_model,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
            data_collator=DistilTrainCollator(
                tokenizer, teacher_tokenizer=teacher_tokenizer, max_p_len=data_args.p_max_len,
                max_q_len=data_args.q_max_len
            ),
        )

    if trainer.is_world_process_zero():
        logger.info(f"saving tokenizer to {training_args.output_dir}")
        tokenizer.save_pretrained(training_args.output_dir)

    train_dataset.trainer = trainer

    if val_dataset:
        val_dataset.trainer = trainer

    trainer.train()  # TODO: resume training

    if val_dataset:
        eval_result = trainer.evaluate(eval_dataset=val_dataset)
    else:
        eval_result = {}

    logger.info(f"evaluation result: {eval_result}")

    trainer.save_model()

    pd.DataFrame(trainer.state.log_history).to_csv(os.path.join(training_args.output_dir, "trainer_state.csv"))

    with open(os.path.join(training_args.output_dir, "eval_result.json"), "w") as writer:
        json.dump(eval_result, writer)

    if not training_args.disable_distributed:
        cleanup()


if __name__ == "__main__":
    main()
