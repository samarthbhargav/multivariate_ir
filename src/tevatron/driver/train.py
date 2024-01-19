import json
import logging
import os
import sys
import pandas as pd

import torch
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, set_seed, EvalPrediction, EarlyStoppingCallback

from tevatron.arguments import DataArguments, ModelArguments, MVRLTrainingArguments
from tevatron.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.data import QPCollator, TrainDataset
from tevatron.datasets import HFTrainDataset
from tevatron.modeling import DenseModel
from tevatron.modeling.dense_mvrl import MVRLDenseModel
from tevatron.trainer import GCTrainer
from tevatron.trainer import TevatronTrainer as Trainer

logger = logging.getLogger(__name__)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group(rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def compute_metrics(pred: EvalPrediction):
    # super hacky, but the predictions here are the MRR scores
    return {"mrr": float(pred.predictions.mean())}


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
    if mvrl_args.model_type == "default":
        model = DenseModel.build(
            model_args,
            training_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif mvrl_args.model_type == "mvrl_no_distill":
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

    train_dataset = HFTrainDataset(
        tokenizer=tokenizer, data_args=data_args, cache_dir=data_args.data_cache_dir or model_args.cache_dir
    )

    if training_args.local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()
    logger.info(f"train dataset: {train_dataset.dataset}")
    train_dataset = TrainDataset(data_args, train_dataset.process(), tokenizer)

    if training_args.do_eval and data_args.val_dir:
        val_dataset = HFTrainDataset(tokenizer=tokenizer, data_args=data_args,
                                     cache_dir=data_args.data_cache_dir or model_args.cache_dir,
                                     is_validation=True)
        logger.info(f"validation dataset: {val_dataset.dataset}")
        val_dataset = TrainDataset(data_args, val_dataset.process(), tokenizer, is_validation=True)
    else:
        val_dataset = None
        logger.info(f"no validation dataset selected!")

    if training_args.local_rank == 0:
        print("Loading results from main process")
        if not training_args.disable_distributed:
            torch.distributed.barrier()

    trainer_cls = GCTrainer if training_args.grad_cache else Trainer

    if training_args.early_stopping_patience > 0:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience,
                                           early_stopping_threshold=training_args.early_stopping_threshold)]
    else:
        callbacks = None

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        data_collator=QPCollator(tokenizer, max_p_len=data_args.p_max_len, max_q_len=data_args.q_max_len),
    )
    train_dataset.trainer = trainer
    if val_dataset:
        val_dataset.trainer = trainer

    trainer.train()

    logger.info(f"loading best model from {trainer.state.best_model_checkpoint}")
    # load best model
    if mvrl_args.model_type == "default":
        trainer.model = DenseModel.load(
            model_name_or_path=trainer.state.best_model_checkpoint,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif mvrl_args.model_type == "mvrl_no_distill":
        assert "[VAR]" in tokenizer.all_special_tokens
        post_config = AutoConfig.from_pretrained(
            trainer.state.best_model_checkpoint,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
        )
        trainer.model = MVRLDenseModel.load(
            model_args=model_args,
            mvrl_args=mvrl_args,
            model_name_or_path=trainer.state.best_model_checkpoint,
            config=post_config,
            cache_dir=model_args.cache_dir
        )
    else:
        raise NotImplementedError(mvrl_args.model_type)

    if val_dataset:
        eval_result = trainer.evaluate(eval_dataset=val_dataset)

    logger.info(f"evaluation result: {eval_result}")
    trainer.save_model()

    pd.DataFrame(trainer.state.log_history).to_csv(os.path.join(training_args.output_dir, "trainer_state.csv"))

    with open(os.path.join(training_args.output_dir, "eval_result.json"), "w") as writer:
        json.dump(eval_result, writer)

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)

    if not training_args.disable_distributed:
        cleanup()


if __name__ == "__main__":
    main()
