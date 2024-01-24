import logging
import os
from collections import defaultdict
from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from transformers.trainer import Trainer
import pytrec_eval

from .loss import DistributedContrastiveLoss, SimpleContrastiveLoss

logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache

    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False


class TevatronTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(TevatronTrainer, self).__init__(*args, **kwargs)
        self._dist_loss_scale_factor = dist.get_world_size() if self.args.negatives_x_device else 1

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        self.model.load_from(self.state.best_model_checkpoint)

    def _prepare_inputs(
            self, inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepared = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                prepared.append(x.to(self.args.device))
            else:
                prepared.append(super()._prepare_inputs(x))
        return prepared

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs):
        query, passage = inputs
        return model(query=query, passage=passage).loss

    def training_step(self, *args):
        return super(TevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
                Perform an evaluation step on `model` using `inputs`.

                NOTE: in our function, the logits are NOT logits, but (ranking) scores

                Subclass and override to inject custom behavior.

                Args:
                    model (`nn.Module`):
                        The model to evaluate.
                    inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                        The inputs and targets of the model.

                        The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                        argument `labels`. Check your model's documentation for all accepted arguments.
                    prediction_loss_only (`bool`):
                        Whether or not to return the loss only.
                    ignore_keys (`List[str]`, *optional*):
                        A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                        gathering predictions.

                Return:
                    Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
                    logits and labels (each being optional).



        """
        queries, passages, labels = self._prepare_inputs(inputs)
        with torch.no_grad():
            out = model(query=queries, passage=passages)
            scores = out.scores

        if prediction_loss_only:
            return out.loss, None, None

        assert scores.size(0) == queries["input_ids"].size(0)
        assert scores.size(1) == passages["input_ids"].size(0)

        run = defaultdict(dict)
        qrel = defaultdict(dict)
        qids = []
        doc_ids = []
        for l in labels:
            for _, (qid, doc_id, rel) in enumerate(l):
                # only add QID once
                if _ == 0:
                    qids.append(qid)

                doc_ids.append(doc_id)
                if rel > 0:
                    qrel[qid][doc_id] = rel

        for i, qid in enumerate(qids):
            for j, doc_id in enumerate(doc_ids):
                run[qid][doc_id] = float(scores[i, j].cpu())
        res = pytrec_eval.RelevanceEvaluator(qrel, ["recip_rank"]).evaluate(run)
        ret_logits = torch.zeros(len(qids))
        for i, qid in enumerate(qids):
            ret_logits[i] = res[qid]["recip_rank"]

        # TODO: this is super hacky
        # we only need pass the logits with the MRR scores,
        # return a dummy zero vector -- otherwise HF will ignore it
        return out.loss, ret_logits, torch.zeros(len(qids))


def split_dense_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    return [{arg_key: c} for c in chunked_arg_val]


def get_dense_rep(x):
    if x.q_reps is None:
        return x.p_reps
    else:
        return x.q_reps


class GCTrainer(TevatronTrainer):
    def __init__(self, *args, **kwargs):
        logger.info("Initializing Gradient Cache Trainer")
        if not _grad_cache_available:
            raise ValueError(
                "Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache."
            )
        super(GCTrainer, self).__init__(*args, **kwargs)

        loss_fn_cls = DistributedContrastiveLoss if self.args.negatives_x_device else SimpleContrastiveLoss
        loss_fn = loss_fn_cls()

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_dense_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None,
        )

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        queries, passages = self._prepare_inputs(inputs)
        queries, passages = {"query": queries}, {"passage": passages}

        _distributed = self.args.local_rank > -1
        self.gc.models = [model, model]
        loss = self.gc(queries, passages, no_sync_except_last=_distributed)

        return loss / self._dist_loss_scale_factor
