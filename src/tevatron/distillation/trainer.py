import logging
import os
import pytrec_eval
import torch
import torch.distributed as dist
import torch.nn.functional as F
from collections import defaultdict

from torch.nn.parallel import DistributedDataParallel

from tevatron.distillation.arguments import DistilTrainingArguments
from torch import nn
from torch.utils.data import DataLoader
from transformers.trainer import Trainer
from typing import Any, Dict, List, Optional, Tuple, Union

from tevatron.loss import DistributedContrastiveLoss, SimpleContrastiveLoss
from tevatron.trainer import split_dense_inputs, get_dense_rep

logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache

    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False


class DistilTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super(DistilTrainer, self).__init__(*args, **kwargs)
        self.args: DistilTrainingArguments
        self.teacher_model = teacher_model
        self._dist_loss_scale_factor = dist.get_world_size() if self.args.negatives_x_device else 1
        if self.args.negatives_x_device:
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

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
        query, passage, pair = inputs
        student_scores = model(query=query, passage=passage).scores
        with torch.no_grad():
            teacher_scores = self.teacher_model(pair=pair).scores
            if self.args.negatives_x_device:
                teacher_scores = self._dist_gather_tensor(teacher_scores)
        teacher_mat = torch.zeros(student_scores.shape, dtype=student_scores.dtype, device=teacher_scores.device)
        index = torch.arange(teacher_scores.size(0), device=teacher_scores.device)
        teacher_scores = torch.softmax(
            teacher_scores.view(student_scores.size(0), -1) / self.args.teacher_temp, dim=1, dtype=student_scores.dtype
        )
        teacher_mat = torch.scatter(
            teacher_mat, dim=-1, index=index.view(student_scores.size(0), -1), src=teacher_scores
        )
        student_scores = nn.functional.log_softmax(student_scores / self.args.student_temp, dim=1)
        loss = nn.functional.kl_div(student_scores, teacher_mat, reduction="batchmean") * self._dist_loss_scale_factor
        return loss

    def training_step(self, *args):
        return super(DistilTrainer, self).training_step(*args) / self._dist_loss_scale_factor

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

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
        queries, passages, _, labels = self._prepare_inputs(inputs)
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

    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        self.model.load_from(self.state.best_model_checkpoint)


class ListwiseDistilTrainer(DistilTrainer):
    """ MVRL paper: using teacher scores."""

    def compute_loss(self, model, inputs):
        query, passage, pair = inputs
        student_scores = model(query=query, passage=passage).scores
        with torch.no_grad():
            teacher_scores = self.teacher_model(pair=pair).scores
            if self.args.negatives_x_device:
                teacher_scores = self._dist_gather_tensor(teacher_scores)

        index = torch.arange(teacher_scores.size(0), device=teacher_scores.device)

        if self.args.kd_in_batch_negs:
            # in-batch negatives are assigned 0 values
            teacher_mat = torch.zeros(student_scores.shape, dtype=student_scores.dtype, device=teacher_scores.device)

            if self.args.softmax_norm:
                teacher_scores = torch.softmax(
                    teacher_scores.view(student_scores.size(0), -1) / self.args.teacher_temp, dim=1,
                    dtype=student_scores.dtype
                )
                student_scores = torch.softmax(student_scores / self.args.student_temp, dim=1)

            else:
                teacher_scores = teacher_scores.view(student_scores.size(0), -1)

            teacher_mat = torch.scatter(
                teacher_mat, dim=-1, index=index.view(student_scores.size(0), -1),
                src=teacher_scores.to(teacher_mat.dtype)
            )

            teacher_scores = teacher_mat

        else:
            teacher_scores = teacher_scores.view(student_scores.size(0), -1)
            student_scores = torch.gather(student_scores, 1, index.view(student_scores.size(0), -1))

            if self.args.softmax_norm:
                teacher_scores = torch.softmax(teacher_scores / self.args.teacher_temp, dim=1)
                student_scores = torch.softmax(student_scores / self.args.student_temp, dim=1)

        # sort predicted scores
        student_scores_sorted, indices_pred = student_scores.sort(descending=True, dim=-1)

        # sort true w.r.t sorted predicted scores
        true_sorted_by_preds = torch.gather(teacher_scores, dim=1, index=indices_pred)

        # compute all possible pairs
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        pairs_mask = true_diffs > 0

        # inverse rank of passage
        inv_pos_idxs = 1. / torch.arange(1, student_scores.shape[1] + 1).to(teacher_scores.device)
        weights = torch.abs(inv_pos_idxs.view(1, -1, 1) - inv_pos_idxs.view(1, 1, -1))  # [1, topk, topk]

        # score differences (part of exp)
        scores_diffs = (student_scores_sorted[:, :, None] - student_scores_sorted[:, None, :])

        # logsumexp trick to avoid inf
        topk = scores_diffs.size(1)
        scores_diffs = scores_diffs.view(1, -1, 1)
        scores_diffs = F.pad(input=-scores_diffs, pad=(0, 1), mode='constant', value=0)
        scores = torch.logsumexp(scores_diffs, 2, True)
        scores = scores.view(-1, topk, topk)

        losses = scores * weights  # [bz, topk, topk]

        loss = torch.mean(losses[pairs_mask])

        return loss


class ListwiseDistilLabelsTrainer(DistilTrainer):
    """ MVRL paper with labels instead of scores"""

    def __init__(self, teacher_model, data_args, *args, **kwargs):
        super(ListwiseDistilLabelsTrainer, self).__init__(teacher_model, *args, **kwargs)
        self.data_args = data_args

    def compute_loss(self, model, inputs):
        query, passage, pair = inputs
        student_scores = model(query=query, passage=passage).scores

        with torch.no_grad():
            # teacher_scores = self.teacher_model(pair=pair).scores
            # if self.args.negatives_x_device:
            #     teacher_scores = self._dist_gather_tensor(teacher_scores)

            # pseudolabels instead of raw teacher scores
            soft_negs_size, hard_negs_size = (self.data_args.train_n_passages - 1), self.data_args.ann_neg_num

            teacher_scores = torch.ones(student_scores.size(0), 1 + soft_negs_size + hard_negs_size).to(
                student_scores.device)

            btz = teacher_scores.size(0)

            soft_negs_index = torch.arange(1, soft_negs_size + 1)
            soft_negs = torch.ones((btz, len(soft_negs_index)), dtype=int) * soft_negs_index
            soft_negs = soft_negs.to(teacher_scores.device)

            # BM25 negative passages have a label of 0
            teacher_scores = torch.scatter(teacher_scores, dim=-1,
                                           index=soft_negs,
                                           src=torch.zeros((btz, soft_negs.size(1)), device=teacher_scores.device))

            # ANN negative passages have a label of 0
            if self.data_args.ann_neg_num > 0:
                hard_negs_index = torch.arange(soft_negs_size + 1, soft_negs_size + hard_negs_size + 1)
                hard_negs = torch.ones((btz, len(hard_negs_index)), dtype=int) * hard_negs_index
                hard_negs = hard_negs.to(teacher_scores.device)
                teacher_scores = torch.scatter(teacher_scores, dim=-1,
                                               index=hard_negs,
                                               src=torch.zeros((btz, hard_negs.size(1)), device=teacher_scores.device))

            teacher_scores = teacher_scores.view(-1, 1)

        index = torch.arange(teacher_scores.size(0), device=teacher_scores.device)

        if self.args.kd_in_batch_negs:
            # in-batch negatives are assigned -1 values
            teacher_mat = -1 * torch.ones(student_scores.shape, dtype=student_scores.dtype,
                                          device=teacher_scores.device)

            if self.args.softmax_norm:
                teacher_scores = torch.softmax(
                    teacher_scores.view(student_scores.size(0), -1) / self.args.teacher_temp, dim=1,
                    dtype=student_scores.dtype
                )
                student_scores = torch.softmax(student_scores / self.args.student_temp, dim=1)

            else:
                teacher_scores = teacher_scores.view(student_scores.size(0), -1)

            teacher_mat = torch.scatter(
                teacher_mat, dim=-1, index=index.view(student_scores.size(0), -1),
                src=teacher_scores.to(teacher_mat.dtype)
            )

            teacher_scores = teacher_mat

        else:
            teacher_scores = teacher_scores.view(student_scores.size(0), -1)
            student_scores = torch.gather(student_scores, 1, index.view(student_scores.size(0), -1))

            if self.args.softmax_norm:
                teacher_scores = torch.softmax(teacher_scores / self.args.teacher_temp, dim=1)
                student_scores = torch.softmax(student_scores / self.args.student_temp, dim=1)

        # sort predicted scores
        student_scores_sorted, indices_pred = student_scores.sort(descending=True, dim=-1)

        # sort true w.r.t sorted predicted scores
        true_sorted_by_preds = torch.gather(teacher_scores, dim=1, index=indices_pred)

        # compute all possible pairs
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        pairs_mask = true_diffs > 0

        # inverse rank of passage
        inv_pos_idxs = 1. / torch.arange(1, student_scores.shape[1] + 1).to(teacher_scores.device)
        weights = torch.abs(inv_pos_idxs.view(1, -1, 1) - inv_pos_idxs.view(1, 1, -1))  # [1, topk, topk]

        # score differences (part of exp)
        scores_diffs = (student_scores_sorted[:, :, None] - student_scores_sorted[:, None, :])

        # logsumexp trick to avoid inf
        topk = scores_diffs.size(1)
        scores_diffs = scores_diffs.view(1, -1, 1)
        scores_diffs = F.pad(input=-scores_diffs, pad=(0, 1), mode='constant', value=0)
        scores = torch.logsumexp(scores_diffs, 2, True)
        scores = scores.view(-1, topk, topk)

        losses = scores * weights  # [bz, topk, topk]

        loss = torch.mean(losses[pairs_mask])

        return loss


class ListwiseDistilPseudolabelsTrainer(DistilTrainer):
    """ CLDRD paper: pseudolabels.
    The following is valid under the assumption that the teacher model that was used to create the dataset remains the same.
    If you want to use a different teacher model, than the one used for creating the dataset, uncomment the other
    implementation for ListwiseDistilPseudolabelsTrainer"""

    def __init__(self, teacher_model, data_args, *args, **kwargs):
        super(ListwiseDistilPseudolabelsTrainer, self).__init__(teacher_model, *args, **kwargs)
        self.data_args = data_args

        group1_size, group2_size, group3_size = self.data_args.group_1, self.data_args.group_2, self.data_args.group_3
        labels = torch.ones(group1_size + group2_size + group3_size) * (
                1 / torch.arange(1, group1_size + group2_size + group3_size + 1))

        hard_negs_index = torch.arange(group1_size, group1_size + group2_size)
        soft_negs_index = torch.arange((group1_size + group2_size), (group1_size + group2_size + group3_size))

        # 0 for group 2
        labels[hard_negs_index] = 0

        # -1 for group 3
        labels[soft_negs_index] = -1

        self.labels = labels

    def compute_loss(self, model, inputs):
        query, passage, pair = inputs

        student_scores = model(query=query, passage=passage).scores

        # with torch.no_grad():
        #    teacher_scores = self.teacher_model(pair=pair).scores
        #    if self.args.negatives_x_device:
        #         teacher_scores = self._dist_gather_tensor(teacher_scores)

        #    # CL-DRD approach: pseudolabels instead of raw teacher scores
        #    teacher_scores = teacher_scores.view(student_scores.size(0), -1)

        # labels w.r.t teacher
        teacher_scores = torch.ones(student_scores.size(0), self.labels.size(0)) * self.labels
        teacher_scores = teacher_scores.to(student_scores.device)

        index = torch.arange(teacher_scores.size(0) * teacher_scores.size(1), device=teacher_scores.device)
        student_scores = torch.gather(student_scores, 1, index.view(student_scores.size(0), -1))

        if self.args.softmax_norm:
            teacher_scores = torch.softmax(teacher_scores / self.args.teacher_temp, dim=1)
            student_scores = torch.softmax(student_scores / self.args.student_temp, dim=1)

        # sort predicted scores
        student_scores_sorted, indices_pred = student_scores.sort(descending=True, dim=-1)

        # sort true w.r.t sorted predicted scores
        true_sorted_by_preds = torch.gather(teacher_scores, dim=1, index=indices_pred)

        # compute all possible pairs
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        pairs_mask = true_diffs > 0

        # inverse rank of passage
        inv_pos_idxs = 1. / torch.arange(1, student_scores.shape[1] + 1).to(teacher_scores.device)
        weights = torch.abs(inv_pos_idxs.view(1, -1, 1) - inv_pos_idxs.view(1, 1, -1))  # [1, topk, topk]

        # score differences (part of exp)
        scores_diffs = (student_scores_sorted[:, :, None] - student_scores_sorted[:, None, :])

        # logsumexp trick to avoid inf
        topk = scores_diffs.size(1)
        scores_diffs = scores_diffs.view(1, -1, 1)
        scores_diffs = F.pad(input=-scores_diffs, pad=(0, 1), mode='constant', value=0)
        scores = torch.logsumexp(scores_diffs, 2, True)
        scores = scores.view(-1, topk, topk)

        losses = scores * weights  # [bz, topk, topk]

        loss = torch.mean(losses[pairs_mask])

        return loss


class GCListwiseDistilLabelsTrainer(ListwiseDistilLabelsTrainer):
    def __init__(self, teacher_model, *args, **kwargs):
        logger.info("Initializing Gradient Cache Trainer")
        if not _grad_cache_available:
            raise ValueError(
                "Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache."
            )
        super().__init__(teacher_model, *args, **kwargs)

        loss_fn_cls = DistributedContrastiveLoss if self.args.negatives_x_device else SimpleContrastiveLoss
        loss_fn = loss_fn_cls()

        logger.info("wrapping model in DDP")
        self.model = DistributedDataParallel(self.model,
                                             device_ids=[0],
                                             output_device=0)

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_dense_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None,
        )

    def _save(self, output_dir: Optional[str] = None):
        try:
            super()._save(output_dir)
        except AttributeError:
            self.model.module.save(output_dir)

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        queries, passages = self._prepare_inputs(inputs)
        queries, passages = {"query": queries}, {"passage": passages}

        _distributed = self.args.local_rank > -1
        # self.gc.models = [model, model]
        loss = self.gc(queries, passages, no_sync_except_last=_distributed)

        return loss / self._dist_loss_scale_factor

# class ListwiseDistilPseudolabelsTrainer(DistilTrainer):
#     """ CLDRD paper: pseudolabels.
#     The following covers the case where the current teacher model does not match the teacher used to create the dataset."""
#
#     def __init__(self, teacher_model, data_args, *args, **kwargs):
#         super(ListwiseDistilPseudolabelsTrainer, self).__init__(teacher_model, *args, **kwargs)
#         self.data_args = data_args
#
#     def compute_loss(self, model, inputs):
#         query, passage, pair = inputs
#         group1_size, group2_size, group3_size = self.data_args.group_1, self.data_args.group_2, self.data_args.group_3
#
#         student_scores = model(query=query, passage=passage).scores
#
#         with torch.no_grad():
#             teacher_scores = self.teacher_model(pair=pair).scores
#             if self.args.negatives_x_device:
#                 teacher_scores = self._dist_gather_tensor(teacher_scores)
#
#             # CL-DRD approach: pseudolabels instead of raw teacher scores
#             teacher_scores = teacher_scores.view(student_scores.size(0), -1)
#             indices_scores = teacher_scores.argsort(descending=True, dim=-1).argsort(dim=-1)
#
#             # inverse rank for group 1
#             teacher_scores = 1. / (indices_scores + 1)
#
#             btz = teacher_scores.size(0)
#
#             hard_negs_index = torch.arange(group1_size, group1_size + group2_size)
#             hard_negs = torch.ones((btz, len(hard_negs_index)), dtype=int) * hard_negs_index
#             hard_negs = hard_negs.to(teacher_scores.device)
#
#             soft_negs_index = torch.arange((group1_size + group2_size),
#                                            (group1_size + group2_size + group3_size))
#             soft_negs = torch.ones((btz, len(soft_negs_index)), dtype=int) * soft_negs_index
#             soft_negs = soft_negs.to(teacher_scores.device)
#
#             # 0 for group 2
#             teacher_scores = torch.scatter(teacher_scores, dim=-1,
#                                            index=hard_negs,
#                                            src=torch.zeros((btz, hard_negs.size(1)), device=teacher_scores.device))
#
#             # -1 for group 3
#             teacher_scores = torch.scatter(teacher_scores, dim=-1,
#                                            index=soft_negs,
#                                            src=-1 * torch.ones((btz, soft_negs.size(1)), device=teacher_scores.device))
#
#             teacher_scores = teacher_scores.view(-1, 1)
#
#         index = torch.arange(teacher_scores.size(0), device=teacher_scores.device)
#
#         teacher_scores = teacher_scores.view(student_scores.size(0), -1)
#         student_scores = torch.gather(student_scores, 1, index.view(student_scores.size(0), -1))
#
#         if self.args.softmax_norm:
#             teacher_scores = torch.softmax(teacher_scores / self.args.teacher_temp, dim=1)
#             student_scores = torch.softmax(student_scores / self.args.student_temp, dim=1)
#
#         # sort predicted scores
#         student_scores_sorted, indices_pred = student_scores.sort(descending=True, dim=-1)
#
#         # sort true w.r.t sorted predicted scores
#         true_sorted_by_preds = torch.gather(teacher_scores, dim=1, index=indices_pred)
#
#         # compute all possible pairs
#         true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
#         pairs_mask = true_diffs > 0
#
#         # inverse rank of passage
#         inv_pos_idxs = 1. / torch.arange(1, student_scores.shape[1] + 1).to(teacher_scores.device)
#         weights = torch.abs(inv_pos_idxs.view(1, -1, 1) - inv_pos_idxs.view(1, 1, -1))  # [1, topk, topk]
#
#         # score differences (part of exp)
#         scores_diffs = (student_scores_sorted[:, :, None] - student_scores_sorted[:, None, :])
#
#         # logsumexp trick to avoid inf
#         topk = scores_diffs.size(1)
#         scores_diffs = scores_diffs.view(1, -1, 1)
#         scores_diffs = F.pad(input=-scores_diffs, pad=(0, 1), mode='constant', value=0)
#         scores = torch.logsumexp(scores_diffs, 2, True)
#         scores = scores.view(-1, topk, topk)
#
#         losses = scores * weights  # [bz, topk, topk]
#
#         loss = torch.mean(losses[pairs_mask])
#
#         return loss
