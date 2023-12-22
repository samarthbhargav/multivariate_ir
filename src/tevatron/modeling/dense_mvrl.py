import logging
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from . import EncoderOutput
from .dense import DenseModel
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class MVRLDenseModel(DenseModel):
    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            output_dim: int,
            pooler: nn.Module = None,
            untie_encoder: bool = False,
            negatives_x_device: bool = False,
            var_activation="softplus",
            var_activation_params: Dict = None
    ):
        super().__init__(lm_q=lm_q, lm_p=lm_p, pooler=pooler, untie_encoder=untie_encoder,
                         negatives_x_device=negatives_x_device)

        self.projection_dim = int(output_dim / 2) - 1
        self.projection_mean = nn.Linear(output_dim, self.projection_dim, bias=False)

        if var_activation == "softplus":
            assert "beta" in var_activation_params
            self.projection_var = nn.Sequential(
                nn.Linear(output_dim, self.projection_dim, bias=False),
                nn.Softplus(beta=var_activation_params["beta"])
            )
        else:
            # assume that you're producing LogVar
            self.projection_var = nn.Linear(output_dim, self.projection_dim, bias=False)
            raise NotImplementedError("TODO")

        logger.info("projection_var: {self.projection_var}")

    def get_faiss_embed(self, means, vars):
        raise NotImplementedError("TODO")

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps_mean, q_reps_var = self.encode_query(query)
        p_reps_mean, p_reps_var = self.encode_passage(passage)

        # for inference
        if q_reps_mean is None or p_reps_mean is None:
            return EncoderOutput(q_reps=self.get_faiss_embed(q_reps_mean, q_reps_var),
                                 p_reps=self.get_faiss_embed(p_reps_mean, p_reps_var))

        # for training
        if self.training:
            if self.negatives_x_device:
                q_reps_mean = self._dist_gather_tensor(q_reps_mean)
                q_reps_var = self._dist_gather_tensor(q_reps_var)
                p_reps_mean = self._dist_gather_tensor(p_reps_mean)
                p_reps_var = self._dist_gather_tensor(p_reps_var)

            scores = self.compute_similarity(q_reps_mean=q_reps_mean,
                                             q_reps_var=q_reps_var,
                                             p_reps_mean=p_reps_mean,
                                             p_reps_var=p_reps_var)
            scores = scores.view(q_reps_mean.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps_mean.size(0) // q_reps_mean.size(0))

            loss = self.compute_loss(scores, target)
            if self.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
        # for eval
        else:
            scores = self.compute_similarity(q_reps_mean=q_reps_mean,
                                             q_reps_var=q_reps_var,
                                             p_reps_mean=p_reps_mean,
                                             p_reps_var=p_reps_var)
            loss = None

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=self.get_faiss_embed(q_reps_mean, q_reps_var),
            p_reps=self.get_faiss_embed(q_reps_mean, q_reps_var),
        )

    def compute_loss(self, scores, target):
        raise NotImplementedError("TODO")

    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        if self.pooler is not None:
            p_reps = self.pooler(p=p_hidden)  # D * d
        else:
            p_reps = p_hidden[:, 0]
        # TODO: what is pooler here? should we disable it?
        return self.projection_mean(p_reps), self.projection_var(p_hidden[:, 1])

    def encode_query(self, qry):
        if qry is None:
            return None
        qry_out = self.lm_q(**qry, return_dict=True)
        q_hidden = qry_out.last_hidden_state
        if self.pooler is not None:
            q_reps = self.pooler(q=q_hidden)
        else:
            q_reps = q_hidden[:, 0]
        # TODO: what is pooler here? should we disable it?
        return self.projection_mean(q_reps), self.projection_var(q_hidden[:, 1])

    def compute_similarity(self, q_reps_mean, q_reps_var=None, p_reps_mean, p_reps_var):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
