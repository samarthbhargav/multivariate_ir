import copy
import json
import logging
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import MultivariateNormal

from . import EncoderOutput
from .dense import DenseModel
from transformers import PreTrainedModel, TrainingArguments

from ..arguments import ModelArguments, MVRLTrainingArguments

logger = logging.getLogger(__name__)


class MVRLDenseModel(DenseModel):

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            mvrl_args: MVRLTrainingArguments,
            **hf_kwargs,
    ):
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(model_args.model_name_or_path, "query_model")
                _psg_model_path = os.path.join(model_args.model_name_or_path, "passage_model")
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f"loading query model weight from {_qry_model_path}")
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f"loading passage model weight from {_psg_model_path}")
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(_psg_model_path, **hf_kwargs)
            else:
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            output_dim=model_args.projection_in_dim,
            var_activation=mvrl_args.var_activation,
            var_activation_params={"beta": mvrl_args.var_activation_param_b},
            pooler=pooler,
            embed_during_train=mvrl_args.embed_during_train,
            embed_formulation=mvrl_args.embed_formulation,
            negatives_x_device=train_args.negatives_x_device,
            untie_encoder=model_args.untie_encoder,
        )
        return model

    @classmethod
    def load(
            cls,
            model_args,
            model_name_or_path,
            mvrl_args: MVRLTrainingArguments,
            **hf_kwargs,
    ):
        if os.path.isdir(model_name_or_path):
            _qry_model_path = os.path.join(model_name_or_path, "query_model")
            _psg_model_path = os.path.join(model_name_or_path, "passage_model")
            if os.path.exists(_qry_model_path):
                logger.info(f"found separate weight for query/passage encoders")
                logger.info(f"loading query model weight from {_qry_model_path}")
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(_qry_model_path, **hf_kwargs)
                logger.info(f"loading passage model weight from {_psg_model_path}")
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(_psg_model_path, **hf_kwargs)
                untie_encoder = False
            else:
                logger.info(f"try loading tied weight")
                logger.info(f"loading model weight from {model_name_or_path}")
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        else:
            logger.info(f"try loading tied weight")
            logger.info(f"loading model weight from {model_name_or_path}")
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, "pooler.pt")
        pooler_config = os.path.join(model_name_or_path, "pooler_config.json")
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f"found pooler weight and configuration")
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.load_pooler(model_name_or_path, **pooler_config_dict)
        else:
            pooler = None

        model = cls(lm_q=lm_q,
                    lm_p=lm_p,
                    pooler=pooler,
                    untie_encoder=model_args.untie_encoder,
                    output_dim=model_args.projection_in_dim,
                    embed_during_train=mvrl_args.embed_during_train,
                    embed_formulation=mvrl_args.embed_formulation,
                    var_activation=mvrl_args.var_activation,
                    var_activation_params={"beta": mvrl_args.var_activation_param_b})
        return model

    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            output_dim: int,
            pooler: nn.Module = None,
            untie_encoder: bool = False,
            negatives_x_device: bool = False,
            var_activation="softplus",
            embed_during_train=False,
            embed_formulation="original",
            var_activation_params: Dict = None
    ):
        super().__init__(lm_q=lm_q, lm_p=lm_p, pooler=pooler, untie_encoder=untie_encoder,
                         negatives_x_device=negatives_x_device)

        self.projection_dim = int(output_dim / 2) - 1
        self.projection_mean = nn.Linear(output_dim, self.projection_dim, bias=False)

        self.var_activation = var_activation
        self.var_activation_params = var_activation_params

        if self.var_activation == "softplus":
            assert var_activation_params.get("beta") is not None
            self.projection_var = nn.Sequential(
                nn.Linear(output_dim, self.projection_dim, bias=False),
                nn.Softplus(beta=var_activation_params["beta"])
            )
        else:
            self.projection_var = nn.Linear(output_dim, self.projection_dim, bias=False)
            raise NotImplementedError("TODO")

        self.embed_during_train = embed_during_train
        self.embed_formulation = embed_formulation
        assert self.embed_formulation in {"original", "updated"}
        logger.info(f"embed_during_train:{self.embed_during_train}, embed_formulation: {self.embed_formulation}")
        logger.info(f"projection_var: {self.projection_var}")
        logger.info(f"projection_mean: {self.projection_mean}")

    def resize_token_space(self, num_tokens):
        # if untied, resize both the lm_q and lm_p models
        if self.untie_encoder:
            self.lm_q.resize_token_embeddings(num_tokens)
            self.lm_p.resize_token_embeddings(num_tokens)
        else:
            # just resizing once is enough, since they're both the same models
            self.lm_q.resize_token_embeddings(num_tokens)

    def get_faiss_embed(self, mean_var, is_query, eps=1e-9):
        means, var = mean_var
        BZ = means.size(0)
        D = means.size(1)

        if self.embed_formulation == "original":
            rep = torch.ones(BZ, 2 + 2 * D, device=means.device)

            if is_query:
                # 1, \sum var, mean^2, mean
                # rep[:, 1] = (var + eps).prod(1)
                rep[:, 1] = var.prod(1)
                rep[:, 2:2 + D] = means ** 2
                rep[:, 2 + D:] = means
            else:
                # doc prior, -1/\sum var, -1/var, (2*mu)/var
                rep[:, 0] = -1 * (torch.log(var) + (means ** 2) / var).sum()
                rep[:, 1] = (-1 / (var.prod(1) + eps))
                rep[:, 2:2 + D] = (-1 / var)
                rep[:, 2 + D:] = (2 * means) / var

            assert not torch.isinf(rep).any() and not torch.isnan(rep).any(), "obtained infs in representation"
            return rep
        else:
            rep = torch.zeros(BZ, 1 + 3 * D, device=means.device)
            if is_query:
                rep[:, 0] = 1
                rep[:, 1:D + 1] = var
                rep[:, D + 1:2 * D + 1] = means ** 2
                rep[:, 2 * D + 1:] = means
            else:
                rep[:, 0] = torch.log(var).sum()
                rep[:, 1:D + 1] = 1 / var
                rep[:, D + 1:2 * D + 1] = (1 / var)
                rep[:, 2 * D + 1:] = (-2 * means) / var

            assert not torch.isinf(rep).any() and not torch.isnan(rep).any(), "obtained infs in representation"
            return rep

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)

        if q_reps is None:
            return EncoderOutput(q_reps=None, p_reps=self.get_faiss_embed(p_reps, is_query=False))
        if p_reps is None:
            return EncoderOutput(q_reps=self.get_faiss_embed(q_reps, is_query=True), p_reps=None)

        q_reps_mean, q_reps_var = q_reps
        p_reps_mean, p_reps_var = p_reps
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
            q_reps=self.get_faiss_embed((q_reps_mean, q_reps_var), is_query=True),
            p_reps=self.get_faiss_embed((q_reps_mean, q_reps_var), is_query=False),
        )

    def encode_passage(self, psg):
        if psg is None:
            return None
        psg_out = self.lm_p(**psg, return_dict=True)
        p_hidden = psg_out.last_hidden_state
        if self.pooler is not None:
            p_reps = self.pooler(p=p_hidden)  # D * d
        else:
            p_reps = p_hidden[:, 0]
        # assumes VAR token is always after CLS
        # TODO: any way to check the above?
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

        mean = self.projection_mean(q_reps)
        # assumes VAR token is always after CLS
        # TODO: any way to check the above?
        var = self.projection_var(q_hidden[:, 1])
        return mean, var

    def compute_similarity(self, q_reps_mean, p_reps_mean, q_reps_var=None, p_reps_var=None):
        if self.embed_during_train:
            return super().compute_similarity(q_reps=self.get_faiss_embed((q_reps_mean, q_reps_var), is_query=True),
                                              p_reps=self.get_faiss_embed((p_reps_mean, p_reps_var), is_query=False))

        else:
            kl = torch.zeros(q_reps_mean.size(0), p_reps_mean.size(0), device=q_reps_mean.device)
            p = []
            q = []

            for i in range(q_reps_mean.size(0)):
                q.append(MultivariateNormal(q_reps_mean[i, :], torch.diag(q_reps_var[i, :])))

            for i in range(p_reps_mean.size(0)):
                p.append(MultivariateNormal(p_reps_mean[i, :], torch.diag(p_reps_var[i, :])))

            # Q x P
            for (i, j) in np.ndindex(len(q), len(p)):
                kl[i, j] = -torch.distributions.kl_divergence(q[i], p[j])
            return kl

    def save(self, output_dir: str):
        super().save(output_dir)
        torch.save(self.projection_mean.state_dict(), os.path.join(output_dir, "projection_mean"))
        torch.save(self.projection_var.state_dict(), os.path.join(output_dir, "projection_var"))

    def load_from(self, input_dir: str):
        super().load_from(input_dir)
        mean_path = os.path.join(input_dir, "projection_mean")
        logger.info(f"loading projection_mean from {mean_path}")
        self.projection_mean.load_state_dict(torch.load(mean_path, map_location="cpu"))
        var_path = os.path.join(input_dir, "projection_var")
        logger.info(f"loading projection_var from {var_path}")
        self.projection_var.load_state_dict(torch.load(var_path, map_location="cpu"))
