import copy
import json
import logging
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from . import EncoderOutput
from .dense import DenseModel
from transformers import PreTrainedModel, TrainingArguments

from ..arguments import ModelArguments, StochasticArguments

logger = logging.getLogger(__name__)


class ConcreteDropout(nn.Module):
    """
    Source: https://github.com/dscohen/LastLayersBayesianIR/blob/main/models/layers/concete_dropout.py#L8
    Concrete Dropout.
    Implementation of the Concrete Dropout module as described in the
    'Concrete Dropout' paper: https://arxiv.org/pdf/1705.07832
    """

    def __init__(self,
                 weight_regulariser: float,
                 dropout_regulariser: float,
                 init_min: float = 0.1,
                 init_max: float = 0.1) -> None:
        """Concrete Dropout.
        Parameters
        ----------
        weight_regulariser : float
            Weight regulariser term.
        dropout_regulariser : float
            Dropout regulariser term.
        init_min : float
            Initial min value.
        init_max : float
            Initial max value.
        """

        super().__init__()

        self.weight_regulariser = weight_regulariser
        self.dropout_regulariser = dropout_regulariser

        init_min = np.log(init_min) - np.log(1.0 - init_min)
        init_max = np.log(init_max) - np.log(1.0 - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.p = torch.sigmoid(self.p_logit)

        self.regularisation = 0.0

    def forward(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        """Calculates the forward pass.
        The regularisation term for the layer is calculated and assigned to a
        class attribute - this can later be accessed to evaluate the loss.
        Parameters
        ----------
        x : Tensor
            Input to the Concrete Dropout.
        layer : nn.Module
            Layer for which to calculate the Concrete Dropout.
        Returns
        -------
        Tensor
            Output from the dropout layer.
        """

        output = layer(self._concrete_dropout(x))

        sum_of_squares = 0
        for param in layer.parameters():
            sum_of_squares += torch.sum(torch.pow(param, 2))

        weights_reg = self.weight_regulariser * sum_of_squares / (1.0 - self.p)

        dropout_reg = self.p * torch.log(self.p)
        dropout_reg += (1.0 - self.p) * torch.log(1.0 - self.p)
        dropout_reg *= self.dropout_regulariser * x[0].numel()

        self.regularisation = weights_reg + dropout_reg

        return output

    def _concrete_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the Concrete Dropout.
        Parameters
        ----------
        x : Tensor
            Input tensor to the Concrete Dropout layer.
        Returns
        -------
        Tensor
            Outputs from Concrete Dropout.
        """

        eps = 1e-7
        tmp = 0.1

        self.p = torch.sigmoid(self.p_logit)
        u_noise = torch.rand_like(x)

        drop_prob = (torch.log(self.p + eps) -
                     torch.log(1 - self.p + eps) +
                     torch.log(u_noise + eps) -
                     torch.log(1 - u_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / tmp)

        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p

        x = torch.mul(x, random_tensor) / retain_prob

        return x


class StochasticDenseModel(DenseModel):

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            stoch_args: StochasticArguments,
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
            pooler=pooler,
            negatives_x_device=train_args.negatives_x_device,
            output_dim=model_args.projection_in_dim,
            untie_encoder=model_args.untie_encoder,
            n_iters=stoch_args.n_iters,
        )
        return model

    @classmethod
    def load(
            cls,
            model_name_or_path,
            model_args: ModelArguments,
            stoch_args: StochasticArguments,
            **hf_kwargs,
    ):
        # load local
        untie_encoder = True
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
                    untie_encoder=untie_encoder,
                    output_dim=model_args.projection_in_dim,
                    n_iters=stoch_args.n_iters)

        stoch_path_1 = os.path.join(model_name_or_path, "stoch_projection_1")
        logger.info(f"loading stoch_projection_1 from {stoch_path_1}")
        model.stoch_projection_1.load_state_dict(torch.load(stoch_path_1, map_location="cpu"))

        stoch_path_2 = os.path.join(model_name_or_path, "stoch_projection_2")
        logger.info(f"loading stoch_projection_2 from {stoch_path_2}")
        model.stoch_projection_2.load_state_dict(torch.load(stoch_path_2, map_location="cpu"))

        cd1_path = os.path.join(model_name_or_path, "cd1")
        logger.info(f"loading cd1 from {cd1_path}")
        model.cd1.load_state_dict(torch.load(cd1_path, map_location="cpu"))

        cd2_path = os.path.join(model_name_or_path, "cd2")
        logger.info(f"loading cd2 from {cd2_path}")
        model.cd2.load_state_dict(torch.load(cd2_path, map_location="cpu"))

        return model

    @staticmethod
    def get_mean_var(reps: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # returns a mean & variance vector
        # dims= BATCH_SIZE x N_ITERS x N_DIM
        assert reps.ndim == 3 and reps.size(1) > 1
        return reps.mean(1), reps.var(1)

    def __init__(
            self,
            lm_q: PreTrainedModel,
            lm_p: PreTrainedModel,
            output_dim: int,
            pooler: nn.Module = None,
            untie_encoder: bool = False,
            negatives_x_device: bool = False,
            n_iters: int = 100,
    ):
        super().__init__(lm_q=lm_q, lm_p=lm_p, pooler=pooler, untie_encoder=untie_encoder,
                         negatives_x_device=negatives_x_device)

        self.stoch_projection_1 = nn.Linear(output_dim, output_dim, bias=True)
        self.stoch_projection_2 = nn.Linear(output_dim, output_dim, bias=True)
        w, d = 1e-6, 1e-3
        self.cd1 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)
        self.cd2 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)
        self.n_iters = n_iters
        logger.info(f"stoch_projection: {self.stoch_projection_1}, {self.stoch_projection_2}")

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, n_iters=1):
        q_reps = self.encode_query(query, n_iters)
        reg = self.cd1.regularisation + self.cd2.regularisation
        p_reps = self.encode_passage(passage, n_iters)
        reg = reg + self.cd1.regularisation + self.cd2.regularisation

        # inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, p_reps=p_reps)

        # for training
        if self.training:
            if self.negatives_x_device:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps=q_reps,
                                             p_reps=p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))
            # also add the regularization from the CD layers
            loss = self.compute_loss(scores, target) + reg.squeeze()
            if self.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
        # for eval
        else:
            scores = self.compute_similarity(q_reps=q_reps,
                                             p_reps=p_reps)
            loss = None

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def _stoch(self, embeds, n_iters):
        out_reps = []
        for _ in range(n_iters):
            out_features = self.cd1(embeds, torch.nn.Sequential(self.stoch_projection_1, nn.ReLU()))
            out_features = self.cd2(out_features, torch.nn.Sequential(self.stoch_projection_2))
            out_reps.append(out_features)

        # size = n_iters * batch_size * 2, not what we want!
        out_reps = torch.stack(out_reps)
        # convert to batch_size * n_iters * 2
        out_reps.swapaxes_(0, 1)
        return out_reps

    def encode_passage(self, psg, n_iters):
        p_reps = super().encode_passage(psg)
        if p_reps is None:
            return None
        p_reps = self._stoch(embeds=p_reps, n_iters=n_iters)
        # n_iters =1 is typically used only during training,
        # so squeeze the middle dim
        if n_iters == 1:
            p_reps = p_reps.squeeze(1)
        return p_reps

    def encode_query(self, qry, n_iters):
        q_reps = super().encode_query(qry)
        if q_reps is None:
            return None
        q_reps = self._stoch(embeds=q_reps, n_iters=n_iters)
        # n_iters = 1 is typically used only during training,
        # so squeeze the middle dim
        if n_iters == 1:
            q_reps = q_reps.squeeze(1)
        return q_reps

    def save(self, output_dir: str):
        super().save(output_dir)
        torch.save(self.stoch_projection_1.state_dict(), os.path.join(output_dir, "stoch_projection_1"))
        torch.save(self.stoch_projection_2.state_dict(), os.path.join(output_dir, "stoch_projection_2"))
        torch.save(self.cd1.state_dict(), os.path.join(output_dir, "cd1"))
        torch.save(self.cd2.state_dict(), os.path.join(output_dir, "cd2"))

    def load_from(self, input_dir: str):
        super().load_from(input_dir)
        stoch_path_1 = os.path.join(input_dir, "stoch_projection_1")
        logger.info(f"loading stoch_projection_1 from {stoch_path_1}")
        self.stoch_projection_1.load_state_dict(torch.load(stoch_path_1, map_location="cpu"))

        stoch_path_2 = os.path.join(input_dir, "stoch_projection_2")
        logger.info(f"loading stoch_projection_2 from {stoch_path_2}")
        self.stoch_projection_2.load_state_dict(torch.load(stoch_path_2, map_location="cpu"))

        cd1_path = os.path.join(input_dir, "cd1")
        logger.info(f"loading cd1 from {cd1_path}")
        self.cd1.load_state_dict(torch.load(cd1_path, map_location="cpu"))

        cd2_path = os.path.join(input_dir, "cd2")
        logger.info(f"loading cd2 from {cd2_path}")
        self.cd2.load_state_dict(torch.load(cd2_path, map_location="cpu"))
