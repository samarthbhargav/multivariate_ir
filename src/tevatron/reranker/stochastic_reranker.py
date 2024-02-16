import logging
import os

import numpy as np

import torch.nn as nn

from tevatron.arguments import StochasticTrainingArguments
from tevatron.reranker.modeling import RerankerModel, RerankerOutput

from typing import Dict

import torch
from torch import Tensor, nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from tevatron.arguments import ModelArguments
from tevatron.arguments import TevatronTrainingArguments as TrainingArguments

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


class StochasticDenseModel(RerankerModel):
    TRANSFORMER_CLS = AutoModelForSequenceClassification

    def __init__(self, hf_model: PreTrainedModel, train_batch_size: int = None, n_iters: int = 100):
        super().__init__(hf_model, train_batch_size)
        self.n_iters = n_iters
        w, d = 1e-6, 1e-3
        self._stochastic_layer = torch.nn.Linear(self._embedding_size, self._embedding_size, bias=True)
        self._output_projection_layer = torch.nn.Linear(self._embedding_size, 2)
        self.cd1 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)
        self.cd2 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)
        self.m = torch.nn.ReLU()

    def forward(self, pair: Dict[str, Tensor] = None):
        ranker_logits = self.hf_model(**pair, return_dict=True).logits
        if self.train_batch_size:
            grouped_logits = ranker_logits.view(self.train_batch_size, -1)
            loss = self.cross_entropy(grouped_logits, self.target_label)
            return RerankerOutput(loss=loss, scores=ranker_logits)

        return RerankerOutput(loss=None, scores=ranker_logits)

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            stochastic_args: StochasticTrainingArguments,
            **hf_kwargs,
    ):
        if os.path.isdir(model_args.model_name_or_path):
            logger.info(f"loading model weight from local {model_args.model_name_or_path}")
            hf_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        else:
            logger.info(f"loading model weight from huggingface {model_args.model_name_or_path}")
            hf_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        model = cls(hf_model=hf_model, train_batch_size=train_args.per_device_train_batch_size)
        return model

    @classmethod
    def load(
            cls,
            model_name_or_path,
            **hf_kwargs,
    ):
        if os.path.isdir(model_name_or_path):
            logger.info(f"loading model weight from local {model_name_or_path}")
            hf_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        else:
            logger.info(f"loading model weight from huggingface {model_name_or_path}")
            hf_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        model = cls(hf_model=hf_model)
        return model

    def save(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
