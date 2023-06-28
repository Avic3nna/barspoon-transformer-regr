import math
import re
from collections.abc import Sequence
from typing import Any, Literal, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor, nn
from torchmetrics.functional.classification.auroc import _multilabel_auroc_compute
from torchmetrics.utilities.data import dim_zero_cat, select_topk

__all__ = [
    "LitEncDecTransformer",
    "EncDecTransformer",
    "ParallelLinear",
    "LitMilClassificaionMixin",
    "TopKMuliLabelAUROC",
]


class EncDecTransformer(nn.Module):
    """An encoder decoder architecture for multilabel classification tasks

    This architecture is a modified encoder decoder stack: First, we encode the
    input tokens using an encoder stack.  We then decode these tokens using a
    set of class tokens, one per output label.  Finally, we forward each of the
    decoded tokens through a fully connected layer to get a label-wise
    prediction.

              +---+
        t1 -->|   |--+
         .    | E |  |
         .    | x |  |
         .    | n |  |
        tn -->|   |--+
              +---+  |
                     v
                   +---+   +---+
        c1 ------->|   |-->|FC1|--> s1
         .         | D |   +---+     .
         .         | x |             .
         .         | k |   +---+     .
        ck ------->|   |-->|FCk|--> sk
                   +---+   +---+
    """

    def __init__(
        self,
        d_features: int,
        n_targets: int,
        *,
        d_model: int = 512,
        num_encoder_heads: int = 8,
        num_decoder_heads: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
    ) -> None:
        super().__init__()

        # one class token per output class
        self.class_tokens = nn.Parameter(torch.rand(n_targets, d_model))

        self.projector = nn.Sequential(nn.Linear(d_features, d_model), nn.ReLU())

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_encoder_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_decoder_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.heads = ParallelLinear(
            in_features=d_model, out_features=1, n_parallel=n_targets
        )

    def forward(self, tile_tokens):
        batch_size, _, _ = tile_tokens.shape

        tile_tokens = self.projector(tile_tokens)  # shape: [bs, seq_len, d_model]
        tile_tokens = self.transformer_encoder(tile_tokens)

        class_tokens = self.class_tokens.expand(batch_size, -1, -1)
        class_tokens = self.transformer_decoder(tgt=class_tokens, memory=tile_tokens)

        # apply the corresponding head to each class token
        logits = self.heads(class_tokens).squeeze(-1)

        return logits


class ParallelLinear(nn.Module):
    """Parallelly applies multiple linear layers.

    For an input of shape (N, F) or (B, N, F), this layer applies a separate
    linear layer to each of the N channels of the input.
    """

    def __init__(self, in_features: int, out_features: int, n_parallel: int):
        super().__init__()
        self.in_features, self.out_features, self.n_parallel = (
            in_features,
            out_features,
            n_parallel,
        )
        self.weight = nn.Parameter(torch.empty((n_parallel, in_features, out_features)))
        self.bias = nn.Parameter(torch.empty((n_parallel, out_features)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Adapted from torch.nn.Linear
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.out_features)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        assert x.ndim in [2, 3], (
            "ParallelLinear is only defined for inputs of shape "
            "(n_parallel, in_features) and (batch_size, n_parallel, in_features)"
        )
        return (x.unsqueeze(-2) @ self.weight).squeeze(-2) + self.bias

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, n_parallel={self.n_parallel}"


class LitMilClassificationMixin(pl.LightningModule):
    def __init__(
        self,
        *,
        target_labels: Sequence[str],
        weights: Sequence[torch.Tensor],
        # other hparams
        learning_rate: float = 1e-4,
        **hparams: Any,
    ) -> None:
        super().__init__()
        _ = hparams  # So we don't get unused parameter warnings

        self.learning_rate = learning_rate

        # use the same metrics for training, validation and testing
        # global_metrics = torchmetrics.MetricCollection(
        #     [
        #         TopKMultilabelAUROC(
        #             num_labels=n_targets, topk=max(int(n_targets * 0.2), 1)
        #         )
        #     ]
        # )
        target_aurocs = torchmetrics.MetricCollection(
            {
                sanatize(target_label): SafeMulticlassAUROC(num_classes=len(weight))
                for target_label, weight in zip(target_labels, weights)
            }
        )
        for step_name in ["train", "val", "test"]:
            # setattr(
            #     self,
            #     f"{step_name}_global_metrics",
            #     global_metrics.clone(prefix=f"{step_name}_"),
            # )
            setattr(
                self,
                f"{step_name}_target_aurocs",
                target_aurocs.clone(prefix=f"{step_name}_"),
            )

        self.target_labels = target_labels
        self.weights = weights

        self.save_hyperparameters()

    def step(self, batch: Tuple[Tensor, Tensor], step_name=None):
        bags, targets = batch
        logits = self(bags)

        # The column ranges belonging to each target
        target_edges = np.cumsum([0, *(len(w) for w in self.weights)])
        # Calculate the cross entropy loss for each target, then sum them
        loss = sum(
            F.cross_entropy(
                logits[:, left:right],
                targets[:, left:right].type_as(logits),
                weight=weight.type_as(logits),
            )
            for left, right, weight in zip(
                target_edges[:-1],  # Leftmost column belonging to target
                target_edges[1:],  # Rightmost column belonging to target
                self.weights,
                # strict=True,  # Python 3.9 hates it
            )
        )

        if step_name:
            self.log(
                f"{step_name}_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

            #     # Update global metrics
            #     global_metrics = getattr(self, f"{step_name}_global_metrics")
            #     global_metrics.update(logits, targets.long())
            #     self.log_dict(
            #         global_metrics,
            #         on_step=False,
            #         on_epoch=True,
            #         prog_bar=True,
            #         sync_dist=True,
            #     )

            # Update target-wise metrics
            for target_label, left, right in zip(
                self.target_labels,
                target_edges[:-1],
                target_edges[1:],
                # strict=True,  # Python 3.9 hates it
            ):
                target_auroc = getattr(self, f"{step_name}_target_aurocs")[
                    sanatize(target_label)
                ]
                is_na = (targets[:, left:right] == 0).all(dim=1)
                target_auroc.update(
                    logits[~is_na, left:right],
                    targets[~is_na, left:right].argmax(dim=1),
                )
                self.log(
                    f"{step_name}_{target_label}_auroc",
                    target_auroc,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, step_name="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, step_name="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, step_name="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, Tensor):
            bag = batch
        else:
            bag, _ = batch
        logits = self(bag)

        target_edges = np.cumsum([0, *(len(w) for w in self.weights)])
        softmaxed = torch.cat(
            [
                torch.softmax(logits[:, left:right], 1)
                for left, right in zip(target_edges[:-1], target_edges[1:])
            ],
            dim=1,
        )
        return softmaxed

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def sanatize(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", x)


class SafeMulticlassAUROC(torchmetrics.classification.MulticlassAUROC):
    """A Multiclass AUROC that doesn't blow up when no targets are given"""

    def compute(self) -> torch.Tensor:
        # Add faux entry if there are none so far
        if len(self.preds) == 0:
            self.update(torch.zeros(1, self.num_classes), torch.zeros(1).long())
        elif len(dim_zero_cat(self.preds)) == 0:
            self.update(
                torch.zeros(1, self.num_classes).type_as(self.preds[0]),
                torch.zeros(1).long().type_as(self.target[0]),
            )
        return super().compute()


class LitEncDecTransformer(LitMilClassificationMixin):
    def __init__(
        self,
        *,
        d_features: int,
        target_labels: Sequence[str],
        weights: Sequence[torch.Tensor],
        # model parameters
        d_model: int = 512,
        num_encoder_heads: int = 8,
        num_decoder_heads: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        # other hparams
        learning_rate: float = 1e-4,
        **hparams: Any,
    ) -> None:
        super().__init__(
            target_labels=target_labels,
            weights=weights,
            learning_rate=learning_rate,
        )
        _ = hparams  # so we don't get unused parameter warnings

        self.model = EncDecTransformer(
            d_features=d_features,
            n_targets=sum(len(w) for w in weights),
            d_model=d_model,
            num_encoder_heads=num_encoder_heads,
            num_decoder_heads=num_decoder_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
        )

        self.save_hyperparameters()

    def forward(self, tile_tokens):
        return self.model(tile_tokens)
