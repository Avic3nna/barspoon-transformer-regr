# %%
import re
from typing import Any, Dict, Mapping, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor, nn
from torchmetrics.utilities.data import dim_zero_cat

__all__ = [
    "LitEncDecTransformer",
    "EncDecTransformer",
    "LitMilClassificationMixin",
    "SafeMulticlassAUROC",
]


class EncDecTransformer(nn.Module):
    """An encoder decoder architecture for multilabel classification tasks

    This architecture is a modified version of the one found in [Attention Is
    All You Need][1]: First, we project the features into a lower-dimensional
    feature space, to prevent the transformer architecture's complexity from
    exploding for high-dimensional features.  We add sinusodial [positional
    encodings][1].  We then encode these projected input tokens using a
    transformer encoder stack.  Next, we decode these tokens using a set of
    class tokens, one per output label.  Finally, we forward each of the decoded
    tokens through a fully connected layer to get a label-wise prediction.

                  PE1
                   |
             +--+  v   +---+
        t1 --|FC|--+-->|   |--+
         .   +--+      | E |  |
         .             | x |  |
         .   +--+      | n |  |
        tn --|FC|--+-->|   |--+
             +--+  ^   +---+  |
                   |          |
                  PEn         v
                            +---+   +---+
        c1 ---------------->|   |-->|FC1|--> s1
         .                  | D |   +---+     .
         .                  | x |             .
         .                  | k |   +---+     .
        ck ---------------->|   |-->|FCk|--> sk
                            +---+   +---+

    We opted for this architecture instead of a more traditional [Vision
    Transformer][2] to improve performance for multi-label predictions with many
    labels.  Our experiments have shown that adding too many class tokens to a
    vision transformer decreases its performance, as the same weights have to
    both process the tiles' information and the class token's processing.  Using
    an encoder-decoder architecture alleviates these issues, as the data-flow of
    the class tokens is completely independent of the encoding of the tiles.
    Furthermore, analysis has shown that there is almost no interaction between
    the different classes in the decoder.  While this points to the decoder
    being more powerful than needed in practice, this also means that each
    label's prediction is mostly independent of the others.  As a consequence,
    noisy labels will not negatively impact the accuracy of non-noisy ones.

    In our experiments so far we did not see any improvement by adding
    positional encodings.  We tried

     1. [Sinusodal encodings][1]
     2. Adding absolute positions to the feature vector, scaled down so the
        maximum value in the training dataset is 1.

    Since neither reduced performance and the author percieves the first one to
    be more elegant (as the magnitude of the positional encodings is bounded),
    we opted to keep the positional encoding regardless in the hopes of it
    improving performance on future tasks.

    The architecture _differs_ from the one descibed in [Attention Is All You
    Need][1] as follows:

     1. There is an initial projection stage to reduce the dimension of the
        feature vectors and allow us to use the transformer with arbitrary
        features.
     2. Instead of the language translation task described in [Attention Is All
        You Need][1], where the tokens of the words translated so far are used
        to predict the next word in the sequence, we use a set of fixed, learned
        class tokens in conjunction with equally as many independent fully
        connected layers to predict multiple labels at once.

    [1]: https://arxiv.org/abs/1706.03762 "Attention Is All You Need"
    [2]: https://arxiv.org/abs/2010.11929
        "An Image is Worth 16x16 Words:
         Transformers for Image Recognition at Scale"
    """

    def __init__(
        self,
        d_features: int,
        target_n_outs: Dict[str, int],
        *,
        d_model: int = 256,
        num_encoder_heads: int = 5,
        num_decoder_heads: int = 5,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 768,
        positional_encoding: bool = True,
    ) -> None:
        super().__init__()

        self.projector = nn.Sequential(nn.Linear(d_features, d_model), nn.ReLU())

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_encoder_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            # norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.target_labels = target_n_outs.keys()

        # One class token per output label
        self.class_tokens = nn.ParameterDict(
            {
                sanitize(target_label): torch.rand(d_model)
                for target_label in target_n_outs
            }
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_decoder_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            # norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.heads = nn.ModuleDict(
            {
                sanitize(target_label): nn.Linear(
                    in_features=d_model, out_features=n_out
                )
                for target_label, n_out in target_n_outs.items()
            }
        )

        self.positional_encoding = positional_encoding

    def forward(
        self,
        tile_tokens: torch.Tensor,
        tile_positions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, _ = tile_tokens.shape

        tile_tokens = self.projector(tile_tokens)  # shape: [bs, seq_len, d_model]

        if self.positional_encoding:
            # Add positional encodings
            d_model = tile_tokens.size(-1)
            x = tile_positions.unsqueeze(-1) / 100_000 ** (
                torch.arange(d_model // 4).type_as(tile_positions) / d_model
            )
            positional_encodings = torch.cat(
                [
                    torch.sin(x).flatten(start_dim=-2),
                    torch.cos(x).flatten(start_dim=-2),
                ],
                dim=-1,
            )
            tile_tokens = tile_tokens + positional_encodings

        tile_tokens = self.transformer_encoder(tile_tokens)

        class_tokens = torch.stack(
            [self.class_tokens[sanitize(t)] for t in self.target_labels]
        ).expand(batch_size, -1, -1)
        class_tokens = self.transformer_decoder(tgt=class_tokens, memory=tile_tokens)

        # Apply the corresponding head to each class token

        #TODO:
        # logits = {
        #     target_label: self.heads[sanitize(target_label)](class_token)
        #     for target_label, class_token in zip(
        #         self.target_labels,
        #         class_tokens.permute(1, 0, 2),  # Permute to [target, batch, d_model]
        #         strict=True,
        #     )
        # }

        logits = {
            target_label: self.heads[sanitize(target_label)](class_tokens[:, idx]).squeeze(dim=1)
            for idx, target_label in enumerate(self.target_labels)
        }

        return logits


class LitMilClassificationMixin(pl.LightningModule):
    """Makes a module into a multilabel, multiclass Lightning one"""

    def __init__(
        self,
        *,
        target_labels: list,
        # Other hparams
        learning_rate: float = 1e-4,
        **hparams: Any,
    ) -> None:
        super().__init__()
        _ = hparams  # So we don't get unused parameter warnings

        self.learning_rate = learning_rate
        
        #TODO: MSE instead of AUROC
        target_mse = torchmetrics.MetricCollection(
            {
                sanitize(target_label): torchmetrics.MeanSquaredError() #TODO: swap with Safe variant?
                for target_label in target_labels
            }
        )
        for step_name in ["train", "val", "test"]:
            setattr(
                self,
                f"{step_name}_target_MSE",
                target_mse.clone(prefix=f"{step_name}_"),
            )

        # self.weights = weights
        self.target_labels = target_labels

        self.save_hyperparameters()

    def step(self, batch: Tuple[Tensor, Tensor], step_name=None):
        feats, coords, targets = batch
        logits = self(feats, coords)
        # Calculate the cross entropy loss for each target, then sum them
        
        # #TODO: MSE instead of CE
        # loss = sum(
        #     F.cross_entropy(
        #         (l := logits[target_label]),
        #         targets[target_label].type_as(l),
        #         weight=weight.type_as(l),
        #     )
        #     for target_label, weight in self.weights.items()
        # )
        # loss = sum(
        #     F.mse_loss(
        #         (l := logits[target_label]),
        #         targets[target_label].type_as(l),
        #         reduction='mean',  # You can change this to 'sum' if needed
        #     )
        #     for target_label in self.target_labels
        # )
        total_loss=0.0
        for target_label in self.target_labels:
            nan_idx = torch.isnan(targets[target_label])

            # The Kullback-Leibler divergence loss
            # kl_loss = nn.KLDivLoss(reduction="batchmean")
            # input = F.log_softmax(logits[target_label][~nan_idx], dim=-1)
            # target = F.softmax(targets[target_label][~nan_idx].type_as(input), dim=-1)
            # loss = kl_loss(input, target)
            loss = F.mse_loss(
                (l := logits[target_label][~nan_idx]),
                targets[target_label][~nan_idx].type_as(l),
                reduction='mean'  # You can change this to 'sum' if needed
            )
            # loss = F.l1_loss(
            #     (l := logits[target_label][~nan_idx]),
            #     targets[target_label][~nan_idx].type_as(l)
            # )
            total_loss += loss
        total_loss = total_loss/len(self.target_labels)


        if step_name:
            self.log(
                f"{step_name}_loss",
                total_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

            #TODO: MSE instead of AUROC
            # Update target-wise metrics
            for target_label in self.target_labels:
                target_mse = getattr(self, f"{step_name}_target_MSE")[
                    sanitize(target_label)
                ]
                is_na = torch.isnan((targets[target_label]))
                target_mse.update(
                    logits[target_label][~is_na],
                    targets[target_label][~is_na],
                )
                self.log(
                    f"{step_name}_{target_label}_MSE",
                    target_mse,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        return total_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, step_name="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, step_name="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, step_name="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 2:
            feats, positions = batch
        else:
            feats, positions, _ = batch
        logits = self(feats, positions)
        softmaxed = {
            target_label: x for target_label, x in logits.items() #TODO: add softmax of x?
        }
        return softmaxed

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def sanitize(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", x)

#TODO:
# class SafeMulticlassAUROC(torchmetrics.classification.MulticlassAUROC):
#     """A Multiclass AUROC that doesn't blow up when no targets are given"""

#     def compute(self) -> torch.Tensor:
#         # Add faux entry if there are none so far
#         if len(self.preds) == 0:
#             self.update(torch.zeros(1, self.num_classes), torch.zeros(1).long())
#         elif len(dim_zero_cat(self.preds)) == 0:
#             self.update(
#                 torch.zeros(1, self.num_classes).type_as(self.preds[0]),
#                 torch.zeros(1).long().type_as(self.target[0]),
#             )
#         return super().compute()


class SafeMSEMetric(torchmetrics.MeanSquaredError):
    """A Mean Squared Error metric that doesn't blow up when no targets are given"""

    def compute(self) -> torch.Tensor:
        breakpoint()
        if len(self.preds) == 0:
            self.update(torch.zeros(1), torch.zeros(1))
        elif len(self.preds) > 0 and len(self.target) == 0:
            self.update(
                torch.zeros(1).type_as(self.preds[0]),
                torch.zeros(1).type_as(self.preds[0]),
            )
        return super().compute()


class LitEncDecTransformer(LitMilClassificationMixin):
    def __init__(
        self,
        *,
        d_features: int,
        target_labels: list,
        # Model parameters
        d_model: int = 512,
        num_encoder_heads: int = 8,
        num_decoder_heads: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 2048,
        positional_encoding: bool = True,
        # Other hparams
        learning_rate: float = 1e-4,
        **hparams: Any,
    ) -> None:
        super().__init__(
            target_labels=target_labels, #TODO change to target labels
            learning_rate=learning_rate,
        )
        _ = hparams  # so we don't get unused parameter warnings

        self.model = EncDecTransformer(
            d_features=d_features,
            target_n_outs={t: 1 for t in target_labels}, #TODO change to target labels
            d_model=d_model,
            num_encoder_heads=num_encoder_heads,
            num_decoder_heads=num_decoder_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            positional_encoding=positional_encoding,
        )

        self.save_hyperparameters()

    def forward(self, *args):
        return self.model(*args)
