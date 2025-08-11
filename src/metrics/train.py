import wandb
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError

from src.metrics.abstract import (
    CrossEntropyMetric,
    KLDMetric,
)


class NodeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class TrainLoss(nn.Module):
    """Train with Cross entropy"""

    def __init__(self, lambda_train=5, kld=False):
        super().__init__()
        self.lambda_train = lambda_train
        if not kld:
            self.node_loss = CrossEntropyMetric()
            self.edge_loss = CrossEntropyMetric()
        else:
            self.node_loss = KLDMetric()
            self.edge_loss = KLDMetric()
        self.y_loss = CrossEntropyMetric()

    def forward(
        self,
        masked_pred_E,
        true_E,
    ):

        true_E = torch.reshape(true_E, (-1, true_E.size(-1))).long()  # (bs * n * n, de)
        masked_pred_E = torch.reshape(
            masked_pred_E, (-1, masked_pred_E.size(-1))
        )  # (bs * n * n, de)

        mask_E = (true_E != 0.0).any(dim=-1)
        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        to_log = {
            "train_loss/batch_CE": loss_E.detach(),
            "train_loss/E_CE": (self.edge_loss.compute() if true_E.numel() > 0 else -1),
        }
        if wandb.run:
            wandb.log(to_log, commit=True)
        return self.lambda_train * loss_E

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = (
            self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        )
        epoch_edge_loss = (
            self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        )
        epoch_y_loss = (
            self.train_y_loss.compute() if self.y_loss.total_samples > 0 else -1
        )

        to_log = {
            "train_epoch/x_CE": epoch_node_loss,
            "train_epoch/E_CE": epoch_edge_loss,
            "train_epoch/y_CE": epoch_y_loss,
        }
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log