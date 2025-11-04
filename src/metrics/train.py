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


class DirectScoreLoss(nn.Module):
    """Masked loss for direct score regression with optional relative scaling."""

    def __init__(
        self,
        weight=1.0,
        *,
        use_relative=True,
        relative_eps=1.0,
        relative_power=1.0,
        diff_clip=None,
    ):
        super().__init__()
        self.weight = weight
        self.use_relative = use_relative
        self.relative_eps = relative_eps
        self.relative_power = relative_power
        self.diff_clip = diff_clip

    def build_scale(self, target):
        if not self.use_relative:
            return None

        scale = target.abs().detach().clamp_min(self.relative_eps)
        if self.relative_power != 1.0:
            scale = scale ** self.relative_power
        return scale

    @staticmethod
    def masked_mse(pred, target, mask, *, scale=None, clip=None):
        diff = pred - target
        if scale is not None:
            diff = diff / scale

        diff_sq = diff ** 2
        if clip is not None:
            diff_sq = diff_sq.clamp(max=clip)

        mask = mask.to(diff_sq.dtype)
        masked = diff_sq * mask

        if diff_sq.ndim <= 1:
            denom = mask.sum().clamp_min(torch.finfo(diff_sq.dtype).tiny)
            return masked.sum() / denom

        # Average per-graph so larger masks do not dominate the batch.
        masked_sum = masked.flatten(start_dim=1).sum(dim=1)
        denom = mask.flatten(start_dim=1).sum(dim=1)
        valid = denom > 0

        if valid.any():
            per_item = masked_sum[valid] / denom[valid]
            return per_item.mean()

        return masked_sum.new_zeros(())

    def forward(self, pred, target, mask):
        scale = self.build_scale(target)
        loss = self.masked_mse(pred, target, mask, scale=scale, clip=self.diff_clip)
        raw_mse = self.masked_mse(pred, target, mask)

        to_log = {
            "train_loss/batch_score_rel_mse": loss.detach(),
            "train_loss/batch_score_mse": raw_mse.detach(),
        }
        if wandb.run:
            wandb.log(to_log, commit=True)
        return self.weight * loss
