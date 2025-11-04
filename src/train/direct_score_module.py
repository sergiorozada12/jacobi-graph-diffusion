import torch
import wandb

from src.train.trainer import DiffusionGraphModule
from src.metrics.train import DirectScoreLoss
from src.sde.score import JacobiScore
from src.utils import node_flags


class DirectScoreModule(DiffusionGraphModule):
    """Lightning module for training a transformer to predict diffusion scores directly."""

    def __init__(self, cfg, sampling_metrics, ref_metrics, node_dist):
        super().__init__(cfg, sampling_metrics, ref_metrics, node_dist)
        self.train_loss = DirectScoreLoss(weight=getattr(cfg.train, "score_loss_weight", 1.0))
        self._jacobi_helper = self._build_jacobi_helper(cfg.sde)

    def _build_jacobi_helper(self, cfg_sde):
        helper = JacobiScore.__new__(JacobiScore)
        helper.order = cfg_sde.order
        helper.eps = cfg_sde.eps_score
        helper.eps_dist = cfg_sde.eps_score_dist
        helper.alpha = cfg_sde.alpha
        helper.beta = cfg_sde.beta
        helper.jacobi_a = helper.beta - 1.0
        helper.jacobi_b = helper.alpha - 1.0
        return helper

    @staticmethod
    def _edge_mask(flags):
        mask = (flags[:, :, None] * flags[:, None, :]).float()
        diag = torch.eye(mask.size(-1), device=mask.device)
        return mask * (1.0 - diag.unsqueeze(0))

    def _analytic_jacobi_score(self, adj_clean, adj_noisy, t):
        return self._jacobi_helper.jacobi_score(adj_clean, adj_noisy, t).float()

    def training_step(self, batch, batch_idx):
        X, A = batch
        B, _, _ = X.shape
        flags = node_flags(A)

        t = self._sample_time(B)
        A_t, A_t_sample = self._perturb_data(A, flags, t)

        E_t = torch.cat([(1 - A_t).unsqueeze(-1), A_t.unsqueeze(-1)], dim=-1).float()
        E_t_sample = torch.cat([(1 - A_t_sample).unsqueeze(-1), A_t_sample.unsqueeze(-1)], dim=-1).float()

        features_input = E_t_sample if self.use_sampled_features else E_t
        extra_pred = self.feature_extractor(features_input, flags)
        y = torch.cat((extra_pred.y.float(), t.unsqueeze(1)), dim=1).float()

        pred = self.model(extra_pred.X.float(), extra_pred.E.float(), y, flags)
        pred_score = pred.E[..., 0]

        target_score = self._analytic_jacobi_score(A.float(), A_t, t).to(pred_score.dtype)
        mask = self._edge_mask(flags)
        loss = self.train_loss(pred_score, target_score, mask)
        return {"loss": loss}

    def _val_denoiser(self, X, A):
        B, _, _ = X.shape
        flags = node_flags(A)

        t = self._sample_time(B)
        A_t, A_t_sample = self._perturb_data(A, flags, t)
        E_t = torch.cat([(1 - A_t).unsqueeze(-1), A_t.unsqueeze(-1)], dim=-1).float()
        E_t_sample = torch.cat([(1 - A_t_sample).unsqueeze(-1), A_t_sample.unsqueeze(-1)], dim=-1).float()

        with torch.no_grad():
            features_input = E_t_sample if self.use_sampled_features else E_t
            extra_pred = self.feature_extractor(features_input, flags)
            y = torch.cat((extra_pred.y.float(), t.unsqueeze(1)), dim=1).float()
            model = self._get_eval_model()
            pred = model(extra_pred.X.float(), extra_pred.E.float(), y, flags)
            pred_score = pred.E[..., 0]

            target_score = self._analytic_jacobi_score(A.float(), A_t, t).to(pred_score.dtype)
            mask = self._edge_mask(flags)
            scale = self.train_loss.build_scale(target_score)
            rel_mse = DirectScoreLoss.masked_mse(
                pred_score,
                target_score,
                mask,
                scale=scale,
                clip=self.train_loss.diff_clip,
            ).detach()
            raw_mse = DirectScoreLoss.masked_mse(pred_score, target_score, mask).detach()

        if wandb.run:
            wandb.log(
                {
                    "val/score_rel_mse": rel_mse,
                    "val/score_mse": raw_mse,
                },
                commit=True,
            )
