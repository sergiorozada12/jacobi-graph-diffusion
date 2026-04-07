import torch
import torch.nn.functional as F
import wandb

from src.metrics.train import TrainLoss
from src.train.base_module import DiffusionBaseModule
from src.utils import gen_noise, node_flags
from src.visualization.plots import close_figure, plot_graph_comparison


class DiffusionGraphModule(DiffusionBaseModule):
    def __init__(self, cfg, sampling_metrics, ref_metrics, node_dist):
        train_loss = TrainLoss(lambda_train=cfg.train.lambda_train)
        super().__init__(
            cfg,
            sampling_metrics,
            ref_metrics,
            node_dist,
            train_loss=train_loss,
        )

    def _training_step_impl(self, batch_idx, X, adj, observed_mask):
        batch = self._prepare_batch(adj, observed_mask)
        flags = batch["flags"]
        pred = self._run_model(batch["extra_pred"], batch["y"], flags)
        loss = self.train_loss(masked_pred_E=pred.E, true_E=batch["edge_true"])
        return {"loss": loss}

    def _validation_step_impl(self, X, adj, observed_mask):
        batch = self._prepare_batch(adj, observed_mask)
        flags = batch["flags"]
        t = batch["t"]

        with torch.no_grad():
            model = self._get_eval_model()
            features_input = batch["features_input"]
            extra_pred = self.feature_extractor(features_input, flags)
            y = torch.cat((extra_pred.y.float(), t.unsqueeze(1)), dim=1).float()
            pred = model(extra_pred.X.float(), extra_pred.E.float(), y, flags)
            adj_pred = F.softmax(pred.E, dim=-1)[..., 1:].sum(dim=-1).float()

        fig = plot_graph_comparison(
            adj_true=adj[0],
            adj_recon=adj_pred[0],
            adj_noisy=batch["adj_t"][0],
            t_val=t[0].item(),
        )
        wandb.log({"val/denoiser": wandb.Image(fig)})
        close_figure(fig)

    def _prepare_batch(self, adj: torch.Tensor, observed_mask):
        flags = node_flags(adj, observed_mask)
        batch_size = adj.size(0)
        t = self._sample_time(batch_size)
        adj_t, adj_t_sample = self._perturb_data(adj, flags, t)

        edge_true = self._edge_channels(adj)
        edge_t = self._edge_channels(adj_t)
        edge_t_sample = self._edge_channels(adj_t_sample)

        features_input = edge_t_sample if self.use_sampled_features else edge_t
        extra_pred = self.feature_extractor(features_input, flags)
        y = torch.cat((extra_pred.y.float(), t.unsqueeze(1)), dim=1).float()

        return {
            "flags": flags,
            "t": t,
            "adj_t": adj_t,
            "adj_t_sample": adj_t_sample,
            "edge_true": edge_true,
            "edge_t": edge_t,
            "edge_t_sample": edge_t_sample,
            "features_input": features_input,
            "extra_pred": extra_pred,
            "y": y,
        }

    def _perturb_data(self, adj0: torch.Tensor, flags: torch.Tensor, T: torch.Tensor):
        B = adj0.shape[0]
        mask = (flags[:, :, None] * flags[:, None, :]).float()
        dt = 1.0 / self.sde.N
        adj = adj0.clone()

        n_full = torch.floor(T / dt).clamp(max=self.sde.N - 1).long()
        max_full = int(n_full.max().item()) if n_full.numel() > 0 else 0

        for i in range(max_full):
            t_val = i * dt
            vec_t = torch.full((B,), t_val, device=self.device, dtype=T.dtype)

            active = n_full > i
            if not active.any():
                break

            mean, std_all = self.sde.transition(adj, vec_t, dt)
            mean = mean * mask
            std_all = std_all * mask
            noise = gen_noise(adj, flags)

            std_all = std_all.clamp(min=1e-4)
            step = mean + std_all * noise
            step = torch.clamp(step, 1e-4, 1.0 - 1e-4)

            active_mask = active.view(B, 1, 1)
            adj = torch.where(active_mask, step, adj)

        t_full = n_full.to(dtype=T.dtype) * dt
        dt_rem = (T - t_full).clamp_min(0.0)
        has_remainder = dt_rem > 1e-12

        if has_remainder.any():
            vec_t = t_full
            mean, std_all = self.sde.transition(adj, vec_t, dt_rem)
            mean = mean * mask
            std_all = std_all * mask
            noise = gen_noise(adj, flags)

            std_all = std_all.clamp(min=1e-4)
            step = mean + std_all * noise
            step = torch.clamp(step, 1e-4, 1.0 - 1e-4)

            rem_mask = has_remainder.view(B, 1, 1)
            adj = torch.where(rem_mask, step, adj)

        adj_triu = torch.triu(adj, diagonal=1)
        adj_triu_sample = torch.bernoulli(adj_triu)

        adj = adj_triu + adj_triu.transpose(-1, -2)
        adj_sample = adj_triu_sample + adj_triu_sample.transpose(-1, -2)

        return adj, adj_sample


class DiffusionWeightedGraphModule(DiffusionBaseModule):
    def __init__(self, cfg, sampling_metrics, ref_metrics, node_dist, *, use_relative=False):
        from src.metrics.train import MaskedMSELoss

        loss_weight = getattr(cfg.train, "lambda_train", 1.0)
        train_loss = MaskedMSELoss(
            weight=loss_weight,
            use_relative=use_relative,
        )
        super().__init__(
            cfg,
            sampling_metrics,
            ref_metrics,
            node_dist,
            train_loss=train_loss,
            output_dims_override={"E": 1},
        )

    def _training_step_impl(self, batch_idx, X, adj, observed_mask):
        batch = self._prepare_batch(adj, observed_mask)
        flags = batch["flags"]
        pred = self._run_model(batch["extra_pred"], batch["y"], flags)
        pred_edges = pred.E[..., 0]
        target_edges = adj.float()
        mask = self._edge_mask(flags)
        loss = self.train_loss(pred_edges, target_edges)
        return {"loss": loss}

    def _validation_step_impl(self, X, adj, observed_mask):
        batch = self._prepare_batch(adj, observed_mask)
        flags = batch["flags"]
        t = batch["t"]

        with torch.no_grad():
            model = self._get_eval_model()
            features_input = batch["features_input"]
            extra_pred = self.feature_extractor(features_input, flags)
            y = torch.cat((extra_pred.y.float(), t.unsqueeze(1)), dim=1).float()
            pred = model(extra_pred.X.float(), extra_pred.E.float(), y, flags)
            adj_pred = pred.E[..., 0].float()

            mask = self._edge_mask(flags)
            mse = (mask * (adj_pred - adj).pow(2)).sum()
            denom = mask.sum().clamp_min(torch.finfo(adj_pred.dtype).tiny)
            mse = (mse / denom).detach()

        if wandb.run:
            wandb.log({"val/weighted_mse": mse.item()}, commit=True)

        fig = plot_graph_comparison(
            adj_true=adj[0],
            adj_recon=adj_pred[0],
            adj_noisy=batch["adj_t"][0],
            t_val=t[0].item(),
        )
        wandb.log({"val/denoiser": wandb.Image(fig)})
        close_figure(fig)

    def _prepare_batch(self, adj: torch.Tensor, observed_mask):
        flags = node_flags(adj, observed_mask)
        batch_size = adj.size(0)
        t = self._sample_time(batch_size)
        adj_t, adj_t_sample = self._perturb_data(adj, flags, t)

        edge_true = self._edge_channels(adj)
        edge_t = self._edge_channels(adj_t)
        edge_t_sample = self._edge_channels(adj_t_sample)

        features_input = edge_t_sample if self.use_sampled_features else edge_t
        extra_pred = self.feature_extractor(features_input, flags)
        y = torch.cat((extra_pred.y.float(), t.unsqueeze(1)), dim=1).float()

        return {
            "flags": flags,
            "t": t,
            "adj_t": adj_t,
            "adj_t_sample": adj_t_sample,
            "edge_true": edge_true,
            "edge_t": edge_t,
            "edge_t_sample": edge_t_sample,
            "features_input": features_input,
            "extra_pred": extra_pred,
            "y": y,
        }

    def _perturb_data(self, adj0: torch.Tensor, flags: torch.Tensor, T: torch.Tensor):
        B = adj0.shape[0]
        mask = (flags[:, :, None] * flags[:, None, :]).float()
        dt = 1.0 / self.sde.N
        adj = adj0.clone()

        n_full = torch.floor(T / dt).clamp(max=self.sde.N - 1).long()
        max_full = int(n_full.max().item()) if n_full.numel() > 0 else 0

        for i in range(max_full):
            t_val = i * dt
            vec_t = torch.full((B,), t_val, device=self.device, dtype=T.dtype)

            active = n_full > i
            if not active.any():
                break

            mean, std_all = self.sde.transition(adj, vec_t, dt)
            mean = mean * mask
            std_all = std_all * mask
            noise = gen_noise(adj, flags)

            std_all = std_all.clamp(min=1e-4)
            step = mean + std_all * noise
            step = torch.clamp(step, 0.0, 1.0)

            active_mask = active.view(B, 1, 1)
            adj = torch.where(active_mask, step, adj)

        t_full = n_full.to(dtype=T.dtype) * dt
        dt_rem = (T - t_full).clamp_min(0.0)
        has_remainder = dt_rem > 1e-12

        if has_remainder.any():
            vec_t = t_full
            mean, std_all = self.sde.transition(adj, vec_t, dt_rem)
            mean = mean * mask
            std_all = std_all * mask
            noise = gen_noise(adj, flags)

            std_all = std_all.clamp(min=1e-4)
            step = mean + std_all * noise
            step = torch.clamp(step, 0.0, 1.0)

            rem_mask = has_remainder.view(B, 1, 1)
            adj = torch.where(rem_mask, step, adj)

        adj = torch.triu(adj, diagonal=1)
        adj = adj + adj.transpose(-1, -2)

        return adj, adj
