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
        adj_t, adj_t_sample = self._perturb_data(adj, flags, t, sample=True)

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
        adj_t, adj_t_sample = self._perturb_data(adj, flags, t, sample=False, clamp_range=(0.0, 1.0))

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

