import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

from src.models.marginal_network import MarginalNetwork
from src.sde.sde import JacobiSDE
from src.features.extra_features import NodeFeatureAugmentor
from src.utils import node_flags, gen_noise


class DiffusionGraphModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = MarginalNetwork(**cfg.model)
        self.sde = self._build_sde(cfg.sde)
        self.feature_extractor = NodeFeatureAugmentor(features=cfg.train.features)

        self.dataset_name = cfg.data.data
        self.loss_eps = cfg.train.eps
        self.lambda_adj = cfg.train.lambda_adj
        self.lambda_x = cfg.train.lambda_x

    def _build_sde(self, cfg_sde):
        return JacobiSDE(
            alpha=cfg_sde.alpha,
            beta=cfg_sde.beta,
            N=cfg_sde.num_scales,
            speed=cfg_sde.speed)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay)

        if self.cfg.train.lr_schedule:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.cfg.train.lr_decay)
            return [opt], [scheduler]
        return opt

    def training_step(self, batch, batch_idx):
        # Data
        x, adj = batch
        flags = node_flags(adj)

        # Noise
        t = torch.rand(adj.shape[0], device=self.device) * (self.sde.T - self.loss_eps) + self.loss_eps
        adj_t_dist, _, _ = self._perturb_data(adj, flags, t)
        adj_t = torch.bernoulli(adj_t_dist)

        # Features
        x_target = self.feature_extractor.augment(x, adj)
        x_aug = self.feature_extractor.augment(x, adj_t)

        # Loss
        x_pred, adj_pred = self.model(x_aug, adj_t, flags)
        loss = self._loss(x_target, x_pred, adj, adj_pred, flags)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def on_after_backward(self):
        grad_norm = self.cfg.train.grad_norm
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_norm)

    def validation_step(self, batch, batch_idx):
        # Data
        x, adj = batch
        flags = node_flags(adj)

        # Noise
        t = torch.rand(adj.shape[0], device=self.device) * (self.sde.T - self.loss_eps) + self.loss_eps
        adj_t_dist, _, _ = self._perturb_data(adj, flags, t)
        adj_t = torch.bernoulli(adj_t_dist)

        # Features
        x_target = self.feature_extractor.augment(x, adj)
        x_aug = self.feature_extractor.augment(x, adj_t)

        # Loss
        x_pred, adj_pred = self.model(x_aug, adj_t, flags)
        with torch.no_grad():
            loss = self._loss(x_target, x_pred, adj, adj_pred, flags)
            self.log("val/loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        fig = self._plot_graph_comparison(
           adj_true=adj[0],
           adj_recon=adj_pred[0],
           adj_noisy=adj_t[0],
           t_val=t[0].item(),
           loss_val=loss.item()
        )
        wandb.log({"policy_big_plot": wandb.Image(fig)})
        plt.close(fig)
        return loss

    def on_fit_end(self):
        torch.save(self.model.state_dict(), f"checkpoints/{self.cfg.data.data}.pth")

    def _loss(self, x_true, x_pred, adj, adj_pred, flags):
        # Adj loss
        mask_adj = (flags[:, :, None] * flags[:, None, :]).float()  # B × N × N
        losses_adj = F.binary_cross_entropy(adj_pred, adj, reduction='none')  # B × N × N
        losses_adj = losses_adj * mask_adj
        num_valid_adj = mask_adj.sum(dim=(1, 2)).clamp(min=1)
        loss_adj = losses_adj.sum(dim=(1, 2)) / num_valid_adj  # B

        # X loss
        mask_x = flags.float().unsqueeze(-1)  # B × N × 1
        losses_x = F.mse_loss(x_pred, x_true, reduction='none')  # B × N × D
        losses_x = losses_x * mask_x
        num_valid_x = mask_x.sum(dim=(1, 2)).clamp(min=1)  # B

        loss_x = losses_x.sum(dim=(1, 2)) / num_valid_x  # B

        # Total loss
        total_loss = self.lambda_adj * loss_adj + self.lambda_x * loss_x  # B
        return total_loss.mean()
    
    def _perturb_data(self, adj0, flags, T):
        B = adj0.shape[0]
        mask = (flags[:, :, None] * flags[:, None, :]).float()
        dt = 1.0 / self.sde.N
        N_max = int((T.max() / dt).ceil().item())
        adj = adj0.clone()
        std = torch.zeros_like(adj)

        for i in range(N_max):
            t_val = (i * dt)
            vec_t = torch.full((B,), t_val, device=self.device)

            active = (T > t_val)
            if not active.any():
                break

            mean, std_all = self.sde.transition(adj, vec_t, dt)
            mean = mean * mask
            std_all = std_all * mask
            noise = gen_noise(adj, flags)

            std_all = std_all.clamp(min=1e-4)
            step = mean + std_all * noise
            step = torch.clamp(step, 1e-4, 1.0 - 1e-4)

            adj = torch.where(active.view(B, 1, 1), step, adj)
            std = torch.where(active.view(B, 1, 1), std_all, std)

        return adj, adj - adj0, std

    def _plot_graph_comparison(self, adj_true, adj_recon, adj_noisy, t_val=None, loss_val=None):
        adj_true_np = adj_true.detach().cpu().numpy()
        adj_recon_np = adj_recon.detach().cpu().numpy()
        adj_noisy_np = adj_noisy.detach().cpu().numpy()

        adj_min = min(adj_true_np.min(), adj_recon_np.min(), adj_noisy_np.min())
        adj_max = max(adj_true_np.max(), adj_recon_np.max(), adj_noisy_np.max())

        fig, axs = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
        title = f"Sampled t = {t_val:.2f} | Loss = {loss_val:.4f}" if t_val is not None and loss_val is not None else "Graph Comparison"
        fig.suptitle(title, fontsize=14)

        im0 = axs[0].imshow(adj_true_np, cmap='viridis', vmin=adj_min, vmax=adj_max)
        axs[0].set_title("True Adjacency")
        axs[0].axis("off")

        im1 = axs[1].imshow(adj_recon_np, cmap='viridis', vmin=adj_min, vmax=adj_max)
        axs[1].set_title("Reconstructed Adjacency")
        axs[1].axis("off")

        im2 = axs[2].imshow(adj_noisy_np, cmap='viridis', vmin=adj_min, vmax=adj_max)
        axs[2].set_title("Noisy Adjacency")
        axs[2].axis("off")

        fig.colorbar(im0, ax=axs[0], shrink=0.9, location='right', pad=0.01).set_label("Adjacency Value")
        fig.colorbar(im1, ax=axs[1], shrink=0.9, location='right', pad=0.01).set_label("Adjacency Value")
        fig.colorbar(im2, ax=axs[2], shrink=0.9, location='right', pad=0.01).set_label("Adjacency Value")

        return fig
