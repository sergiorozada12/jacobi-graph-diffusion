import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

from src.metrics.train import TrainLoss
from src.models.transformer_model import GraphTransformer
from src.sde.sde import JacobiSDE
from src.sample.sampler import Sampler
from src.features.extra_features import ExtraFeatures
from src.utils import node_flags, gen_noise


class DiffusionGraphModule(pl.LightningModule):
    def __init__(self, cfg, sampling_metrics, ref_metrics):
        super().__init__()
        self.cfg = cfg

        self.train_loss = TrainLoss(lambda_train=cfg.train.lambda_train)

        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=cfg.model.input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=cfg.model.output_dims,
            act_fn_in=torch.nn.ReLU(),
            act_fn_out=torch.nn.ReLU(),
        )

        self.sampler = Sampler(cfg=cfg, model=self.model)
        self.sampling_metrics = sampling_metrics
        self.ref_metrics = ref_metrics
        self.sde = self._build_sde(cfg.sde)
        self.feature_extractor = ExtraFeatures(
            extra_features_type=cfg.model.extra_features_type,
            rrwp_steps=cfg.model.rrwp_steps,
            max_n_nodes=cfg.data.max_node_num,
        )

        self.dataset_name = cfg.data.data
        self.loss_eps = cfg.train.eps

    def _build_sde(self, cfg_sde):
        return JacobiSDE(
            alpha=cfg_sde.alpha,
            beta=cfg_sde.beta,
            N=cfg_sde.num_scales,
            speed=cfg_sde.speed)

    def configure_optimizers(self):    
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=self.cfg.train.amsgrad,
            weight_decay=self.cfg.train.weight_decay,
        )
        return opt

    def training_step(self, batch, batch_idx):
        # Data
        X, A = batch
        B, _, _ = X.shape
        flags = node_flags(A)

        # Noise
        t = torch.rand(B, device=self.device) * (self.sde.T - self.loss_eps) + self.loss_eps
        A_t = self._perturb_data(A, flags, t)

        E = torch.cat([(1 - A).unsqueeze(-1), A.unsqueeze(-1)], dim=-1).float()
        E_t = torch.cat([(1 - A_t).unsqueeze(-1), A_t.unsqueeze(-1)], dim=-1).float()

        # Features
        extra_pred = self.feature_extractor(E_t, flags)
        y = torch.cat((extra_pred.y.float(), t.unsqueeze(1)), dim=1).float()

        # Prediction
        pred = self.model(extra_pred.X.float(), extra_pred.E.float(), y, flags)
        loss = self.train_loss(
            masked_pred_E=pred.E,
            true_E=E,
        )

        return {'loss': loss}
    
    def _val_denoiser(self, X, A):
        B, _, _ = X.shape
        flags = node_flags(A)

        t = torch.rand(B, device=self.device) * (self.sde.T - self.loss_eps) + self.loss_eps
        A_t = self._perturb_data(A, flags, t)
        E_t = torch.cat([(1 - A_t).unsqueeze(-1), A_t.unsqueeze(-1)], dim=-1).float()

        with torch.no_grad():
            extra_pred = self.feature_extractor(E_t, flags)
            y = torch.cat((extra_pred.y.float(), t.unsqueeze(1)), dim=1).float()
            pred = self.model(extra_pred.X.float(), extra_pred.E.float(), y, flags)
            A_pred = F.softmax(pred.E, dim=-1)[..., 1:].sum(dim=-1).float()
        
        fig = self._plot_graph_comparison(
           adj_true=A[0],
           adj_recon=A_pred[0],
           adj_noisy=A_t[0],
           t_val=t[0].item(),
        )
        wandb.log({"val/denoiser": wandb.Image(fig)})
        plt.close(fig)

    def _val_sampler(self):
        samples, fig = self.sampler.sample()
        wandb.log({"val/sampler": wandb.Image(fig)})
        plt.close(fig)

        self.sampling_metrics.reset()
        _ = self.sampling_metrics.forward(
            samples,
            ref_metrics=self.ref_metrics,
            local_rank=self.local_rank,
            test=False,
        )

    def validation_step(self, batch, batch_idx):
        X, A = batch
        self._val_denoiser(X, A)
        self._val_sampler()
        return

    def on_fit_end(self):
        torch.save(self.model.state_dict(), f"checkpoints/{self.cfg.data.data}/weights.pth")

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

        adj_triu = torch.bernoulli(torch.triu(adj, diagonal=1))
        adj_sample = adj_triu + adj_triu.transpose(-1, -2)

        return adj_sample

    def _plot_graph_comparison(self, adj_true, adj_recon, adj_noisy, t_val=None):
        adj_true_np = adj_true.detach().cpu().numpy()
        adj_recon_np = adj_recon.detach().cpu().numpy()
        adj_noisy_np = adj_noisy.detach().cpu().numpy()

        adj_min = min(adj_true_np.min(), adj_recon_np.min(), adj_noisy_np.min())
        adj_max = max(adj_true_np.max(), adj_recon_np.max(), adj_noisy_np.max())

        fig, axs = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
        title = f"Sampled t = {t_val:.2f}" if t_val is not None else "Graph Comparison"
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
