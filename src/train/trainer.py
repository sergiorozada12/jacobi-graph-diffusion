import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

from src.metrics.train import TrainLossDiscrete
from src.models.marginal_network import MarginalNetwork
from src.models.transformer_model import GraphTransformer
from src.sde.sde import JacobiSDE
from src.features.extra_features import ExtraFeatures
from src.utils import node_flags, gen_noise


class DiffusionGraphModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.train_loss = TrainLossDiscrete(
            lambda_train = [5, 0]  # X=1, E = lambda[0], y = lambda[1]
        )

        # self.model = MarginalNetwork(**cfg.model)
        self.model = GraphTransformer(
            n_layers=8,
            input_dims = {
                "X": 20,
                "E": 20,
                "y": 5 + 1,  # +1 for the time conditioning
            },
            hidden_mlp_dims={
                'X': 128,
                'E': 64,
                'y': 128
            },
            hidden_dims={
                'dx': 256,
                'de': 64,
                'dy': 64,
                'n_head': 8,
                'dim_ffX': 256,
                'dim_ffE': 64,
                'dim_ffy': 256
            },
            output_dims = {
                "X": 20,
                "E": 2,
                "y": 0,
            },
            act_fn_in=torch.nn.ReLU(),
            act_fn_out=torch.nn.ReLU(),
        )

        self.sde = self._build_sde(cfg.sde)
        # self.feature_extractor = NodeFeatureAugmentor(features=cfg.train.features)
        self.feature_extractor = ExtraFeatures(
            extra_features_type='rrwp',
            rrwp_steps=20,
            max_n_nodes=40,
        )

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
        #opt = torch.optim.Adam(
        #    self.model.parameters(),
        #    lr=self.cfg.train.lr,
        #    weight_decay=self.cfg.train.weight_decay)

        #if self.cfg.train.lr_schedule:
        #    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.cfg.train.lr_decay)
        #    return [opt], [scheduler]
    
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.0002, #0.0002,
            amsgrad=True,
            weight_decay=1e-12,
        )
        return opt

    def training_step(self, batch, batch_idx):
        # Data
        X, A = batch
        B, N, _ = X.shape
        flags = node_flags(A)

        # Noise
        t = torch.rand(B, device=self.device) * (self.sde.T - self.loss_eps) + self.loss_eps
        A_t_dist, _, _ = self._perturb_data(A, flags, t)
        A_t_triu = torch.bernoulli(torch.triu(A_t_dist, diagonal=1))
        A_t = A_t_triu + A_t_triu.transpose(-1, -2)
        #adj_t = torch.bernoulli(adj_t_dist)

        E = torch.cat([(1 - A).unsqueeze(-1), A.unsqueeze(-1)], dim=-1).float()
        E_t = torch.cat([(1 - A_t).unsqueeze(-1), A_t.unsqueeze(-1)], dim=-1).float()

        # Features
        #x_target = self.feature_extractor.augment(x, adj)
        #x_aug = self.feature_extractor.augment(x, adj_t)

        #extra_target = self.feature_extractor(E, flags)
        extra_pred = self.feature_extractor(E_t, flags)

        # Prediction
        # x_pred, adj_pred = self.model(x_aug, adj_t, t, flags)
        y = torch.cat((extra_pred.y.float(), t.unsqueeze(1)), dim=1).float()
        pred = self.model(extra_pred.X.float(), extra_pred.E.float(), y, flags)

        # Loss
        # loss, loss_adj, loss_x = self._loss(x_target, x_pred, adj, adj_pred, flags)
        #X_target = extra_target.X
        #X_pred = pred.X
        #A_pred = F.softmax(pred.E, dim=-1)[..., 1:].sum(dim=-1).float()

        #loss, loss_adj, loss_x = self._loss(extra_target.X, pred.X, A, A_pred, flags)

        loss = self.train_loss(
            #masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            #pred_y=pred.y,
            #true_X=X,
            true_E=E,
            #true_y=data.y,
            #log=i % self.log_every_steps == 0,
        )
        #self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        #self.log("train/loss_adj", loss_adj, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        #self.log("train/loss_x", loss_x, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        # Data
        X, A = batch
        B, N, _ = X.shape
        flags = node_flags(A)

        # Noise
        t = torch.rand(B, device=self.device) * (self.sde.T - self.loss_eps) + self.loss_eps
        A_t_dist, _, _ = self._perturb_data(A, flags, t)
        A_t_triu = torch.bernoulli(torch.triu(A_t_dist, diagonal=1))
        A_t = A_t_triu + A_t_triu.transpose(-1, -2)
        #adj_t = torch.bernoulli(adj_t_dist)

        E = torch.cat([(1 - A).unsqueeze(-1), A.unsqueeze(-1)], dim=-1).float()
        E_t = torch.cat([(1 - A_t).unsqueeze(-1), A_t.unsqueeze(-1)], dim=-1).float()

        # Features
        #x_target = self.feature_extractor.augment(x, adj)
        #x_aug = self.feature_extractor.augment(x, adj_t)

        #extra_target = self.feature_extractor(E, flags)
        extra_pred = self.feature_extractor(E_t, flags)

        with torch.no_grad():
            # Prediction
            # x_pred, adj_pred = self.model(x_aug, adj_t, t, flags)
            y = torch.cat((extra_pred.y.float(), t.unsqueeze(1)), dim=1).float()
            pred = self.model(extra_pred.X.float(), extra_pred.E.float(), y, flags)

            # Loss
            # loss, loss_adj, loss_x = self._loss(x_target, x_pred, adj, adj_pred, flags)
            #X_target = extra_target.X
            #X_pred = pred.X
            A_pred = F.softmax(pred.E, dim=-1)[..., 1:].sum(dim=-1).float() 

            #loss, loss_adj, loss_x = self._loss(X_target, X_pred, A, A_pred, flags)
            #self.log("val/loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            #self.log("val/loss_adj", loss_adj, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            #self.log("val/loss_x", loss_x, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        fig = self._plot_graph_comparison(
           adj_true=A[0],
           adj_recon=A_pred[0],
           adj_noisy=A_t[0],
           t_val=t[0].item(),
           loss_val=0.0
           #loss_val=loss.item()
        )
        wandb.log({"policy_big_plot": wandb.Image(fig)})
        plt.close(fig)

        #return loss
        return

    def on_after_backward(self):
        grad_norm = self.cfg.train.grad_norm
        #if grad_norm is not None:
            ##torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_norm)

    def ___validation_step(self, batch, batch_idx):
        # Data
        x, adj = batch
        flags = node_flags(adj)

        # Noise
        t = torch.rand(adj.shape[0], device=self.device) * (self.sde.T - self.loss_eps) + self.loss_eps
        adj_t_dist, _, _ = self._perturb_data(adj, flags, t)
        #adj_t = torch.bernoulli(adj_t_dist)

        upper_dist = torch.triu(adj_t_dist, diagonal=1)
        upper = torch.bernoulli(upper_dist)
        adj_t = upper + upper.transpose(-1, -2)

        # Features
        x_target = self.feature_extractor.augment(x, adj)
        x_aug = self.feature_extractor.augment(x, adj_t)

        # Loss
        x_pred, adj_pred = self.model(x_aug, adj_t, t, flags)
        with torch.no_grad():
            loss, loss_adj, loss_x = self._loss(x_target, x_pred, adj, adj_pred, flags)
            self.log("val/loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log("val/loss_adj", loss_adj, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log("val/loss_x", loss_x, on_step=True, on_epoch=False, prog_bar=True, logger=True)

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
        # Masks
        B, N, _ = x_true.shape

        diag_mask = torch.eye(N)
        diag_mask = ~diag_mask.type_as(adj).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(B, -1, -1)

        mask_adj = (flags[:, :, None] * flags[:, None, :]).float()  # B × N × N
        mask_x = flags.float().unsqueeze(-1)  # B × N × 1

        # Adj loss
        losses_adj = F.binary_cross_entropy(adj_pred, adj, reduction='none')  # B × N × N
        losses_adj = losses_adj * mask_adj * diag_mask
        num_valid_adj = mask_adj.sum(dim=(1, 2)).clamp(min=1)
        loss_adj = losses_adj.sum(dim=(1, 2)) / num_valid_adj  # B

        # X loss
        losses_x = F.mse_loss(x_pred, x_true, reduction='none')  # B × N × D
        losses_x = losses_x * mask_x
        num_valid_x = mask_x.sum(dim=(1, 2)).clamp(min=1)  # B

        loss_x = losses_x.sum(dim=(1, 2)) / num_valid_x  # B

        # Total loss
        total_loss = self.lambda_adj * loss_adj + self.lambda_x * loss_x  # B
        return total_loss.mean(), loss_adj.mean(), loss_x.mean()
    
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
