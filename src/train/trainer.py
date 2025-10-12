import copy
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from contextlib import contextmanager

from src.metrics.train import TrainLoss
from src.models.transformer_model import GraphTransformer
from src.sde.sde import JacobiSDE
from src.sample.sampler import Sampler
from src.features.extra_features import ExtraFeatures
from src.utils import node_flags, gen_noise, build_time_schedule
from src.visualization.plots import close_figure, plot_graph_comparison


class DiffusionGraphModule(pl.LightningModule):
    def __init__(self, cfg, sampling_metrics, ref_metrics, node_dist):
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

        self.sampler = Sampler(cfg=cfg, model=self.model, node_dist=node_dist)
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
        self._ran_sampling_metrics = False
        self.use_sampled_features = getattr(cfg.model, "use_sampled_features", True)
        self.use_ema = getattr(cfg.train, "use_ema", False)
        self.ema_decay = getattr(cfg.train, "ema_decay", 0.999)
        if self.use_ema:
            self.ema_model = copy.deepcopy(self.model)
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            self.ema_model.eval()

        # Pre-compute time discretisation used by sampler to mirror during training.
        self.time_schedule_steps = build_time_schedule(
            N=cfg.sde.num_scales,
            T=self.sde.T,
            eps=cfg.sampler.eps_time,
            kind=getattr(cfg.sde, "time_schedule", "log"),
            power=getattr(cfg.sde, "time_schedule_power", 2.0),
        )

    def _build_sde(self, cfg_sde):
        return JacobiSDE(
            alpha=cfg_sde.alpha,
            beta=cfg_sde.beta,
            N=cfg_sde.num_scales,
            s_min=cfg_sde.s_min,
            s_max=cfg_sde.s_max,
            eps=cfg_sde.eps_sde,
            max_force=cfg_sde.max_force,
        )

    def _get_eval_model(self):
        if self.use_ema:
            ema_device = next(self.ema_model.parameters()).device
            if ema_device != self.device:
                self.ema_model.to(self.device)
            return self.ema_model
        return self.model

    @torch.no_grad()
    def _update_ema(self):
        if not self.use_ema:
            return

        ema_params = dict(self.ema_model.named_parameters())
        model_params = dict(self.model.named_parameters())
        for name, param in model_params.items():
            ema_param = ema_params[name]
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)

        ema_buffers = dict(self.ema_model.named_buffers())
        model_buffers = dict(self.model.named_buffers())
        for name, buffer in model_buffers.items():
            ema_buffers[name].data.copy_(buffer.data)

    @contextmanager
    def _using_sampler_model(self, model):
        if self.sampler.model is model:
            yield
            return

        original_model = self.sampler.model
        original_mode = getattr(original_model, "training", False)
        target_mode = getattr(model, "training", False)
        model.eval()
        self.sampler.set_model(model)
        try:
            yield
        finally:
            self.sampler.set_model(original_model)
            if original_model is not None:
                original_model.train(original_mode)
            if model is not None:
                model.train(target_mode)

    def configure_optimizers(self):    
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=self.cfg.train.amsgrad,
            weight_decay=self.cfg.train.weight_decay,
        )
        return opt

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self._update_ema()

    def on_train_start(self):
        if self.use_ema:
            self.ema_model.to(self.device)
            self.ema_model.load_state_dict(self.model.state_dict())

    def _sample_time(self, batch_size):
        schedule = self.time_schedule_steps.to(self.device)
        if schedule.numel() < 2:
            return torch.full((batch_size,), schedule[-1], device=self.device)

        idx = torch.randint(0, schedule.numel() - 1, (batch_size,), device=self.device)
        upper = schedule[idx]
        lower = schedule[idx + 1]
        u = torch.rand(batch_size, device=self.device)
        t = lower + (upper - lower) * u
        return t.clamp(min=self.loss_eps)

    def training_step(self, batch, batch_idx):
        # Data
        X, A = batch
        B, _, _ = X.shape
        flags = node_flags(A)

        # Noise
        t = self._sample_time(B)
        A_t, A_t_sample = self._perturb_data(A, flags, t)

        E = torch.cat([(1 - A).unsqueeze(-1), A.unsqueeze(-1)], dim=-1).float()
        E_t = torch.cat([(1 - A_t).unsqueeze(-1), A_t.unsqueeze(-1)], dim=-1).float()
        E_t_sample = torch.cat([(1 - A_t_sample).unsqueeze(-1), A_t_sample.unsqueeze(-1)], dim=-1).float()

        # Features
        features_input = E_t_sample if self.use_sampled_features else E_t
        extra_pred = self.feature_extractor(features_input, flags)
        y = torch.cat((extra_pred.y.float(), t.unsqueeze(1)), dim=1).float()

        # Prediction
        pred = self.model(extra_pred.X.float(), extra_pred.E.float(), y, flags)
        #pred = self.model(extra_pred.X.float(), E_t.float(), y, flags)
        #E_inp = torch.cat([extra_pred.E.float(), E_t.float()], dim=-1) 
        #pred = self.model(extra_pred.X.float(), E_inp, y, flags)
        loss = self.train_loss(
            masked_pred_E=pred.E,
            true_E=E,
        )

        return {'loss': loss}
    
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
            #pred = self.model(extra_pred.X.float(), E_t.float(), y, flags)
            #E_inp = torch.cat([extra_pred.E.float(), E_t.float()], dim=-1)
            #pred = self.model(extra_pred.X.float(), E_inp, y, flags)
            A_pred = F.softmax(pred.E, dim=-1)[..., 1:].sum(dim=-1).float()

        fig = plot_graph_comparison(
           adj_true=A[0],
           adj_recon=A_pred[0],
           adj_noisy=A_t[0],
           t_val=t[0].item(),
        )
        wandb.log({"val/denoiser": wandb.Image(fig)})
        close_figure(fig)

    def _val_sampler(self):
        eval_model = self._get_eval_model()
        with self._using_sampler_model(eval_model):
            samples, fig = self.sampler.sample()
        wandb.log({"val/sampler": wandb.Image(fig)})
        close_figure(fig)

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
        if not self._ran_sampling_metrics:
            self._val_sampler()
            self._ran_sampling_metrics = True
        return

    def on_validation_epoch_start(self):
        self._ran_sampling_metrics = False

    def on_fit_end(self):
        ckpt_dir = f"checkpoints/{self.cfg.data.data}"
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), f"{ckpt_dir}/weights.pth")
        if self.use_ema:
            torch.save(self.ema_model.state_dict(), f"{ckpt_dir}/weights_ema.pth")

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

        adj_triu = torch.triu(adj, diagonal=1)
        adj_triu_sample = torch.bernoulli(adj_triu)

        adj = adj_triu + adj_triu.transpose(-1, -2)
        adj_sample = adj_triu_sample + adj_triu_sample.transpose(-1, -2)

        return adj, adj_sample
