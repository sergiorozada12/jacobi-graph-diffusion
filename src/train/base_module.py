import copy
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb

from src.features.extra_features import ExtraFeatures
from src.models.transformer_model import GraphTransformer
from src.sample.sampler import Sampler
from src.sde.sde import JacobiSDE
from src.utils import build_time_schedule, gen_noise, node_flags
from src.visualization.plots import close_figure, plot_edge_weight_histograms, plot_graph_comparison


class DiffusionBaseModule(pl.LightningModule):
    def __init__(
        self,
        cfg,
        sampling_metrics,
        ref_metrics,
        node_dist,
        *,
        train_loss,
        dataset_info: Optional[Dict] = None,
        output_dims_override: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.sampling_metrics = sampling_metrics
        self.ref_metrics = ref_metrics
        self.node_dist = node_dist
        self.train_loss = train_loss
        self.dataset_info = dataset_info

        self.dataset_name = cfg.data.data
        self._ran_sampling_metrics = False

        self.use_sampled_features = getattr(cfg.model, "use_sampled_features", True)
        self.use_ema = getattr(cfg.train, "use_ema", False)
        self.ema_decay = getattr(cfg.train, "ema_decay", 0.999)
        self._cached_ref_graphs: Optional[List] = None

        output_dims = dict(cfg.model.output_dims)
        if output_dims_override:
            output_dims.update(output_dims_override)

        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=cfg.model.input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=output_dims,
            act_fn_in=torch.nn.ReLU(),
            act_fn_out=torch.nn.ReLU(),
        )

        self.sampler = Sampler(cfg=cfg, model=self.model, node_dist=node_dist)
        self.sde = self._build_sde(cfg.sde)
        self.feature_extractor = ExtraFeatures(
            extra_features_type=cfg.model.extra_features_type,
            rrwp_steps=cfg.model.rrwp_steps,
            max_n_nodes=cfg.data.max_node_num,
        )

        if self.use_ema:
            self.ema_model = copy.deepcopy(self.model)
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            self.ema_model.eval()
        else:
            self.ema_model = None

        self.time_schedule_steps = build_time_schedule(
            num_scales=cfg.sde.num_scales,
            T=self.sde.T,
            eps=cfg.train.eps_time_train,
            kind=cfg.train.time_schedule_train,
            power=cfg.train.time_schedule_power_train,
        )

    def _build_sde(self, cfg_sde):
        return JacobiSDE(
            num_scales=cfg_sde.num_scales,
            alpha=cfg_sde.alpha,
            beta=cfg_sde.beta,
            s_min=cfg_sde.s_min,
            s_max=cfg_sde.s_max,
        ).to(self.device)

    def _get_eval_model(self):
        if self.use_ema and self.ema_model is not None:
            ema_device = next(self.ema_model.parameters()).device
            if ema_device != self.device:
                self.ema_model.to(self.device)
            return self.ema_model
        return self.model

    def _update_ema(self):
        if not self.use_ema or self.ema_model is None:
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
        original_mode = getattr(original_model, "training", False) if original_model else False
        target_mode = getattr(model, "training", False) if model else False
        if model is not None:
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
        if self.use_ema and self.ema_model is not None:
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
        return t

    @staticmethod
    def _edge_mask(flags: torch.Tensor) -> torch.Tensor:
        mask = (flags[:, :, None] * flags[:, None, :]).float()
        diag = torch.eye(mask.size(-1), device=mask.device)
        return mask * (1.0 - diag.unsqueeze(0))

    @staticmethod
    def _edge_channels(adj: torch.Tensor) -> torch.Tensor:
        adj = adj.clamp(0.0, 1.0)
        complement = (1.0 - adj).unsqueeze(-1)
        return torch.cat([complement, adj.unsqueeze(-1)], dim=-1)

    def _run_model(self, extra_pred, y, flags):
        return self.model(extra_pred.X.float(), extra_pred.E.float(), y, flags)

    def _perturb_data(self, data, flags, t, sde=None, sample=True):
        if sde is None:
            sde = self.sde
        
        # Check if we are using Stick-Breaking for categorical data
        is_stick_breaking = hasattr(sde, "x_to_v")
        
        if is_stick_breaking:
            # Map categorical classes (X or E) to v-space in [0,1]
            v0 = sde.x_to_v(data)
            v_t = self._numerical_integration(v0, flags, t, sde.base_sde)
            data_t = sde.v_to_x(v_t)
            # For stick-breaking models, we return the v-space variables instead
            # of the argmax sample, as v_t is the actual state space for the SDE
            return data_t, v_t
        else:
            # Binary/Continuous case (standard adjacencies)
            data_t = self._numerical_integration(data, flags, t, sde)
            # Binary sampling uses Bernoulli
            data_sample = torch.bernoulli(data_t) if (sample and data.ndim == 3) else data_t
            return data_t, data_sample

    def _numerical_integration(self, data0, flags, T, sde):
        B = data0.shape[0]
        # Multi-dimensional mask support (Nodes [B,N,1] or Adjs [B,N,N,1])
        if data0.ndim == 3:
            mask = flags.unsqueeze(-1)
        elif data0.ndim == 4:
            mask = (flags[:, :, None] * flags[:, None, :]).unsqueeze(-1)
        else:
            mask = flags
            
        dt = 1.0 / sde.N
        data = data0.clone()
        
        n_full = torch.floor(T / dt).clamp(max=sde.N - 1).long()
        max_full = int(n_full.max().item()) if n_full.numel() > 0 else 0
        
        for i in range(max_full):
            t_val = i * dt
            vec_t = torch.full((B,), t_val, device=self.device, dtype=T.dtype)

            active = n_full > i
            if not active.any():
                break

            mean, std_all = sde.transition(data, vec_t, dt)
            mean = mean * mask
            std_all = std_all * mask
            noise = gen_noise(data, flags)

            std_all = std_all.clamp(min=1e-4)
            step = mean + std_all * noise
            step = torch.clamp(step, 1e-4, 1.0 - 1e-4)

            active_mask = active.view(-1, *( (1,) * (data.ndim - 1) ))
            data = torch.where(active_mask, step, data)

        t_full = n_full.to(dtype=T.dtype) * dt
        dt_rem = (T - t_full).clamp_min(0.0)
        has_remainder = dt_rem > 1e-12

        if has_remainder.any():
            mean, std_all = sde.transition(data, t_full, dt_rem)
            mean = mean * mask
            std_all = std_all * mask
            noise = gen_noise(data, flags)
            std_all = std_all.clamp(min=1e-4)
            step = mean + std_all * noise
            step = torch.clamp(step, 1e-4, 1.0 - 1e-4)

            rem_mask = has_remainder.view(-1, *( (1,) * (data.ndim - 1) ))
            data = torch.where(rem_mask, step, data)
            
        return data

    @staticmethod
    def _extract_batch_tensors(batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(batch, (list, tuple)):
            # Support TensorDataset (X, adj, mask) or (X, adj, y, mask)
            if len(batch) >= 3:
                return batch[0], batch[1], batch[2]
            if len(batch) == 2:
                return batch[0], batch[1], None
        else:
            # Support PyG DataBatch
            return getattr(batch, "x", None), getattr(batch, "edge_index", None), getattr(batch, "mask", None)
        raise ValueError(f"Unexpected batch structure: {type(batch)}")

    def training_step(self, batch, batch_idx):
        X, adj, observed_mask = self._extract_batch_tensors(batch)
        return self._training_step_impl(batch_idx, X, adj, observed_mask)

    def validation_step(self, batch, batch_idx):
        X, adj, observed_mask = self._extract_batch_tensors(batch)
        self._validation_step_impl(X, adj, observed_mask)
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
        if self.use_ema and self.ema_model is not None:
            torch.save(self.ema_model.state_dict(), f"{ckpt_dir}/weights_ema.pth")

    def _val_sampler(self):
        eval_model = self._get_eval_model()
        val_use_full = getattr(self.cfg.sampler, "val_use_full_graph", False)
        val_keep_isolates = getattr(self.cfg.sampler, "val_keep_isolates", False)
        val_use_fixed_nodelist = getattr(self.cfg.sampler, "val_use_fixed_nodelist", False)
        nodelist = list(range(self.cfg.data.max_node_num)) if val_use_fixed_nodelist else None
        with self._using_sampler_model(eval_model):
            samples, fig, adj_samples = self.sampler.sample(
                keep_isolates=val_keep_isolates,
                use_node_dist=not val_use_full,
                nodelist=nodelist,
                keep_zero_weights=False,
                return_adjs=True,
            )
        if fig is not None:
            wandb.log({"val/sampler": wandb.Image(fig)})
            close_figure(fig)

        self.sampling_metrics.reset()
        metrics = self.sampling_metrics.forward(
            samples,
            ref_metrics=self.ref_metrics,
            local_rank=self.local_rank,
            test=False,
        )

        # Log molecular metrics (validity, uniqueness, novelty) if returned as a dict
        if isinstance(metrics, dict):
            to_log = {f"val/{k}": v for k, v in metrics.items()}
            for key, value in to_log.items():
                self.log(key, value, prog_bar=True)
            if wandb.run:
                wandb.log(to_log, commit=False)

        # Subclass hook — e.g. DiffusionMolModule uses this to log molecule images
        self._on_val_samples(samples)

        self._maybe_log_weight_histograms(samples, adj_samples)

    def _on_val_samples(self, samples):
        """Hook called after validation sampling. Override in subclasses to log extra artefacts."""
        pass

    def _training_step_impl(self, batch_idx, X, adj, observed_mask):
        raise NotImplementedError

    def _validation_step_impl(self, X, adj, observed_mask):
        raise NotImplementedError

    def _load_reference_graphs(self) -> Optional[List]:
        if self._cached_ref_graphs is not None:
            return self._cached_ref_graphs

        if self.dataset_name != "metrofi":
            return None

        dataset_path = Path(self.cfg.data.dir) / f"{self.cfg.data.data}.pkl"
        if not dataset_path.exists():
            return None

        try:
            dataset = pd.read_pickle(dataset_path)
        except Exception as exc:
            print(f"Warning: failed to load reference graphs from {dataset_path}: {exc}")
            return None

        ref_graphs = dataset.get("test", []) or []
        self._cached_ref_graphs = ref_graphs
        return ref_graphs

    def _maybe_log_weight_histograms(self, generated_graphs, adj_samples=None):
        if self.dataset_name != "metrofi":
            return
        if not wandb.run:
            return

        ref_graphs = self._load_reference_graphs()
        if not ref_graphs:
            return

        def _interference_range(graphs):
            vals = []
            for g in graphs:
                for _, _, data in g.edges(data=True):
                    raw = data.get("interference_raw", data.get("weight", None))
                    if raw is not None:
                        vals.append(float(raw))
            if not vals:
                return None
            return min(vals), max(vals)

        ref_range = _interference_range(ref_graphs)
        ref_min, ref_max = (ref_range if ref_range else (0.0, 1.0))
        scale = ref_max - ref_min if ref_max is not None and ref_min is not None else 1.0
        if scale <= 0:
            scale = 1.0

        gen_scaled = []
        if adj_samples is not None:
            # Use dense adjacencies so zero-weight edges are retained and rescaled.
            adj_rescaled = adj_samples * scale + ref_min
            adj_np = adj_rescaled.detach().cpu().numpy()
            nodelist = list(range(adj_np.shape[-1]))
            gen_scaled = adjs_to_graphs(
                adj_np,
                is_cuda=False,
                keep_isolates=True,
                nodelist=nodelist,
                keep_zero_weights=True,
            )
        else:
            for g in generated_graphs:
                g_copy = g.copy()
                for u, v, data in g_copy.edges(data=True):
                    w_norm = float(data.get("weight", 0.0))
                    w_raw = w_norm * scale + ref_min
                    data["weight"] = w_raw
                    data["interference"] = w_raw
                    data["interference_raw"] = w_raw
                gen_scaled.append(g_copy)

        try:
            fig = plot_edge_weight_histograms(
                ref_graphs,
                gen_scaled,
                dataset_name="metrofi test vs generated edge weights",
            )
        except Exception as exc:
            print(f"Warning: could not plot edge weight histograms: {exc}")
            return

        wandb.log({"val/edge_weight_hist": wandb.Image(fig)}, commit=False)
        close_figure(fig)
