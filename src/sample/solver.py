import math
import torch
import numpy as np
from tqdm import trange

from src.sde.score import JacobiScore
from src.utils import mask_adjs, mask_x, gen_noise, assert_symmetric_and_masked, build_time_schedule
from src.visualization.plots import (
    plot_graph_snapshots,
    plot_heatmap_snapshots,
    save_figure,
)


class EulerMaruyamaPredictor:
    def __init__(self, sde, score_fn):
        self.sde = sde
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn

    @torch.no_grad()
    def update(self, adj, flags, t, dt):
        assert_symmetric_and_masked(adj, flags)
        device, dtype = adj.device, adj.dtype

        # Regular EM
        sqrt_mdt = torch.sqrt(torch.tensor(-dt, device=device, dtype=dtype))
        noise = gen_noise(adj, flags).to(device=device, dtype=dtype)
        drift, diffusion = self.rsde.sde(adj, flags, t)
        adj_mean = adj + drift * dt
        adj_new  = adj_mean + diffusion * sqrt_mdt * noise
        """
        # Tamed EM
        sqrt_mdt = torch.sqrt(torch.tensor(-dt, device=device, dtype=dtype))
        noise    = gen_noise(adj, flags).to(device=device, dtype=dtype)
        drift, diffusion = self.rsde.sde(adj, flags, t)
        #tame_drift = drift / (1.0 - dt)
        tame_drift = drift / (1.0 - dt * drift.abs())
        adj_mean = adj + tame_drift * dt
        adj_new  = adj_mean + diffusion * sqrt_mdt * noise
        """

        adj_new  = adj_new.clamp(0.0, 1.0)
        adj_mean = adj_mean.clamp(0.0, 1.0)

        flags_mask = (flags[:, :, None] * flags[:, None, :]).float()

        adj_new_triu = torch.triu(adj_new, diagonal=1) * flags_mask
        adj_new = adj_new_triu + adj_new_triu.transpose(-1, -2)

        adj_mean_triu = torch.triu(adj_mean, diagonal=1) * flags_mask
        adj_mean = adj_mean_triu + adj_mean_triu.transpose(-1, -2)

        assert_symmetric_and_masked(adj_new, flags)
        assert_symmetric_and_masked(adj_mean, flags)
        return adj_new, adj_mean

class LangevinCorrector:
    def __init__(self, sde, score_fn, snr, scale_eps, n_steps, eps):
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.scale_eps = scale_eps
        self.n_steps = n_steps
        self.eps = eps

    def update(self, adj, flags, t):
        device, dtype = adj.device, adj.dtype
        B = adj.shape[0]

        for _ in range(self.n_steps):
            score = self.score_fn.compute_score(adj, flags, t)
            precond = (adj * (1.0 - adj)).clamp_min(1e-12)
            score = precond * score

            noise = gen_noise(adj, flags).to(device=device, dtype=dtype)
            mask = flags.unsqueeze(2) * flags.unsqueeze(1)
            def masked_norm(X):
                Xf = (X * mask).reshape(B, -1)
                return Xf.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            f_norm = masked_norm(score)
            n_norm = masked_norm(noise)
            step_size = 2.0 * (self.snr * n_norm / f_norm).pow(2)  # (B,1)
            step_size = step_size.clamp(1e-20, 1e-10).view(B, 1, 1)

            # Drift + noise
            adj_mean = adj + step_size * score
            adj_new  = adj_mean + torch.sqrt(2.0 * step_size) * noise * self.scale_eps

            adj_new  = adj_new.clamp(0.0, 1.0)
            adj_mean = adj_mean.clamp(0.0, 1.0)

            adj = adj_new

        return adj_new, adj_mean


class PCSolver:
    def __init__(
            self,
            sde,
            shape_adj,
            model,
            node_features,
            rrwp_steps,
            max_n_nodes,
            snr=0.1,
            scale_eps=1.0,
            n_steps=1,
            denoise=True,
            eps=1e-3,
            device="cuda",
            order=10,
            sample_target=True,
            eps_corrector=1e-5,
            eps_score=1e-10,
            eps_score_dist=1e-5,
            use_corrector=False,
            time_schedule="log",
            time_schedule_power=2.0,
            use_sampled_features=True,
            score_mode="graph",
        ):
        self.sde = sde
        self.shape_adj = shape_adj
        self.denoise = denoise
        self.eps = eps
        self.device = device
        self.n_steps = n_steps
        self.use_corrector = use_corrector
        self.time_schedule = time_schedule
        self.time_schedule_power = time_schedule_power
        self.score_mode = score_mode
        
        jacobi_score = JacobiScore(
            model=model,
            extra_features=node_features,
            rrwp_steps=rrwp_steps,
            max_n_nodes=max_n_nodes,
            order=order,
            sample_target=sample_target,
            eps_score=eps_score,
            eps_score_dist=eps_score_dist,
            use_sampled_features=use_sampled_features,
            alpha=sde.alpha,
            beta=sde.beta,
            direct_model_score=(score_mode == "direct_score"),
        )

        self.predictor = EulerMaruyamaPredictor(sde, jacobi_score)
        self.corrector = LangevinCorrector(sde, jacobi_score, snr, scale_eps, n_steps, eps_corrector)

    def solve(self, flags):
        with torch.no_grad():
            adj = self.sde.prior_sampling(self.shape_adj).to(self.device)
            adj = mask_adjs(adj, flags)

            history = []
            N = self.sde.N
            ts = build_time_schedule(
                N=N,
                T=self.sde.T,
                eps=self.eps,
                kind=self.time_schedule,
                power=self.time_schedule_power,
            ).to(self.device, dtype=adj.dtype)
            for i in trange(0, N, desc="[Sampling]", position=1, leave=False):
                t, dt  = ts[i], (ts[i+1] - ts[i]).item()
                vec_t  = torch.full((self.shape_adj[0],), t, device=self.device, dtype=adj.dtype)

                adj, adj_mean = self.predictor.update(adj, flags, vec_t, dt)
                if self.use_corrector:
                    adj, adj_mean = self.corrector.update(adj, flags, vec_t)
                
                history.append(adj[0].detach().cpu())
            
        flags0 = flags[0].cpu().bool()
        active_idx = flags0.nonzero(as_tuple=True)[0]

        # pick 100 evenly spaced snapshots
        idxs = np.linspace(0, len(history) - 1, 100, dtype=int)
        snapshots = [history[i].numpy()[np.ix_(active_idx, active_idx)] for i in idxs]

        graph_fig = plot_graph_snapshots(
            snapshots,
            grid_shape=(10, 10),
            threshold=0.5,
            layout_seed=42,
            node_size=20,
            edge_width=0.8,
        )
        save_figure(graph_fig, "tests/history_graphs.png", dpi=150)

        heatmap_fig = plot_heatmap_snapshots(
            snapshots,
            grid_shape=(10, 10),
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
        )
        save_figure(heatmap_fig, "tests/history_heatmaps.png", dpi=150)

        return ((adj_mean if self.denoise else adj), N * (self.n_steps + 1))
