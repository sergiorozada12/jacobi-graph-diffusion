import torch
import numpy as np
from tqdm import trange
import networkx as nx
import matplotlib.pyplot as plt

from src.sde.score import JacobiScore
from src.utils import mask_adjs, mask_x, gen_noise, assert_symmetric_and_masked


def make_timesteps(T, eps, N, kind="log", power=2.0):
    import math
    if kind == "log":
        # geometric spacing (many small steps near eps)
        t = torch.exp(torch.linspace(math.log(T), math.log(eps), N+1))
    elif kind == "log_power":
        # t = T^(1 - u^p) * eps^(u^p)
        w = u.pow(power)
        t = (T ** (1.0 - w)) * (eps ** w)
    elif kind == "double_log":
        # more extreme than geometric: convex warp in log-space
        # a controls aggressiveness; larger a => more packed near eps
        a = power if power is not None else 3.0
        # m(u) in [0,1], convex increasing
        m = 1.0 - torch.exp(-a * u)
        logT, logE = math.log(T), math.log(eps)
        t = torch.exp(logE + m * (logT - logE))
    elif kind == "cosine":
        u = torch.linspace(0, 1, N+1)
        w = 0.5 * (1 - torch.cos(math.pi * u))  # 0→0, 1→1
        t = T - (T - eps) * w                   # denser near both ends (gentle)
    elif kind == "power":
        u = torch.linspace(0, 1, N+1)
        w = u.pow(power)                        # power>1 densifies near eps
        t = T - (T - eps) * w
    else:
        t = torch.linspace(T, eps, N+1)
    return t  # length N+1, decreasing


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
        target_snr = self.snr
        seps = self.scale_eps
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
        ):
        self.sde = sde
        self.shape_adj = shape_adj
        self.denoise = denoise
        self.eps = eps
        self.device = device
        self.n_steps = n_steps
        self.use_corrector = use_corrector
        
        jacobi_score = JacobiScore(
            model=model,
            extra_features=node_features,
            rrwp_steps=rrwp_steps,
            max_n_nodes=max_n_nodes,
            order=order,
            sample_target=sample_target,
            eps_score=eps_score,
            eps_score_dist=eps_score_dist
        )

        self.predictor = EulerMaruyamaPredictor(sde, jacobi_score)
        self.corrector = LangevinCorrector(sde, jacobi_score, snr, scale_eps, n_steps, eps_corrector)

    def solve(self, flags):
        with torch.no_grad():
            adj = self.sde.prior_sampling(self.shape_adj).to(self.device)
            adj = mask_adjs(adj, flags)

            history = []
            N = self.sde.N
            ts = make_timesteps(self.sde.T, self.eps, N, kind="log").to(self.device, dtype=adj.dtype)
            for i in trange(0, N, desc="[Sampling]", position=1, leave=False):
                t, dt  = ts[i], (ts[i+1] - ts[i]).item()
                vec_t  = torch.full((self.shape_adj[0],), t, device=self.device, dtype=adj.dtype)

                adj, adj_mean = self.predictor.update(adj, flags, vec_t, dt)
                if self.use_corrector:
                    adj, adj_mean = self.corrector.update(adj, flags, vec_t)
                
                history.append(adj[0].detach().cpu())
            
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np

        flags0 = flags[0].cpu().bool()
        active_idx = flags0.nonzero(as_tuple=True)[0]

        # pick 100 evenly spaced snapshots
        idxs = np.linspace(0, len(history) - 1, 100, dtype=int)
        snapshots = [history[i].numpy()[np.ix_(active_idx, active_idx)] for i in idxs]

        fig, axes = plt.subplots(10, 10, figsize=(18, 18))
        axes = axes.reshape(-1)

        # fixed layout from last snapshot
        G_last = nx.from_numpy_array((snapshots[-1] > 0.5).astype(int))
        pos = nx.spring_layout(G_last, seed=42)

        for ax, A in zip(axes, snapshots):
            G = nx.from_numpy_array((A > 0.5).astype(int))
            ax.axis("off")
            nx.draw_networkx(G, pos=pos, with_labels=False, node_size=20, width=0.8, ax=ax)

        plt.tight_layout()
        plt.savefig("history_graphs.png", dpi=150)
        plt.close(fig)

        # also plot heatmaps
        fig, axes = plt.subplots(10, 10, figsize=(18, 18))
        axes = axes.reshape(-1)
        for ax, A in zip(axes, snapshots):
            ax.imshow(A, vmin=0, vmax=1, cmap="Greys")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig("history_heatmaps.png", dpi=150)
        plt.close(fig)

        return ((adj_mean if self.denoise else adj), N * (self.n_steps + 1))
