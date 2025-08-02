import torch
import numpy as np
from tqdm import trange

from src.sde.score import JacobiScore
from src.utils import mask_adjs, mask_x, gen_noise


class EulerMaruyamaPredictor:
    def __init__(self, sde, score_fn):
        self.sde = sde
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn

    def update(self, x, adj, flags, t):
        dt = -1.0 / self.rsde.N
        noise = gen_noise(adj, flags)
        drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=True)

        adj_mean = adj + drift * dt
        adj = adj_mean + diffusion * np.sqrt(-dt) * noise

        adj = torch.clamp(adj, 1e-5, 1 - 1e-5)
        adj_mean = torch.clamp(adj_mean, 1e-5, 1 - 1e-5)

        return adj, adj_mean


class LangevinCorrector:
    def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.scale_eps = scale_eps
        self.n_steps = n_steps

    def update(self, x, adj, flags, t):
        n_steps = self.n_steps
        target_snr = self.snr
        seps = self.scale_eps

        for _ in range(n_steps):
            grad = self.score_fn.compute_score(x, adj, flags, t)
            noise = gen_noise(adj, flags)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = 2 * (target_snr * noise_norm / grad_norm) ** 2
            adj_mean = adj + step_size * grad
            adj = adj_mean + torch.sqrt(step_size * 2) * noise * seps

            adj = torch.clamp(adj, 1e-5, 1 - 1e-5)
            adj_mean = torch.clamp(adj_mean, 1e-5, 1 - 1e-5)
        return adj, adj_mean


class PCSolver:
    def __init__(
            self,
            sde,
            shape_x,
            shape_adj,
            model,
            node_features,
            k_eig,
            snr=0.1,
            scale_eps=1.0,
            n_steps=1,
            denoise=True,
            eps=1e-3,
            device="cuda",
            order=10,
        ):

        self.sde = sde
        self.shape_x = shape_x
        self.shape_adj = shape_adj
        self.denoise = denoise
        self.eps = eps
        self.device = device
        self.n_steps = n_steps
        
        jacobi_score = JacobiScore(
            model=model,
            node_features=node_features,
            k_eig=k_eig,
            order=order
        )

        self.predictor = EulerMaruyamaPredictor(sde, jacobi_score)
        self.corrector = LangevinCorrector(sde, jacobi_score, snr, scale_eps, n_steps)

    def solve(self, flags):
        with torch.no_grad():
            x = torch.ones(self.shape_x, device=self.device)
            adj = self.sde.prior_sampling(self.shape_adj).to(self.device)

            x = mask_x(x, flags)
            adj = mask_adjs(adj, flags)
            diff_steps = self.sde.N
            timesteps = torch.linspace(self.sde.T, self.eps, diff_steps, device=self.device)

            for i in trange(0, (diff_steps), desc="[Sampling]", position=1, leave=False):
                t = timesteps[i]
                vec_t = torch.ones(self.shape_adj[0], device=t.device) * t
                adj, adj_mean = self.corrector.update(x, adj, flags, vec_t)
                adj, adj_mean = self.predictor.update(x, adj, flags, vec_t)

        return (
            (adj_mean if self.denoise else adj),
            diff_steps * (self.n_steps + 1),
        )
