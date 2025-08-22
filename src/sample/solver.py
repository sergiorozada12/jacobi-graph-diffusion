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

    @torch.no_grad()
    def update(self, adj, flags, t):
        dt = -1.0 / self.rsde.N
        device, dtype = adj.device, adj.dtype

        noise = gen_noise(adj, flags).to(device=device, dtype=dtype)
        drift, diffusion = self.rsde.sde(adj, flags, t)

        # EM update
        sqrt_mdt = torch.sqrt(torch.tensor(-dt, device=device, dtype=dtype))
        adj_mean = adj + drift * dt
        adj_new  = adj_mean + diffusion * sqrt_mdt * noise
        print('scd', (drift * dt).mean(), (drift * dt).min(), (drift * dt).max())

        # Sanitize & project back to valid domain
        adj_new  = torch.nan_to_num(adj_new,  nan=0.0, posinf=1.0, neginf=0.0)
        adj_mean = torch.nan_to_num(adj_mean, nan=0.0, posinf=1.0, neginf=0.0)

        # Enforce symmetry & zero diagonal, then mask, then clamp
        adj_new = torch.triu(adj_new, 1); adj_new = adj_new + adj_new.transpose(-1, -2)
        adj_mean = torch.triu(adj_mean, 1); adj_mean = adj_mean + adj_mean.transpose(-1, -2)

        #mask = (flags[:, :, None] * flags[:, None, :]).to(dtype=dtype, device=device)
        adj_new  = (adj_new).clamp(0.0, 1.0)
        adj_mean = (adj_mean).clamp(0.0, 1.0)

        return adj_new, adj_mean


class LangevinCorrector:
    def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.scale_eps = scale_eps
        self.n_steps = n_steps

    @torch.no_grad()
    def update(self, adj, flags, t):
        target_snr = self.snr
        seps = self.scale_eps
        device, dtype = adj.device, adj.dtype
        B = adj.shape[0]
        mask = (flags[:, :, None] * flags[:, None, :]).to(dtype=dtype, device=device)

        for _ in range(self.n_steps):
            # diffusion & score
            _, diffusion = self.sde.sde(adj, t)
            score = self.score_fn.compute_score(adj.clamp(1e-5, 1-1e-5), flags, t)
            force = (diffusion ** 2) * score

            # Clip per sample
            #M = mask.sum(dim=(1,2)).clamp(min=1.0)                  # active entries per sample
            #tau_time = 100.0 * torch.sqrt(t).view(B,1) * torch.sqrt(M).view(B,1)  # (B,1)
            #fvec  = force.reshape(B, -1)
            #fnorm = fvec.norm(dim=-1, keepdim=True).clamp_min(1e-12)             # (B,1)
            #scale = (tau_time / fnorm).clamp(max=1.0)                             # (B,1)
            #force = (fvec * scale).view_as(force)

            # Target-SNR step size per sample
            noise = gen_noise(adj, flags).to(device=device, dtype=dtype)
            f_norm = force.reshape(B, -1).norm(dim=-1, keepdim=True).clamp_min(1e-12)  # (B,1)
            n_norm = noise.reshape(B, -1).norm(dim=-1, keepdim=True).clamp_min(1e-12)  # (B,1)
            step_size = 2.0 * (target_snr * n_norm / f_norm).pow(2)  # (B,1)
            step_size = step_size.view(B, 1, 1)

            # Update
            adj_mean = adj + step_size * force
            adj_new  = adj_mean + torch.sqrt(2.0 * step_size) * noise * seps
            #print('scale', scale.mean())
            print('stepsize', step_size.mean())

            # Sanitize non-finite
            adj_new  = torch.nan_to_num(adj_new,  nan=0.0, posinf=1.0, neginf=0.0)
            adj_mean = torch.nan_to_num(adj_mean, nan=0.0, posinf=1.0, neginf=0.0)

            # Symmetry, mask, clamp
            adj_new = torch.triu(adj_new, 1); adj_new = adj_new + adj_new.transpose(-1, -2)
            adj_mean = torch.triu(adj_mean, 1); adj_mean = adj_mean + adj_mean.transpose(-1, -2)

            adj_new  = (adj_new).clamp(0.0, 1.0)
            adj_mean = (adj_mean).clamp(0.0, 1.0)

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
        ):
        self.sde = sde
        self.shape_adj = shape_adj
        self.denoise = denoise
        self.eps = eps
        self.device = device
        self.n_steps = n_steps
        
        jacobi_score = JacobiScore(
            model=model,
            extra_features=node_features,
            rrwp_steps=rrwp_steps,
            max_n_nodes=max_n_nodes,
            order=order
        )

        self.predictor = EulerMaruyamaPredictor(sde, jacobi_score)
        self.corrector = LangevinCorrector(sde, jacobi_score, snr, scale_eps, n_steps)

    def solve(self, flags):
        with torch.no_grad():
            adj = self.sde.prior_sampling(self.shape_adj).to(self.device)
            adj = mask_adjs(adj, flags)
            diff_steps = self.sde.N
            timesteps = torch.linspace(self.sde.T, self.eps, diff_steps, device=self.device)
            for i in trange(0, (diff_steps), desc="[Sampling]", position=1, leave=False):
                t = timesteps[i]
                vec_t = torch.ones(self.shape_adj[0], device=t.device) * t
                #adj, adj_mean = self.corrector.update(adj, flags, vec_t)
                adj, adj_mean = self.predictor.update(adj, flags, vec_t)

        return (
            (adj_mean if self.denoise else adj),
            diff_steps * (self.n_steps + 1),
        )
