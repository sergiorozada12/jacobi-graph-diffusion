import torch
import numpy as np

from src.utils import mask_adjs

class JacobiSDEOld:
    def __init__(self, N=1000, alpha=1.0, beta=1.0, speed=2.0, eps=1e-7):
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.speed = speed
        self.eps = eps
        self.device = None

    @property
    def T(self):
        return 1.0

    def to(self, device):
        self.device = device
        return self

    def sde(self, x, t):
        x = x.clamp(self.eps, 1.0 - self.eps)
        drift = (self.speed / 2) * (self.alpha * (1 - x) - self.beta * x)
        diffusion = torch.sqrt(self.speed * x * (1.0 - x))
        return drift, diffusion

    def prior_sampling(self, shape):
        x = torch.rand(*shape, device=self.device).triu(1)
        return x + x.transpose(-1, -2)

    def reverse(self, score_fn):
        parent_sde = self.sde
        N = self.N
        eps = self.eps
        speed = self.speed
        alpha = self.alpha
        beta = self.beta

        class ReverseJacobiSDE(JacobiSDE):
            def __init__(self):
                super().__init__(N, alpha, beta, speed, eps)

            def sde(self, x, flags, t):
                drift, diffusion = parent_sde(x, t)
                score = score_fn.compute_score(x, flags, t)
                drift = drift - diffusion**2 * score
                return drift, diffusion

        return ReverseJacobiSDE()

    def transition(self, x, t, dt):
        drift, diffusion = self.sde(x, t)
        mean = x + drift * dt
        std = diffusion * np.sqrt(dt)
        return mean, std


class JacobiSDE:
    """
    Jacobi diffusion on [0,1] with cosine speed:
      s(t) = s_min + (s_max - s_min) * sin^2(pi t / 2),  t in [0,1]
    """
    def __init__(self, N=1000, alpha=1.0, beta=1.0, s_min=0.1, s_max=2.0, eps=1e-5):
        self.N = int(N)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.eps = float(eps)
        self.device = None

    @property
    def T(self):
        return 1.0

    def to(self, device):
        self.device = device
        return self

    def _speed(self, t):
        tt = torch.as_tensor(t, dtype=torch.float32, device=self.device).clamp(0.0, self.T)
        return self.s_min + (self.s_max - self.s_min) * torch.sin(0.5 * np.pi * tt) ** 2

    def sde(self, x, t):
        #x = x.clamp(self.eps, 1.0 - self.eps)
        s = self._speed(t)
        while s.ndim < x.ndim:
            s = s.unsqueeze(-1)
        drift = 0.5 * s * (self.alpha * (1.0 - x) - self.beta * x)
        diffusion = torch.sqrt(s * x * (1.0 - x))
        #print(diffusion[0, 0, :5].cpu().detach().numpy())
        #diffusion = diffusion.clamp(min=1e-3, max=1.0 - 1e-3)
        # drift = drift.clamp(min=-1.0, max=1.0)
        return drift, diffusion

    def transition(self, x, t, dt):
        drift, diffusion = self.sde(x, t)
        mean = x + drift * dt
        std = diffusion * np.sqrt(float(dt))
        return mean, std

    def reverse(self, score_fn):
        parent = self

        class ReverseJacobiSDE(JacobiSDE):
            def __init__(self):
                super().__init__(parent.N, parent.alpha, parent.beta,
                                 parent.s_min, parent.s_max, parent.eps)
                self.device = parent.device

            def sde(self, x, flags, t):
                x1 = x.clamp(1e-5, 1.0 - 1e-5) # This 0.1, 0.99 // All 1e-5
                x2 = x.clamp(1e-5, 1.0 - 1e-5) # This all 0 // All 1e-5

                # Base drift and diffusion from SDE
                drift, diffusion = parent.sde(x1, t)
                drift     = mask_adjs(drift, flags)
                diffusion = mask_adjs(diffusion, flags)

                # Score-based force
                score = score_fn.compute_score(x2, flags, t)
                force = (diffusion ** 2) * score

                #B = force.shape[0]
                #fvec  = force.reshape(B, -1)
                #fnorm = fvec.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # (B,1)
                #force = (fvec / fnorm).view_as(force)

                force = force.clamp(-1000.0, 1000.0)

                # Scale force per sample
                #B = force.shape[0]
                #M = (flags[:, :, None] * flags[:, None, :]).sum(dim=(1, 2))        # active entries
                #tau_time = 1.0 * torch.sqrt(t).view(B, 1) * torch.sqrt(M).view(B, 1)     # (B,1)
                #fvec  = force.reshape(B, -1)
                #fnorm = fvec.norm(dim=-1, keepdim=True).clamp_min(1e-12)           # (B,1)
                #scale = (tau_time / fnorm).clamp(max=1.0)                          # (B,1)
                #print('scale', scale.mean())
                #force = (fvec * scale).view_as(force)

                print('force', force.mean(), force.max(), force.min())
                print('drift', drift.mean(), drift.max(), drift.min())
                print('diff', diffusion.mean(), diffusion.max(), diffusion.min())
                # Correct drift with scaled force
                drift = drift - force

                return drift, diffusion

        return ReverseJacobiSDE()

    def prior_sampling(self, shape):
        x = torch.rand(*shape, device=self.device).triu(1)
        return x + x.transpose(-1, -2)
