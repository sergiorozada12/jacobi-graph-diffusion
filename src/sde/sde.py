import torch
import numpy as np

from src.utils import mask_adjs


class JacobiSDE:
    """
    Jacobi diffusion on [0,1] with cosine speed:
      s(t) = s_min + (s_max - s_min) * sin^2(pi t / 2),  t in [0,1]
    """
    def __init__(self, N=1000, alpha=1.0, beta=1.0, s_min=0.1, s_max=2.0, eps=1e-5, max_force=1000.0):
        self.N = int(N)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.eps = float(eps)
        self.max_force = max_force
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
        s = self._speed(t)
        while s.ndim < x.ndim:
            s = s.unsqueeze(-1)
        drift = 0.5 * s * (self.alpha * (1.0 - x) - self.beta * x)
        diffusion_2 = (s * x * (1.0 - x))
        diffusion = torch.sqrt(diffusion_2).clamp_min(self.eps) # PLAY WITH THIS THRESHOLD!
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
                super().__init__(
                    parent.N,
                    parent.alpha,
                    parent.beta,
                    parent.s_min,
                    parent.s_max,
                    parent.eps,
                    parent.max_force,
                )
                self.device = parent.device
    
            def sde(self, x, flags, t):
                #x = x.clamp(self.eps, 1.0 - self.eps)
    
                # Base drift and diffusion from SDE
                drift, diffusion = parent.sde(x, t)
                drift     = mask_adjs(drift, flags)
                diffusion = mask_adjs(diffusion, flags)

                #x = x.clamp(self.eps, 1.0 - self.eps)
    
                # Score-based force
                score = score_fn.compute_score(x, flags, t)
                force = (diffusion ** 2) * score
                #force = force.clamp(-self.max_force, self.max_force) # THIS SEEMS TO HELP FOR ACC NOT SO MUCH FOR METRICS
    
                # Correct drift with scaled force
                drift = drift - force
    
                return drift, diffusion
            
            def sde_logit(self, x, flags, t):
                x = x.clamp(self.eps, 1.0 - self.eps)
                den = (x * (1.0 - x)).clamp_min(1e-12)

                mu_x, sigma_x = parent.sde(x, t)
                mu_x     = mask_adjs(mu_x, flags)
                sigma_x  = mask_adjs(sigma_x, flags)
                sigma_x2 = sigma_x ** 2

                mu_z    = mu_x / den - 0.5 * sigma_x2 * (1.0 - 2.0 * x) / (den ** 2)
                sigma_z = sigma_x / den

                score_x = score_fn.compute_score(x, flags, t)
                score_z = score_x * den + (1.0 - 2.0 * x)

                drift_z = mu_z - (sigma_z ** 2) * score_z
                drift_z = mask_adjs(drift_z, flags)
                sigma_z = mask_adjs(sigma_z, flags)
                return drift_z, sigma_z
    
        return ReverseJacobiSDE()

    def prior_sampling(self, shape):
        device = self.device
        if device is None:
            device = torch.device("cpu")
        else:
            device = torch.device(device)

        if self.alpha == 1.0 and self.beta == 1.0:
            base = torch.rand(*shape, device=device)
        else:
            dtype = torch.get_default_dtype()
            alpha = torch.tensor(self.alpha, dtype=dtype)
            beta = torch.tensor(self.beta, dtype=dtype)
            base = torch.distributions.Beta(alpha, beta).sample(shape)
            base = base.to(device)
        base = base.triu(1)
        return base + base.transpose(-1, -2)
