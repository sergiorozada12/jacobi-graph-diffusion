import torch
import numpy as np


class JacobiSDE:
    def __init__(self, N=1000, alpha=1.0, beta=1.0, speed=2.0, eps=1e-5):
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
