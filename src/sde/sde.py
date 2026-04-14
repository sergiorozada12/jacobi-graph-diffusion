import math
import torch
import torch.nn as nn
import numpy as np
from src.utils import mask_adjs, mask_x


class JacobiSDE(nn.Module):
    """
    Jacobi diffusion on [0,1] with cosine speed:
      s(t) = s_min + (s_max - s_min) * sin^2(pi t / 2),  t in [0,1]
    """
    def __init__(self, num_scales=1000, alpha=1.0, beta=1.0, s_min=0.1, s_max=2.0, eps=1e-5):
        super().__init__()
        self.num_scales = int(num_scales)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.eps = float(eps)
        self.device = None

    @property
    def T(self):
        return 1.0

    @property
    def N(self):
        return self.num_scales

    @N.setter
    def N(self, value):
        self.num_scales = int(value)

    def to(self, device):
        self.device = device
        return self

    def _speed(self, t, device):
        tt = torch.as_tensor(t, dtype=torch.float32, device=device).clamp(0.0, self.T)
        return self.s_min + (self.s_max - self.s_min) * torch.sin(0.5 * np.pi * tt) ** 2

    def _drift_diffusion_and_grad(self, x, t):
        s = self._speed(t, device=x.device)
        while s.ndim < x.ndim:
            s = s.unsqueeze(-1)

        drift = 0.5 * s * (self.alpha * (1.0 - x) - self.beta * x)
        diffusion_2 = (s * x * (1.0 - x))
        diffusion = torch.sqrt(diffusion_2).clamp_min(self.eps)
        diffusion_grad = 0.5 * s * (1.0 - 2.0 * x) / diffusion.clamp_min(self.eps)
        return drift, diffusion, diffusion_grad

    def sde(self, x, t):
        drift, diffusion, _ = self._drift_diffusion_and_grad(x, t)
        return drift, diffusion

    def sde_with_diffusion_grad(self, x, t):
        return self._drift_diffusion_and_grad(x, t)

    def transition(self, x, t, dt):
        drift, diffusion = self.sde(x, t)

        if torch.is_tensor(dt):
            dt_tensor = dt.to(device=x.device, dtype=x.dtype).clamp_min(0.0)
            while dt_tensor.ndim < x.ndim:
                dt_tensor = dt_tensor.unsqueeze(-1)
            mean = x + drift * dt_tensor
            std = diffusion * torch.sqrt(dt_tensor).to(diffusion.dtype)
        else:
            mean = x + drift * dt
            std = diffusion * math.sqrt(float(dt))
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
                )
                self.device = parent.device
    
            def sde(self, x, flags, t):
                drift, diffusion, _ = self.sde_with_diffusion_grad(x, flags, t)
                return drift, diffusion

            def sde_with_diffusion_grad(self, x, flags, t):
                # Compute score internally
                score = score_fn.compute_score(x, flags, t)
                return self.drift_from_score(x, flags, t, score)

            def drift_from_score(self, x, flags, t, score):
                """Compute reverse drift/diffusion using a pre-computed score."""
                drift, diffusion, diffusion_grad = parent._drift_diffusion_and_grad(x, t)
                
                # Apply appropriate masking based on rank
                if x.ndim == 3:
                    # [B, N, K] -> Node features
                    drift          = mask_x(drift, flags)
                    diffusion      = mask_x(diffusion, flags)
                    diffusion_grad = mask_x(diffusion_grad, flags)
                else:
                    # [B, N, N] or [B, N, N, K] -> Adjacency/Bonds
                    drift          = mask_adjs(drift, flags)
                    diffusion      = mask_adjs(diffusion, flags)
                    diffusion_grad = mask_adjs(diffusion_grad, flags)
    
                # Correct drift with score-based force
                force = (diffusion ** 2) * score
                drift = drift - force
    
                return drift, diffusion, diffusion_grad
            
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
        if base.ndim == 3 and base.shape[1] == base.shape[2]:
            return base + base.transpose(-1, -2)
        elif base.ndim == 4 and base.shape[1] == base.shape[2]:
             # Categorical Adjacency: [B, N, N, K]
             return 0.5 * (base + base.transpose(1, 2))
        else:
             return base


class StickBreakingJacobiSDE(nn.Module):
    """
    K-class categorical diffusion using Stick-Breaking transform to K-1 Jacobi v-variables.
    """
    def __init__(self, K, alpha=1.0, beta=1.0, s_min=0.1, s_max=2.0, num_scales=1000, eps=1e-5):
        super().__init__()
        self.K = K
        self.eps = eps
        # Each v_i is an independent Jacobi diffusion
        self.base_sde = JacobiSDE(num_scales=num_scales, alpha=alpha, beta=beta, s_min=s_min, s_max=s_max, eps=eps)

    def to(self, device):
        super().to(device)
        self.base_sde.to(device)
        return self

    @property
    def T(self):
        return self.base_sde.T

    @property
    def num_scales(self):
        return self.base_sde.num_scales

    @num_scales.setter
    def num_scales(self, value):
        self.base_sde.num_scales = value

    @property
    def N(self):
        return self.base_sde.N

    @N.setter
    def N(self, value):
        self.base_sde.N = value

    @property
    def alpha(self):
        return self.base_sde.alpha

    @property
    def beta(self):
        return self.base_sde.beta

    def to(self, device):
        self.base_sde.to(device)
        return self

    def v_to_x(self, v):
        """[..., K-1] -> [..., K]"""
        bs = v.shape[:-1]
        K = self.K
        x = torch.zeros((*bs, K), device=v.device, dtype=v.dtype)
        
        rem = torch.ones(bs, device=v.device, dtype=v.dtype)
        for i in range(K - 1):
            x[..., i] = rem * v[..., i]
            rem = rem * (1.0 - v[..., i])
        x[..., K - 1] = rem
        return x

    def x_to_v(self, x):
        """[..., K] -> [..., K-1]"""
        bs = x.shape[:-1]
        K = self.K
        v = torch.zeros((*bs, K - 1), device=x.device, dtype=x.dtype)
        
        rem = torch.ones(bs, device=x.device, dtype=x.dtype)
        for i in range(K - 1):
            v[..., i] = (x[..., i] / rem).clamp(0.0, 1.0)
            # Handle potential div by zero gracefully
            v[..., i] = torch.where(rem > self.eps, v[..., i], torch.zeros_like(v[..., i]))
            rem = (rem - x[..., i]).clamp_min(0.0)
        return v

    def drift_diffusion_and_grad(self, v, t):
        return self.base_sde._drift_diffusion_and_grad(v, t)

    def transition(self, v, t, dt):
        return self.base_sde.transition(v, t, dt)

    def prior_sampling(self, shape):
        """Returns v samples from the stationary distribution (product of Betas)"""
        # Note: True Dirichlet stationary would require alpha/beta to vary per component,
        # but for simplicity and score-matching consistency, we often use uniform.
        return self.base_sde.prior_sampling((*shape, self.K - 1))

    def reverse(self, score_fn):
        parent = self
        
        class ReverseStickBreakingJacobiSDE:
            def __init__(self):
                self.K = parent.K
                self.base_rev = parent.base_sde.reverse(score_fn)
                self.eps = parent.eps

            def sde(self, x, flags, t):
                drift, diffusion, _ = self.sde_with_diffusion_grad(x, flags, t)
                return drift, diffusion

            def sde_with_diffusion_grad(self, x, flags, t):
                from src.utils import PlaceHolder as PH
                if isinstance(x, PH):
                    # Joint mode: compute score on the full PlaceHolder to preserve X context
                    score = score_fn.compute_score(x, flags, t)  # returns PlaceHolder(X=..., E=..., y=None)
                    return self.drift_from_score(x, flags, t, score)
                # Edge-only mode: delegate fully to base_rev
                return self.base_rev.sde_with_diffusion_grad(x, flags, t)

            def drift_from_score(self, x, flags, t, score):
                from src.utils import PlaceHolder as PH
                if isinstance(x, PH):
                    # Apply physics per component
                    drift_E, diff_E, dgrad_E = parent.base_sde._drift_diffusion_and_grad(x.E, t)
                    drift_E  = mask_adjs(drift_E, flags)
                    diff_E   = mask_adjs(diff_E, flags)
                    dgrad_E  = mask_adjs(dgrad_E, flags)
                    score_E  = score.E if isinstance(score, PH) else score
                    drift_E  = drift_E - (diff_E ** 2) * score_E

                    drift_X = diff_X = dgrad_X = None
                    if x.X is not None:
                        drift_X, diff_X, dgrad_X = parent.base_sde._drift_diffusion_and_grad(x.X, t)
                        drift_X  = mask_x(drift_X, flags)
                        diff_X   = mask_x(diff_X, flags)
                        dgrad_X  = mask_x(dgrad_X, flags)
                        score_X  = score.X if isinstance(score, PH) else None
                        if score_X is not None:
                            drift_X = drift_X - (diff_X ** 2) * score_X

                    return (
                        PH(X=drift_X, E=drift_E,  y=None),
                        PH(X=diff_X,  E=diff_E,   y=None),
                        PH(X=dgrad_X, E=dgrad_E,  y=None),
                    )
                return self.base_rev.drift_from_score(x, flags, t, score)

            def sde_logit(self, x, flags, t):
                return self.base_rev.sde_logit(x, flags, t)

        return ReverseStickBreakingJacobiSDE()
