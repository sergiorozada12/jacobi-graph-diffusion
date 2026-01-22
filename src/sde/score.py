import torch
import numpy as np

from src.features.extra_features import ExtraFeatures
import torch.nn.functional as F
from src.utils import assert_symmetric_and_masked, assert_symmetric_and_masked_E


class JacobiScore:
    def __init__(
            self, 
            model, 
            extra_features, 
            rrwp_steps, 
            max_n_nodes, 
            order=10, 
            eps_score=1e-10, 
            eps_score_dist=1e-5, 
            sample_target=True,
            use_sampled_features=True,
            alpha=1.0,
            beta=1.0,
            direct_model_score=False,
        ):
        self.order = order
        self.eps = eps_score
        self.eps_dist = eps_score_dist
        self.model = model
        self.feature_extractor = ExtraFeatures(
            extra_features_type=extra_features,
            rrwp_steps=rrwp_steps,
            max_n_nodes=max_n_nodes,
        )
        self.sample_target = sample_target
        self.decay_cutoff = 1e-12
        self.use_sampled_features = use_sampled_features
        self.direct_model_score = direct_model_score
        if self.model is not None:
            self.model.eval()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.jacobi_a = self.beta - 1.0
        self.jacobi_b = self.alpha - 1.0

    def _jacobi_polynomials(self, x, order, a, b):
        if order <= 0:
            raise ValueError("Order must be positive")

        device, dtype = x.device, x.dtype
        a_t = torch.as_tensor(a, dtype=dtype, device=device)
        b_t = torch.as_tensor(b, dtype=dtype, device=device)
        ab = a_t + b_t
        diff = a_t - b_t
        a_sq_minus_b_sq = diff * (a_t + b_t)

        polys = []
        p_prev = torch.ones_like(x)
        polys.append(p_prev)

        if order == 1:
            return torch.stack(polys, dim=-1)

        p_curr = 0.5 * ((2.0 + ab) * x + diff)
        polys.append(p_curr)

        for n in range(1, order - 1):
            n_t = torch.as_tensor(float(n), dtype=dtype, device=device)
            two_n = 2.0 * n_t
            ab_n = two_n + ab
            ab_np1 = ab_n + 1.0
            ab_np2 = ab_np1 + 1.0

            numerator1 = ab_np1 * (ab_n * ab_np2 * x + a_sq_minus_b_sq)
            numerator2 = 2.0 * (n_t + a_t) * (n_t + b_t) * ab_np2
            denominator = 2.0 * (n_t + 1.0) * (n_t + ab + 1.0) * ab_n

            p_next = (numerator1 * p_curr - numerator2 * p_prev) / denominator
            polys.append(p_next)
            p_prev, p_curr = p_curr, p_next

        return torch.stack(polys, dim=-1)

    def jacobi_poly_and_derivative(self, x):
        P_stack = self._jacobi_polynomials(x, self.order, self.jacobi_a, self.jacobi_b)

        if self.order == 1:
            dP_stack = torch.zeros_like(P_stack)
            return P_stack, dP_stack

        shifted = self._jacobi_polynomials(
            x,
            self.order - 1,
            self.jacobi_a + 1.0,
            self.jacobi_b + 1.0,
        )

        device, dtype = x.device, x.dtype
        n = torch.arange(1, self.order, dtype=dtype, device=device)
        factors = 0.5 * (n + self.jacobi_a + self.jacobi_b + 1.0)
        shape = (1,) * x.ndim + (self.order - 1,)
        factors = factors.view(shape)

        dP_stack = torch.zeros_like(P_stack)
        dP_stack[..., 1:] = shifted * factors
        return P_stack, dP_stack

    def jacobi_score(self, adj_0, adj, t):
        orig_dtype = adj.dtype
        adj_0 = adj_0.to(torch.float64)
        adj = adj.to(torch.float64)
        t = t.to(torch.float64)

        x0 = 2.0 * adj_0 - 1.0  # [B, N, N]
        xt = 2.0 * adj - 1.0    # [B, N, N]

        P_x0, _ = self.jacobi_poly_and_derivative(x0)     # [B, N, N, order]
        P_xt, dP_xt = self.jacobi_poly_and_derivative(xt) # [B, N, N, order]

        device = adj.device
        n = torch.arange(self.order, dtype=torch.float64, device=device)
        lambdas = 0.5 * n * (n + self.alpha + self.beta - 1.0)
        lambdas = lambdas[None, None, None, :]

        a = torch.tensor(self.jacobi_a, dtype=torch.float64, device=device)
        b = torch.tensor(self.jacobi_b, dtype=torch.float64, device=device)
        raw_weights = 2.0 * n + a + b + 1.0

        log_weights = torch.empty_like(raw_weights)
        general_mask = ~( (n == 0) & (torch.abs(raw_weights) < 1e-12) )
        if general_mask.any():
            n_gen = n[general_mask]
            raw_gen = raw_weights[general_mask]
            log_weights[general_mask] = (
                torch.log(torch.abs(raw_gen))
                + torch.lgamma(n_gen + a + 1.0)
                + torch.lgamma(n_gen + b + 1.0)
                - torch.lgamma(n_gen + 1.0)
                - torch.lgamma(n_gen + a + b + 1.0)
            )

        if (~general_mask).any():
            log_weights[~general_mask] = torch.lgamma(a + 1.0) + torch.lgamma(b + 1.0)

        log_decay = -t[:, None, None, None] * lambdas + log_weights      # [B,1,1,order]
        log_decay_max = log_decay.max(dim=-1, keepdim=True).values       # [B,1,1,1]
        decay_scaled = torch.exp(log_decay - log_decay_max)              # [B,1,1,order]

        weighted_density = (decay_scaled * P_xt * P_x0).sum(dim=-1)      # [B,N,N]
        weighted_grad = (decay_scaled * dP_xt * P_x0).sum(dim=-1)        # [B,N,N]
        scale = torch.exp(log_decay_max.squeeze(-1))                     # [B,1,1]

        density = weighted_density * scale                               # [B,N,N]
        grad_xt = 2.0 * weighted_grad * scale                            # [B,N,N]

        score = grad_xt / density.clamp_min(self.eps)
        return score.to(orig_dtype)

    def compute_score(self, A_t_dist, flags, t):
        assert_symmetric_and_masked(A_t_dist, flags)
        flags_mask = (flags[:, :, None] * flags[:, None, :]).float()
        A_t_dist = A_t_dist * flags_mask

        A_t_triu = torch.triu(A_t_dist, diagonal=1)
        if self.use_sampled_features:
            A_t_triu_sample = torch.bernoulli(A_t_triu)
            A_t_sample = A_t_triu_sample + A_t_triu_sample.transpose(-1, -2)
        else:
            A_t_sample = A_t_triu + A_t_triu.transpose(-1, -2)

        A_t = A_t_triu + A_t_triu.transpose(-1, -2)

        assert_symmetric_and_masked(A_t, flags)
        E_t = torch.cat([(1 - A_t).unsqueeze(-1), A_t.unsqueeze(-1)], dim=-1).float()
        E_t_sample = torch.cat([(1 - A_t_sample).unsqueeze(-1), A_t_sample.unsqueeze(-1)], dim=-1).float()
        feature_input = E_t_sample if self.use_sampled_features else E_t
        extra_pred = self.feature_extractor(feature_input, flags)
        y = torch.cat((extra_pred.y.float(), t.unsqueeze(1)), dim=1).float()

        pred = self.model(extra_pred.X.float(), extra_pred.E.float(), y, flags)

        if self.direct_model_score:
            score_raw = pred.E[..., 0]
            score_triu = torch.triu(score_raw, diagonal=1)
            score = score_triu + score_triu.transpose(-1, -2)
            return score * flags_mask

        assert_symmetric_and_masked_E(pred.E, flags)
        A_0_dist = F.softmax(pred.E, dim=-1)[..., 1:].sum(dim=-1).float()
        A_0_dist = A_0_dist * flags_mask

        if self.sample_target:
            A_0_triu = torch.bernoulli(torch.triu(A_0_dist, diagonal=1))
            A_0_triu = A_0_triu * flags_mask
        else:
            A_0_triu = torch.triu(A_0_dist, diagonal=1)
        A_0 = A_0_triu + A_0_triu.transpose(-1, -2)

        score_raw = self.jacobi_score(A_0, A_t_dist, t).float()

        score_triu = torch.triu(score_raw, diagonal=1)
        score = score_triu + score_triu.transpose(-1, -2)
        return score * flags_mask
