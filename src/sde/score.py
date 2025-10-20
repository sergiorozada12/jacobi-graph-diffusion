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
            use_sampled_features=True
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
        self.model.eval()

    def legendre_poly_and_derivative(self, x):
        P = [torch.ones_like(x), x]  # P₀(x), P₁(x)
        for n in range(1, self.order - 1):
            pn = ((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1)
            P.append(pn)
        P_stack = torch.stack(P, dim=-1)  # shape [..., order]

        dP = [torch.zeros_like(x), torch.ones_like(x)]
        for n in range(1, self.order - 1):
            dp = ((2 * n + 1) * (P[n] + x * dP[n]) - n * dP[n - 1]) / (n + 1)
            dP.append(dp)
        dP_stack = torch.stack(dP, dim=-1)

        return P_stack, dP_stack

    def legendre_score(self, adj_0, adj, t):
        orig_dtype = adj.dtype
        adj_0 = adj_0.to(torch.float64)
        adj = adj.to(torch.float64)
        t = t.to(torch.float64)

        x0 = 2.0 * adj_0 - 1.0  # [B, N, N]
        xt = 2.0 * adj - 1.0    # [B, N, N]

        P_x0, _     = self.legendre_poly_and_derivative(x0)     # [B, N, N, order]
        P_xt, dP_xt = self.legendre_poly_and_derivative(xt)     # [B, N, N, order]

        n = torch.arange(self.order, dtype=torch.float64, device=adj.device)
        lambdas = 0.5 * (n * (n + 1))[None, None, None, :]               # [1, 1, 1, order]
        log_weights = torch.log(2 * n + 1)                               # [order]
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
        assert_symmetric_and_masked_E(pred.E, flags)
        A_0_dist = F.softmax(pred.E, dim=-1)[..., 1:].sum(dim=-1).float()
        A_0_dist = A_0_dist * flags_mask

        if self.sample_target:
            A_0_triu = torch.bernoulli(torch.triu(A_0_dist, diagonal=1)) # This critical for TREE
            A_0_triu = A_0_triu * flags_mask
        else:
            A_0_triu = torch.triu(A_0_dist, diagonal=1) # This critical for EGO
        A_0 = A_0_triu + A_0_triu.transpose(-1, -2)

        score_raw = self.legendre_score(A_0, A_t_dist, t).float()

        score_triu = torch.triu(score_raw, diagonal=1)
        score = score_triu + score_triu.transpose(-1, -2)
        return score * flags_mask
