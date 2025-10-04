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
            sample_target=True
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
    
    @staticmethod
    def _t_min_from_order(K: int, tau: float = 1e-6) -> float:
        # ensures tail term exp(-t * n(n+1)/2) <= tau at n=K-1
        import math
        if K <= 1: return 0.0
        return 2.0 * math.log(1.0 / tau) / ( (K-1) * K )

    def _clenshaw_legendre(self, x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """
        S(x) = sum_{n=0}^{K-1} coeffs[..., n] * P_n(x) via Clenshaw for Legendre.
        After the loop: return S = c0 + x*b1 - b2  (not b0).
        """
        x = x.to(torch.float64)
        c = coeffs.to(torch.float64)
        *batch, K = c.shape
        if K == 0:
            return torch.zeros_like(x, dtype=torch.float64)

        b_kp1 = torch.zeros_like(x, dtype=torch.float64)  # b_{k+1}
        b_kp2 = torch.zeros_like(x, dtype=torch.float64)  # b_{k+2}
        # k runs from K-1 down to 1
        for k in range(K - 1, 0, -1):
            kf = float(k)
            a = (2.0 * kf + 1.0) / (kf + 1.0)
            b = (kf + 1.0) / (kf + 2.0) if (k + 2) <= K else 0.0
            b_k = c[..., k] + a * x * b_kp1 - b * b_kp2
            b_kp2, b_kp1 = b_kp1, b_k
        # final combine with c0:
        S = c[..., 0] + x * b_kp1 - b_kp2
        return S

    def legendre_score_clenshaw(self, adj_0, adj, t):
        # map to [-1,1]
        x0 = (2.0 * adj_0 - 1.0)
        xt = (2.0 * adj    - 1.0)

        B = adj.shape[0]
        K = int(self.order)

        # time weights: match your original ( (2n+1) * exp(-t * λ_n) )
        n = torch.arange(K, dtype=adj.dtype, device=adj.device)                      # [K]
        lambdas = 0.5 * (n * (n + 1))                                               # [K]
        logw = torch.log(2.0 * n + 1.0)                                             # [K]
        decay = torch.exp(-t[:, None, None, None] * lambdas + logw)                 # [B,1,1,K]

        # P_n(x0) via your recurrence
        P_x0, _ = self.legendre_poly_and_derivative(x0)                             # [B,N,N,K]
        c = (decay * P_x0).to(torch.float64)                                        # [B,N,N,K]

        # Denominator: sum c_n P_n(xt)
        den = self._clenshaw_legendre(xt, c)                                        # [B,N,N], f64

        # Numerator using identity: (1 - x^2) P_n' = n (P_{n-1} - x P_n)
        n64 = n.to(torch.float64).view(1,1,1,K)
        nc = n64 * c                                                                 # [B,N,N,K]
        Sx = self._clenshaw_legendre(xt, nc)                                         # Σ n c_n P_n(xt)

        if K >= 2:
            d = torch.zeros_like(c)
            d[..., :-1] = n64[..., 1:] * c[..., 1:]                                  # (m+1) c_{m+1}
            Sshift = self._clenshaw_legendre(xt, d)                                   # Σ (m+1)c_{m+1} P_m(xt)
        else:
            Sshift = torch.zeros_like(Sx)

        xt64 = xt.to(torch.float64)
        one_minus_x2 = (1.0 - xt64.pow(2)).clamp_min(1e-12)
        # derivative wrt xt:
        dlog_den_dxt = (Sshift - xt64 * Sx) / one_minus_x2
        # chain rule d/d(adj) = 2 * d/d(xt)
        score = (2.0 * dlog_den_dxt / den.clamp_min(float(self.eps))).to(adj.dtype)
        return score

    def legendre_score(self, adj_0, adj, t):
        x0 = 2.0 * adj_0 - 1.0  # [B, N, N]
        xt = 2.0 * adj   - 1.0  # [B, N, N]

        P_x0, _     = self.legendre_poly_and_derivative(x0)     # [B, N, N, order]
        P_xt, dP_xt = self.legendre_poly_and_derivative(xt)     # [B, N, N, order]

        n = torch.arange(self.order, dtype=adj.dtype, device=adj.device)
        lambdas = 0.5 * (n * (n + 1))[None, None, None, :]               # [1, 1, 1, order]
        logdn = - torch.log(2 * n + 1)
        decay = torch.exp(-t[:, None, None, None] * lambdas - logdn)

        density = (decay * P_xt * P_x0).to(torch.float64).sum(dim=-1)                # [B, N, N]
        grad_xt = 2.0 * (decay * dP_xt * P_x0).to(torch.float64).sum(dim=-1)         # [B, N, N]

        score = grad_xt / density.clamp_min(self.eps)
        return score.to(adj.dtype)
    
    def legendre_score_autograd(self, adj_0, adj, t):
        with torch.enable_grad():
            x0 = 2.0 * adj_0 - 1.0
            xt = (2.0 * adj   - 1.0).detach().requires_grad_(True)

            P_x0, _ = self.legendre_poly_and_derivative(x0)      # adj.dtype
            P_xt, _ = self.legendre_poly_and_derivative(xt)      # adj.dtype

            n = torch.arange(self.order, dtype=adj.dtype, device=adj.device)
            lambdas = 0.5 * (n * (n + 1))[None, None, None, :]
            logdn = -torch.log(2 * n + 1)
            decay = torch.exp(-t[:, None, None, None] * lambdas - logdn)     # adj.dtype

            density = (decay * P_xt * P_x0).to(torch.float64).sum(dim=-1)
            dden_dx = torch.autograd.grad(density.sum(), xt, retain_graph=False, create_graph=False)[0]
            score   = 2.0 * dden_dx / density.clamp_min(self.eps)
            #log_density = torch.log(density)                                    # [B,N,N]
            #score = 2.0 * torch.autograd.grad(log_density.sum(), xt, retain_graph=False, create_graph=False)[0]  # [B,N,N]
        return score.to(adj.dtype)

    def compute_score(self, A_t_dist, flags, t):
        #if t[0].item() < 0.2:
        #    self.order = 50
        assert_symmetric_and_masked(A_t_dist, flags)
        flags_mask = (flags[:, :, None] * flags[:, None, :]).float()
        A_t_dist = A_t_dist * flags_mask

        A_t_triu = torch.triu(A_t_dist, diagonal=1)
        A_t_triu_sample = torch.bernoulli(A_t_triu)

        A_t = A_t_triu + A_t_triu.transpose(-1, -2)
        A_t_sample = A_t_triu_sample + A_t_triu_sample.transpose(-1, -2)

        assert_symmetric_and_masked(A_t, flags)
        E_t = torch.cat([(1 - A_t).unsqueeze(-1), A_t.unsqueeze(-1)], dim=-1).float()
        E_t_sample = torch.cat([(1 - A_t_sample).unsqueeze(-1), A_t_sample.unsqueeze(-1)], dim=-1).float()
        extra_pred = self.feature_extractor(E_t_sample, flags)
        y = torch.cat((extra_pred.y.float(), t.unsqueeze(1)), dim=1).float()
        # assert_symmetric_and_masked_E(extra_pred.E, flags) # This is not symmetric

        pred = self.model(extra_pred.X.float(), extra_pred.E.float(), y, flags)
        #pred = self.model(extra_pred.X.float(), E_t.float(), y, flags)
        #E_inp = torch.cat([extra_pred.E.float(), E_t.float()], dim=-1) 
        #pred = self.model(extra_pred.X.float(), E_inp, y, flags)
        assert_symmetric_and_masked_E(pred.E, flags)
        A_0_dist = F.softmax(pred.E, dim=-1)[..., 1:].sum(dim=-1).float()
        A_0_dist = A_0_dist * flags_mask

        if self.sample_target:
            A_0_triu = torch.bernoulli(torch.triu(A_0_dist, diagonal=1)) # This critical for TREE
            A_0_triu = A_0_triu * flags_mask
        else:
            A_0_triu = torch.triu(A_0_dist, diagonal=1) # This critical for EGO
        A_0 = A_0_triu + A_0_triu.transpose(-1, -2)
        #assert_symmetric_and_masked(A_0, flags)

        score_raw_1 = self.legendre_score(A_0, A_t_dist, t).float()
        #score_raw_2 = self.legendre_score_autograd(A_0, A_t_dist, t).float()
        #score_raw_3 = self.legendre_score_clenshaw(A_0, A_t_dist, t).float()

        score_triu = torch.triu(score_raw_1, diagonal=1)
        score = score_triu + score_triu.transpose(-1, -2)
        return score * flags_mask
