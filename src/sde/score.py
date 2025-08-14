import torch

from src.features.extra_features import ExtraFeatures
import torch.nn.functional as F

class JacobiScore:
    def __init__(self, model, extra_features, rrwp_steps, max_n_nodes, order=10, eps=1e-12):
        self.order = order
        self.eps = eps
        self.model = model
        self.feature_extractor = ExtraFeatures(
            extra_features_type=extra_features,
            rrwp_steps=rrwp_steps,
            max_n_nodes=max_n_nodes,
        )

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
        x0 = 2.0 * adj_0 - 1.0  # [B, N, N]
        xt = 2.0 * adj   - 1.0  # [B, N, N]

        P_x0, _     = self.legendre_poly_and_derivative(x0)     # [B, N, N, order]
        P_xt, dP_xt = self.legendre_poly_and_derivative(xt)     # [B, N, N, order]

        n = torch.arange(self.order, dtype=torch.double, device=adj.device)
        lambdas = 0.5 * (n * (n + 1))[None, None, None, :]               # [1, 1, 1, order]
        logdn = - torch.log(2 * n + 1)
        decay = torch.exp(-t[:, None, None, None] * lambdas - logdn)

        density = (decay * P_xt * P_x0).sum(dim=-1)                # [B, N, N]
        grad_xt = 2.0 * (decay * dP_xt * P_x0).sum(dim=-1)         # [B, N, N]

        score = grad_xt / density.clamp_min(self.eps)
        return score

    def compute_score(self, A_t_dist, flags, t):
        A_t_triu = torch.bernoulli(torch.triu(A_t_dist, diagonal=1))
        A_t = A_t_triu + A_t_triu.transpose(-1, -2)
        E_t = torch.cat([(1 - A_t).unsqueeze(-1), A_t.unsqueeze(-1)], dim=-1).float()
        extra_pred = self.feature_extractor(E_t, flags)
        y = torch.cat((extra_pred.y.float(), t.unsqueeze(1)), dim=1).float()

        pred = self.model(extra_pred.X.float(), extra_pred.E.float(), y, flags)
        A_0_dist = F.softmax(pred.E, dim=-1)[..., 1:].sum(dim=-1).float()

        flags_mask = (flags[:, :, None] * flags[:, None, :]).float()
        A_0_dist = A_0_dist * flags_mask

        A_0_triu = torch.bernoulli(torch.triu(A_0_dist, diagonal=1))
        A_0 = A_0_triu + A_0_triu.transpose(-1, -2)

        score = self.legendre_score(A_0, A_t_dist, t).float()
        return score
