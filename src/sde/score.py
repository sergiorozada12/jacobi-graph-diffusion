import torch

from src.features.extra_features import NodeFeatureAugmentor


class JacobiScore:
    def __init__(self, model, node_features, k_eig, order = 10, eps = 1e-12):
        self.order = order
        self.eps = eps
        self.model = model
        self.feature_extractor = NodeFeatureAugmentor(features=node_features, k_eig=k_eig)

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

    def compute_score(self, x, adj, flags, t):
        adj_inp = torch.bernoulli(adj)
        x_aug = self.feature_extractor.augment(x, adj_inp)
        _, adj_0 = self.model(x_aug, adj_inp, flags)
        score = self.legendre_score(adj_0, adj, t).float()
        return score
