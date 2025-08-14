import torch

class NodeFeatureAugmentor:
    def __init__(self, features=None, k_eig=4):
        self.features = features or ["degree", "clustering", "cycles3", "cycles4", "eigenvectors"]
        self.k_eig = k_eig

    def augment(self, x, adj):
        B, N, _ = x.shape
        feat_list = []

        if "degree" in self.features:
            degree = adj.sum(dim=-1, keepdim=True)  # [B, N, 1]
            degree /= degree.max().clamp(min=1e-6)
            feat_list.append(degree)

        if "clustering" in self.features or "cycles3" in self.features:
            A2 = torch.bmm(adj, adj)
            A3 = torch.bmm(A2, adj)
            diag_A3 = torch.diagonal(A3, dim1=1, dim2=2)

            if "clustering" in self.features:
                deg = adj.sum(dim=-1)
                denom = deg * (deg - 1)
                denom = denom.clamp(min=1e-6)
                clustering = diag_A3 / denom
                feat_list.append(clustering.unsqueeze(-1))

            if "cycles3" in self.features:
                cycles3 = diag_A3 / diag_A3.max().clamp(min=1e-6)
                feat_list.append(cycles3.unsqueeze(-1))

        if "cycles4" in self.features:
            A2 = torch.bmm(adj, adj)
            A4 = torch.bmm(A2, A2)
            diag_A4 = torch.diagonal(A4, dim1=1, dim2=2)
            cycles4 = diag_A4 - 2 * adj.sum(dim=-1)
            cycles4 = cycles4 / cycles4.max().clamp(min=1e-6)
            feat_list.append(cycles4.unsqueeze(-1))

        if "eigenvectors" in self.features:
            eigvecs_list = []
            for b in range(B):
                A = adj[b]
                D = torch.diag(A.sum(dim=-1))
                L = D - A
                try:
                    eigvals, eigvecs = torch.linalg.eigh(L)  # [N], [N, N]
                    eigvecs = eigvecs[:, :self.k_eig]  # [N, k]
                except RuntimeError:
                    eigvecs = torch.zeros(N, self.k_eig, device=adj.device, dtype=adj.dtype)
                eigvecs_list.append(eigvecs)
            eigvecs_stack = torch.stack(eigvecs_list, dim=0)  # [B, N, k]
            feat_list.append(eigvecs_stack)

        if feat_list:
            x_aug = torch.cat([x] + feat_list, dim=-1)
        else:
            x_aug = x

        return x_aug
